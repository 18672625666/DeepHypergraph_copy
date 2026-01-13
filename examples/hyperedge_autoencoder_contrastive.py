"""
Hyperedge Autoencoder with Contrastive Learning

This script implements a Hyperedge Autoencoder using HGNN/HGNNP encoders and decoders
to optimize initial hyperedge features through contrastive learning.

Key Features:
1. Encoder-Decoder architecture using HGNN/HGNNP layers
2. Node feature masking for data augmentation
3. Hyperedge-level contrastive learning with InfoNCE loss
4. Training until loss stabilization
5. Extraction of optimized hyperedge representations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Tuple, Optional
import time
from copy import deepcopy

import dhg
from dhg import Hypergraph
from dhg.nn import HGNNConv, HGNNPConv
from dhg.random import set_seed


class HyperedgeEncoder(nn.Module):
    """
    Encoder module using HGNN or HGNNP convolution layers.
    Maps node features to hyperedge embeddings via v2e (vertex to hyperedge) propagation.
    
    Args:
        in_dim: Input feature dimension
        hidden_dim: Hidden/output feature dimension
        use_hgnnp: If True, uses HGNNP convolution, otherwise uses HGNN
        use_bn: Whether to use batch normalization
        drop_rate: Dropout rate
    """
    
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        use_hgnnp: bool = False,
        use_bn: bool = False,
        drop_rate: float = 0.5,
    ):
        super().__init__()
        ConvLayer = HGNNPConv if use_hgnnp else HGNNConv
        self.conv = ConvLayer(in_dim, hidden_dim, use_bn=use_bn, drop_rate=drop_rate, is_last=False)
        
    def forward(self, X: torch.Tensor, hg: Hypergraph) -> torch.Tensor:
        """
        Forward pass: node features -> hyperedge embeddings
        
        Args:
            X: Node feature matrix of size (num_nodes, in_dim)
            hg: Hypergraph structure
            
        Returns:
            Hyperedge embeddings of size (num_edges, hidden_dim)
        """
        # Apply convolution on node features
        X = self.conv(X, hg)
        # Aggregate node features to hyperedge embeddings
        E = hg.v2e(X, aggr="mean")
        return E


class HyperedgeDecoder(nn.Module):
    """
    Decoder module using HGNN or HGNNP convolution layers.
    Maps hyperedge embeddings back to node features via e2v (hyperedge to vertex) propagation.
    
    Args:
        hidden_dim: Hidden/input feature dimension
        out_dim: Output feature dimension (should match original node feature dimension)
        use_hgnnp: If True, uses HGNNP convolution, otherwise uses HGNN
        use_bn: Whether to use batch normalization
        drop_rate: Dropout rate
    """
    
    def __init__(
        self,
        hidden_dim: int,
        out_dim: int,
        use_hgnnp: bool = False,
        use_bn: bool = False,
        drop_rate: float = 0.5,
    ):
        super().__init__()
        ConvLayer = HGNNPConv if use_hgnnp else HGNNConv
        self.conv = ConvLayer(hidden_dim, out_dim, use_bn=use_bn, drop_rate=drop_rate, is_last=True)
        
    def forward(self, E: torch.Tensor, hg: Hypergraph) -> torch.Tensor:
        """
        Forward pass: hyperedge embeddings -> node features
        
        Args:
            E: Hyperedge embeddings of size (num_edges, hidden_dim)
            hg: Hypergraph structure
            
        Returns:
            Reconstructed node features of size (num_nodes, out_dim)
        """
        # Propagate hyperedge embeddings to nodes
        X = hg.e2v(E, aggr="mean")
        # Apply convolution on reconstructed node features
        X = self.conv(X, hg)
        return X


class HyperedgeAutoencoder(nn.Module):
    """
    Complete Hyperedge Autoencoder with encoder and decoder.
    
    Args:
        in_dim: Input feature dimension
        hidden_dim: Hidden feature dimension for hyperedge embeddings
        use_hgnnp: If True, uses HGNNP convolution, otherwise uses HGNN
        use_bn: Whether to use batch normalization
        drop_rate: Dropout rate
    """
    
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        use_hgnnp: bool = False,
        use_bn: bool = False,
        drop_rate: float = 0.3,
    ):
        super().__init__()
        self.encoder = HyperedgeEncoder(in_dim, hidden_dim, use_hgnnp, use_bn, drop_rate)
        self.decoder = HyperedgeDecoder(hidden_dim, in_dim, use_hgnnp, use_bn, drop_rate)
        
    def forward(self, X: torch.Tensor, hg: Hypergraph) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through encoder and decoder.
        
        Args:
            X: Node feature matrix of size (num_nodes, in_dim)
            hg: Hypergraph structure
            
        Returns:
            E: Hyperedge embeddings of size (num_edges, hidden_dim)
            X_recon: Reconstructed node features of size (num_nodes, in_dim)
        """
        E = self.encoder(X, hg)
        X_recon = self.decoder(E, hg)
        return E, X_recon
    
    def encode(self, X: torch.Tensor, hg: Hypergraph) -> torch.Tensor:
        """
        Encode node features to hyperedge embeddings.
        
        Args:
            X: Node feature matrix
            hg: Hypergraph structure
            
        Returns:
            Hyperedge embeddings
        """
        return self.encoder(X, hg)


def mask_node_features(X: torch.Tensor, mask_rate: float = 0.3) -> torch.Tensor:
    """
    Mask node features for data augmentation by randomly masking entire nodes.
    
    This function implements node-level masking where entire node feature vectors
    are set to zero. This is a common augmentation strategy in graph contrastive
    learning that forces the model to be robust to node dropout.
    
    Note: Alternative strategies include:
    - Feature-level masking: mask individual features within nodes (X_aug[mask_nodes, mask_features] = 0)
    - Gaussian noise: add random noise instead of zeroing out
    - Edge dropout: remove hyperedge connections (requires hypergraph modification)
    
    Args:
        X: Node feature matrix of size (num_nodes, feature_dim)
        mask_rate: Proportion of nodes to mask (not individual features)
        
    Returns:
        Masked node feature matrix with the same shape as input
    """
    X_aug = X.clone()
    # Node-level masking: randomly select nodes and zero out all their features
    mask = torch.rand(X.shape[0]) < mask_rate
    X_aug[mask] = 0.0
    return X_aug


def info_nce_loss(
    anchor_emb: torch.Tensor,
    augmented_emb: torch.Tensor,
    temperature: float = 0.5
) -> torch.Tensor:
    """
    Compute InfoNCE (contrastive) loss for hyperedge embeddings.
    
    Positive pairs: Same hyperedge from anchor and augmented graphs
    Negative pairs: Different hyperedges
    
    Args:
        anchor_emb: Hyperedge embeddings from anchor graph (num_edges, hidden_dim)
        augmented_emb: Hyperedge embeddings from augmented graph (num_edges, hidden_dim)
        temperature: Temperature parameter for softmax
        
    Returns:
        InfoNCE loss value
    """
    # Normalize embeddings
    anchor_emb = F.normalize(anchor_emb, dim=1)
    augmented_emb = F.normalize(augmented_emb, dim=1)
    
    # Compute similarity matrix: (num_edges, num_edges)
    similarity_matrix = torch.matmul(anchor_emb, augmented_emb.T) / temperature
    
    # Labels: diagonal elements are positive pairs (same hyperedge)
    batch_size = anchor_emb.shape[0]
    labels = torch.arange(batch_size, device=anchor_emb.device)
    
    # Compute cross-entropy loss (InfoNCE)
    # For each anchor, the positive is its corresponding augmented version
    loss = F.cross_entropy(similarity_matrix, labels)
    
    return loss


def reconstruction_loss(X_recon: torch.Tensor, X_target: torch.Tensor) -> torch.Tensor:
    """
    Compute reconstruction loss (MSE) between reconstructed and target node features.
    
    Args:
        X_recon: Reconstructed node features
        X_target: Target node features
        
    Returns:
        MSE loss value
    """
    return F.mse_loss(X_recon, X_target)


def train_epoch(
    model: HyperedgeAutoencoder,
    X: torch.Tensor,
    hg: Hypergraph,
    optimizer: optim.Optimizer,
    mask_rate: float = 0.3,
    temperature: float = 0.5,
    recon_weight: float = 1.0,
    contrast_weight: float = 1.0,
) -> Tuple[float, float, float]:
    """
    Train the model for one epoch.
    
    Args:
        model: Hyperedge autoencoder model
        X: Node feature matrix
        hg: Hypergraph structure
        optimizer: Optimizer
        mask_rate: Node feature masking rate
        temperature: Temperature for InfoNCE loss
        recon_weight: Weight for reconstruction loss
        contrast_weight: Weight for contrastive loss
        
    Returns:
        total_loss: Combined loss value
        recon_loss_val: Reconstruction loss value
        contrast_loss_val: Contrastive loss value
    """
    model.train()
    optimizer.zero_grad()
    
    # Create augmented view by masking node features
    X_aug = mask_node_features(X, mask_rate)
    
    # Forward pass on anchor (original) graph
    E_anchor, X_recon_anchor = model(X, hg)
    
    # Forward pass on augmented graph
    E_aug, X_recon_aug = model(X_aug, hg)
    
    # Compute reconstruction loss (reconstruct original features)
    recon_loss_anchor = reconstruction_loss(X_recon_anchor, X)
    recon_loss_aug = reconstruction_loss(X_recon_aug, X)
    recon_loss_val = (recon_loss_anchor + recon_loss_aug) / 2.0
    
    # Compute contrastive loss (InfoNCE) at hyperedge level
    contrast_loss_val = info_nce_loss(E_anchor, E_aug, temperature)
    
    # Combined loss
    total_loss = recon_weight * recon_loss_val + contrast_weight * contrast_loss_val
    
    # Backward pass
    total_loss.backward()
    optimizer.step()
    
    return total_loss.item(), recon_loss_val.item(), contrast_loss_val.item()


def check_convergence(
    loss_history: list,
    window: int = 10,
    threshold: float = 1e-4
) -> bool:
    """
    Check if training has converged by examining recent loss history.
    
    Args:
        loss_history: List of recent loss values
        window: Number of recent epochs to consider
        threshold: Convergence threshold (standard deviation of recent losses)
        
    Returns:
        True if converged, False otherwise
    """
    if len(loss_history) < window:
        return False
    
    recent_losses = loss_history[-window:]
    std = torch.std(torch.tensor(recent_losses)).item()
    return std < threshold


def train_hyperedge_autoencoder(
    X: torch.Tensor,
    hg: Hypergraph,
    in_dim: int,
    hidden_dim: int = 64,
    use_hgnnp: bool = False,
    use_bn: bool = False,
    drop_rate: float = 0.3,
    lr: float = 0.001,
    weight_decay: float = 5e-4,
    mask_rate: float = 0.3,
    temperature: float = 0.5,
    recon_weight: float = 1.0,
    contrast_weight: float = 1.0,
    max_epochs: int = 500,
    patience: int = 50,
    convergence_window: int = 10,
    convergence_threshold: float = 1e-4,
    device: torch.device = None,
    verbose: bool = True,
) -> Tuple[HyperedgeAutoencoder, torch.Tensor, dict]:
    """
    Train the Hyperedge Autoencoder with contrastive learning.
    
    Args:
        X: Node feature matrix
        hg: Hypergraph structure
        in_dim: Input feature dimension
        hidden_dim: Hidden dimension for hyperedge embeddings
        use_hgnnp: Whether to use HGNNP (True) or HGNN (False)
        use_bn: Whether to use batch normalization
        drop_rate: Dropout rate
        lr: Learning rate
        weight_decay: Weight decay for optimizer
        mask_rate: Node feature masking rate for augmentation
        temperature: Temperature for InfoNCE loss
        recon_weight: Weight for reconstruction loss
        contrast_weight: Weight for contrastive loss
        max_epochs: Maximum number of training epochs
        patience: Early stopping patience
        convergence_window: Window size for convergence checking
        convergence_threshold: Threshold for convergence
        device: Device to train on
        verbose: Whether to print training progress
        
    Returns:
        model: Trained Hyperedge Autoencoder
        hyperedge_embeddings: Final anchor hyperedge representations
        history: Training history dictionary
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Move data to device
    X = X.to(device)
    hg = hg.to(device)
    
    # Initialize model
    model = HyperedgeAutoencoder(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        use_hgnnp=use_hgnnp,
        use_bn=use_bn,
        drop_rate=drop_rate,
    ).to(device)
    
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Training history
    history = {
        "total_loss": [],
        "recon_loss": [],
        "contrast_loss": [],
    }
    
    best_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0
    
    if verbose:
        print(f"Training Hyperedge Autoencoder on {device}")
        print(f"Model: {'HGNNP' if use_hgnnp else 'HGNN'}")
        print(f"Architecture: {in_dim} -> {hidden_dim} -> {in_dim}")
        print(f"Num nodes: {X.shape[0]}, Num hyperedges: {hg.num_e}")
        print("-" * 80)
    
    # Training loop
    for epoch in range(max_epochs):
        start_time = time.time()
        
        # Train one epoch
        total_loss, recon_loss_val, contrast_loss_val = train_epoch(
            model=model,
            X=X,
            hg=hg,
            optimizer=optimizer,
            mask_rate=mask_rate,
            temperature=temperature,
            recon_weight=recon_weight,
            contrast_weight=contrast_weight,
        )
        
        # Record history
        history["total_loss"].append(total_loss)
        history["recon_loss"].append(recon_loss_val)
        history["contrast_loss"].append(contrast_loss_val)
        
        # Check for improvement
        if total_loss < best_loss:
            best_loss = total_loss
            best_state = deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        
        # Print progress
        if verbose and (epoch % 10 == 0 or epoch == max_epochs - 1):
            elapsed = time.time() - start_time
            print(
                f"Epoch {epoch:3d} | "
                f"Loss: {total_loss:.4f} | "
                f"Recon: {recon_loss_val:.4f} | "
                f"Contrast: {contrast_loss_val:.4f} | "
                f"Time: {elapsed:.3f}s"
            )
        
        # Check convergence
        if check_convergence(history["total_loss"], convergence_window, convergence_threshold):
            if verbose:
                print(f"\nConverged at epoch {epoch}!")
            break
        
        # Early stopping
        if epochs_without_improvement >= patience:
            if verbose:
                print(f"\nEarly stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break
    
    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)
    
    # Extract final hyperedge embeddings
    model.eval()
    with torch.no_grad():
        hyperedge_embeddings = model.encode(X, hg)
    
    if verbose:
        print("-" * 80)
        print(f"Training completed!")
        print(f"Best loss: {best_loss:.4f}")
        print(f"Final hyperedge embeddings shape: {hyperedge_embeddings.shape}")
    
    return model, hyperedge_embeddings, history


def main():
    """
    Example usage of the Hyperedge Autoencoder with contrastive learning.
    """
    # Set random seed for reproducibility
    set_seed(2024)
    
    # Create example hypergraph
    print("Creating example hypergraph...")
    num_nodes = 100
    num_features = 32
    
    # Generate random node features
    X = torch.randn(num_nodes, num_features)
    
    # Create hypergraph with random hyperedges
    hyperedge_list = [
        [0, 1, 2, 3],
        [1, 4, 5],
        [2, 3, 6, 7, 8],
        [5, 9, 10, 11],
        [7, 12, 13],
        [10, 14, 15, 16, 17],
        [15, 18, 19],
        [20, 21, 22, 23, 24],
    ]
    
    # Add more random hyperedges
    for i in range(20):
        size = torch.randint(3, 8, (1,)).item()
        edge = torch.randint(0, num_nodes, (size,)).tolist()
        hyperedge_list.append(edge)
    
    hg = Hypergraph(num_nodes, hyperedge_list)
    
    print(f"Hypergraph created: {num_nodes} nodes, {hg.num_e} hyperedges\n")
    
    # Configuration
    use_hgnnp = False  # Set to True to use HGNNP instead of HGNN
    hidden_dim = 64
    
    # Train Hyperedge Autoencoder
    model, hyperedge_embeddings, history = train_hyperedge_autoencoder(
        X=X,
        hg=hg,
        in_dim=num_features,
        hidden_dim=hidden_dim,
        use_hgnnp=use_hgnnp,
        use_bn=False,
        drop_rate=0.3,
        lr=0.001,
        weight_decay=5e-4,
        mask_rate=0.3,
        temperature=0.5,
        recon_weight=1.0,
        contrast_weight=1.0,
        max_epochs=200,
        patience=50,
        convergence_window=10,
        convergence_threshold=1e-4,
        verbose=True,
    )
    
    print("\n" + "=" * 80)
    print("Training Summary:")
    print("=" * 80)
    print(f"Model type: {'HGNNP' if use_hgnnp else 'HGNN'}")
    print(f"Input dimension: {num_features}")
    print(f"Hidden dimension: {hidden_dim}")
    print(f"Number of training epochs: {len(history['total_loss'])}")
    print(f"Final total loss: {history['total_loss'][-1]:.4f}")
    print(f"Final reconstruction loss: {history['recon_loss'][-1]:.4f}")
    print(f"Final contrastive loss: {history['contrast_loss'][-1]:.4f}")
    print(f"Hyperedge embeddings shape: {hyperedge_embeddings.shape}")
    print("=" * 80)
    
    # Save model (optional)
    # torch.save(model.state_dict(), "hyperedge_autoencoder.pth")
    # torch.save(hyperedge_embeddings, "hyperedge_embeddings.pth")
    
    return model, hyperedge_embeddings, history


if __name__ == "__main__":
    main()
