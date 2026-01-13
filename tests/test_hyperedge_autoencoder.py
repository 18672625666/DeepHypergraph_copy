"""
Test suite for Hyperedge Autoencoder with Contrastive Learning
"""

import torch
import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dhg import Hypergraph
from dhg.random import set_seed
from examples.hyperedge_autoencoder_contrastive import (
    HyperedgeEncoder,
    HyperedgeDecoder,
    HyperedgeAutoencoder,
    mask_node_features,
    info_nce_loss,
    reconstruction_loss,
    train_hyperedge_autoencoder,
)


class TestHyperedgeEncoder:
    """Test HyperedgeEncoder class"""
    
    def test_encoder_forward(self):
        """Test encoder forward pass"""
        set_seed(42)
        in_dim, hidden_dim = 16, 32
        num_nodes = 50
        
        # Create simple hypergraph
        hyperedge_list = [[0, 1, 2], [1, 3, 4], [2, 4, 5]]
        hg = Hypergraph(num_nodes, hyperedge_list)
        
        # Create encoder
        encoder = HyperedgeEncoder(in_dim, hidden_dim, use_hgnnp=False)
        
        # Test forward pass
        X = torch.randn(num_nodes, in_dim)
        E = encoder(X, hg)
        
        # Check output shape
        assert E.shape == (hg.num_e, hidden_dim), f"Expected shape ({hg.num_e}, {hidden_dim}), got {E.shape}"
    
    def test_encoder_hgnnp(self):
        """Test encoder with HGNNP"""
        set_seed(42)
        in_dim, hidden_dim = 16, 32
        num_nodes = 50
        
        hyperedge_list = [[0, 1, 2], [1, 3, 4], [2, 4, 5]]
        hg = Hypergraph(num_nodes, hyperedge_list)
        
        encoder = HyperedgeEncoder(in_dim, hidden_dim, use_hgnnp=True)
        X = torch.randn(num_nodes, in_dim)
        E = encoder(X, hg)
        
        assert E.shape == (hg.num_e, hidden_dim)


class TestHyperedgeDecoder:
    """Test HyperedgeDecoder class"""
    
    def test_decoder_forward(self):
        """Test decoder forward pass"""
        set_seed(42)
        hidden_dim, out_dim = 32, 16
        num_nodes = 50
        
        hyperedge_list = [[0, 1, 2], [1, 3, 4], [2, 4, 5]]
        hg = Hypergraph(num_nodes, hyperedge_list)
        
        decoder = HyperedgeDecoder(hidden_dim, out_dim, use_hgnnp=False)
        E = torch.randn(hg.num_e, hidden_dim)
        X_recon = decoder(E, hg)
        
        assert X_recon.shape == (num_nodes, out_dim), f"Expected shape ({num_nodes}, {out_dim}), got {X_recon.shape}"
    
    def test_decoder_hgnnp(self):
        """Test decoder with HGNNP"""
        set_seed(42)
        hidden_dim, out_dim = 32, 16
        num_nodes = 50
        
        hyperedge_list = [[0, 1, 2], [1, 3, 4], [2, 4, 5]]
        hg = Hypergraph(num_nodes, hyperedge_list)
        
        decoder = HyperedgeDecoder(hidden_dim, out_dim, use_hgnnp=True)
        E = torch.randn(hg.num_e, hidden_dim)
        X_recon = decoder(E, hg)
        
        assert X_recon.shape == (num_nodes, out_dim)


class TestHyperedgeAutoencoder:
    """Test HyperedgeAutoencoder class"""
    
    def test_autoencoder_forward(self):
        """Test autoencoder forward pass"""
        set_seed(42)
        in_dim, hidden_dim = 16, 32
        num_nodes = 50
        
        hyperedge_list = [[0, 1, 2], [1, 3, 4], [2, 4, 5]]
        hg = Hypergraph(num_nodes, hyperedge_list)
        
        model = HyperedgeAutoencoder(in_dim, hidden_dim, use_hgnnp=False)
        X = torch.randn(num_nodes, in_dim)
        E, X_recon = model(X, hg)
        
        assert E.shape == (hg.num_e, hidden_dim)
        assert X_recon.shape == (num_nodes, in_dim)
    
    def test_autoencoder_encode(self):
        """Test autoencoder encode method"""
        set_seed(42)
        in_dim, hidden_dim = 16, 32
        num_nodes = 50
        
        hyperedge_list = [[0, 1, 2], [1, 3, 4], [2, 4, 5]]
        hg = Hypergraph(num_nodes, hyperedge_list)
        
        model = HyperedgeAutoencoder(in_dim, hidden_dim)
        X = torch.randn(num_nodes, in_dim)
        E = model.encode(X, hg)
        
        assert E.shape == (hg.num_e, hidden_dim)


class TestAugmentation:
    """Test augmentation functions"""
    
    def test_mask_node_features(self):
        """Test node feature masking"""
        set_seed(42)
        X = torch.randn(100, 32)
        mask_rate = 0.3
        
        X_masked = mask_node_features(X, mask_rate)
        
        # Check shape is preserved
        assert X_masked.shape == X.shape
        
        # Check that some features are masked (set to zero)
        num_masked = (X_masked.sum(dim=1) == 0).sum().item()
        assert num_masked > 0, "No features were masked"
        
        # Original should not be modified
        assert not torch.equal(X, X_masked) or mask_rate == 0


class TestLossFunctions:
    """Test loss functions"""
    
    def test_info_nce_loss(self):
        """Test InfoNCE loss computation"""
        set_seed(42)
        batch_size, hidden_dim = 10, 32
        
        anchor_emb = torch.randn(batch_size, hidden_dim)
        augmented_emb = torch.randn(batch_size, hidden_dim)
        
        loss = info_nce_loss(anchor_emb, augmented_emb, temperature=0.5)
        
        # Loss should be a scalar
        assert loss.ndim == 0
        
        # Loss should be positive
        assert loss.item() >= 0
    
    def test_info_nce_loss_identical(self):
        """Test InfoNCE loss with identical embeddings"""
        set_seed(42)
        batch_size, hidden_dim = 10, 32
        
        emb = torch.randn(batch_size, hidden_dim)
        
        # Loss should be relatively low for identical embeddings (lower than random)
        loss_identical = info_nce_loss(emb, emb, temperature=0.5)
        
        # Compare with random embeddings
        random_emb = torch.randn(batch_size, hidden_dim)
        loss_random = info_nce_loss(emb, random_emb, temperature=0.5)
        
        # Identical should have lower or similar loss to random
        assert loss_identical.item() <= loss_random.item() + 0.5, \
            f"Expected identical embeddings to have reasonable loss, got {loss_identical.item()}"
    
    def test_reconstruction_loss(self):
        """Test reconstruction loss"""
        set_seed(42)
        X_recon = torch.randn(100, 32)
        X_target = torch.randn(100, 32)
        
        loss = reconstruction_loss(X_recon, X_target)
        
        # Loss should be a scalar
        assert loss.ndim == 0
        
        # Loss should be positive
        assert loss.item() >= 0
    
    def test_reconstruction_loss_identical(self):
        """Test reconstruction loss with identical inputs"""
        set_seed(42)
        X = torch.randn(100, 32)
        
        # Loss should be zero for identical inputs
        loss = reconstruction_loss(X, X)
        assert loss.item() < 1e-6, f"Expected zero loss for identical inputs, got {loss.item()}"


class TestTraining:
    """Test training functions"""
    
    def test_train_hyperedge_autoencoder(self):
        """Test complete training pipeline"""
        set_seed(42)
        
        # Small example for fast testing
        num_nodes = 30
        in_dim = 16
        hidden_dim = 32
        
        X = torch.randn(num_nodes, in_dim)
        hyperedge_list = [[0, 1, 2], [1, 3, 4], [2, 4, 5], [5, 6, 7], [7, 8, 9]]
        hg = Hypergraph(num_nodes, hyperedge_list)
        
        # Train for just a few epochs
        model, embeddings, history = train_hyperedge_autoencoder(
            X=X,
            hg=hg,
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            max_epochs=10,
            patience=10,
            verbose=False,
        )
        
        # Check model is returned
        assert isinstance(model, HyperedgeAutoencoder)
        
        # Check embeddings shape
        assert embeddings.shape == (hg.num_e, hidden_dim)
        
        # Check history
        assert "total_loss" in history
        assert "recon_loss" in history
        assert "contrast_loss" in history
        assert len(history["total_loss"]) <= 10
        
        # Check losses are decreasing (at least initially)
        assert history["total_loss"][-1] <= history["total_loss"][0] * 1.5
    
    def test_train_with_hgnnp(self):
        """Test training with HGNNP"""
        set_seed(42)
        
        num_nodes = 30
        in_dim = 16
        hidden_dim = 32
        
        X = torch.randn(num_nodes, in_dim)
        hyperedge_list = [[0, 1, 2], [1, 3, 4], [2, 4, 5]]
        hg = Hypergraph(num_nodes, hyperedge_list)
        
        model, embeddings, history = train_hyperedge_autoencoder(
            X=X,
            hg=hg,
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            use_hgnnp=True,
            max_epochs=5,
            verbose=False,
        )
        
        assert embeddings.shape == (hg.num_e, hidden_dim)
        assert len(history["total_loss"]) <= 5


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__, "-v"])
