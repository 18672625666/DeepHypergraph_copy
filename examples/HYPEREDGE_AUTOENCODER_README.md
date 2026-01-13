# Hyperedge Autoencoder with Contrastive Learning

A complete implementation of a Hyperedge Autoencoder using HGNN/HGNNP encoders and decoders to optimize initial hyperedge features through contrastive learning.

## Overview

This implementation provides a deep learning framework for learning optimized hyperedge representations using:
- **Encoder-Decoder Architecture**: HGNN or HGNNP layers for encoding and decoding
- **Data Augmentation**: Node feature masking to create contrastive views
- **Contrastive Learning**: Hyperedge-level contrastive learning with InfoNCE loss
- **Training**: Automated training with loss stabilization detection and early stopping

## Key Features

### Architecture
- **Encoder**: Maps node features to hyperedge embeddings (in_dim → hidden_dim)
- **Decoder**: Reconstructs node features from hyperedge embeddings (hidden_dim → in_dim)
- Supports both HGNN and HGNNP convolution layers
- Configurable batch normalization and dropout

### Contrastive Learning
- **Positive Pairs**: Same hyperedge from anchor and augmented graphs
- **Negative Pairs**: Different hyperedges
- **InfoNCE Loss**: Standard contrastive loss for self-supervised learning
- **Augmentation**: Random node feature masking

### Training
- Combined reconstruction loss (MSE) and contrastive loss (InfoNCE)
- Automatic convergence detection
- Early stopping with patience
- Comprehensive training history tracking

## Installation

Ensure you have the DHG library and its dependencies installed:

```bash
pip install torch scipy numpy scikit-learn matplotlib
cd /path/to/DeepHypergraph_copy
pip install -e .
```

## Quick Start

### Basic Usage

```python
import torch
from dhg import Hypergraph
from dhg.random import set_seed
from examples.hyperedge_autoencoder_contrastive import train_hyperedge_autoencoder

# Set random seed for reproducibility
set_seed(2024)

# Create your hypergraph
num_nodes = 100
num_features = 32
X = torch.randn(num_nodes, num_features)

hyperedge_list = [
    [0, 1, 2, 3],
    [1, 4, 5],
    [2, 3, 6, 7, 8],
    # ... more hyperedges
]
hg = Hypergraph(num_nodes, hyperedge_list)

# Train the autoencoder
model, hyperedge_embeddings, history = train_hyperedge_autoencoder(
    X=X,
    hg=hg,
    in_dim=num_features,
    hidden_dim=64,
    use_hgnnp=False,  # Set to True to use HGNNP
    max_epochs=200,
    verbose=True,
)

# Use the optimized hyperedge embeddings
print(f"Hyperedge embeddings shape: {hyperedge_embeddings.shape}")
```

### Running the Example

```bash
cd /path/to/DeepHypergraph_copy
python examples/hyperedge_autoencoder_contrastive.py
```

### Advanced Configuration

```python
model, embeddings, history = train_hyperedge_autoencoder(
    X=X,
    hg=hg,
    in_dim=32,
    hidden_dim=64,
    use_hgnnp=False,          # Use HGNN (False) or HGNNP (True)
    use_bn=False,              # Batch normalization
    drop_rate=0.3,             # Dropout rate
    lr=0.001,                  # Learning rate
    weight_decay=5e-4,         # Weight decay
    mask_rate=0.3,             # Node feature masking rate
    temperature=0.5,           # Temperature for InfoNCE loss
    recon_weight=1.0,          # Reconstruction loss weight
    contrast_weight=1.0,       # Contrastive loss weight
    max_epochs=500,            # Maximum epochs
    patience=50,               # Early stopping patience
    convergence_window=10,     # Convergence check window
    convergence_threshold=1e-4, # Convergence threshold
    verbose=True,
)
```

## API Documentation

### Classes

#### `HyperedgeEncoder`
Encoder module using HGNN or HGNNP convolution layers.

**Parameters:**
- `in_dim` (int): Input feature dimension
- `hidden_dim` (int): Hidden/output feature dimension
- `use_hgnnp` (bool): If True, uses HGNNP; otherwise uses HGNN
- `use_bn` (bool): Whether to use batch normalization
- `drop_rate` (float): Dropout rate

#### `HyperedgeDecoder`
Decoder module using HGNN or HGNNP convolution layers.

**Parameters:**
- `hidden_dim` (int): Hidden/input feature dimension
- `out_dim` (int): Output feature dimension
- `use_hgnnp` (bool): If True, uses HGNNP; otherwise uses HGNN
- `use_bn` (bool): Whether to use batch normalization
- `drop_rate` (float): Dropout rate

#### `HyperedgeAutoencoder`
Complete autoencoder with encoder and decoder.

**Parameters:**
- `in_dim` (int): Input feature dimension
- `hidden_dim` (int): Hidden feature dimension
- `use_hgnnp` (bool): If True, uses HGNNP; otherwise uses HGNN
- `use_bn` (bool): Whether to use batch normalization
- `drop_rate` (float): Dropout rate

### Functions

#### `mask_node_features(X, mask_rate)`
Mask node features for data augmentation.

**Parameters:**
- `X` (torch.Tensor): Node feature matrix
- `mask_rate` (float): Proportion of features to mask

**Returns:**
- Masked node feature matrix

#### `info_nce_loss(anchor_emb, augmented_emb, temperature)`
Compute InfoNCE (contrastive) loss for hyperedge embeddings.

**Parameters:**
- `anchor_emb` (torch.Tensor): Hyperedge embeddings from anchor graph
- `augmented_emb` (torch.Tensor): Hyperedge embeddings from augmented graph
- `temperature` (float): Temperature parameter for softmax

**Returns:**
- InfoNCE loss value

#### `train_hyperedge_autoencoder(...)`
Train the Hyperedge Autoencoder with contrastive learning.

**Returns:**
- `model`: Trained Hyperedge Autoencoder
- `hyperedge_embeddings`: Final anchor hyperedge representations
- `history`: Training history dictionary

## Example Output

```
Training Hyperedge Autoencoder on cpu
Model: HGNN
Architecture: 32 -> 64 -> 32
Num nodes: 100, Num hyperedges: 28
--------------------------------------------------------------------------------
Epoch   0 | Loss: 3.7671 | Recon: 0.9948 | Contrast: 2.7723 | Time: 0.007s
Epoch  10 | Loss: 3.7429 | Recon: 0.9816 | Contrast: 2.7612 | Time: 0.002s
...
Epoch 190 | Loss: 3.0017 | Recon: 0.9147 | Contrast: 2.0870 | Time: 0.002s
--------------------------------------------------------------------------------
Training completed!
Best loss: 2.9736
Final hyperedge embeddings shape: torch.Size([28, 64])
```

## Testing

Run the test suite to validate the implementation:

```bash
cd /path/to/DeepHypergraph_copy
python -m pytest tests/test_hyperedge_autoencoder.py -v
```

All tests should pass:
- Encoder forward pass tests
- Decoder forward pass tests
- Autoencoder forward pass tests
- Augmentation tests
- Loss function tests
- Training pipeline tests

## Implementation Details

### Contrastive Learning Strategy

1. **Anchor Graph**: Original node features → hyperedge embeddings
2. **Augmented Graph**: Masked node features → hyperedge embeddings
3. **Positive Pairs**: (anchor_edge_i, augmented_edge_i) for same hyperedge i
4. **Negative Pairs**: (anchor_edge_i, augmented_edge_j) for different hyperedges i ≠ j

### Loss Function

```
Total Loss = α × Reconstruction Loss + β × Contrastive Loss
```

Where:
- **Reconstruction Loss**: MSE between reconstructed and original node features
- **Contrastive Loss**: InfoNCE loss for hyperedge embeddings
- α, β: Configurable weights (default: 1.0, 1.0)

### Message Passing

The implementation uses DHG's hypergraph operations:
- **v2e**: Vertex-to-hyperedge aggregation (encoder)
- **e2v**: Hyperedge-to-vertex aggregation (decoder)
- **Smoothing**: HGNN/HGNNP convolution for feature transformation

## Use Cases

1. **Hyperedge Feature Optimization**: Learn better representations for hyperedges
2. **Hypergraph Pre-training**: Pre-train models for downstream tasks
3. **Anomaly Detection**: Identify unusual hyperedges based on reconstruction error
4. **Hypergraph Clustering**: Use learned embeddings for hyperedge clustering
5. **Link Prediction**: Predict new hyperedges using learned representations

## Performance Tips

1. **GPU Acceleration**: Use CUDA-enabled device for faster training
2. **Batch Normalization**: Enable for better convergence with deep models
3. **Dropout**: Adjust drop_rate to prevent overfitting
4. **Learning Rate**: Start with 0.001 and adjust based on convergence
5. **Temperature**: Lower values (0.1-0.3) for harder contrastive learning

## Citation

If you use this implementation in your research, please cite:

```bibtex
@misc{hyperedge_autoencoder,
  title={Hyperedge Autoencoder with Contrastive Learning},
  author={DeepHypergraph Contributors},
  year={2024},
  howpublished={\url{https://github.com/iMoonLab/DeepHypergraph}},
}
```

## References

- **HGNN**: Feng et al. "Hypergraph Neural Networks" (AAAI 2019)
- **HGNN+**: Gao et al. "HGNN+: General Hypergraph Neural Networks" (IEEE T-PAMI 2022)
- **InfoNCE**: van den Oord et al. "Representation Learning with Contrastive Predictive Coding" (2018)
- **DHG**: Gao et al. "Deep Hypergraph Library" (2022)

## License

This implementation is part of the DeepHypergraph (DHG) library and is licensed under Apache-2.0.
