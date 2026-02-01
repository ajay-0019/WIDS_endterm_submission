# GPT Transformer From Scratch

A character-level transformer built from scratch using PyTorch. Trained on Shakespeare's works to generate similar text.

## Overview

This project implements a decoder-only GPT-style transformer with 210,000 parameters. The model achieves 1.83 validation loss and generates coherent Shakespearean dialogue after training for 5000 iterations.

**Training Time:** 15 minutes on GPU  
**Dataset:** 1.1 million characters from Shakespeare's complete works

## Model Architecture

The transformer consists of several key components:

- **Token Embeddings:** Maps 65 unique characters to 64-dimensional vectors
- **Positional Embeddings:** Adds position information for up to 32 characters
- **Multi-Head Attention:** 4 parallel attention heads operating in 16-dimensional subspaces
- **Feed-Forward Networks:** Two linear layers expanding to 256 dimensions with ReLU activation
- **Residual Connections:** Enable gradient flow through deep networks
- **Layer Normalization:** Stabilizes training by normalizing activations

The complete model stacks 4 transformer blocks between embeddings and output projection.

## Hyperparameters

- Embedding dimension: 64
- Number of layers: 4
- Attention heads: 4
- Context length: 32 characters
- Batch size: 16
- Learning rate: 1e-3
- Training iterations: 5000

## Results

**Bigram Baseline (loss 2.7):**
