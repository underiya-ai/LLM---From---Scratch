# GPT From Scratch (PyTorch Implementation)

This project implements a simplified GPT (Generative Pre-trained Transformer) model from scratch using PyTorch.

The goal is to understand how modern Large Language Models (LLMs) like GPT work internally — from raw text processing to text generation.

---

## Project Pipeline

This project is built step-by-step:
Step-01 → Data Cleaning
Step-02 → Tokenization (Scratch + tiktoken)
Step-03 → Dataset & DataLoader (GPT-style)
Step-04 → Transformer Block
Step-05 → Full GPT Model  


---

## Step-01: Data Cleaning

- Removed HTML tags
- Removed URLs
- Removed numbers and special characters


Output: Clean text ready for tokenization

---

## Step-02: Tokenization

###  Scratch Tokenizer
- Built custom tokenizer using Python
- Created vocabulary
- Implemented encoder and decoder

### tiktoken (GPT-2 Tokenizer)
- Used GPT-2 tokenizer
- Added support for special token:


 Output: Text converted into token IDs

---

## Step-03: Dataset & DataLoader

### ✔️ GPT-style Dataset
- Used sliding window approach
- Created input-target pairs

Example:  Input: [10, 25, 90, 12]
          Target: [25, 90, 12, 45] 


### ✔️ Train-Validation Split
- 90% training data
- 10% validation data

### ✔️ DataLoader
- Batch processing
- Shuffling for training
- Efficient data loading

---

## Step-04: Transformer Block

Implemented core transformer components:

### 🔹 Multi-Head Self Attention
- Query, Key, Value mechanism
- Parallel attention heads

### 🔹 Feed Forward Network
- Linear → ReLU → Linear

### 🔹 Residual Connections
- Skip connections for stable training

### 🔹 Layer Normalization
- Normalizes activations

---

## 🤖 Step-05: GPT Model

Full GPT architecture includes:

### 🔹 Token Embedding
- Converts token IDs into vectors

### 🔹 Positional Embedding
- Adds position information

### 🔹 Transformer Blocks
- Stacked multiple layers

### 🔹 Final LayerNorm

### 🔹 Output Head
- Predicts next token probabilities

---

## 🧠 Model Configuration

```python
GPT_CONFIG = {
    "vocab_size": 50257,
    "context_length": 64,
    "emb_dim": 64,
    "n_heads": 4,
    "n_layers": 2,
    "drop_rate": 0.1
}

Training

Loss Function: CrossEntropyLoss
Optimizer: AdamW
Input: token sequences
Target: shifted token sequences

Text Generation

Predict next token using softmax
Append predicted token to sequence
Repeat to generate text

