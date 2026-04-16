# Step-04: Transformer Block

## Objective
Implement the core building block of modern Language Models (LLMs) — the Transformer.

---

## What is a Transformer?

A Transformer is a neural network architecture that processes sequences using attention mechanisms instead of recurrence.

It is the foundation of models like GPT, BERT, and ChatGPT.

---

## High-Level Architecture

Input → Embedding → Positional Encoding → Transformer Block → Output

---

## Components of Transformer Block

### 1. Multi-Head Self Attention 

Allows the model to focus on different parts of the sequence simultaneously.

#### Key Idea:
Each word looks at other words and decides:
“Which words are important for me?”

---

### Steps inside Attention:

- Query (Q)
- Key (K)
- Value (V)

### Formula:
Attention(Q, K, V) = softmax(QKᵀ / √d) V

---

### 2. Multi-Head Mechanism

Instead of one attention, we use multiple heads.

#### Why?
- Each head learns different relationships
- Improves model understanding

---

### 3. Feed Forward Network (FFN)

A simple neural network applied after attention.

Structure:
- Linear → ReLU → Linear

Purpose:
- Adds non-linearity
- Learns complex patterns

---

### 4. Residual Connections

We add input back to output:

x = x + Attention(x)

#### Why?
- Prevents vanishing gradients
- Helps deep networks train

---

### 5. Layer Normalization

Normalizes values for stable training.

Applied after each block:
- After Attention
- After Feed Forward

---

### 6. Dropout

Randomly drops neurons during training.

#### Purpose:
- Prevent overfitting

---

## 🔁 Transformer Block Flow

``` id="flow001"
Input (x)
   ↓
LayerNorm
   ↓
Multi-Head Attention
   ↓
Add (Residual)
   ↓
LayerNorm
   ↓
Feed Forward
   ↓
Add (Residual)
   ↓
Output