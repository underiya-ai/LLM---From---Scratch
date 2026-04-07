# Step-02: Tokenization

## Objective
Convert raw text into tokens and numerical representations for training a Language Model (LLM).

---

## 🧠 What is Tokenization?

Tokenization is the process of breaking down text into smaller units called tokens.

Example:
"I love AI" → ["I", "love", "AI"]

---

## Files

- tokenizer_scratch.py → Tokenization implemented from scratch
- tokenizer_tiktoken.py → Tokenization using tiktoken library
- cleaned.txt → Input text (from Step-01)
- README.md → Documentation

---

## Approach 1: Tokenization From Scratch

### Steps:
- Split text into tokens using regex
- Handle punctuation separately
- Create vocabulary (unique tokens)
- Convert tokens → numerical IDs
- Decode IDs → text

### Key Concepts:
- Word-level tokenization
- Vocabulary creation
- Encoding & decoding

---

## Sample Code in tokenizer_scratch.py 

## Sample Output (Scratch Tokenizer)

### Input:
This is my first LLM project

### Encoded:
[12, 45, 67, 23, ...]

### Decoded:
this is my first llm project

## Tokenization using tiktoken

Used OpenAI's tiktoken library for fast and efficient tokenization.

### Features:
- Subword tokenization
- Optimized for LLMs
- Same tokenizer used in GPT models

---

## Sample Output

### Encoded Tokens:
[15496, 11, 616, 3290, ...]

### Decoded Text:
hello this is my first llm project ...