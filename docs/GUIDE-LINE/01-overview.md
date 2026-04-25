# System Documentation: Overview

Welcome to the official documentation of **ML-V1**, a custom machine learning framework designed to provide full control, flexibility, and high performance for AI researchers and developers.

## What is ML-V1?

**ML-V1** is a low-to-mid-level machine learning library built with **TypeScript** and accelerated by a **Rust (N-API)** backend. This project was born from the need for a transparent ML ecosystem, where every mathematical operation and training loop logic can be manually inspected and modified without depending on complex commercial frameworks.

## Vision and Goals

The project is designed with several main goals:
- **Full Control**: Providing the ability to manage every technical detail, from matrix shapes to parameter update mechanisms.
- **Hybrid Efficiency**: Combining the convenience of coding in TypeScript with the execution speed of critical numerical operations using Rust.
- **Architecture Research**: Serving as a playground to experiment with custom model architectures such as Transformers, Dimensionality Reduction, and more.

---

## Core Architecture

The system is divided into several integrated main modules:

### 1. Data Structure & Math (`src/matrix` & `src/math`)
The heart of this framework is the `Matrix` class based on `Float32Array`. 
- **Flat Memory**: Uses contiguous memory for cache efficiency.
- **Math Primitives**: Provides basic operations like `dotProduct`, `add`, `sumAxis`, and `clipGradients`.

### 2. Hybrid Backend (`src-rust`)
For time-consuming operations (hot paths), ML-V1 automatically delegates to the Rust backend if available.
- **Native Acceleration**: Speeds up heavy operations like *Multi-Head Attention* and *Layer Normalization*.
- **Fallback Mechanism**: If the native binary is not found, the system automatically switches to pure JavaScript implementations without stopping the process.

### 3. Neural Network Components (`src/layers`)
This module provides building blocks for composing models:
- **Linear/Dense**: Fully-connected layers with optimizer support.
- **Recurrent**: `RNN`, `LSTM`, and `GRU` for hidden-state-based sequence modeling.
- **Attention**: Implementation of *Self-Attention* and *Multi-Head Attention* with causal masking schemes.
- **Normalization**: *Layer Normalization* for training stability.
- **Specialized**: *Embedding*, *Dropout*, *Positional Encoding*, and *Flatten*.

### 4. Model Composition (`src/models`)
High-level abstractions for managing data flow:
- **Sequential**: Linear layer stacking.
- **Transformers**: Complete NLP architecture with full-sequence causal language modeling training and last-token inference for generation.
- **Dimensionality Reduction**: Specialized models for data dimension reduction.

### 5. Text Preprocessing (`src/tokenizer`)
Implementation of the **Byte Pair Encoding (BPE) Tokenizer** supporting:
- Vocabulary training from raw datasets.
- Text encoding/decoding to token IDs.
- Management of special tokens and padding.

---

## Main Features

- **Matrix-Driven**: All operations center on efficient matrix manipulation.
- **BPE-Native**: Built-in support for advanced tokenization.
- **Optimizer & Loss Functions**: Various choices like Adam optimizer, MSE, and Softmax Cross-Entropy.
- **Training Workflow**: Intuitive API with `.forward()`, `.backward()`, and `.fit()` methods.

---

## Performance Philosophy

ML-V1 prioritizes performance through:
1. **Pre-allocated Buffers**: Reducing Garbage Collection (GC) frequency during intense training loops.
2. **Native Dispatching**: Using `napi-rs` to minimize overhead between JavaScript and Rust layers.
3. **Optimized Hot-Paths**: Manual implementation of critical operations to ensure lowest latency.

---

> [!NOTE]
> This project is under active development (v2.1.0). Always ensure compatibility between the library version and the native backend used.

**Next Steps:**
Continue to the [Installation](02-installation.md) section to start setting up your development environment.
