# 🗺️ Embedding Layer API Reference

The **[Embedding](file:///home/akhyar/Dokumen/Code/NODE_JS/Oxide-JS/packages/layers/src/layers/Embedding.ts)** layer acts as a lookup table that maps discrete integer tokens (such as word or subword IDs from a tokenizer) into continuous, dense multi-dimensional vector representations.

---

## 📐 Mathematical Formulation

Given an input matrix $\mathbf{X}$ of shape `[batchSize, seqLen]` containing integer token IDs:
$$\mathbf{X}_{i,j} \in \{0, 1, \dots, V - 1\}$$
where $V$ is the vocabulary size (`inputDim`).

The embedding layer maintains a learnable lookup weight table $\mathbf{E}$ (the embeddings matrix) of shape `[V, D]`, where $D$ is the embedding dimension (`outputDim`).

The forward pass produces an output tensor $\mathbf{Y}$ of shape `[batchSize * seqLen, D]` by fetching index coordinates:
$$\mathbf{Y}_{k, \cdot} = \mathbf{E}_{\mathbf{X}_k, \cdot}$$
where $\mathbf{X}_k$ is the flat 1D token ID at flattened index $k = i \cdot \text{seqLen} + j$.

### 🔄 Backward Propagation & Autodiff
During backpropagation, the incoming gradient $\frac{\partial L}{\partial \mathbf{Y}}$ of shape `[batchSize * seqLen, D]` is scattered and accumulated back into the embeddings parameter gradient $\frac{\partial L}{\partial \mathbf{E}}$ of shape `[V, D]` based on the active lookup indices:
$$\frac{\partial L}{\partial \mathbf{E}_{v, d}} = \sum_{k \text{ where } \mathbf{X}_k = v} \frac{\partial L}{\partial \mathbf{Y}_{k, d}}$$
The integer input $\mathbf{X}$ does not require gradients, so its returned derivative is `null`.

---

## 📌 Configuration Parameters (`EmbeddingConfig`)
- `inputDim: number` — Size of the vocabulary (maximum token ID + 1).
- `outputDim: number` — Dimension of the dense embedding vectors.
- `inputLength?: number` — Optional length of input sequences.
- `embeddingsInitializer?: string` — Initializer for the embeddings matrix. Default is `"random"`.

---

## 📌 Properties
- `embeddings: Matrix | undefined` — Trainable embedding lookup table parameter matrix of shape `[inputDim, outputDim]`.

---

## 🛠️ Usage Example

This example demonstrates how to initialize an embedding layer, map token sequences, and trigger autograd backward sweeps to compute embedding weight updates.

```ts
import { Embedding } from "@oxide-js/layers";
import { Matrix, engine } from "@oxide-js/core";

// 1. Instantiate the Embedding layer (vocabSize = 10, embedDim = 4)
const embedLayer = new Embedding({
  name: "token_embeddings",
  inputDim: 10,
  outputDim: 4,
  embeddingsInitializer: "random"
});

// 2. Feed sequence inputs [batch=2, seqLen=3] containing token indices
// Batch 1: [2, 0, 5]
// Batch 2: [1, 9, 2]
const tokenInputs = Matrix.fromFlat(new Float32Array([
  2, 0, 5,
  1, 9, 2
]), [2, 3]);

// 3. Compute forward pass (builds the embedding weight matrix internally)
const outputs = embedLayer.forward(tokenInputs);
console.log("Output Shape (flat batch * seqLen, embedDim):", outputs._shape); // [6, 4]

// 4. Wrap inside autograd tape to trace gradient propagation
const tape = engine.grad(() => {
  const lookups = embedLayer.forward(tokenInputs);
  return lookups; // standard representation
});

// Assume upstream gradient is ones of shape [6, 4]
const upstreamGrad = Matrix.fromFlat(new Float32Array(6 * 4).fill(1.0), [6, 4]);
tape.backward(tape.result, upstreamGrad);

// Verify computed weight gradients for the embedding matrix
console.log("\nEmbedding Table Weights Gradient:");
embedLayer.embeddings?.grad?.print(); // Accumulates derivatives for indices [0, 1, 2, 5, 9]
```
