# 🌌 Attention Layers API Reference

Attention mechanisms in **@oxide-js/layers** enable sequence-to-sequence modeling, cross-attention mapping, and self-attention blocks that form the backbone of state-of-the-art Transformer models. They support standard dot-product **Attention** and causal **Multi-Head Attention (MHA)**.

---

## ⚡ 1. `Attention` Layer (Standard Dot-Product)

The **[Attention](file:///home/akhyar/Dokumen/Code/NODE_JS/Oxide-JS/packages/layers/src/layers/Attention.ts)** layer implements standard scaled dot-product query-key-value projections mapping.

### 📐 Mathematical Formulation
Given query inputs $\mathbf{X}_Q$ and key/value inputs $\mathbf{X}_K$:
1. **Query, Key, Value Projections**:
   $$\mathbf{Q} = \mathbf{X}_Q \mathbf{W}_Q + \mathbf{b}_Q$$
   $$\mathbf{K} = \mathbf{X}_K \mathbf{W}_K + \mathbf{b}_K$$
   $$\mathbf{V} = \mathbf{X}_K \mathbf{W}_V + \mathbf{b}_V$$
2. **Scaled Dot-Product Attention**:
   $$\text{Scores}_{i,j} = \frac{\mathbf{Q}_{i, \cdot} \mathbf{K}_{j, \cdot}^T}{\sqrt{d_k}}$$
   $$\mathbf{P} = \text{Softmax}(\text{Scores})$$
   $$\mathbf{Y} = \mathbf{P} \mathbf{V}$$
where $d_k$ is the projection dimension (`units`).

---

## 🎭 2. `MultiHeadAttention` Layer (MHA)

The **[MultiHeadAttention](file:///home/akhyar/Dokumen/Code/NODE_JS/Oxide-JS/packages/layers/src/layers/MultiHeadAttention.ts)** layer projects queries, keys, and values into multiple subspaces, allowing the model to jointly attend to information from different representation spaces at different positions.

### 📐 Multi-Head Process
Given query, key, and value vectors:
1. **Multi-Head Projections**:
   $$\mathbf{Q}_h = \mathbf{X}_Q \mathbf{W}_{Q,h} + \mathbf{b}_{Q,h}$$
   $$\mathbf{K}_h = \mathbf{X}_K \mathbf{W}_{K,h} + \mathbf{b}_{K,h}$$
   $$\mathbf{V}_h = \mathbf{X}_V \mathbf{W}_{V,h} + \mathbf{b}_{V,h}$$
   where $h \in [1, H]$ is the head index and $H$ is `numHeads`.
2. **Scaled Attention Per Head**:
   $$\mathbf{A}_h = \text{Softmax}\left(\frac{\mathbf{Q}_h \mathbf{K}_h^T}{\sqrt{d_k}}\right) \mathbf{V}_h$$
3. **Concatenation & Output Projection**:
   $$\text{ConcatOut} = [\mathbf{A}_1, \mathbf{A}_2, \dots, \mathbf{A}_H]$$
   $$\mathbf{Y} = \text{ConcatOut} \cdot \mathbf{W}_O + \mathbf{b}_O$$
   where $\mathbf{W}_O$ is the final projection matrix of shape `[H * valueDim, outputDim]`.

---

## 📌 Configurations & API Interfaces

### 📌 `AttentionConfig`
- `units: number` — Dimensionality of projected queries, keys, and values.
- `useBias?: boolean` — Default is `true`.
- `sequenceLength?: number` — Step sequence length. Required if input shape is 2D.
- `inputDim?: number` — Channels per step.

### 📌 `MultiHeadAttentionConfig`
- `numHeads: number` — Number of attention heads.
- `keyDim: number` — Size of each attention head for Query and Key.
- `valueDim?: number` — Size of each attention head for Value. Defaults to `keyDim`.
- `outputDim?: number` — Final projection dimension. Defaults to `inputDim`.
- `useBias?: boolean` — Default is `true`.

---

## 🛠️ Cross-Attention & Caching (`setExternal`)

Both `Attention` and `MultiHeadAttention` layers support setting external query, key, or value matrices. This is highly useful for:
* **Decoder Cross-Attention**: Query is from the decoder hidden states; Key/Value are from the encoder output.
* **Inference Key-Value Caching**: Passing pre-allocated cached keys/values to avoid redundant step recalculations.

```ts
layer.setExternal({
  query: queryMatrix,
  key: keyMatrix,
  value: valueMatrix,
  trainableQuery: false, // locks query backpropagation
  trainableKey: false,
  trainableValue: false
});
```

---

## 🛠️ Usage Examples

### 📌 Example (Self-Attention Block)
This example demonstrates a complete self-attention pass over sequence embeddings.

```ts
import { MultiHeadAttention } from "@oxide-js/layers";
import { Matrix } from "@oxide-js/core";

// 1. Instantiate MultiHeadAttention (heads=2, keyDim=4, inputDim=8)
const mha = new MultiHeadAttention({
  name: "self_attention",
  numHeads: 2,
  keyDim: 4,
  valueDim: 4,
  sequenceLength: 3,
  inputDim: 8
});

// 2. Feed sequence inputs [batch * seqLen, inputDim] -> [2 * 3, 8] = [6, 8]
const inputs = Matrix.fromFlat(new Float32Array(6 * 8).map((_, i) => i * 0.05), [6, 8]);

const outputs = mha.forward(inputs);
console.log("MHA Output Shape (batch * seqLen, outputDim):", outputs._shape); // [6, 8]
outputs.print();
```

---

### 📌 Example (Decoder Cross-Attention)
This example demonstrates cross-attention, where a query matrix attends to an external key-value database.

```ts
import { Attention } from "@oxide-js/layers";
import { Matrix } from "@oxide-js/core";

// 1. Instantiate standard Attention layer
const crossAttention = new Attention({
  name: "cross_attention",
  units: 6,
  sequenceLength: 2, // Query sequence length
  inputDim: 6
});

// 2. Set up Query [batch * seqLenQ, inputDim] -> [2 * 2, 6] = [4, 6]
const query = Matrix.fromFlat(new Float32Array(4 * 6).fill(0.8), [4, 6]);

// 3. Set up External Keys/Values database [batch * seqLenK, inputDim] -> [2 * 3, 6] = [6, 6]
const keyValDatabase = Matrix.fromFlat(new Float32Array(6 * 6).fill(0.2), [6, 6]);

// 4. Register external database on the attention layer
crossAttention.setExternal({
  key: keyValDatabase,
  trainableKey: false
});

// 5. Run forward pass (Query attends to the External database)
const outputs = crossAttention.forward(query);
console.log("Cross Attention Output Shape (batch * seqLenQ, units):", outputs._shape); // [4, 6]
outputs.print();
```
