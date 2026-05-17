# 📐 Convolutional & Pooling Layers API Reference

Convolutional and Pooling layers in **@oxide-js/layers** enable spatial modeling for multi-dimensional sequential or structured data (such as time series, speech, and images). They leverage high-performance **im2col (grid2col) native mathematical mappings** to transform spatial grids into 2D patch matrices, achieving maximum speed through optimized native GEMM kernels.

---

## 📸 1. `Conv2D` & `Conv1D` Layers

### 📐 Mathematical Formulation (Conv2D)
Given an input volume $\mathbf{X}$ of shape `[batchSize, height, width, channels]`, a 2D convolution layer applies $F$ filters. Each filter contains a learnable kernel weight $\mathbf{W}$ of spatial size `[kernelHeight, kernelWidth, channels]`.

1. **`grid2col` Patch Extraction**:
   Slices overlapping patch grids across the spatial dimensions according to `strides` and `padding` rules. This creates a flattened 2D patch matrix of shape:
   $$\text{Shape} = [B \cdot H_{\text{out}} \cdot W_{\text{out}}, \, K_H \cdot K_W \cdot C]$$
2. **Accelerated Matrix Multiplication (GEMM)**:
   Computes matrix products of the patch matrix with the flattened kernels weight matrix of shape `[K_H \cdot K_W \cdot C, F]`.
3. **Bias & Activation**:
   $$\mathbf{Y} = \sigma(\text{PatchMatrix} \cdot \mathbf{W} + \mathbf{b})$$
   resulting in a dense shape `[batchSize * H_out * W_out, filters]`.

### 📌 Configuration Parameters (`Conv2DConfig`)
- `filters: number` — The number of output filter channels.
- `kernelSize: number | [number, number]` — Spatial size of the filters.
- `strides?: number | [number, number]` — Step size along height and width. Default is `[1, 1]`.
- `padding?: "valid" | "same"` — Padding algorithm. Default is `"valid"`.
- `activation?: string` — Non-linear activation function. Default is `"linear"`.
- `useBias?: boolean` — Default is `true`.
- `imageShape?: [number, number]` — Height and width of input matrices. Required if input shape is 2D.
- `inputDim?: number` — Number of input channels.

### 📌 Properties
- `kernel: Matrix` — Trainable kernels weight matrix of shape `[kernelRows * kernelCols * channels, filters]`.
- `bias: Matrix` — Trainable bias vector of shape `[filters, 1]`.

### 📌 Example (Conv2D)
```ts
import { Conv2D } from "@oxide-js/layers";
import { Matrix } from "@oxide-js/core";

// 1. Instantiate Conv2D layer
const convLayer = new Conv2D({
  name: "conv2d_1",
  filters: 8,
  kernelSize: 3,
  strides: 1,
  padding: "same",
  activation: "relu",
  imageShape: [8, 8], // 8x8 input image
  inputDim: 3         // RGB channels
});

// 2. Feed inputs [batch * height * width, channels] -> [1 * 64, 3]
const inputs = Matrix.fromFlat(new Float32Array(64 * 3).fill(0.5), [64, 3]);
const outputs = convLayer.forward(inputs);

console.log("Outputs Shape (batch * H_out * W_out, filters):", outputs._shape); // [64, 8]
```

### 📌 Example (Conv1D)
```ts
import { Conv1D } from "@oxide-js/layers";
import { Matrix } from "@oxide-js/core";

// 1. Instantiate Conv1D layer
const conv1d = new Conv1D({
  name: "conv1d_1",
  filters: 16,
  kernelSize: 3,
  strides: 1,
  padding: "valid",
  inputLength: 10, // sequence length
  inputDim: 8      // features per step
});

// 2. Feed inputs [batch * seqLen, inputDim] -> [2 * 10, 8]
const inputs = Matrix.fromFlat(new Float32Array(20 * 8).fill(1.0), [20, 8]);
const outputs = conv1d.forward(inputs);
console.log("Outputs Shape:", outputs._shape); // [16, 16] (2 samples * 8 steps_out)
```

---

## ⚡ 2. Pooling Layers

Pooling layers reduce spatial dimensions (downsampling) by computing maximum or average activations inside sliding windows, preserving channels.

### 📌 Catalog

* **`MaxPooling2D`** — Downsamples by taking the maximum value.
* **`AveragePooling2D`** — Downsamples by taking the average value.
* **`MaxPooling1D`** — Max downsampling for sequence features.
* **`AveragePooling1D`** — Average downsampling for sequence features.

---

### 📌 Example (MaxPooling2D)
```ts
import { MaxPooling2D } from "@oxide-js/layers";
import { Matrix } from "@oxide-js/core";

// 1. Instantiate Max Pooling 2D layer
const maxPool = new MaxPooling2D({
  name: "maxpool_2d",
  poolSize: 2,
  strides: 2,
  imageShape: [4, 4], // 4x4 input grid
  inputDim: 1         // 1 channel
});

// 2. Feed 4x4 inputs
const inputs = Matrix.fromFlat(new Float32Array([
  1, 2,  5, 6,
  3, 4,  7, 8,
  
  9, 10, 13, 14,
  11, 12, 15, 16
]), [16, 1]);

const outputs = maxPool.forward(inputs);
console.log("Pooled Output Shape (batch * H_out * W_out, channels):", outputs._shape); // [4, 1]
outputs.print(); // Displays max elements of each 2x2 patch: [[4], [8], [12], [16]]
```

---

### 📌 Example (AveragePooling2D)
```ts
import { AveragePooling2D } from "@oxide-js/layers";
import { Matrix } from "@oxide-js/core";

// 1. Instantiate Average Pooling 2D
const avgPool = new AveragePooling2D({
  name: "avgpool_2d",
  poolSize: 2,
  strides: 2,
  imageShape: [4, 4],
  inputDim: 1
});

const inputs = Matrix.fromFlat(new Float32Array([
  1, 2, 5, 6,
  3, 4, 7, 8,
  1, 2, 5, 6,
  3, 4, 7, 8
]), [16, 1]);

const outputs = avgPool.forward(inputs);
outputs.print(); // Displays average elements of each 2x2 patch
```

---

### 📌 Example (MaxPooling1D)
```ts
import { MaxPooling1D } from "@oxide-js/layers";
import { Matrix } from "@oxide-js/core";

// 1. Instantiate Max Pooling 1D
const maxPool1d = new MaxPooling1D({
  name: "maxpool_1d",
  poolSize: 2,
  strides: 2,
  inputLength: 6,
  inputDim: 2
});

// 2. Feed inputs [batch * seqLen, inputDim] -> [6, 2]
const inputs = Matrix.fromFlat(new Float32Array([
  1, 10,  2, 20,
  3, 30,  4, 40,
  5, 50,  6, 60
]), [6, 2]);

const outputs = maxPool1d.forward(inputs);
console.log("MaxPooling1D output shape:", outputs._shape); // [3, 2]
outputs.print();
```

---

### 📌 Example (AveragePooling1D)
```ts
import { AveragePooling1D } from "@oxide-js/layers";
import { Matrix } from "@oxide-js/core";

// 1. Instantiate Average Pooling 1D
const avgPool1d = new AveragePooling1D({
  name: "avgpool_1d",
  poolSize: 2,
  strides: 2,
  inputLength: 4,
  inputDim: 1
});

const inputs = Matrix.fromFlat(new Float32Array([
  1, 3, 5, 7
]), [4, 1]);

const outputs = avgPool1d.forward(inputs);
outputs.print(); // Displays averages: [[2], [6]]
```
