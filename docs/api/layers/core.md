# 🎛️ Core Layers API Reference

Core layers in **@oxide-js/layers** form the building blocks of deep neural networks. They manage dense connections, non-linear activation steps, dropout gates, flattening multi-dimensional tensors, and shape restructuring.

---

## 🚀 1. `Dense` Layer

The **[Dense](file:///home/akhyar/Dokumen/Code/NODE_JS/Oxide-JS/packages/layers/src/layers/Dense.ts)** layer is a regular deeply-connected neural network layer. It implements the operation: $\mathbf{Y} = \sigma(\mathbf{X} \mathbf{W} + \mathbf{b})$, where $\mathbf{W}$ is the weights matrix (kernel), $\mathbf{b}$ is the bias vector, and $\sigma$ is the element-wise activation function.

### 📌 Configuration Parameters (`DenseConfig`)
- `units: number` — Dimensionality of the output space.
- `activation?: ActivationType` — Activation function to use. Default is `"linear"`.
- `useBias?: boolean` — Whether the layer uses a bias vector. Default is `true`.
- `kernelInitializer?: string` — Initializer for the kernel weights matrix. Default is `"glorot_normal"` (Xavier).
- `biasInitializer?: string` — Initializer for the bias vector. Default is `"zeros"`.

### 📌 Properties
- `kernel: Matrix | undefined` — Trainable weight matrix of shape `[inputFeatures, units]`.
- `bias: Matrix | undefined` — Trainable bias vector of shape `[units, 1]`.

### 📌 Example
```ts
import { Dense } from "@oxide-js/layers";
import { Matrix } from "@oxide-js/core";

// 1. Instantiate dense layer with ReLU activation
const denseLayer = new Dense({
  name: "dense_1",
  units: 4,
  activation: "relu",
  useBias: true
});

// 2. Feed inputs matrix [batch=2, features=3]
const inputs = Matrix.fromFlat(new Float32Array([
  0.5, -1.0,  2.0,
  1.0,  0.0, -0.5
]), [2, 3]);

const outputs = denseLayer.forward(inputs);
outputs.print(); // Displays shape [2, 4] with ReLU applied
```

---

## ⚡ 2. `Activation` Layer

The **[Activation](file:///home/akhyar/Dokumen/Code/NODE_JS/Oxide-JS/packages/layers/src/layers/Activation.ts)** layer applies an element-wise non-linear activation function to the inputs without changing their dimensions.

### 📌 Configuration Parameters (`ActivationConfig`)
- `activation: ActivationType` — The name of the activation function to apply (e.g. `"sigmoid"`, `"tanh"`, `"relu"`, `"softmax"`).

### 📌 Example
```ts
import { Activation } from "@oxide-js/layers";
import { Matrix } from "@oxide-js/core";

// 1. Instantiate activation layer
const sigmoidLayer = new Activation({ activation: "sigmoid" });

// 2. Feed inputs matrix [batch=1, features=3]
const inputs = Matrix.fromFlat(new Float32Array([-1.0, 0.0, 1.0]), [1, 3]);
const outputs = sigmoidLayer.forward(inputs);

outputs.print(); // Output elements mapped between (0, 1)
```

---

## 🛡️ 3. `Dropout` Layer

The **[Dropout](file:///home/akhyar/Dokumen/Code/NODE_JS/Oxide-JS/packages/layers/src/layers/Dropout.ts)** layer randomly sets input elements to 0 with a frequency of `rate` during training time to prevent overfitting. Outputs are scaled by $\frac{1}{1 - \text{rate}}$ so that their sum remains unchanged. During evaluation mode (`layer.eval()`), the layer acts as an identity function.

### 📌 Configuration Parameters (`DropoutConfig`)
- `rate: number` — Float between 0 and 1 representing the fraction of the input units to drop.

### 📌 Example
```ts
import { Dropout } from "@oxide-js/layers";
import { Matrix } from "@oxide-js/core";

// 1. Instantiate dropout layer with 20% drop rate
const dropoutLayer = new Dropout({ rate: 0.2 });

// 2. Feed inputs
const inputs = Matrix.fromFlat(new Float32Array([1.0, 2.0, 3.0, 4.0]), [1, 4]);

// 3. Training mode (some elements are zeroed out and scaled)
dropoutLayer.train();
const trainOutputs = dropoutLayer.forward(inputs);
console.log("Training outputs:");
trainOutputs.print();

// 4. Inference/Evaluation mode (acts as a pass-through)
dropoutLayer.eval();
const evalOutputs = dropoutLayer.forward(inputs);
console.log("Evaluation outputs (pass-through):");
evalOutputs.print(); // [1.0, 2.0, 3.0, 4.0]
```

---

## 📐 4. `Flatten` Layer

The **[Flatten](file:///home/akhyar/Dokumen/Code/NODE_JS/Oxide-JS/packages/layers/src/layers/Flatten.ts)** layer reshapes a multi-dimensional input tensor shape `[batch, d1, d2, ...]` into a 2D matrix shape `[batch, d1 * d2 * ...]`. This is standard when transitioning from convolutional or recurrent layers to fully connected Dense layers.

### 📌 Configuration Parameters (`FlattenConfig`)
- Accepts standard `name` and `trainable` configuration fields.

### 📌 Example
```ts
import { Flatten } from "@oxide-js/layers";
import { Matrix } from "@oxide-js/core";

// 1. Instantiate flattening layer
const flattenLayer = new Flatten({ name: "flat" });

// 2. Assume inputs came from a layer with shape [batch=2, seq=3, features=2]
// Total features per sample = 3 * 2 = 6
const inputs = Matrix.fromFlat(new Float32Array([
  1.0, 2.0,  3.0, 4.0,  5.0, 6.0,   // sample 1
  7.0, 8.0,  9.0, 0.0,  1.0, 2.0    // sample 2
]), [2, 6]);

// Set logical inputs dimensions representation
inputs._shape = [2, 6]; 

const outputs = flattenLayer.forward(inputs);
console.log("Flattened shape:", outputs._shape); // [2, 6]
outputs.print();
```

---

## 🔄 5. `Reshape` Layer

The **[Reshape](file:///home/akhyar/Dokumen/Code/NODE_JS/Oxide-JS/packages/layers/src/layers/Reshape.ts)** layer restructures input dimensions into a specified target shape. It preserves the batch size (index 0) and supports a single dynamic dimension of `-1` to automatically infer size based on available elements.

### 📌 Configuration Parameters (`ReshapeConfig`)
- `targetShape: number[]` — The target dimensions list (excluding the batch dimension).

### 📌 Example
```ts
import { Reshape } from "@oxide-js/layers";
import { Matrix } from "@oxide-js/core";

// 1. Instantiate reshaping layer to convert [batch, 6] -> [batch, 3, 2]
const reshapeLayer = new Reshape({
  name: "reshape_1",
  targetShape: [3, 2]
});

// 2. Feed inputs [batch=2, features=6]
const inputs = Matrix.fromFlat(new Float32Array([
  1, 2, 3, 4, 5, 6,
  7, 8, 9, 10, 11, 12
]), [2, 6]);

const outputs = reshapeLayer.forward(inputs);
console.log("Reshaped logical shape:", outputs._shape); // [2, 6] (physically stored as 2D)
outputs.print();
```
