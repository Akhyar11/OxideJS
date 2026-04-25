# Complete API Reference: ML-V1

This page contains comprehensive technical documentation for all components within the **ML-V1** library. This documentation is organized progressively, starting with core components like **`Matrix`**.

---

## 1. Core Data Structure: `Matrix` Class (`src/matrix`)

The `Matrix` class is the backbone of all numerical operations. It uses **`Float32Array`** for data storage to ensure memory efficiency and maximum access speed.

### A. Main Properties
- **`_data: Float32Array`**: Flat data buffer. Element access $(i, j)$ is calculated internally with the index `i * cols + j`.
- **`_shape: [rows, cols]`**: Matrix dimensions (e.g., `[2, 3]` for 2 rows and 3 columns).

### B. Initialization & Creation

#### `constructor({ array: number[][] })`
Creates a matrix from a standard 2D array.
```ts
const m = new Matrix({ 
  array: [
    [1, 2], 
    [3, 4]
  ] 
});
// Internal Result: _data = [1, 2, 3, 4], _shape = [2, 2]
```

#### `static fromFlat(data, shape)`
Creates a matrix directly from flat data. Faster because there is no conversion process (looping) from nested arrays.
```ts
const rawData = new Float32Array([10, 20, 30, 40]);
const m = Matrix.fromFlat(rawData, [2, 2]);
// Matrix:
// [[10, 20],
//  [30, 40]]
```

### C. Element Access & Modification

#### `get(i, j)` & `set(i, j, val)`
Quickly access and change elements at specific positions.
```ts
const m = new Matrix({ array: [[1, 2], [3, 4]] });

console.log(m.get(0, 1)); // Output: 2 (Row 0, Column 1)

m.set(1, 0, 99); 
// Matrix now:
// [[1,  2],
//  [99, 4]]
```

#### `getCol(index)` & `setCol(index, data)`
Manipulate entire columns of data using typed arrays.
```ts
const m = new Matrix({ array: [[1, 2], [3, 4]] });

const col1 = m.getCol(1); 
// col1 = Float32Array([2, 4])

m.setCol(0, new Float32Array([10, 30]));
// Matrix now:
// [[10, 2],
//  [30, 4]]
```

### D. Element-wise Operations (In-Place on Instance)

#### `add(a)`, `sub(a)`, `mul(a)`, `div(a)`
Basic arithmetic operations that **directly modify the current matrix instance**.

> [!IMPORTANT]
> Unlike the helpers in the `mj` module (`mj.add`, `mj.sub`, etc.), the `Matrix.add/sub/mul/div` instance methods **do not** return a new matrix.
```ts
const a = new Matrix({ array: [[1, 2], [3, 4]] });

a.add(10);
// a now: [[11, 12], [13, 14]]

const b = new Matrix({ array: [[1, 2], [3, 4]] });
b.mul(b);
// b now: [[1, 4], [9, 16]] (Hadamard product in-place)
```

### E. In-Place Operations (Maximum Optimization)

Modify data directly on the original `_data` buffer. Highly memory-efficient as it does not allocate a new matrix. Automatically accelerated if the Rust backend is available.

#### `addInPlace(other)`, `subInPlace(other)`, `mulInPlace(other)`
```ts
const m = new Matrix({ array: [[1, 2], [3, 4]] });
m.addInPlace(5); 

// m CHANGES to:
// [[6, 7],
//  [8, 9]]
```

### F. Transformation & Utilities

#### `reshape(shape)` & `flatten()`
Change dimension interpretation without changing the data order in memory.
```ts
const m = new Matrix({ array: [[1, 2], [3, 4]] }); // [2, 2]

m.reshape([1, 4]); 
// m now: [[1, 2, 3, 4]] (1 Row, 4 Columns)

m.flatten(); 
// m now: [[1, 2, 3, 4]] (Flat vector)
```

#### `clone()` & `map(func)`
```ts
const original = new Matrix({ array: [[1, 2], [3, 4]] });

// Literal clone
const copy = original.clone(); 

// Custom transformation per element (in-place)
original.map(v => v * 2);
// original: [[2, 4], [6, 8]]
```

Note:
- `clone()` returns a new `Matrix` object.
- `map(func)` modifies the current matrix content and does not return a new `Matrix`.

#### `print()`
Displays the matrix structure in a table format in the console for easier debugging.
```ts
m.print(); 
// ┌─────────┬────┬────┐
// │ (index) │ 0  │ 1  │
// ├─────────┼────┼────┤
// │    0    │ 2  │ 4  │
// │    1    │ 6  │ 8  │
// └─────────┴────┴────┘
```

---

## 2. Math Module (`src/math`)

The `math` module (often aliased as `mj`) is a collection of pure functions for tensor/matrix processing. Almost all functions support operands of type `Matrix` or `number` (scalar).

---

### A. Main Operations & Linear Algebra

#### `mj.dotProduct(a, b, out?, transA?, transB?)`
The most critical function in machine learning for Matrix Multiplication. This operation is automatically accelerated by the Rust backend for large workloads.

> [!IMPORTANT]
> **Dimension Rule**: For an operation `(M x K) * (K x N)`, the final result is always `(M x N)`. The number of columns of the first matrix must match the number of rows of the second matrix.

##### 1. Basic Usage
```ts
const a = mj.matrix([
  [1, 2], 
  [3, 4]
]); // [2x2]

const b = mj.matrix([
  [5, 6], 
  [7, 8]
]); // [2x2]

const res = mj.dotProduct(a, b);
// a * b = 
// [[(1*5 + 2*7), (1*6 + 2*8)],
//  [(3*5 + 4*7), (3*6 + 4*8)]]
// res: [[19, 22], [43, 50]]
```

##### 2. On-the-fly Transposition (Advanced Feature)
You can perform multiplication against a transposed matrix without needing to perform a memory-intensive physical transpose operation.
- `transA = true`: Treats matrix `a` as `a.transpose()`.
- `transB = true`: Treats matrix `b` as `b.transpose()`.

```ts
// Example: A * B^T
// A [2x3], B [2x3] -> B is manipulated as [3x2] on-the-fly
const res = mj.dotProduct(a, b, undefined, false, true); 
```

##### 3. Optimization with `out` Parameter
For high performance inside training loops, use a pre-allocated matrix.
```ts
const output = mj.zeros([2, 2]);
mj.dotProduct(a, b, output); // Result written directly to 'output'
```

#### `mj.transpose(a: Matrix)`
Generates a new matrix by swapping rows and columns.
```ts
const t = mj.transpose(mj.matrix([[1, 2], [3, 4]]));
// t: [[1, 3], [2, 4]]
```

#### `mj.concat(a, b): Matrix`
Joins two matrices/vectors. Currently optimized for row vector concatenation (shape `[1, N]`).
```ts
const a = mj.matrix([[1, 2]]);
const b = mj.matrix([[3, 4]]);
const res = mj.concat(a, b); // res: [[1, 2, 3, 4]]
```

#### `mj.addBias(a, bias): void`
Adds a bias vector in-place to a matrix using *broadcasting* techniques (adding the same vector to every column).
```ts
const input = mj.matrix([[1, 2], [3, 4]]); // [2x2]
const bias = mj.matrix([[10], [20]]);      // [2x1] column vector
mj.addBias(input, bias);
// input: [[11, 12], [23, 24]]
```

#### `mj.norm(a): number`
Calculates the L2 Norm or Euclidean length of a matrix (square root of the sum of the squares of all elements).
```ts
const length = mj.norm(weights);
```

---

### B. Arithmetic & Element-wise Functions

These functions process each element independently and support an optional `out` parameter for memory optimization.

| Function | Description | Example |
| :--- | :--- | :--- |
| `add`, `sub` | Addition & Subtraction | `mj.add(m, 10)` |
| `mul`, `div` | Multiplication & Division | `mj.mul(m1, m2)` |
| `absm(a)` | Absolute value per element | `mj.absm(m)` |
| `expm(a)` | Exponential (`e^x`) | `mj.expm(m)` |
| `logm(a)` | Natural logarithm (`ln`) | `mj.logm(m)` |
| `map(a, f)` | Custom function per element | `mj.map(m, x => x * x)` |

```ts
const res = mj.add(a, b, outputBuffer); // Avoids new allocation
```

Important contract notes:
- `mj.add`, `mj.sub`, and `mj.mul` can use an optional `out` parameter to reuse a buffer.
- `mj.div` currently does not accept an `out` parameter.
- `mj.map(a, f)` returns a new `Matrix`. This differs from `matrix.map(f)` on a `Matrix` instance, which works in-place.
- For `mj.add(a, b, out)` and `mj.sub(a, b, out)`, the `out` buffer must be separate from the `a` and `b` buffers.

---

### C. Reduction Operations (Producing One Number)

These functions summarize the entire contents of a matrix into a single value. Very useful for calculating loss, performance metrics, or data aggregation.

#### `mj.mean(a)`
Calculates the average value of all elements in the matrix.
```ts
const a = mj.matrix([[2, 4], [6, 8]]);
const avg = mj.mean(a); // (2+4+6+8) / 4 = 5
```

#### `mj.dotSum(a)`
Calculates the total sum ($\sum$) of all elements.
```ts
const a = mj.matrix([[1, 2], [3, 4]]);
const total = mj.dotSum(a); // 1+2+3+4 = 10
```

#### `mj.dotMul(a)`
Calculates the total product ($\prod$) of all elements.
```ts
const a = mj.matrix([[1, 2], [3, 4]]);
const total = mj.dotMul(a); // 1*2*3*4 = 24
```

#### `mj.dotSub(a)`
Calculates the total sequential subtraction of all elements (starting from 0).
```ts
const a = mj.matrix([[1, 2], [3, 4]]);
const res = mj.dotSub(a); // 0 - 1 - 2 - 3 - 4 = -10
```

#### `mj.dotDiv(a)`
Calculates the total sequential division of all elements (starting from 1).
```ts
const a = mj.matrix([[2, 5]]);
const res = mj.dotDiv(a); // 1 / 2 / 5 = 0.1
```

---

### D. Statistics & Axis Reduction

#### `mj.sumAxis(a, axis, out?)`
Sums values along a specific direction.
- **`axis 1`**: Sums rows (Result: Column Matrix `[rows x 1]`).
- **`axis 0`**: Sums columns (Result: Row Matrix `[1 x cols]`).

```ts
const a = mj.matrix([[1, 2], [3, 4]]);
const sumRows = mj.sumAxis(a, 1); // [[3], [7]]
```

---

### E. Generators & Matrix Initialization

These functions are used to create new matrices with specific initial values according to model architecture requirements. All functions in this category accept one main parameter: **`shape: [rows, cols]`**.

#### `mj.zeros([rows, cols])`
Creates a matrix where all elements are **0**. Highly efficient as it uses default `Float32Array` initialization.
```ts
const z = mj.zeros([2, 3]);
// Result:
// [[0, 0, 0],
//  [0, 0, 0]]
```

#### `mj.ones([rows, cols])`
Creates a matrix where all elements are **1**.
```ts
const o = mj.ones([2, 2]);
// Result:
// [[1, 1],
//  [1, 1]]
```

#### `mj.random([rows, cols])`
Creates a matrix with uniform random values between **0** and **1**.
```ts
const r = mj.random([3, 1]); // Random column vector
```

#### `mj.xavier([rows, cols])`
Xavier (Glorot) initialization that keeps activation variance constant across layers. Highly recommended for layers using **Sigmoid** or **Tanh** activation functions.
```ts
const w = mj.xavier([128, 64]);
```

#### `mj.he([rows, cols])`
He (Kaiming) initialization optimized for layers with **ReLU** activation functions. This prevents *vanishing gradients* in very deep networks.
```ts
const w = mj.he([64, 10]);
```

---

### F. Specialized Operations (Deep Learning)

#### `mj.convolution(a, kernel)`
Basic 2D convolution operation for spatial feature extraction. Filter (kernel) slides over input to produce new activation paths.
```ts
const input = mj.matrix([
  [1, 1, 1, 0, 0],
  [0, 1, 1, 1, 0],
  [0, 0, 1, 1, 1],
  [0, 0, 1, 1, 0],
  [0, 1, 1, 0, 0]
]);

const kernel = mj.matrix([
  [1, 0, 1],
  [0, 1, 0],
  [1, 0, 1]
]);

const features = mj.convolution(input, kernel);
// Calculation Result (3x3):
// [[4, 3, 4],
//  [2, 4, 3],
//  [2, 3, 4]]
```

#### `mj.clipGradients(a, limit)`
Limits the absolute values in a matrix **in-place** to be within the range `[-limit, limit]`. This is crucial for preventing *Exploding Gradients* where gradient values become too large and damage training stability.
```ts
const grads = mj.matrix([
  [0.5,  5.0], // 5.0 exceeds limit
  [-10.2, 0.1] // -10.2 is smaller than -limit
]);

mj.clipGradients(grads, 1.0);

// grads now CHANGES to:
// [[0.5,  1.0],
//  [-1.0, 0.1]]
```

#### `mj.reshape(a, shape)` & `mj.flatten(a)`
Standalone functions for matrix shape manipulation without new allocation.
```ts
const a = mj.matrix([[1, 2], [3, 4]]); // [2, 2]

const reshaped = mj.reshape(a, [1, 4]);
// reshaped: [[1, 2, 3, 4]]

const flat = mj.flatten(a);
// flat: [[1, 2, 3, 4]]
```

---

## 3. Activation Functions (`src/activation`)

Activation functions introduce non-linearity into neural networks, allowing the model to learn complex patterns. In ML-V1, almost all activation functions return a tuple **`[Matrix, Matrix]`**:
1.  **Activation Result (Forward)**: Output passed to the next layer.
2.  **Gradient/Derivative (Backward)**: Used to calculate error correction during backpropagation.

---

### A. Sigmoid
Transforms input values to the range **(0, 1)**. Very commonly used in output layers for binary classification.

```ts
import { sigmoid } from "./src/activation";

const input = mj.matrix([[-1, 0, 2]]);
const [out, grad] = sigmoid(input);

// out (Activation Result):
// [[0.268, 0.5, 0.880]]

// grad (Derivative):
// [[0.196, 0.25, 0.105]]
```

---

### B. ReLU (Rectified Linear Unit)
The most popular activation function. Transforms all negative values to **0** and leaves positive values unchanged.

```ts
import { relu } from "./src/activation";

const input = mj.matrix([[ -1.5, 0.5, 2.0 ]]);
const [out, grad] = relu(input);

// out (Only positive values pass):
// [[ 0, 0.5, 2.0 ]]

// grad (1 if input > 0, else 0):
// [[ 0, 1, 1 ]]
```

---

### C. Tanh (Hyperbolic Tangent)
Similar to sigmoid but its output range is **(-1, 1)**. It often provides better performance in hidden layers compared to sigmoid.

```ts
import { tanh } from "./src/activation";

const input = mj.matrix([[ -1, 0, 1 ]]);
const [out, grad] = tanh(input);

// out (Mapped to range -1 to 1):
// [[ -0.761, 0, 0.761 ]]

// grad (1 - out^2):
// [[ 0.419, 1, 0.419 ]]
```

---

### D. Softmax (Multi-Class Output)
Produces a probability distribution where the sum of all elements is **1.0**.

#### `softmax(a, row = false)`
- **`row = true`**: Calculates probability per row (Standard batch).
- **`row = false`**: Calculates probability per column.

```ts
import { softmax } from "./src/activation";

const logits = mj.matrix([[ 1, 2, 3 ]]);
const [probs, dSoftmax] = softmax(logits, true);

// probs (Total sum is 1.0):
// [[ 0.09, 0.24, 0.66 ]]
```

Notes:
- The second value from `softmax(...)` is an approximation of the diagonal gradient `s * (1 - s)`, not the full Softmax Jacobian.
- For Softmax backpropagation against incoming error, use `softmaxBackward(...)` or `softmaxBackwardInto(...)`.

---

### E. Leaky ReLU (lRelu)
A ReLU variant that provides a small value (leak) for negative inputs (multiplier $10^{-5}$) to prevent the "dying neuron" problem.

```ts
import { lRelu } from "./src/activation";

const input = mj.matrix([[ -1, 1 ]]);
const [out, grad] = lRelu(input);

// out:
// [[ -0.00001, 1 ]]
```

---

### F. Linear (Identity)
Usually used in output layers for **Regression** tasks.

```ts
import linear from "./src/activation"; // Default export

const [out, grad] = linear(inputMatrix);
// out: identical to input
// grad: contains the number 1 (since derivative of x is 1)
```

---

> [!TIP]
> When performing a *Manual Training Loop*, ensure you save the `grad` matrix (the second element of the tuple) for use in calculating weight updates (Update Weights).

---

## 4. Layers & Models (`src/layers` & `src/models`)

This section documents the main building blocks for constructing artificial neural networks, from model containers to various specialized layer types.

---

### A. Model Container: `Sequential`

`Sequential` is a wrapper model that allows you to stack layers sequentially.

#### `constructor()`
Creates a new model instance.
```ts
import { Sequential } from "./src/models";
const model = new Sequential();
```

#### `add(layer)`
Adds a layer to the execution sequence.
```ts
model.add(new Dense({ units: 4, outputUnits: 2 }));
```

#### `forward(input)` & `predict(input)`
Passes data through all layers. `predict` automatically disables training mode (like Dropout).
```ts
const output = model.predict(inputMatrix);
```

#### `compile({ alpha, optimizer, error, clipGradient })`
Configures learning parameters globally for all layers in the model.
- **`alpha`**: Learning rate.
- **`optimizer`**: Optimizer name (e.g., `"adam"`).
- **`error`**: Global loss function name (e.g., `"mse"`, `"softmaxCrossEntropy"`).
- **`clipGradient`**: Custom gradient clipping limit (number or boolean).

#### `fit(X, y, epochs, config?): FitResult`
Automatically trains the model using input and target data pairs. Supports batching, validation split, early stopping, shuffle, verbose logging, and per-epoch callbacks.

##### Supported Signatures

```ts
// 1. New config-based API (recommended)
const result = model.fit(X, y, epochs, config?: FitConfig): FitResult;

// 2. Legacy callback (backward compatible)
model.fit(X, y, epochs, (loss: number) => void): FitResult;
```

##### `FitConfig` Parameters

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `batchSize` | `number` | `max(1, floor(N/10))` | Number of samples per mini-batch |
| `validationSplit` | `number` | `0` | Proportion of data for validation (0–1, exclusive) |
| `earlyStoppingPatience` | `number` | `Infinity` | Epochs without improvement before training stops |
| `shuffle` | `boolean` | `true` | Shuffle training order every epoch |
| `verbose` | `boolean` | `false` | Print loss progress to console every epoch |
| `onEpochEnd` | `(epoch, loss, valLoss?) => void` | `() => {}` | Callback after each epoch finishes |
| `monitorMetric` | `"loss" \| "valLoss"` | `"valLoss"` if validation exists, else `"loss"` | Metric monitored for early stopping |
| `minDelta` | `number` | `0` | Minimum change considered an improvement |
| `mode` | `"min" \| "max"` | `"min"` | `"min"` = stop if no decrease, `"max"` = stop if no increase |
| `trimPadding` | `boolean` | `true` | Dynamically trims PAD from each batch before forward/backward. Only active for full-sequence targets (Y.shape[0] === X.shape[0]) and models supporting `getPadTokenId()` / `setPositionOffset()` (e.g., Transformers). For other models or legacy targets Y=[1,batch], training proceeds normally without trimming. |
| `paddingSide` | `"left" \| "right"` | `"right"` | Padding side on input data. `"right"` trims trailing PAD (recommended for full-sequence causal LM). `"left"` trims leading PAD and adjusts positional encoding offset. |

##### `FitResult` Return Value

```ts
interface FitResult {
  history: {
    loss: number[];      // Training loss per epoch
    valLoss?: number[];  // Validation loss per epoch (exists if validationSplit > 0)
  };
  bestEpoch: number;       // Index of epoch with best loss (0-indexed)
  bestLoss: number;        // Best recorded loss value
  stoppedEarly: boolean;   // true if early stopping was triggered
  stoppingEpoch?: number;  // Epoch where early stopping occurred
}
```

##### Usage Examples

###### New API (Recommended)
```ts
const result = model.fit(trainData, labels, 100, {
  batchSize: 16,
  validationSplit: 0.2,
  earlyStoppingPatience: 10,
  verbose: true,
  onEpochEnd: (epoch, loss, valLoss) => {
    console.log(`Epoch ${epoch}: loss=${loss.toFixed(4)}, valLoss=${valLoss?.toFixed(4)}`);
  },
});

console.log(`Best epoch: ${result.bestEpoch}, Best loss: ${result.bestLoss}`);
console.log("Training history:", result.history.loss);
```

###### Legacy Callback (Still Supported)
```ts
model.fit(trainData, labels, 100, (loss) => {
  console.log(`Current Loss: ${loss}`);
});
```

---

### B. Autoencoder Model: `DimentionalityReduction`

`DimentionalityReduction` is a derivative of `Sequential` for simple encoder-decoder / autoencoder scenarios.

Additional behaviors:
- Splits `layers` into `layersEncode` and `layersDecode`.
- Encoder boundary determined by the first layer with `status === "outputReduction"`.
- Calling `fit(X, epochs, config)` automatically assumes the target is the same as the input (`X -> X`).

#### `constructor({ layers })`
```ts
import { DimentionalityReduction } from "./src/models";
import { Dense } from "./src/layers";

const model = new DimentionalityReduction({
  layers: [
    new Dense({ units: 8, outputUnits: 4, activation: "relu", status: "input" }),
    new Dense({ units: 4, outputUnits: 2, activation: "relu", status: "outputReduction" }),
    new Dense({ units: 2, outputUnits: 4, activation: "relu" }),
    new Dense({ units: 4, outputUnits: 8, activation: "linear", status: "output", loss: "mse" }),
  ],
});
```

#### `encode(x)` & `decode(enc)`
```ts
const latent = model.encode(inputMatrix);
const reconstructed = model.decode(latent);
```

#### `fit(X, epochs, config?)`
Autoencoder training shortcut that calls `super.fit(X, X, epochs, config)`.

```ts
const result = model.fit(trainX, 50, { batchSize: 8, verbose: true });
```

---

### C. Transformers Model

A complete Transformer architecture model (based on the `Sequential` architecture) for causal language modeling.

Important changes in this version:
- Training path now uses **full-sequence causal LM**.
- Default inference path remains **last-token logits**, but `predict()` can now be switched to full-sequence.
- Architecture now supports **multi-block depth** via `numBlocks`.
- Correct training target is **shifted next-token targets** with shape `[seqLen, batch]`.
- Legacy `backward(y)` contract with target `[1, batch]` is still accepted as a limited compatibility path, but it is no longer the recommended training path.

#### Shape Contracts

- **Input token IDs**: `Matrix` with shape `[seqLen, batch]`.
- **Training logits** (`model.train(); model.forward(x)` or `model.forwardFullSequence(x)`): `[vocabSize, seqLen * batch]`.
- **Inference logits**:
  - `model.eval(); model.forward(x)` or `model.forwardNextToken(x)`: `[vocabSize, batch]`.
  - `model.predict(x)`: follows `predictMode`.
- **Training target**: `Matrix` sparse index `[seqLen, batch]`.
- **Legacy target**: `Matrix` sparse index `[1, batch]`.

Training logits column order is **sample-major**:
- sample 0 positions `0..seqLen-1`
- sample 1 positions `0..seqLen-1`
- and so on.

Valid positions for full-sequence loss:
- causal shift must be valid.
- current input token is not `padTokenId`.
- shifted target token is also not `padTokenId`.

#### `constructor(config)`
- **`units`**: Model dimension (`d_model`).
- **`seqLen`**: Input sequence length.
- **`vocabSize`**: Vocabulary size.
- **`heads`**: Number of attention heads (default: 8).
- **`numBlocks`**: Number of stacked Transformer blocks (default: 1).
- **`dropoutRate`**: Dropout rate (default: 0.1).
- **`alpha`**: Learning rate (default: 0.01).
- **`padTokenId`**: Padding token to be ignored in embedding, attention mask, and full-sequence loss.
- **`clipGradient`**: Global gradient clipping limit for all sub-layers (default: 5.0).
- **`predictMode`**: Default output mode for `predict()`. Options: `"next-token"` (default) or `"full-sequence"`.

```ts
import { Transformers } from "./src/models";

const model = new Transformers({
  units: 128,
  seqLen: 50,
  vocabSize: 5000,
  heads: 8,
  numBlocks: 4,
  padTokenId: 0,
  clipGradient: 1.5,
  predictMode: "next-token",
});
```

#### Internal Architecture

Each Transformer block contains:
- `LayerNormalization -> MultiHeadAttention -> Dropout -> Residual`
- `LayerNormalization -> Dense(4x units, relu) -> Dropout -> Dense(units, linear) -> Dropout -> Residual`

If `numBlocks = 1`, the behavior and basic topology are equivalent to the old single-block version.

If `numBlocks > 1`, all blocks are executed sequentially during forward passes and unrolled in reverse during backward passes.

#### `forward(input)`

The behavior of `forward()` now depends on the model mode:
- In `model.train()` mode: returns full-sequence logits `[vocabSize, seqLen * batch]`.
- In `model.eval()` mode: returns last-token logits `[vocabSize, batch]`.

This intentionally separates the default training path and inference path without removing existing generation ergonomics.

#### `forwardFullSequence(input)`

Forces the training/full-sequence path regardless of the current mode.

Use this if you want to be explicit that the desired output is logits for the entire sequence.

#### `forwardNextToken(input)`

Forces the last-token path regardless of the default `forward()` method.

The behavior depends on the model mode:
- In `model.eval()` mode: returns last-token logits `[vocabSize, batch]` via a fast inference projector.
- In `model.train()` mode: still returns logits `[vocabSize, batch]`, but through a path compatible with `backward([1, batch])` for legacy next-token training.

Use this for sampling the next token in a generation loop, or for legacy training loops that only train on the last token.

#### `predict(input)`

`predict()` is now the single inference entry point, and its output shape follows the `predictMode` set in the constructor:

- `predictMode: "next-token"` -> logits `[vocabSize, batch]`.
- `predictMode: "full-sequence"` -> logits `[vocabSize, seqLen * batch]`.

The default is `"next-token"` to maintain compatibility with older generation loops.

#### `setPredictMode(mode)` / `getPredictMode()`

Use these to change the `predict()` mode without creating a new instance.

```ts
model.setPredictMode("full-sequence");
const logitsAll = model.predict(x); // [vocabSize, seqLen * batch]

model.setPredictMode("next-token");
const nextLogits = model.predict(x); // [vocabSize, batch]
```

#### `backward(target)`

Recommended target contract:
- **shape**: `[seqLen, batch]`.
- **content**: next-token targets shifted one position to the left.
- **last row**: generally filled with `padTokenId` as there is no subsequent token.

Loss is calculated for all valid non-pad positions, not just the single last position.

Compatibility path:
- A `[1, batch]` target is still accepted for legacy loops that only train on the last token.
- This path is maintained only to minimize breaking changes, not as a new best practice.
- If using this path, call `model.train(); model.forwardNextToken(x); model.backward(yLastToken)` to ensure logits and backward cache are consistent.

#### `save(path)` / `load(path)`

`Transformers` still uses a flat-array serialization format similar to the `Sequential` model but can now save multiple blocks.

Important rules:
- Old single-block models can still be loaded.
- New multi-block models can also be saved/loaded.
- The instance calling `load()` must be created with the same `numBlocks` as the model artifact.

Safe example:

```ts
const model = new Transformers({
  units: 128,
  seqLen: 50,
  vocabSize: 5000,
  heads: 8,
  numBlocks: 4,
  padTokenId: 0,
});

model.load("transformer_model.json");
```

If the model artifact has a different number of blocks than the current instance, `load()` will throw an explicit error.

#### Full-Sequence Training Example

```ts
import mj from "./src/math";
import { Transformers } from "./src/models";

const padTokenId = 0;
const model = new Transformers({
  units: 64,
  seqLen: 6,
  vocabSize: 2000,
  heads: 8,
  numBlocks: 2,
  alpha: 0.001,
  padTokenId,
});

const x = mj.matrix([
  [0, 0],
  [11, 21],
  [12, 22],
  [13, 23],
  [14, 24],
  [15, 25],
]);

const y = mj.matrix([
  [0, 0],
  [12, 22],
  [13, 23],
  [14, 24],
  [15, 25],
  [0, 0],
]);

model.train();
const logits = model.forward(x); // [vocabSize, seqLen * batch]
model.backward(y);
console.log(logits._shape, model.loss);
```

#### Inference / Generation Example

```ts
import mj from "./src/math";
import { Transformers } from "./src/models";

const model = new Transformers({
  units: 64,
  seqLen: 6,
  vocabSize: 2000,
  heads: 8,
  numBlocks: 2,
  alpha: 0.001,
  padTokenId: 0,
  predictMode: "next-token",
});

model.eval();
const x = mj.matrix([
  [0],
  [11],
  [12],
  [13],
  [14],
  [15],
]);

const nextTokenLogits = model.predict(x); // [vocabSize, 1]
```

#### Full-Sequence Inference Example via `predict()`

```ts
import mj from "./src/math";
import { Transformers } from "./src/models";

const model = new Transformers({
  units: 64,
  seqLen: 6,
  vocabSize: 2000,
  heads: 8,
  numBlocks: 2,
  alpha: 0.001,
  padTokenId: 0,
  predictMode: "full-sequence",
});

const x = mj.matrix([
  [0],
  [11],
  [12],
  [13],
  [14],
  [15],
]);

const logits = model.predict(x); // [vocabSize, seqLen]
```

#### Legacy Next-Token Training Example

```ts
import mj from "./src/math";
import { Transformers } from "./src/models";

const model = new Transformers({
  units: 64,
  seqLen: 6,
  vocabSize: 2000,
  heads: 8,
  numBlocks: 2,
  alpha: 0.001,
  padTokenId: 0,
});

const x = mj.matrix([
  [11],
  [12],
  [13],
  [14],
  [15],
  [16],
]);

const yLastToken = mj.matrix([[17]]);

model.train();
const logits = model.forwardNextToken(x); // [vocabSize, 1]
model.backward(yLastToken); // compatibility path [1, batch]
console.log(logits._shape, model.loss);
```

#### Best Practices

- Use shifted targets `[seqLen, batch]`, not single targets `[1, batch]`, for correct LM training.
- Maintain consistency in `seqLen`, `vocabSize`, and `padTokenId` across the tokenizer, preprocessing, and model.
- Start with `numBlocks=2` or `numBlocks=4` for deeper models, then benchmark as runtime costs will increase significantly.
- Ensure padding positions in the target remain `padTokenId` so that loss is not calculated for padding areas.
- Use `predictMode: "next-token"`, then `model.predict()` in generation loops to keep sampling from last-token logits.
- Use `model.forwardFullSequence()` if you need to inspect logits for all positions during evaluation or debugging.
- To unify the inference API into a single method, use `predict()` + `predictMode`, and treat `forwardNextToken()` / `forwardFullSequence()` as advanced explicit methods.
- Enable `trimPadding: true` (default) in `fit()` for optimal performance when data contains significant padding.

#### Dynamic Padding Trim

As of version 2.2.0, `Transformers` supports dynamic per-batch PAD trimming during training via the `trimPadding` and `paddingSide` options in `FitConfig`.

##### When to use `paddingSide="right"` (default)

Use this when the dataset is already in **right-padded** format:
```
[token0, token1, ..., tokenN, PAD, PAD]
```
- Positional encoding for original tokens still starts from position 0.
- Dynamic trimming cuts trailing PAD at the end.
- `positionOffset = 0`.

Usage example:
```ts
model.fit(trainX, trainY, epochs, {
  batchSize: 8,
  trimPadding: true,
  paddingSide: "right",
  shuffle: true
});
```

##### When to use `paddingSide="left"`

Use this when legacy datasets are still in **left-padded** format:
```
[PAD, PAD, token0, token1, ..., tokenN]
```
- The library trims leading PAD and sets `positionOffset = firstUsefulPos`.
- Absolute positional encoding for original tokens remains the same after trimming.

Usage example:
```ts
model.fit(trainX, trainY, epochs, {
  batchSize: 8,
  trimPadding: true,
  paddingSide: "left",
  shuffle: true
});
```

##### Correctness Notes

- `trimPadding` is only active for full-sequence targets with shape `Y=[seqLen, batch]`.
- Legacy last-token targets `Y=[1, batch]` are not trimmed.
- PAD is still ignored in loss/gradient calculation (via `buildShiftedLossGradient`).
- Trimming does not modify non-PAD tokens.
- for left-padding, `positionOffset` keeps positional encoding consistent.

##### Performance Notes

- `trimPadding` does not change the model's `maxSeqLen`; `seqLen`/`contextLen` can still be 1024.
- What changes is the `effectiveSeqLen` per batch (usually much smaller).
- Attention cost drops from O(seqLen²) to O(effectiveSeqLen²).
- Dense output projection cost drops from `vocabSize × seqLen × batch` to `vocabSize × effectiveSeqLen × batch`.
- For best results, use right-padded data and set `paddingSide: "right"`.

##### Configuration Example for Long Contexts

```ts
const model = new Transformers({
  units: 64,
  seqLen: 1024,
  vocabSize,
  heads: 8,
  numBlocks: 2,
  padTokenId: 0
});

model.fit(trainX, trainY, 80, {
  batchSize: 8,
  trimPadding: true,
  paddingSide: "right",
  shuffle: true
});
```

##### API Bridge Methods

Transformers exposes several public methods for manual or advanced use:

| Method | Description |
| :--- | :--- |
| `getPadTokenId(): number \| null` | Returns `padTokenId` from the embedding layer. |
| `setPositionOffset(n: number): this` | Sets the PE position offset for the next batch (used for left-padding trim). |
| `resetPositionOffset(): this` | Resets the position offset back to 0. Automatically called by `fit()` after each batch. |
| `resizeVocab(newVocabSize: number)` | Expand the embedding vocabulary and output projector to sync with the growing tokenizer. |
| `enableProfiling(enabled = true): this` | Enables stage-by-stage internal profiler. |
| `disableProfiling(): this` | Disables the internal profiler. |
| `resetProfiling(): void` | Clears stored profiler statistics. |
| `getProfilingReport(reset = false)` | Returns a `{ totalMs, avgMs, count }` summary per profiler stage. |

#### Migration Note

Before this refactor, transformer training only used the last-token representation to predict the next token.

After the refactor:
- Default training uses a full-sequence causal LM objective.
- The output shape of `forward()` in training mode changes from `[vocabSize, batch]` to `[vocabSize, seqLen * batch]`.
- `backward()` ideally receives a shifted target of `[seqLen, batch]`.
- Last-token inference paths are maintained via `forwardNextToken()` and `predict()` when `predictMode="next-token"`.
- Legacy last-token training compatibility is also available through `forwardNextToken()` when the model is in training mode.
- The architecture can now be scaled in depth with `numBlocks` without changing the primary training/inference APIs.
- `trimPadding: true` (default) is active for Transformers and reduces computational costs on batches with significant padding.

If previous training data was left-padded, set `paddingSide: "left"`.
If creating new datasets for full-sequence causal LM, use right-padding and set `paddingSide: "right"`.
If you want the old behavior without trimming, set `trimPadding: false`.

---

---

### D. Dense Layer (Fully Connected)

A standard layer where every input is connected to every output.

#### `constructor(config)`
- **`units`**: Number of input neurons.
- **`outputUnits`**: Number of output neurons.
- **`activation`**: Activation function name (e.g., `"relu"`, `"sigmoid"`).
- **`alpha`**: Layer-specific learning rate.
- **`loss`**: Loss function for the output layer (e.g., `"mse"`, `"softmaxCrossEntropy"`).
- **`optimizer`**: Optimization algorithm (e.g., `"sgd"`, `"adam"`).
- **`status`**: Layer status (`"input"`, `"hidden"`, `"output"`, etc. depending on architecture).
- **`clipGradient`**: Specific gradient clipping limit for this layer (default: 5.0).

> [!IMPORTANT]
> `Dense` will throw an error if you combine `activation: "softmax"` with `loss: "softmaxCrossEntropy"`, as that would apply softmax twice.

```ts
import { Dense } from "./src/layers";

const layer = new Dense({
  units: 128,
  outputUnits: 64,
  activation: "relu",
  optimizer: "adam"
});
```

---

### E. Embedding Layer

Used to transform word indices (integers) into dense vectors. Crucial for NLP tasks.

#### `constructor(config)`
- **`vocabSize`**: Total dictionary size.
- **`embeddingDim`**: Vector dimension for each word.
- **`alpha`**: Layer-specific embedding learning rate.
- **`status`**: Layer status in the model graph.
- **`optimizer`**: Optimizer used for the embedding table.
- **`padTokenId`**: PAD token ID to be ignored during backward passes.

```ts
import { Embedding } from "./src/layers";

const embed = new Embedding({
  vocabSize: 5000,
  embeddingDim: 128
});
```

---

### F. Multi-Head Attention

The core of the Transformer architecture that allows the model to focus on different parts of the input simultaneously.

#### `constructor(config)`
- **`units`**: Internal dimension (must be divisible by the number of `heads`).
- **`heads`**: Number of parallel attention mechanisms.
- **`seqLen`**: Maximum input sequence length.
- **`alpha`**: Layer-specific attention learning rate.
- **`status`**: Layer status in the model graph.
- **`clipGradient`**: Gradient clipping limit (default: 5.0).

```ts
import { MultiHeadAttention } from "./src/layers";

const attention = new MultiHeadAttention({
  units: 512,
  heads: 8,
  seqLen: 128
});
```

---

### G. Recurrent Layer Family (`RNN`, `LSTM`, `GRU`)

The recurrent layer family is used for sequential data with the input format **`[features, seqLen]`** for a single sequence sample.

#### General Conventions
- **Input**: `Matrix` with shape `[units, seqLen]`.
- **`returnSequences: false`**: output shape `[hiddenUnits, 1]` for `RNN`/`LSTM`, or `[hiddenUnits * 2, 1]` for bidirectional `GRU`.
- **`returnSequences: true`**: output shape `[hiddenUnits, seqLen]` for `RNN`/`LSTM`, or `[hiddenUnits * 2, seqLen]` for bidirectional `GRU`.
- **`stateful: true`**: hidden state is carried over to the next `forward()` call until `resetState()` is called.
- **`returnState`**: currently **not supported** across the entire recurrent family and will throw an explicit error when `forward()` is called.
- **`Sequential.fit()`**: for the generic recurrent path, currently use **`batchSize=1`**. If `stateful=true`, avoid `shuffle=true` and `validationSplit > 0`.
- **`getState()` / `resetState()`**: available on the entire recurrent family for state inspection and resetting.
- **`save()` / `load()`**: the entire recurrent family saves weights, important configurations, and stateful hidden states.

#### `RNN(config)`
Basic recurrent layer with one hidden state and Backpropagation Through Time (BPTT).

##### `constructor(config)`
- **`units`**: Number of input features per time step.
- **`hiddenUnits`**: Number of hidden state units.
- **`activation`**: Recurrent activation (`"tanh"` by default, or `"relu"`).
- **`returnSequences`**: If `true`, returns output for every time step.
- **`returnState`**: Stored in configuration, but not yet supported during inference/training.
- **`stateful`**: If `true`, the last hidden state is maintained between calls.
- **`optimizer`**: Recurrent parameter optimizer.
- **`clipGradient`**: Gradient clipping limit (default: `5.0`).
- **`loss`**: Loss name stored for output layer compatibility/state serialization.

```ts
import mj from "./src/math";
import { RNN } from "./src/layers";

const layer = new RNN({
  units: 8,
  hiddenUnits: 16,
  activation: "tanh",
  returnSequences: true,
  stateful: false,
});

const x = mj.matrix([
  [1, 2, 3],
  [0, 1, 0],
  [1, 0, 1],
  [0, 0, 1],
  [1, 1, 1],
  [0, 1, 1],
  [1, 0, 0],
  [0, 0, 0],
]); // [8, 3]

const out = layer.forward(x); // [16, 3] since returnSequences=true
```

#### `LSTM(config)`
Recurrent layer with **cell state** and input/forget/output gates to handle longer sequence dependencies.

##### `constructor(config)`
- **`units`**: Number of input features per time step.
- **`hiddenUnits`**: Number of hidden state and cell state units.
- **`returnSequences`**: If `true`, returns output for every time step.
- **`returnState`**: Not yet supported and will throw an explicit error during `forward()`.
- **`stateful`**: If `true`, hidden state and cell state are maintained between calls.
- **`optimizer`**: LSTM gate parameter optimizer.
- **`clipGradient`**: Gradient clipping limit (default: `5.0`).
- **`getState()`**: Returns an object `{ h, c }`.

```ts
import mj from "./src/math";
import { LSTM } from "./src/layers";

const layer = new LSTM({
  units: 8,
  hiddenUnits: 32,
  returnSequences: false,
  stateful: true,
});

const out = layer.forward(
  mj.matrix([
    [1, 2, 3, 4],
    [0, 1, 0, 1],
    [1, 0, 1, 0],
    [0, 0, 1, 1],
    [1, 1, 1, 1],
    [0, 1, 1, 0],
    [1, 0, 0, 1],
    [0, 0, 0, 1],
  ])
); // [32, 1]

layer.resetState(); // Clear stateful hidden/cell states
```

#### `GRU(config)`
Recurrent layer with **update/reset** gates. This implementation also supports **`bidirectional`** mode.

##### `constructor(config)`
- **`units`**: Number of input features per time step.
- **`hiddenUnits`**: Number of hidden state units per direction.
- **`bidirectional`**: If `true`, runs GRU forward and backward and concatenates their outputs.
- **`returnSequences`**: If `true`, returns output for every time step.
- **`returnState`**: Not yet supported and will throw an explicit error during `forward()`.
- **`stateful`**: If `true`, hidden states per direction are maintained between calls.
- **`optimizer`**: GRU gate parameter optimizer.
- **`clipGradient`**: Gradient clipping limit (default: `5.0`).
- **`getState()`**: Returns an object `{ forward, backward? }` depending on bidirectional mode.

```ts
import mj from "./src/math";
import { GRU } from "./src/layers";

const layer = new GRU({
  units: 8,
  hiddenUnits: 16,
  bidirectional: true,
  returnSequences: true,
});

const out = layer.forward(
  mj.matrix([
    [1, 2, 3],
    [0, 1, 0],
    [1, 0, 1],
    [0, 0, 1],
    [1, 1, 1],
    [0, 1, 1],
    [1, 0, 0],
    [0, 0, 0],
  ])
); // [32, 3] = 16 forward + 16 backward
```

#### Practical Notes
- For sequence modeling within `Sequential`, recurrent layers are usually followed by a `Dense` output layer.
- If `returnSequences=false`, the layer returns the representation from the last time step.
- `Sequential.fit()` on the generic recurrent path is safe for simple learning tests, but is still limited to `batchSize=1`.
- The `save()`/`load()` methods preserve recurrent weights and internal stateful states.
- `RNN.getState()` returns a `Matrix`, `LSTM.getState()` returns `{ h, c }`, and `GRU.getState()` returns `{ forward, backward? }`.

---

### H. Other Utility Layers

- **`Flatten`**: Flattens a matrix into one dimension (usually before a Dense layer).
- **`Dropout({ rate })`**: Randomly deactivates neurons to prevent overfitting.
- **`LayerNormalization({ units, clipGradient })`**: Stabilizes the distribution of values within the network.
- **`Convolution({ kernelSize, inputShape, activation, clipGradient })`**: 2D filter operation for spatial data (Images).
- **`SelfAttention({ units, alpha, clipGradient })`**: Basic attention mechanism for a single input.

---

## 5. Preprocessing & Tokenizer (`src/tokenizer`)

Before text can be processed by a machine learning model, it must be converted into numbers. ML-V1 uses the highly efficient **BPE (Byte Pair Encoding)** algorithm to handle large vocabularies and Out-of-Vocabulary (OOV) words.

---

### A. BPETokenizer

`BPETokenizer` works by splitting rare words into subwords (word pieces) and keeping popular words as single whole tokens.

#### `constructor(config)`
- **`vocabSize`**: Target vocabulary size (e.g., `5000`).
- **`minFrequency`**: Minimum number of character pair occurrences to be merged (default: `2`).
- **`preTokenizer`**: Built-in pre-tokenizer name or custom `(text: string) => string[]` function. Default: `"char"`.
- **`specialTokens`**: Additional special tokens to be maintained in the vocabulary.

```ts
import { BPETokenizer } from "./src/tokenizer";

const tokenizer = new BPETokenizer({
  vocabSize: 1000,
  minFrequency: 2
});
```

#### Unicode and Multilingual Tokenization

ML-V1 supports custom and built-in pre-tokenizers for non-Latin text.

Supported modes:
- `char`
- `unicode-grapheme`
- `unicode-word`
- `whitespace`
- `script-aware`

`"char"` is the default for backward compatibility and splits text into Unicode code points without breaking surrogate pairs. For multilingual corpora, prefer `"unicode-grapheme"` when you want safer character clusters, or `"script-aware"` when the corpus mixes Latin, Arabic, Japanese, Mandarin Chinese, Thai, Korean, Javanese, emoji, math symbols, and punctuation.

```ts
import { BPETokenizer } from "@akhyar11/ml-v1";

const tokenizer = new BPETokenizer({
  vocabSize: 1000,
  preTokenizer: "script-aware"
});

tokenizer.train([
  "hello world",
  "مرحبا بالعالم",
  "こんにちは世界",
  "你好世界",
  "ภาษาไทย",
  "한국어테스트",
  "ꦱꦺꦴꦥꦺꦴ",
  "x² + y² = z²",
  "hello ꦱꦺꦴꦥꦺꦴ 😊 你好"
]);
```

Javanese example:

```ts
import { scriptAwarePreTokenizer } from "@akhyar11/ml-v1";

const tokens = scriptAwarePreTokenizer("ꦱꦺꦴꦥꦺꦴ");
// ["ꦱꦺꦴ", "ꦥꦺꦴ"]
```

BPE alone is not enough for every writing system. Pre-tokenization is required for many scripts because word boundaries, combining marks, and emoji sequences are not represented well by naive string splitting. `script-aware` is a general built-in mode; users can pass custom pre-tokenizers for language-specific needs. `Intl.Segmenter` improves grapheme and word segmentation when available. Fallback behavior is deterministic but may be less linguistically accurate.

Built-in pre-tokenizer names are saved in tokenizer JSON files. Custom pre-tokenizer functions are not serialized; saved metadata records `"custom"`, and users must pass the same function again when loading:

```ts
const loaded = BPETokenizer.load("./model/vocab.json", { preTokenizer: myPreTokenizer });
```

#### `train(texts: string[])`
Trains the tokenizer to recognize word patterns from a corpus.
```ts
const corpus = ["saya makan nasi", "kamu makan roti"];
tokenizer.train(corpus);
```

#### `update(texts: string[], newVocabSize?)`
Continues training the tokenizer from a new corpus without resetting old IDs. Useful when the vocabulary needs to be expanded gradually.
```ts
tokenizer.update(["matematika diskrit", "logika proposisional"], 1200);
```

#### `encode(text)` & `decode(ids)`
Converts text to numbers and vice versa.
```ts
const ids = tokenizer.encode("saya makan"); 
// ids: [12, 45, 67]

const text = tokenizer.decode(ids);
// text: "saya makan"
```

#### `encodeWithSpecial(text)`
Automatically adds **BOS** (Beginning of Sequence) at the start and **EOS** (End of Sequence) at the end. Very useful for generative/Transformer models.

#### `padSequence(ids, maxLength)`
Adds **PAD** tokens so that all sequences have the same length for batch processing.
```ts
const padded = tokenizer.padSequence([1, 2], 5);
// padded: [1, 2, 0, 0, 0] (assuming PAD_ID = 0)
```

#### Helper IDs & Vocabulary

The tokenizer also exposes several public helpers:

| Method | Description |
| :--- | :--- |
| `getVocabSize()` | Number of tokens currently stored in the vocabulary map. |
| `getVocabularyCapacity()` | Effective ID capacity (`maxTokenId + 1`). This is the safest way to sync `Embedding`/`Transformers.vocabSize`. |
| `getTokenId(token)` | Get the ID of a specific token. |
| `getToken(id)` | Get the token associated with a specific ID. |
| `getPadId()` | Get the PAD token ID. |

---

### B. Saving & Loading

Tokenizers can be saved to `.json` files so they don't need to be retrained every time the application runs.

```ts
// Save to file
tokenizer.save("./model/vocab.json");

// Load back
const loadedTokenizer = BPETokenizer.load("./model/vocab.json");
```

---

> [!CAUTION]
> For **Embedding** layers or **Transformers** models, the safest vocabulary size is `tokenizer.getVocabularyCapacity()`, not just `tokenizer.getVocabSize()`.
> `getVocabSize()` counts the number of stored token entries, while `getVocabularyCapacity()` counts `maxTokenId + 1`. This is important because token IDs may not always be dense after updates or placeholder reuse. Using a size that is too small can lead to *index out of bounds* errors in the model.

---

## 6. Optimization Algorithms (`src/optimizer`)

The optimizer is responsible for updating model weights and biases based on gradients calculated during backpropagation.

---

### A. Available Optimizer Types

You can choose an optimizer type via a string literal when initializing a layer or model.

| Name | Description | Recommendation |
| :--- | :--- | :--- |
| **`"sgd"`** | Simple Stochastic Gradient Descent. | Very simple models or debugging. |
| **`"momentum"`**| SGD with direction memory (*velocity*). | Smoother convergence than simple SGD. |
| **`"adam"`** | Adaptive Moment Estimation. | **Best Default** for almost all cases. |
| **`"adaGrad"`** | Per-parameter adaptive learning rate. | Good for sparse data. |
| **`"nag"`** | Nesterov Accelerated Gradient. | More accurate than standard momentum. |

> [!IMPORTANT]
> The valid optimizers in the `setOptimizer(...)` internal helper are currently: `sgd`, `momentum`, `nag`, `adaGrad`, and `adam`.
> `rmsprop` is not yet available in the library.

---

### B. Adam (Adaptive Moment Estimation)

The most popular optimizer because it combines the advantages of Momentum and RMSProp.

#### Characteristics:
- **Native Acceleration**: Optimized using the Rust backend for extremely fast parameter updates.
- **Stability**: Includes a *bias correction* mechanism for the early steps of training.
- **Internal defaults**: `beta1=0.9`, `beta2=0.999`, `epsilon=1e-8`.

```ts
// Example configuration in a Dense layer
const layer = new Dense({
  optimizer: "adam",
  alpha: 0.001 // Learning Rate
});
```

---

### C. Other Available Optimizers

- **`SGD`**: Directly updates `alpha * gradient`.
- **`Momentum`**: Stores internal velocity with `beta = 0.9`.
- **`NAG`**: A momentum variant with look-ahead updates, also using `beta = 0.9`.
- **`AdaGrad`**: Accumulates squared gradients and adjusts the effective learning rate per parameter.

#### D. Working Mechanism

Every optimizer in ML-V1 implements a `calculate(grad, alpha)` method that returns an **Update** matrix, which is then subtracted from the original weights *in-place*.

```ts
// Internal update logic performed by the layer
const update = optimizer.calculate(gradWeight, alpha);
weight.subInPlace(update);
```

Notes:
- Normal users usually do not need to create optimizers manually; layers will allocate internal optimizers through constructors or `compile(...)`.
- `calculate(...)` returns the update value, not the final weights after the update.

---

### Conclusion
You now have a complete reference for all core **ML-V1** APIs. Use this guide alongside [01-overview.md](./01-overview.md) and [03-tutorial.md](./03-tutorial.md) to build powerful and efficient AI applications.

---

> [!TIP]
> For intensive operations in a training loop, always prioritize using **`get()`**, **`set()`**, and **`InPlace`** methods to avoid performance bottlenecks.
