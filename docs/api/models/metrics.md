# 📊 Performance Metrics & Data Helpers API Reference

Evaluation metrics in **@oxide-js/models** measure the predictive performance of models against target datasets. In addition, the package provides data preprocessing utilities for splitting datasets and creating mini-batches.

---

## 🎛️ 1. Evaluation Metrics (`metrics.ts`)

The **[metrics.ts](file:///home/akhyar/Dokumen/Code/NODE_JS/Oxide-JS/packages/models/src/metrics.ts)** module contains implementations of standard evaluation algorithms.

### 📐 Multi-Class Accuracy (`accuracy` / `categoricalAccuracy`)
Compares the index of the maximum value (argmax) in each prediction row to the index of the maximum value in the corresponding target row.
* **Formula**:
  $$\text{Accuracy} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}\left(\text{argmax}(\hat{\mathbf{y}}_i) = \text{argmax}(\mathbf{y}_i)\right)$$
* **Usage**: Best suited for multi-class classification tasks using `softmax` outputs and one-hot targets.

### 📐 Binary Accuracy (`binaryAccuracy`)
Computes accuracy for binary classification tasks by rounding predictions and targets to `0` or `1` using a threshold of `0.5`.
* **Formula**:
  $$\text{BinaryAccuracy} = \frac{1}{M} \sum_{k=1}^M \mathbb{I}\left([\hat{y}_k > 0.5] = [y_k > 0.5]\right)$$
* **Usage**: Suited for binary classification using `sigmoid` outputs.

### 📐 Mean Absolute Error (`mae` / `meanAbsoluteError`)
Calculates the average absolute difference between predicted values and targets.
* **Formula**:
  $$\text{MAE} = \frac{1}{M} \sum_{k=1}^M \left| \hat{y}_k - y_k \right|$$
* **Usage**: Standard metric for regression tasks.

### 📐 Mean Squared Error (`mse` / `meanSquaredError`)
Calculates the average squared difference between predictions and targets.
* **Formula**:
  $$\text{MSE} = \frac{1}{M} \sum_{k=1}^M (\hat{y}_k - y_k)^2$$
* **Usage**: Captures larger errors heavily; standard metric for regression tasks.

---

## ⚡ 2. Compile Config & Metric Resolvers

When compile config receives string metrics names, `resolveMetric()` automatically translates them:

```ts
import { computeMetric, getMetricName } from "@oxide-js/models";

// Programmatic metric calculation
const score = computeMetric("accuracy", yPred, yTrue);
console.log(getMetricName("accuracy")); // "accuracy"
```

---

## 📂 3. Data Processing Utilities (`data.ts`)

The **[data.ts](file:///home/akhyar/Dokumen/Code/NODE_JS/Oxide-JS/packages/models/src/data.ts)** module provides essential tools to prepare and slice dataset matrices before passing them to the training pipeline:

### 🧩 `trainValidationSplit`
Splits input data and targets into distinct training and validation sets.
```ts
import { trainValidationSplit } from "@oxide-js/models/data";

const { xTrain, yTrain, xVal, yVal } = trainValidationSplit(
  inputs,
  targets,
  0.2, // 20% validation split
  true // shuffle before splitting
);
```

### 🧩 `createBatches`
Partitions dataset matrices into a collection of mini-batches for batch gradient descent.
```ts
import { createBatches } from "@oxide-js/models/data";

const batches = createBatches(
  xTrain,
  yTrain,
  32, // batch size
  true // shuffle batches
);

// Process batches in training loop
for (const batch of batches) {
  const xBatch = batch.x;
  const yBatch = batch.y;
}
```

---

## 🛠️ Usage Example (Metrics & Data Split)

This standalone example demonstrates how to split a dataset, batch it, and compute performance metrics.

```ts
import { Matrix } from "@oxide-js/core";
import { trainValidationSplit, createBatches } from "@oxide-js/models/data";
import { accuracy, mae, mse } from "@oxide-js/models";

// 1. Setup raw input dataset [6 samples, 2 classes]
const inputs = Matrix.fromFlat(new Float32Array([
  0.1, 0.9,  // argmax index = 1
  0.8, 0.2,  // argmax index = 0
  0.3, 0.7,  // argmax index = 1
  0.9, 0.1,  // argmax index = 0
  0.2, 0.8,  // argmax index = 1
  0.7, 0.3   // argmax index = 0
]), [6, 2]);

const targets = Matrix.fromFlat(new Float32Array([
  0, 1,      // index 1
  1, 0,      // index 0
  1, 0,      // index 0 (prediction mismatch!)
  1, 0,      // index 0
  0, 1,      // index 1
  0, 1       // index 1 (prediction mismatch!)
]), [6, 2]);

// 2. Perform train/validation split (33% validation -> 4 train, 2 val)
console.log("Splitting dataset...");
const { xTrain, yTrain, xVal, yVal } = trainValidationSplit(inputs, targets, 0.33, false);
console.log(`xTrain shape: [${xTrain._shape}], xVal shape: [${xVal._shape}]`);

// 3. Partition training data into mini-batches (size=2)
const batches = createBatches(xTrain, yTrain, 2, false);
console.log(`Created ${batches.length} training batches.`);

// 4. Calculate metrics directly on the entire dataset
const acc = accuracy(inputs, targets);
const maeVal = mae(inputs, targets);
const mseVal = mse(inputs, targets);

console.log("\nComputed Dataset Metrics:");
console.log(`- Multi-Class Accuracy : ${(acc * 100).toFixed(2)}% (Expected: 4/6 correct = 66.67%)`);
console.log(`- Mean Absolute Error  : ${maeVal.toFixed(4)}`);
console.log(`- Mean Squared Error   : ${mseVal.toFixed(4)}`);
```
