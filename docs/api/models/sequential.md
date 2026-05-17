# 🥞 Sequential Model API Reference

The **[Sequential](file:///home/akhyar/Dokumen/Code/NODE_JS/Oxide-JS/packages/models/src/Sequential.ts)** model container is a specialized subclass of `BaseModel` designed for stacked, feed-forward neural network topologies. It processes input matrices sequentially, forwarding outputs from one layer directly into the inputs of the next layer.

---

## 📐 Stack Architecture & Mechanics

### 1. Layers Aggregation (`add`)
Call `add(layer)` to stack layers at the end of the execution sequence:
* **Unique Names Resolver**: `add()` automatically resolves naming conflicts by assigning unique identifiers (e.g. if two `Dense` layers are added without custom names, they are named `Dense_1` and `Dense_2`).
* **Graph Compilation State**: Adding a layer marks the model as unbuilt (`isBuilt = false`), triggering a rebuild pass on the next forward pass.

### 2. Forward Propagation & Build
When `forward()` is called with inputs:
* **Dynamic Builder**: If the model is not built, it captures the input shape dynamically (`inputs._shape`) and calls `build(inputShape)`.
* **Automatic Shape Propagation**: Each layer's `build()` is executed in sequence, and the output shape of one layer is passed as the input shape to compile the parameters of the subsequent layer.

---

## 🛠️ Usage Example

This example demonstrates how to build, compile, train, evaluate, serialize, and deserialize a classification network using `Sequential`.

```ts
import { Sequential } from "@oxide-js/models";
import { Dense, Dropout } from "@oxide-js/layers";
import { Matrix } from "@oxide-js/core";

// 1. Compose the Sequential model stack
const model = new Sequential();
model.add(new Dense({ name: "dense_input", units: 4, outputUnits: 8, activation: "relu" }));
model.add(new Dropout({ name: "dropout_regularizer", rate: 0.1 }));
model.add(new Dense({ name: "dense_output", units: 8, outputUnits: 2, activation: "softmax" }));

// 2. Compile settings
model.compile({
  optimizer: "adam",
  loss: "softmaxCrossEntropy",
  learningRate: 0.05,
  metrics: ["accuracy"]
});

// 3. Prepare synthetic classification inputs
// 4 samples, 4 features
const trainInputs = Matrix.fromFlat(new Float32Array([
  0.1, 0.2, 0.7, 0.9,
  0.9, 0.8, 0.1, 0.2,
  0.2, 0.1, 0.8, 0.8,
  0.8, 0.9, 0.2, 0.1
]), [4, 4]);

// 4 samples, 2 classes (one-hot vectors)
const trainTargets = Matrix.fromFlat(new Float32Array([
  0.0, 1.0,
  1.0, 0.0,
  0.0, 1.0,
  1.0, 0.0
]), [4, 2]);

// 4. Fit the model stack
console.log("Fitting Sequential Stack...");
model.fit(trainInputs, trainTargets, {
  epochs: 10,
  batchSize: 2,
  verbose: 1
});

// 5. Evaluate the model stack
console.log("\nEvaluating Sequential Stack...");
const evalResult = model.evaluate(trainInputs, trainTargets);
console.log(`Evaluation loss: ${evalResult.loss?.toFixed(6)}`);
console.log("Accuracy metric:", evalResult.metrics.accuracy);

// 6. Predict results
console.log("\nPredicting values...");
const outputs = model.predict(trainInputs);
console.log("Predictions shape:", outputs._shape);
outputs.print();

// 7. Output architecture diagnostics summary
console.log("\nSequential Diagnostics Summary:");
model.summary();
```
