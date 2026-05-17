# ⚖️ Normalization Layers API Reference

Normalization layers in **@oxide-js/layers** stabilize and accelerate deep neural network training by normalizing layer inputs. They mitigate vanishing or exploding gradients and allow for higher learning rates.

---

## 🚀 1. `LayerNormalization` Layer

The **[LayerNormalization](file:///home/akhyar/Dokumen/Code/NODE_JS/Oxide-JS/packages/layers/src/layers/LayerNormalization.ts)** layer normalizes activations across the feature dimension (channels/columns) independently for each training sample (row) in a batch. 

### 📐 Mathematical Definition
For an input vector $\mathbf{x}$ of features:
$$\mu = \frac{1}{H} \sum_{i=1}^{H} x_i, \quad \sigma^2 = \frac{1}{H} \sum_{i=1}^{H} (x_i - \mu)^2$$
$$\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}$$
$$y_i = \gamma_i \hat{x}_i + \beta_i$$
where:
* $H$ is the number of features.
* $\gamma$ is a trainable scaling parameter initialized to `1.0`.
* $\beta$ is a trainable bias parameter initialized to `0.0`.
* $\epsilon$ is a small float constant for numerical stability.

### 📌 Configuration Parameters (`LayerNormalizationConfig`)
- `epsilon?: number` — A small float added to variance to avoid dividing by zero. Default is `1e-5`.

### 📌 Properties
- `gamma: Matrix` — Trainable scaling vector of shape `[1, features]`.
- `beta: Matrix` — Trainable shifting bias vector of shape `[1, features]`.

### 📌 Example
```ts
import { LayerNormalization } from "@oxide-js/layers";
import { Matrix } from "@oxide-js/core";

// 1. Instantiate layer normalization
const lnLayer = new LayerNormalization({ epsilon: 1e-5 });

// 2. Feed inputs [batch=2, features=3]
const inputs = Matrix.fromFlat(new Float32Array([
  10.0,  5.0,  0.0,  // sample 1
   1.0,  2.0,  3.0   // sample 2
]), [2, 3]);

const outputs = lnLayer.forward(inputs);
outputs.print(); // Normalizes row-wise, scaled by gamma (1) and beta (0)
```

---

## 📊 2. `BatchNormalization` Layer

The **[BatchNormalization](file:///home/akhyar/Dokumen/Code/NODE_JS/Oxide-JS/packages/layers/src/layers/BatchNormalization.ts)** layer normalizes inputs across the batch dimension (rows) for each feature column. It maintains moving statistics during training to use during inference.

### 📐 Mathematical Definition
During **training mode** (`layer.train()`):
1. Compute mini-batch mean $\mu_B$ and variance $\sigma_B^2$ for each column:
   $$\mu_B = \frac{1}{m} \sum_{i=1}^{m} x_i, \quad \sigma_B^2 = \frac{1}{m} \sum_{i=1}^{m} (x_i - \mu_B)^2$$
2. Normalize inputs:
   $$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$
3. Update non-trainable moving statistics:
   $$\mu_{\text{moving}} = \text{momentum} \cdot \mu_{\text{moving}} + (1 - \text{momentum}) \cdot \mu_B$$
   $$\sigma^2_{\text{moving}} = \text{momentum} \cdot \sigma^2_{\text{moving}} + (1 - \text{momentum}) \cdot \sigma_B^2$$
4. Compute final output scaled by trainable parameters $\gamma$ and $\beta$:
   $$y_i = \gamma \hat{x}_i + \beta$$

During **evaluation mode** (`layer.eval()`):
Uses stored moving averages $\mu_{\text{moving}}$ and $\sigma^2_{\text{moving}}$ directly instead of batch statistics.

### 📌 Configuration Parameters (`BatchNormalizationConfig`)
- `epsilon?: number` — Small float added to variance to avoid division-by-zero. Default is `1e-5`.
- `momentum?: number` — Momentum factor used to calculate the moving average. Default is `0.99`.

### 📌 Properties
- `gamma: Matrix` — Trainable scaling vector of shape `[1, features]`.
- `beta: Matrix` — Trainable bias vector of shape `[1, features]`.
- `movingMean: Matrix` — **Non-trainable** moving average of shape `[1, features]`.
- `movingVariance: Matrix` — **Non-trainable** moving variance of shape `[1, features]`.

### 📌 Example
```ts
import { BatchNormalization } from "@oxide-js/layers";
import { Matrix } from "@oxide-js/core";

// 1. Instantiate batch normalization layer
const bnLayer = new BatchNormalization({
  epsilon: 1e-5,
  momentum: 0.9
});

// 2. Feed inputs [batch=3, features=2]
const inputs = Matrix.fromFlat(new Float32Array([
  1.0, 10.0,
  2.0, 20.0,
  3.0, 30.0
]), [3, 2]);

// 3. Train pass (normalizes column-wise and updates moving stats)
bnLayer.train();
const trainOut = bnLayer.forward(inputs);
console.log("Training Outputs (normalized per column):");
trainOut.print();

// 4. Eval pass (uses moving statistics)
bnLayer.eval();
const evalOut = bnLayer.forward(inputs);
console.log("Evaluation Outputs (uses moving stats):");
evalOut.print();
```
