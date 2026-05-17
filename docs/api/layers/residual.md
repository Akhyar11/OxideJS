# 🔄 Residual (Skip Connection) Layer API Reference

The **[Residual](file:///home/akhyar/Dokumen/Code/NODE_JS/Oxide-JS/packages/layers/src/layers/Residual.ts)** block layer implements skip connections (shortcuts), which are foundational for deep architectures like ResNet or Transformer sub-blocks. Skip connections allow gradients to flow directly through the network, mitigating vanishing gradient problems during backpropagation.

---

## 📐 Mathematical Formulation

Given an input matrix $\mathbf{X}$:
1. **Main Path**:
   Calculates the forward transformations through the main layer:
   $$\mathbf{F}(\mathbf{X}) = \text{layer.forward}(\mathbf{X})$$
2. **Shortcut Path**:
   * **Identity Connection**: If no projection shortcut layer is registered, the shortcut remains the original inputs:
     $$\text{Shortcut}(\mathbf{X}) = \mathbf{X}$$
     *Note: This requires that the shape of $\mathbf{F}(\mathbf{X})$ is exactly identical to the shape of $\mathbf{X}$.*
   * **Projection Connection**: If a shortcut layer is registered (e.g. a Dense or Conv2D layer to match dimensions):
     $$\text{Shortcut}(\mathbf{X}) = \text{shortcut.forward}(\mathbf{X})$$
3. **Element-wise Addition**:
   Sums the main path and shortcut path outputs:
   $$\mathbf{Y} = \mathbf{F}(\mathbf{X}) + \text{Shortcut}(\mathbf{X})$$

### 🔄 Gradient Flow
During backpropagation, the gradient of the loss with respect to the input $\mathbf{X}$ is distributed directly across both paths:
$$\frac{\partial L}{\partial \mathbf{X}} = \frac{\partial L}{\partial \mathbf{F}(\mathbf{X})} \cdot \frac{\partial \mathbf{F}(\mathbf{X})}{\partial \mathbf{X}} + \frac{\partial L}{\partial \text{Shortcut}(\mathbf{X})} \cdot \frac{\partial \text{Shortcut}(\mathbf{X})}{\partial \mathbf{X}}$$
This additive gradient path prevents gradients from vanishing even when the main path layers are extremely deep.

---

## 📌 Configuration Parameters (`ResidualConfig`)
- `layer: BaseLayer` — The main computational layer (e.g. Dense, Conv2D, or attention block).
- `shortcut?: BaseLayer` — Optional projection layer to adjust input dimensions to match the main path output shape.

---

## 📌 Operational Properties & State Delegates
* **Training Mode Propagation**: Calling `layer.train()` or `layer.eval()` on the `Residual` block automatically propagates the state downstream to both the `layer` and the `shortcut`.
* **Weights & Parameter Tracking**: Dynamically aggregates trainable/non-trainable weights and counts parameter arrays from all nested layers.
* **Gradients Purge**: `clearGradients()` systematically clears gradients across all internal sub-layers.

---

## 🛠️ Usage Examples

### 📌 Example (Dense Block with Identity Skip)
This example demonstrates a basic residual block where the inputs pass through a Dense layer and are added directly to the original inputs.

```ts
import { Residual, Dense } from "@oxide-js/layers";
import { Matrix, engine } from "@oxide-js/core";

// 1. Instantiate the Dense layer (units = 3)
const denseLayer = new Dense({ units: 3, activation: "relu" });

// 2. Instantiate the Residual block (no shortcut, requires input features === output units)
const residualBlock = new Residual({
  name: "dense_residual",
  layer: denseLayer
});

// 3. Feed inputs [batch=2, features=3]
const inputs = Matrix.fromFlat(new Float32Array([
  1.0, 2.0, 3.0,
  0.5, 0.0, -1.0
]), [2, 3]);

// 4. Compute forward pass (performs Dense output + inputs in-place)
const outputs = residualBlock.forward(inputs);
console.log("Residual output shape:", outputs._shape); // [2, 3]
outputs.print();
```

---

### 📌 Example (Dense Block with Projection Shortcut)
This example demonstrates a residual block where the inputs are projected to a larger space, requiring a shortcut Dense layer to align dimensions.

```ts
import { Residual, Dense } from "@oxide-js/layers";
import { Matrix } from "@oxide-js/core";

// 1. Main Path: projects features 2 -> 4
const mainPathDense = new Dense({
  name: "main_dense",
  units: 4,
  activation: "relu"
});

// 2. Shortcut Path: projects features 2 -> 4 to match the main path
const shortcutDense = new Dense({
  name: "shortcut_projector",
  units: 4,
  useBias: false
});

// 3. Instantiate the Residual block with the projection shortcut
const projectionResidual = new Residual({
  name: "projection_residual",
  layer: mainPathDense,
  shortcut: shortcutDense
});

// 4. Feed inputs [batch=2, features=2]
const inputs = Matrix.fromFlat(new Float32Array([
  1.0, 2.0,
  3.0, 4.0
]), [2, 2]);

// 5. Compute forward pass
const outputs = projectionResidual.forward(inputs);
console.log("Projection Residual Output Shape (batch, units):", outputs._shape); // [2, 4]
outputs.print();
```
