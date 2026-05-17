# 🏗️ BaseLayer Specification API Reference

The **[BaseLayer](file:///home/akhyar/Dokumen/Code/NODE_JS/Oxide-JS/packages/layers/src/base/BaseLayer.ts)** class is the primary abstract base class that defines the lifecycle, interface, and execution contract for all layers in the **@oxide-js/layers** package. Every neural layer (from Dense to self-attention or RNNs) extends `BaseLayer` to integrate seamlessly with training loops, dynamic autograd gradient recorders, weight loaders, and model serialization graphs.

---

## 📌 Properties & Accessors

| Property | Type | Access | Description |
| :--- | :--- | :--- | :--- |
| **`name`** | `string` | Public | Unique identifying name of the layer instance. |
| **`trainable`** | `boolean` | Public | If `false`, the layer parameters are locked (frozen) and will not update. |
| **`dtype`** | `"float32" \| "float64"` | Public | Buffer precision. Default is `"float32"`. |
| **`training`** | `boolean` | Public | Indicates if the layer is running in training mode (enabling Dropout/BatchNorm updates) or evaluation mode. |
| **`isBuilt`** | `boolean` | Public | Tracks whether weights have been allocated based on the incoming shape. |
| **`inputShape`** | `number[]` | Public | Dimensions of the input tensor registered during `build()`. |
| **`outputShape`** | `number[]` | Public | Deterministic dimensions of output tensors calculated by the layer. |
| **`weights`** | `Matrix[]` | Public (Getter) | Retrieves all parameter matrices associated with this layer. |
| **`trainableWeights`** | `Matrix[]` | Public (Getter) | Retrieves active parameter matrices that accumulate gradients during backpropagation. |
| **`nonTrainableWeights`** | `Matrix[]` | Public (Getter) | Retrieves frozen parameter matrices. |

---

## 🌀 Base Lifecycle Methods

### 1. Mode Management
- **`train(): void`**
  Sets the layer status to training mode. Actively enables Dropout gates and triggers running statistics updates inside Batch Normalization layers.
- **`eval(): void`**
  Sets the layer status to evaluation mode (deactivates Dropout, locks normalization steps).

### 2. Dimension Calculations & Parameter Allocation
- **`abstract computeOutputShape(inputShape: number[]): number[]`**
  Must be overridden by sub-classes to dynamically calculate output dimensions based on input shapes.
- **`build(inputShape: number[]): void`**
  Allocates weights and biases parameter matrices based on the final item length of input features. Sets `isBuilt = true`.
- **`forward(inputs: Matrix, options?: ForwardOptions | boolean): Matrix`**
  The central public entry point to perform forward calculations. Automatically triggers `build()` if the layer is not yet built, validates shape expectations, and invokes the internal `compute` method.

### 3. Parameters Register
- **`protected addParameter(name: string, param: Matrix, trainable?: boolean, logicalShape?: number[]): Matrix`**
  Registers a target parameter matrix inside the internal parameters map, assigning standard autograd tracking indicators.
- **`getParameter(name: string): Matrix | undefined`**
  Retrieves a parameter by name (e.g. `"kernel"`, `"bias"`).
- **`getTrainableParameters(): Matrix[]`**
  Retrieves parameters to be fed to optimizers. Returns empty array if `trainable` is set to `false`.
- **`clearGradients(): void`**
  Clears the stored gradients (`matrix.grad = null`) of all parameters registered inside this layer.

### 4. Serialization (Keras Interoperability)
- **`getConfig(): Record<string, any>`**
  Returns structural configuration options (units, activations, names) formatted into a standard Keras-compatible layout.
- **`getKerasConfig(): Record<string, any>`**
  Formats the constructor class name and config objects.
- **`getWeights(): { name: string, shape: number[], physicalShape: number[], dtype: string, data: Float32Array }[]`**
  Extracts the raw physical data array buffers along with shape manifests matching Keras weights specifications.
- **`setWeights(weightsData: Array, options?: { strict: boolean }): void`**
  Loads raw parameter data back into active layer matrices.

---

## 🛠️ Subclassing BaseLayer Example

This example demonstrates how to build a custom mathematical layer (`CustomScaleBias`) by extending the abstract `BaseLayer` class, showing how to define parameter builders, implement mathematical calculations via `mj`, and execute a complete forward/backward pass.

```ts
import { BaseLayer, LayerConfig, type ForwardOptions } from "@oxide-js/layers";
import { Matrix, mj } from "@oxide-js/core";

// 1. Define configuration interfaces
interface ScaleBiasConfig extends LayerConfig {
  scaleValue?: number;
}

// 2. Subclass BaseLayer
export class CustomScaleBias extends BaseLayer {
  public scaleValue: number;

  public get scale(): Matrix {
    return this.getParameter("scale")!;
  }

  public get bias(): Matrix {
    return this.getParameter("bias")!;
  }

  constructor(config?: ScaleBiasConfig) {
    super(config);
    this.scaleValue = config?.scaleValue ?? 2.0;
  }

  // Calculate output shape dynamically
  public computeOutputShape(inputShape: number[]): number[] {
    return [...inputShape]; // Element-wise modification preserves shape
  }

  // Allocate parameters [rows, cols]
  public build(inputShape: number[]): void {
    super.build(inputShape);
    
    // We expect 2D matrix input [batch, features]
    const cols = inputShape[inputShape.length - 1];

    // Allocate scale matrix initialized to constant scaling factor
    const scaleMatrix = mj.matrix([[this.scaleValue]], [1, 1]);
    this.addParameter("scale", scaleMatrix, true);

    // Allocate bias matrix initialized to zeros
    const biasMatrix = mj.zeros([cols, 1]);
    this.addParameter("bias", biasMatrix, true);
  }

  // Define mathematical computation pass
  protected compute(inputs: Matrix, options?: ForwardOptions): Matrix {
    // scaleInputs = inputs * scale
    let scaled = mj.mul(inputs, this.scale);

    // add bias in-place by transposing
    const scaledT = mj.transpose(scaled);
    mj.addBias(scaledT, this.bias);
    
    return mj.transpose(scaledT);
  }
}

// 3. Instantiate and run Custom Layer
const layer = new CustomScaleBias({ name: "my_scale_layer", scaleValue: 3.5 });

const inputs = Matrix.fromFlat(new Float32Array([1.0, 2.0]), [1, 2]);
inputs.requiresGrad = true;

// Run forward pass (automatically builds layer internally)
const outputs = layer.forward(inputs);
outputs.print(); // Displays scaled elements + bias
```
