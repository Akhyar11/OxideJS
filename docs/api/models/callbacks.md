# 🎛️ Training Callbacks System API Reference

Callbacks in **@oxide-js/models** provide hooks into the lifecycle of model training, enabling developers to monitor loss/accuracy metrics, perform checkpointing, implement early stopping on plateaus, or stream stats to visual dashboards.

---

## ⚡ The Callback Interface (`Callback`)

Any object that implements the **[Callback](file:///home/akhyar/Dokumen/Code/NODE_JS/Oxide-JS/packages/models/src/types.ts#L104-L112)** interface can be registered in `model.fit()`.

```ts
export interface Callback {
  onTrainBegin?(logs?: CallbackLogs): void | Promise<void>;
  onTrainEnd?(logs?: CallbackLogs): void | Promise<void>;
  onEpochBegin?(epoch: number, logs?: CallbackLogs): void | Promise<void>;
  onEpochEnd?(epoch: number, logs?: CallbackLogs): void | Promise<void>;
  onBatchBegin?(batch: number, logs?: CallbackLogs): void | Promise<void>;
  onBatchEnd?(batch: number, logs?: CallbackLogs): void | Promise<void>;
  shouldStop?: boolean; // Set to true to stop training loop early
}
```

### 📌 CallbackLogs Schema
The `logs` object is updated at each step of training and passed to the hooks:
* `epoch?: number` — Current active epoch index.
* `batch?: number` — Current active mini-batch index.
* `loss?: number` — Current step average loss.
* `val_loss?: number` — Validation loss (available on epoch end if validation data is active).
* `metrics?: Record<string, number>` — Dictionary of training metrics.
* `val_metrics?: Record<string, number>` — Dictionary of validation metrics.

---

## 📦 Built-In Callbacks Catalog

### 1. `HistoryCallback`
Accumulates losses and metrics logs at the end of each epoch, returning a compiled history list. This is automatically appended inside `model.fit()` to return training history records.
- **Access**: `historyCallback.history` returns an array of `HistoryRecord`.

### 2. `ProgressLogger`
Logs training progress to the console during fitting.
- **Parameters**: `verbose: number` (default `1`).
  - `verbose = 1`: Outputs epoch average loss, validation loss, and metrics on epoch completion.
  - `verbose = 2`: Outputs real-time updates for every mini-batch via `process.stdout.write`.

### 3. `EarlyStopping`
Monitors a target metric and stops training early if the model ceases to improve after a set number of epochs.
- **Configuration options**:
  - `monitor?: string` — The metric to track (default: `"val_loss"`; supports `"loss"`, or metrics like `"val_accuracy"`).
  - `patience?: number` — Number of epochs to wait for an improvement before stopping (default: `3`).
  - `minDelta?: number` — Minimum change in the monitored value to qualify as an improvement (default: `0`).
  - `restoreBestWeights?: boolean` — Restores best weights when stopped (default: `false`).

---

## 🛠️ Usage Example (Custom Logging & EarlyStopping)

This example demonstrates how to build and register a custom CSV logging callback alongside the built-in `EarlyStopping` callback to safely train a model.

```ts
import { Sequential, EarlyStopping, ProgressLogger } from "@oxide-js/models";
import { Dense } from "@oxide-js/layers";
import { Matrix } from "@oxide-js/core";
import type { Callback, CallbackLogs } from "@oxide-js/models";

// 1. Create a Custom CSV Training File Logger Callback
class CSVModelLogger implements Callback {
  private logRows: string[] = [];

  onTrainBegin(): void {
    console.log("[CSV Logger] Initializing log registry...");
    this.logRows.push("epoch,loss,val_loss");
  }

  onEpochEnd(epoch: number, logs?: CallbackLogs): void {
    const loss = logs?.loss !== undefined ? logs.loss.toFixed(6) : "";
    const valLoss = logs?.val_loss !== undefined ? logs.val_loss.toFixed(6) : "";
    
    const row = `${epoch},${loss},${valLoss}`;
    this.logRows.push(row);
    console.log(`[CSV Logger] Recorded: ${row}`);
  }

  onTrainEnd(): void {
    console.log("\n[CSV Logger] Training completed. Generated CSV Log file content:");
    console.log(this.logRows.join("\n"));
  }
}

// 2. Build the Model
const model = new Sequential();
model.add(new Dense({ units: 2, outputUnits: 1 }));

model.compile({
  optimizer: "sgd",
  loss: "mse",
  learningRate: 0.1
});

// Prepare mock inputs [batch=4, features=2] -> Targets [batch=4, output=1]
const inputs = Matrix.fromFlat(new Float32Array([1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0]), [4, 2]);
const targets = Matrix.fromFlat(new Float32Array([2.0, 4.0, 6.0, 8.0]), [4, 1]);

// 3. Register Callbacks inside the fitting pipeline
model.fit(inputs, targets, {
  epochs: 100,
  batchSize: 2,
  verbose: 0, // disable default logs to show custom logger outputs
  validationData: [inputs, targets], // pass validation data to trigger val_loss metrics
  callbacks: [
    new CSVModelLogger(),
    new EarlyStopping({
      monitor: "val_loss",
      patience: 2,
      minDelta: 0.0001
    })
  ]
});
```
