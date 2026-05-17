import { describe, expect, it, vi } from "vitest";
import {
  Sequential,
  HistoryCallback,
  EarlyStopping,
  accuracy,
  categoricalAccuracy,
  binaryAccuracy,
  mae,
  createBatches,
  trainValidationSplit,
  computeMetric
} from "../src/index.js";
import { Dense, Activation } from "@oxide-js/layers";
import { Matrix, engine } from "@oxide-js/core";

describe("BaseModel and Sequential Tests", () => {
  it("should initialize, add layers, and build correctly", () => {
    const model = new Sequential([
      new Dense({ units: 16 }),
      new Activation("relu")
    ], { name: "my_sequential_model" });

    expect(model.name).toBe("my_sequential_model");
    expect(model.layerCount).toBe(2);

    // Build the model with input shape
    model.build([4, 8]); // batch = 4, input_features = 8

    expect(model.isBuilt).toBe(true);
    expect(model.inputShape).toEqual([4, 8]);
    expect(model.outputShape).toEqual([4, 16]);

    // Check unique layer names
    expect(model.getLayer(0).name).toBe("Dense");
    expect(model.getLayer(1).name).toBe("Activation");
  });

  it("should generate unique layer names for identical layer classes", () => {
    const model = new Sequential();
    model.add(new Dense({ units: 16 }));
    model.add(new Dense({ units: 8 }));

    expect(model.getLayer(0).name).toBe("Dense");
    expect(model.getLayer(1).name).toBe("Dense_1");
  });

  it("should forward inputs through the layers cleanly", () => {
    const model = new Sequential([
      new Dense({ units: 4, useBias: false }),
      new Activation("relu")
    ]);

    // Force dense weights to be deterministic
    model.build([2, 3]);
    const denseLayer = model.getLayer(0) as Dense;
    denseLayer.kernel!._data.fill(0.5);

    const x = Matrix.fromFlat(new Float32Array([1.0, 2.0, 3.0, -4.0, 5.0, -6.0]), [2, 3]);
    const out = model.forward(x);
    expect(out._shape).toEqual([2, 4]);
    const results = Array.from(out._data).map(val => Object.is(val, -0) ? 0 : val);
    expect(results).toEqual([3, 3, 3, 3, 0, 0, 0, 0]);
  });

  it("should compile and perform a training step with tape-based autodiff and mock optimizer", () => {
    const model = new Sequential([
      new Dense({ units: 2, useBias: false })
    ]);

    const x = Matrix.fromFlat(new Float32Array([1.0, 2.0, -1.0, -2.0]), [2, 2]);
    const yTrue = Matrix.fromFlat(new Float32Array([0.5, 0.5, -0.5, -0.5]), [2, 2]);

    const mockOptimizer = {
      step: vi.fn(),
      update: vi.fn()
    };

    // MSE loss function returning an autodiff scalar Matrix
    const mseLoss = (yPred: Matrix, yT: Matrix) => {
      // (yPred - yT)^2
      const diff = Matrix.fromFlat(new Float32Array(yPred._data.length), yPred._shape);
      for (let i = 0; i < yPred._data.length; i++) {
        diff._data[i] = yPred._data[i] - yT._data[i];
      }
      const sumSq = Matrix.fromFlat(new Float32Array([0.0]), [1, 1]);
      for (let i = 0; i < diff._data.length; i++) {
        sumSq._data[0] += diff._data[i] * diff._data[i];
      }
      engine.record([yPred], [sumSq], (grad) => {
        const outGrad = Matrix.fromFlat(new Float32Array(yPred._data.length), yPred._shape);
        for (let i = 0; i < yPred._data.length; i++) {
          outGrad._data[i] = 2 * diff._data[i] * grad._data[0];
        }
        return [outGrad];
      });
      return sumSq;
    };

    model.compile({
      optimizer: mockOptimizer,
      loss: mseLoss
    });

    const stepResult = model.trainStep(x, yTrue);
    expect(stepResult.loss).toBeDefined();
    expect(stepResult.yPred).toBeDefined();

    // Check that optimizer.step() or update() was called
    expect(mockOptimizer.step).toHaveBeenCalled();
  });

  it("should print model summary and serialize/deserialize weights successfully", () => {
    const model = new Sequential([
      new Dense({ units: 8, useBias: true }),
      new Activation("relu")
    ], { name: "summary_model" });

    model.build([2, 4]);

    // Check summary prints without crashing
    const logSpy = vi.spyOn(console, "log").mockImplementation(() => {});
    model.summary();
    expect(logSpy).toHaveBeenCalled();
    logSpy.mockRestore();

    // Serialize model
    const serialized = model.serialize();
    expect(serialized.name).toBe("summary_model");
    expect(serialized.layers.length).toBe(2);
    expect(serialized.weights.length).toBe(2); // kernel & bias of Dense

    // Try restoring weights
    const restoredModel = new Sequential([
      new Dense({ units: 8, useBias: true }),
      new Activation("relu")
    ], { name: "restored_model" });

    restoredModel.build([2, 4]);
    restoredModel.setWeights(serialized.weights);

    // Verify weights are transferred
    const origDense = model.getLayer(0) as Dense;
    const destDense = restoredModel.getLayer(0) as Dense;
    expect(Array.from(origDense.kernel!._data)).toEqual(Array.from(destDense.kernel!._data));
    expect(Array.from(origDense.bias!._data)).toEqual(Array.from(destDense.bias!._data));
  });

  it("should forward with ForwardOptions object for future compatibility", () => {
    const model = new Sequential([
      new Dense({ units: 2, useBias: false })
    ]);

    model.build([2, 3]);
    const x = Matrix.fromFlat(new Float32Array([1.0, 2.0, 3.0, 1.0, 2.0, 3.0]), [2, 3]);

    // Forward with boolean (backward compatibility)
    const out1 = model.forward(x, true);
    expect(out1._shape).toEqual([2, 2]);

    // Forward with ForwardOptions object (new way)
    const out2 = model.forward(x, { training: true });
    expect(out2._shape).toEqual([2, 2]);

    // Outputs should be the same
    expect(Array.from(out1._data)).toEqual(Array.from(out2._data));
  });

  it("should support fit with epochs and batchSize", () => {
    const model = new Sequential([
      new Dense({ units: 2, useBias: false })
    ]);

    const x = Matrix.fromFlat(new Float32Array([1.0, 2.0, -1.0, -2.0, 0.5, 0.5]), [3, 2]);
    const y = Matrix.fromFlat(new Float32Array([0.5, 0.5, -0.5, -0.5, 0.2, 0.2]), [3, 2]);

    const mockOptimizer = { step: vi.fn() };
    const mseLoss = (yPred: Matrix, yT: Matrix) => {
      const diff = Matrix.fromFlat(new Float32Array(yPred._data.length), yPred._shape);
      for (let i = 0; i < yPred._data.length; i++) {
        diff._data[i] = yPred._data[i] - yT._data[i];
      }
      const sumSq = Matrix.fromFlat(new Float32Array([0.0]), [1, 1]);
      for (let i = 0; i < diff._data.length; i++) {
        sumSq._data[0] += diff._data[i] * diff._data[i];
      }
      engine.record([yPred], [sumSq], (grad) => {
        const outGrad = Matrix.fromFlat(new Float32Array(yPred._data.length), yPred._shape);
        for (let i = 0; i < yPred._data.length; i++) {
          outGrad._data[i] = 2 * diff._data[i] * grad._data[0];
        }
        return [outGrad];
      });
      return sumSq;
    };

    model.compile({
      optimizer: mockOptimizer,
      loss: mseLoss
    });

    const history = model.fit(x, y, { epochs: 2, batchSize: 2, verbose: 0 });

    expect(history.length).toBe(2);
    expect(history[0].epoch).toBe(1);
    expect(history[1].epoch).toBe(2);
    expect(history[0].loss).toBeDefined();
    expect(history[1].loss).toBeDefined();
  });

  it("should support fit with validationSplit", () => {
    const model = new Sequential([
      new Dense({ units: 1, useBias: false })
    ]);

    const x = Matrix.fromFlat(new Float32Array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), [3, 2]);
    const y = Matrix.fromFlat(new Float32Array([0.5, 1.0, 1.5]), [3, 1]);

    const mockOptimizer = { step: vi.fn() };
    const mseLoss = (yPred: Matrix, yT: Matrix) => {
      const diff = Matrix.fromFlat(new Float32Array(yPred._data.length), yPred._shape);
      for (let i = 0; i < yPred._data.length; i++) {
        diff._data[i] = yPred._data[i] - yT._data[i];
      }
      const sumSq = Matrix.fromFlat(new Float32Array([0.0]), [1, 1]);
      for (let i = 0; i < diff._data.length; i++) {
        sumSq._data[0] += diff._data[i] * diff._data[i];
      }
      engine.record([yPred], [sumSq], (grad) => {
        const outGrad = Matrix.fromFlat(new Float32Array(yPred._data.length), yPred._shape);
        for (let i = 0; i < yPred._data.length; i++) {
          outGrad._data[i] = 2 * diff._data[i] * grad._data[0];
        }
        return [outGrad];
      });
      return sumSq;
    };

    model.compile({
      optimizer: mockOptimizer,
      loss: mseLoss
    });

    const history = model.fit(x, y, {
      epochs: 1,
      batchSize: 1,
      validationSplit: 0.33,
      verbose: 0
    });

    expect(history.length).toBe(1);
    expect(history[0].val_loss).toBeDefined();
  });

  it("should support predict in eval mode", () => {
    const model = new Sequential([
      new Dense({ units: 2, useBias: false })
    ]);

    model.build([2, 3]);
    const denseLayer = model.getLayer(0) as Dense;
    denseLayer.kernel!._data.fill(0.5);

    const x = Matrix.fromFlat(new Float32Array([1.0, 2.0, 3.0, 1.0, 2.0, 3.0]), [2, 3]);
    const pred = model.predict(x);

    expect(pred._shape).toEqual([2, 2]);
    expect(model.training).toBe(false); // Should be in eval mode after predict
  });

  it("should support evaluate with loss and metrics", () => {
    const model = new Sequential([
      new Dense({ units: 2, useBias: false })
    ]);

    const x = Matrix.fromFlat(new Float32Array([1.0, 2.0, -1.0, -2.0]), [2, 2]);
    const y = Matrix.fromFlat(new Float32Array([0.5, 0.5, -0.5, -0.5]), [2, 2]);

    const mseLoss = (yPred: Matrix, yT: Matrix) => {
      const diff = Matrix.fromFlat(new Float32Array(yPred._data.length), yPred._shape);
      for (let i = 0; i < yPred._data.length; i++) {
        diff._data[i] = yPred._data[i] - yT._data[i];
      }
      const sumSq = Matrix.fromFlat(new Float32Array([0.0]), [1, 1]);
      for (let i = 0; i < diff._data.length; i++) {
        sumSq._data[0] += diff._data[i] * diff._data[i];
      }
      engine.record([yPred], [sumSq], (grad) => {
        const outGrad = Matrix.fromFlat(new Float32Array(yPred._data.length), yPred._shape);
        for (let i = 0; i < yPred._data.length; i++) {
          outGrad._data[i] = 2 * diff._data[i] * grad._data[0];
        }
        return [outGrad];
      });
      return sumSq;
    };

    model.compile({
      optimizer: { step: vi.fn() },
      loss: mseLoss,
      metrics: ["mae"]
    });

    const result = model.evaluate(x, y);
    expect(result.loss).toBeDefined();
    expect(result.yPred).toBeDefined();
    expect(result.metrics).toBeDefined();
    expect(result.metrics.mae).toBeDefined();
  });

  it("should compute accuracy metric correctly", () => {
    // Test categorical accuracy
    const yPred = Matrix.fromFlat(
      new Float32Array([0.1, 0.9, 0.8, 0.2, 0.2, 0.8]),
      [3, 2]
    );
    const yTrue = Matrix.fromFlat(
      new Float32Array([0, 1, 1, 0, 0, 1]),
      [3, 2]
    );

    const acc = accuracy(yPred, yTrue);
    expect(acc).toEqual(1.0); // All predictions match

    const acc2 = categoricalAccuracy(yPred, yTrue);
    expect(acc2).toEqual(1.0);
  });

  it("should compute binary accuracy correctly", () => {
    const yPred = Matrix.fromFlat(
      new Float32Array([0.9, 0.1, 0.8, 0.3, 0.2]),
      [5, 1]
    );
    const yTrue = Matrix.fromFlat(
      new Float32Array([1, 0, 1, 0, 0]),
      [5, 1]
    );

    const acc = binaryAccuracy(yPred, yTrue);
    expect(acc).toEqual(1.0); // All predictions correct with threshold 0.5
  });

  it("should compute MAE metric correctly", () => {
    const yPred = Matrix.fromFlat(
      new Float32Array([1.0, 2.0, 3.0]),
      [3, 1]
    );
    const yTrue = Matrix.fromFlat(
      new Float32Array([1.1, 2.1, 2.9]),
      [3, 1]
    );

    const maeVal = mae(yPred, yTrue);
    expect(Math.abs(maeVal - 0.1)).toBeLessThan(0.01);
  });

  it("should support HistoryCallback", () => {
    const model = new Sequential([
      new Dense({ units: 1, useBias: false })
    ]);

    const x = Matrix.fromFlat(new Float32Array([1.0, 2.0]), [1, 2]);
    const y = Matrix.fromFlat(new Float32Array([0.5]), [1, 1]);

    const mockOptimizer = { step: vi.fn() };
    const mseLoss = (yPred: Matrix, yT: Matrix) => {
      const diff = Matrix.fromFlat(new Float32Array(yPred._data.length), yPred._shape);
      for (let i = 0; i < yPred._data.length; i++) {
        diff._data[i] = yPred._data[i] - yT._data[i];
      }
      const sumSq = Matrix.fromFlat(new Float32Array([0.0]), [1, 1]);
      for (let i = 0; i < diff._data.length; i++) {
        sumSq._data[0] += diff._data[i] * diff._data[i];
      }
      engine.record([yPred], [sumSq], (grad) => {
        const outGrad = Matrix.fromFlat(new Float32Array(yPred._data.length), yPred._shape);
        for (let i = 0; i < yPred._data.length; i++) {
          outGrad._data[i] = 2 * diff._data[i] * grad._data[0];
        }
        return [outGrad];
      });
      return sumSq;
    };

    model.compile({
      optimizer: mockOptimizer,
      loss: mseLoss
    });

    const historyCallback = new HistoryCallback();
    model.fit(x, y, { epochs: 2, batchSize: 1, verbose: 0, callbacks: [historyCallback] });

    expect(historyCallback.history.length).toBe(2);
    expect(historyCallback.history[0].epoch).toBe(1);
    expect(historyCallback.history[1].epoch).toBe(2);
  });

  it("should support EarlyStopping callback", () => {
    const model = new Sequential([
      new Dense({ units: 1, useBias: false })
    ]);

    const x = Matrix.fromFlat(new Float32Array([1.0, 2.0, 3.0]), [3, 1]);
    const y = Matrix.fromFlat(new Float32Array([0.5, 1.0, 1.5]), [3, 1]);

    const mockOptimizer = { step: vi.fn() };
    const mseLoss = (yPred: Matrix, yT: Matrix) => {
      const diff = Matrix.fromFlat(new Float32Array(yPred._data.length), yPred._shape);
      for (let i = 0; i < yPred._data.length; i++) {
        diff._data[i] = yPred._data[i] - yT._data[i];
      }
      const sumSq = Matrix.fromFlat(new Float32Array([0.0]), [1, 1]);
      for (let i = 0; i < diff._data.length; i++) {
        sumSq._data[0] += diff._data[i] * diff._data[i];
      }
      engine.record([yPred], [sumSq], (grad) => {
        const outGrad = Matrix.fromFlat(new Float32Array(yPred._data.length), yPred._shape);
        for (let i = 0; i < yPred._data.length; i++) {
          outGrad._data[i] = 2 * diff._data[i] * grad._data[0];
        }
        return [outGrad];
      });
      return sumSq;
    };

    model.compile({
      optimizer: mockOptimizer,
      loss: mseLoss
    });

    const earlyStopping = new EarlyStopping({
      monitor: "loss",
      patience: 1,
      minDelta: 1000 // High threshold to trigger early stopping
    });

    const history = model.fit(x, y, {
      epochs: 5,
      batchSize: 3,
      verbose: 0,
      callbacks: [earlyStopping]
    });

    expect(history.length).toBeLessThan(5);
    expect(earlyStopping.shouldStop).toBe(true);
  });

  it("should support createBatches utility", () => {
    const x = Matrix.fromFlat(
      new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]),
      [4, 2]
    );
    const y = Matrix.fromFlat(
      new Float32Array([1, 2, 3, 4]),
      [4, 1]
    );

    const batches = createBatches(x, y, 2, false);

    expect(batches.length).toBe(2);
    expect(batches[0].x._shape).toEqual([2, 2]);
    expect(batches[0].y._shape).toEqual([2, 1]);
    expect(batches[1].x._shape).toEqual([2, 2]);
  });

  it("should support trainValidationSplit utility", () => {
    const x = Matrix.fromFlat(
      new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]),
      [4, 2]
    );
    const y = Matrix.fromFlat(
      new Float32Array([1, 2, 3, 4]),
      [4, 1]
    );

    const split = trainValidationSplit(x, y, 0.25, false);

    expect(split.xTrain._shape[0]).toBe(3);
    expect(split.xVal._shape[0]).toBe(1);
    expect(split.yTrain._shape[0]).toBe(3);
    expect(split.yVal._shape[0]).toBe(1);
  });

  it("should support computeMetric utility for string metric names", () => {
    const yPred = Matrix.fromFlat(
      new Float32Array([0.9, 0.1, 0.8, 0.2]),
      [2, 2]
    );
    const yTrue = Matrix.fromFlat(
      new Float32Array([1, 0, 1, 0]),
      [2, 2]
    );

    const accValue = computeMetric("accuracy", yPred, yTrue);
    expect(accValue).toEqual(1.0);

    const maeValue = computeMetric("mae", yPred, yTrue);
    expect(maeValue).toBeGreaterThan(0);
  });

  it("should compile and fit with string loss, string optimizer, and metrics", () => {
    const model = new Sequential([
      new Dense({ units: 1, useBias: false })
    ]);

    const x = Matrix.fromFlat(new Float32Array([1.0, 2.0, 3.0]), [3, 1]);
    const y = Matrix.fromFlat(new Float32Array([0.5, 1.0, 1.5]), [3, 1]);

    model.compile({
      optimizer: "sgd",
      loss: "mse",
      metrics: ["mae"]
    });

    const history = model.fit(x, y, {
      epochs: 2,
      batchSize: 1,
      verbose: 0
    });

    expect(history.length).toBe(2);
    expect(history[0].loss).toBeDefined();
    expect(history[0].metrics?.mae).toBeDefined();
    expect(history[0].metrics?.mae).toBeGreaterThan(0);
  });
});
