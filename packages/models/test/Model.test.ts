import { describe, expect, it, vi } from "vitest";
import { Sequential } from "../src/index.js";
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
});
