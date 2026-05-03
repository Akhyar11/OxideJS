import CategoricalCrossEntropy from "../../src/cost/crossEntropy";
import SoftmaxCrossEntropy from "../../src/cost/softmaxCrossEntropy";
import mj from "../../src/math";
import { setForceDisableNative } from "../../src/math/rust_backend";
import Matrix from "../../src/matrix";
import { Sequential, Transformers } from "../../src/models";
import { Dense } from "../../src/layers";

function assert(condition: boolean, message: string): void {
  if (!condition) {
    throw new Error(message);
  }
}

function assertClose(actual: number, expected: number, tolerance: number, message: string): void {
  if (!Number.isFinite(actual) || Math.abs(actual - expected) > tolerance) {
    throw new Error(`${message}: expected ${expected}, got ${actual}`);
  }
}

class WeightedLossSequential extends Sequential {
  private callCount = 0;

  protected override computeLossAndWeight(_yTrue: Matrix, _yPred: Matrix): { loss: number; weight: number } {
    this.callCount++;
    if (this.callCount === 1) {
      return { loss: 10, weight: 1 };
    }
    return { loss: 4, weight: 3 };
  }
}

class SparseSoftmaxGuardSequential extends Sequential {
  public computeLossPublic(yTrue: Matrix, yPred: Matrix): number {
    return this.computeSampleLoss(yTrue, yPred);
  }
}

function createTransformerForGradientTest(): Transformers {
  return new Transformers({
    units: 8,
    seqLen: 4,
    vocabSize: 5,
    heads: 2,
    numBlocks: 1,
    dropoutRate: 0,
    alpha: 1,
    padTokenId: 0,
  });
}

function buildBatch(samples: number[][]): Matrix {
  const rows = samples[0].length;
  const cols = samples.length;
  const out = mj.zeros([rows, cols]);
  for (let col = 0; col < cols; col++) {
    out.setCol(col, Float32Array.from(samples[col]));
  }
  return out;
}

function runSoftmaxCrossEntropyScalingTest(): void {
  const logits = mj.matrix([
    [2, 0],
    [1, 3],
    [0, -1],
  ]);
  const yTrue = mj.matrix([[0, 1]]);
  const [loss, grad] = SoftmaxCrossEntropy(yTrue, logits);

  assert(Number.isFinite(loss), "softmax cross entropy: loss must be finite");

  const probs = [
    [0.66524096, 0.04661262],
    [0.24472847, 0.93623955],
    [0.09003057, 0.01714783],
  ];
  const expected = [
    [(probs[0][0] - 1) / 2, probs[0][1] / 2],
    [probs[1][0] / 2, (probs[1][1] - 1) / 2],
    [probs[2][0] / 2, probs[2][1] / 2],
  ];

  for (let col = 0; col < 2; col++) {
    let colSum = 0;
    for (let row = 0; row < 3; row++) {
      const actual = grad._data[row * 2 + col];
      colSum += actual;
      assertClose(actual, expected[row][col], 1e-5, `softmax cross entropy grad mismatch row=${row} col=${col}`);
    }
    assertClose(colSum, 0, 1e-6, `softmax cross entropy grad column sum mismatch col=${col}`);
  }
}

function runCategoricalCrossEntropyScalingTest(): void {
  const yTrue = mj.matrix([
    [1, 0],
    [0, 1],
    [0, 0],
  ]);
  const yPred = mj.matrix([
    [0.8, 0.1],
    [0.15, 0.7],
    [0.05, 0.2],
  ]);

  const [loss, grad] = CategoricalCrossEntropy(yTrue, yPred);
  const expectedLoss = (-Math.log(0.8) - Math.log(0.7)) / 2;

  assertClose(loss, expectedLoss, 1e-6, "categorical cross entropy loss mismatch");
  assertClose(grad._data[0], -1 / (0.8 * 2), 1e-6, "categorical cross entropy grad[0,0] mismatch");
  assertClose(grad._data[3], -1 / (0.7 * 2), 1e-6, "categorical cross entropy grad[1,1] mismatch");
  assertClose(grad._data[1], 0, 1e-6, "categorical cross entropy grad[0,1] mismatch");
}

function runTransformerFallbackGradientScalingTest(): void {
  setForceDisableNative(true);
  try {
    const model = createTransformerForGradientTest();
    const batchX = buildBatch([
      [1, 2, 3, 4],
      [1, 2, 3, 4],
    ]);
    const batchY = buildBatch([
      [2, 3, 4, 0],
      [2, 3, 4, 0],
    ]);
    const dense = model.layers[model.layers.length - 1] as Dense;
    const totalTokens = batchX._shape[0] * batchX._shape[1];
    const logits = Matrix.fromFlat(new Float32Array([
      2, 0, 1, -1, 2, 0, 1, -1,
      0, 3, 1, 2, 0, 3, 1, 2,
      -1, 0, 2, 1, -1, 0, 2, 1,
      1, -2, 0, 0, 1, -2, 0, 0,
      0, 1, -1, 3, 0, 1, -1, 3,
    ]), [model.vocabSize, totalTokens]);

    (model as any).lastInputTokens = batchX;
    (dense as any).result = logits;

    const state = (model as any).buildShiftedLossGradient(batchY) as {
      loss: number;
      gradient: Matrix;
      validTokens: number;
    };

    assert(Number.isFinite(state.loss), "transformers fallback gradient: loss must be finite");
    assert(state.validTokens === 6, `transformers fallback gradient: expected 6 valid tokens, got ${state.validTokens}`);

    const expectedGrad = new Float32Array(model.vocabSize * totalTokens);
    const batchSize = batchX._shape[1];
    const seqLen = batchX._shape[0];

    for (let b = 0; b < batchSize; b++) {
      for (let pos = 0; pos < seqLen; pos++) {
        const sourceIndex = pos * batchSize + b;
        const tokenIndex = b * seqLen + pos;
        if (pos === seqLen - 1) {
          continue;
        }

        let maxLogit = -Infinity;
        for (let vocab = 0; vocab < model.vocabSize; vocab++) {
          const value = logits._data[vocab * totalTokens + tokenIndex];
          if (value > maxLogit) maxLogit = value;
        }

        let sumExp = 0;
        for (let vocab = 0; vocab < model.vocabSize; vocab++) {
          sumExp += Math.exp(logits._data[vocab * totalTokens + tokenIndex] - maxLogit);
        }

        for (let vocab = 0; vocab < model.vocabSize; vocab++) {
          expectedGrad[vocab * totalTokens + tokenIndex] =
            Math.exp(logits._data[vocab * totalTokens + tokenIndex] - maxLogit) / sumExp / state.validTokens;
        }

        const target = batchY._data[sourceIndex];
        expectedGrad[target * totalTokens + tokenIndex] -= 1 / state.validTokens;
      }
    }

    for (let i = 0; i < expectedGrad.length; i++) {
      assertClose(
        state.gradient._data[i],
        expectedGrad[i],
        1e-6,
        `transformers fallback gradient mismatch idx=${i}`
      );
    }
  } finally {
    setForceDisableNative(false);
  }
}

function runSequentialLossAggregationTest(): void {
  const model = new WeightedLossSequential();
  const X = [mj.matrix([[1]]), mj.matrix([[2]])];
  const y = [mj.matrix([[1]]), mj.matrix([[2]])];

  const result = model.fit(X, y, 1, {
    batchSize: 1,
    shuffle: false,
    verbose: false,
  });

  assertClose(result.history.loss[0], 5.5, 1e-6, "sequential weighted loss aggregation mismatch");
}

function runSparseSoftmaxGuardTest(): void {
  const model = new SparseSoftmaxGuardSequential({
    layers: [
      new Dense({ units: 2, outputUnits: 3, activation: "softmax", status: "output", loss: "mse" }),
    ],
  });

  let thrown = false;
  try {
    model.computeLossPublic(
      mj.matrix([[1]]),
      mj.matrix([
        [0.7],
        [0.2],
        [0.1],
      ])
    );
  } catch (error: any) {
    thrown = error?.message?.includes("Sparse multiclass target requires activation='linear'");
  }
  assert(thrown, "sparse softmax guard: expected clear double-softmax error");
}

export function runLossScalingCorrectnessSuite(): void {
  runSoftmaxCrossEntropyScalingTest();
  runCategoricalCrossEntropyScalingTest();
  runTransformerFallbackGradientScalingTest();
  runSequentialLossAggregationTest();
  runSparseSoftmaxGuardTest();

  console.log("=== Loss Scaling Correctness ===");
  console.table([
    { check: "softmaxCrossEntropy sparse scaling", status: "pass" },
    { check: "categoricalCrossEntropy batch scaling", status: "pass" },
    { check: "transformers fallback token scaling", status: "pass" },
    { check: "sequential weighted aggregation", status: "pass" },
    { check: "sparse softmax guard", status: "pass" },
  ]);
}
