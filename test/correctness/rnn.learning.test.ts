import { Dense, GRU, LSTM, RNN } from "../../src/layers";
import mj from "../../src/math";
import Matrix from "../../src/matrix";
import { Sequential } from "../../src/models";

type RecurrentFamily = "rnn" | "lstm" | "gru";

type LearningResult = {
  family: RecurrentFamily;
  initialLoss: number;
  finalLoss: number;
  history: number[];
};

function assert(condition: boolean, message: string): void {
  if (!condition) {
    throw new Error(message);
  }
}

function buildSequence(bits: number[]): Matrix {
  const rows = [
    bits.map((bit) => (bit === 0 ? 1 : 0)),
    bits.map((bit) => (bit === 1 ? 1 : 0)),
  ];
  return mj.matrix(rows);
}

function createLearningDataset(): { X: Matrix[]; y: Matrix[] } {
  const patterns = [
    [0, 0, 0, 0],
    [0, 1, 0, 0],
    [1, 0, 1, 0],
    [1, 1, 0, 0],
    [0, 0, 1, 1],
    [0, 1, 1, 1],
    [1, 0, 0, 1],
    [1, 1, 1, 1],
  ];

  const X: Matrix[] = [];
  const y: Matrix[] = [];

  for (let repeat = 0; repeat < 8; repeat++) {
    for (const pattern of patterns) {
      X.push(buildSequence(pattern));
      y.push(mj.matrix([[pattern[pattern.length - 1]]]));
    }
  }

  return { X, y };
}

function createRecurrentLayer(family: RecurrentFamily, returnSequences = false): RNN | LSTM | GRU {
  return family === "rnn"
    ? new RNN({
        units: 2,
        hiddenUnits: 8,
        activation: "tanh",
        returnSequences,
        status: "input",
      })
    : family === "lstm"
      ? new LSTM({
          units: 2,
          hiddenUnits: 8,
          returnSequences,
          status: "input",
        })
      : new GRU({
          units: 2,
          hiddenUnits: 8,
          returnSequences,
          status: "input",
        });
}

function createModel(family: RecurrentFamily): Sequential {
  const model = new Sequential({
    layers: [
      createRecurrentLayer(family),
      new Dense({
        units: 8,
        outputUnits: 2,
        activation: "linear",
        status: "output",
        loss: "softmaxCrossEntropy",
      }),
    ],
  });

  model.compile({
    alpha: 0.01,
    optimizer: "adam",
    error: "softmaxCrossEntropy",
  });

  return model;
}

function numericSnapshot(value: unknown, out: number[] = []): number[] {
  if (typeof value === "number") {
    out.push(value);
    return out;
  }
  if (Array.isArray(value)) {
    for (const item of value) numericSnapshot(item, out);
    return out;
  }
  if (value && typeof value === "object") {
    const record = value as Record<string, unknown>;
    for (const key of Object.keys(record).sort()) {
      if (key.toLowerCase().includes("optimizer")) continue;
      numericSnapshot(record[key], out);
    }
  }
  return out;
}

function assertSnapshotChanged(before: number[], after: number[], label: string): void {
  assert(before.length === after.length, `${label}: snapshot length changed (${before.length} -> ${after.length})`);
  let changed = 0;
  let maxAbsDelta = 0;
  for (let i = 0; i < before.length; i++) {
    const delta = Math.abs(after[i] - before[i]);
    if (delta > 1e-9) changed++;
    if (delta > maxAbsDelta) maxAbsDelta = delta;
  }
  assert(changed > 0, `${label}: expected recurrent weights/state parameters to update, but no parameter changed`);
  assert(Number.isFinite(maxAbsDelta), `${label}: weight update delta must be finite`);
}

function assertFiniteMatrix(matrix: Matrix, label: string): void {
  assert(matrix._data.length > 0, `${label}: matrix must not be empty`);
  for (const value of matrix._data) {
    assert(Number.isFinite(value), `${label}: matrix contains non-finite value ${value}`);
  }
}

function createSequenceBatch(samples: Matrix[]): Matrix {
  assert(samples.length > 0, "createSequenceBatch: samples must not be empty");
  const [units, seqLen] = samples[0]._shape;
  const batchSize = samples.length;
  const batch = mj.zeros([units, seqLen * batchSize]);

  for (let sampleIndex = 0; sampleIndex < batchSize; sampleIndex++) {
    const sample = samples[sampleIndex];
    assert(sample._shape[0] === units && sample._shape[1] === seqLen, "all sequence samples must share shape");
    for (let t = 0; t < seqLen; t++) {
      const targetCol = t * batchSize + sampleIndex;
      for (let row = 0; row < units; row++) {
        batch._data[row * batch._shape[1] + targetCol] = sample._data[row * seqLen + t];
      }
    }
  }

  return batch;
}

function assertMatricesClose(actual: Matrix, expected: Matrix, label: string, tolerance = 1e-5): void {
  assert(
    actual._shape[0] === expected._shape[0] && actual._shape[1] === expected._shape[1],
    `${label}: shape mismatch, expected [${expected._shape}], got [${actual._shape}]`
  );
  for (let i = 0; i < actual._data.length; i++) {
    const delta = Math.abs(actual._data[i] - expected._data[i]);
    assert(delta <= tolerance, `${label}: mismatch at flat index ${i}, delta=${delta}`);
  }
}

function assertLayerBatchFeedMatchesPerSample(family: RecurrentFamily, returnSequences: boolean): void {
  const samples = [
    buildSequence([0, 1, 0, 1]),
    buildSequence([1, 0, 1, 0]),
    buildSequence([1, 1, 0, 1]),
  ];
  const batchSize = samples.length;
  const batch = createSequenceBatch(samples);
  const referenceLayer = createRecurrentLayer(family, returnSequences) as any;
  const serialized = referenceLayer.save();

  const batchLayer = createRecurrentLayer(family, returnSequences) as any;
  batchLayer.load(serialized);
  const batchOut = batchLayer.forwardBatch(batch, batchSize);
  assertFiniteMatrix(batchOut, `${family}.forwardBatch(returnSequences=${returnSequences})`);

  const [hiddenRows] = batchOut._shape;
  const expectedCols = returnSequences ? samples[0]._shape[1] * batchSize : batchSize;
  const expected = mj.zeros([hiddenRows, expectedCols]);

  for (let sampleIndex = 0; sampleIndex < batchSize; sampleIndex++) {
    const perSampleLayer = createRecurrentLayer(family, returnSequences) as any;
    perSampleLayer.load(serialized);
    const out = perSampleLayer.forward(samples[sampleIndex]);
    assertFiniteMatrix(out, `${family}.forward sample ${sampleIndex}`);

    if (returnSequences) {
      for (let t = 0; t < samples[sampleIndex]._shape[1]; t++) {
        const targetCol = t * batchSize + sampleIndex;
        for (let row = 0; row < hiddenRows; row++) {
          expected._data[row * expectedCols + targetCol] = out._data[row * out._shape[1] + t];
        }
      }
    } else {
      for (let row = 0; row < hiddenRows; row++) {
        expected._data[row * expectedCols + sampleIndex] = out._data[row];
      }
    }
  }

  assertMatricesClose(batchOut, expected, `${family}: batched recurrent feed must match per-sample feed`);
}

function assertSequentialRecurrentBatchGuard(family: RecurrentFamily): void {
  const { X, y } = createLearningDataset();
  const model = createModel(family);
  let didThrow = false;
  try {
    model.fit(X.slice(0, 4), y.slice(0, 4), 1, {
      batchSize: 2,
      shuffle: false,
      verbose: false,
    });
  } catch (error) {
    didThrow = error instanceof Error && error.message.includes("batchSize=1");
  }
  assert(
    didThrow,
    `${family}: Sequential.fit must explicitly guard recurrent generic batching until full sequence-batch training is wired end-to-end`
  );
}

function assertSequentialUpdatesRecurrentWeights(family: RecurrentFamily): void {
  const { X, y } = createLearningDataset();
  const model = createModel(family);
  const recurrentLayer = model.layers[0] as any;
  const before = numericSnapshot(recurrentLayer.save());

  const result = model.fit(X.slice(0, 16), y.slice(0, 16), 2, {
    batchSize: 1,
    shuffle: false,
    verbose: false,
  });

  const after = numericSnapshot(recurrentLayer.save());
  assertSnapshotChanged(before, after, `${family}: Sequential.fit recurrent update`);
  assert(result.history.loss.length === 2, `${family}: expected 2 training loss entries`);
  assertFiniteMatrix(model.predict(X[0]), `${family}: prediction after recurrent update`);
}

function runLearningTest(family: RecurrentFamily): LearningResult {
  const { X, y } = createLearningDataset();
  const model = createModel(family);
  const history: number[] = [];

  model.fit(X, y, 10, {
    batchSize: 1,
    shuffle: false,
    verbose: false,
    onEpochEnd: (_epoch, loss) => {
      history.push(loss);
    },
  });

  assert(history.length === 10, `${family}: expected 10 loss entries, got ${history.length}`);

  const initialLoss = history[0];
  const finalLoss = history[history.length - 1];
  const bestLoss = Math.min(...history);

  assert(Number.isFinite(initialLoss), `${family}: initial loss must be finite`);
  assert(Number.isFinite(finalLoss), `${family}: final loss must be finite`);
  assert(
    finalLoss < initialLoss,
    `${family}: expected final loss to be lower than initial loss (${finalLoss} >= ${initialLoss})`
  );
  assert(
    bestLoss <= initialLoss - 0.01,
    `${family}: expected at least one meaningful loss improvement, history=${history.map((v) => v.toFixed(6)).join(", ")}`
  );

  assertSequentialUpdatesRecurrentWeights(family);
  assertLayerBatchFeedMatchesPerSample(family, false);
  assertLayerBatchFeedMatchesPerSample(family, true);
  assertSequentialRecurrentBatchGuard(family);

  return {
    family,
    initialLoss,
    finalLoss,
    history,
  };
}

export function runRecurrentLearningCorrectnessSuite(): LearningResult[] {
  const families: RecurrentFamily[] = ["rnn", "lstm", "gru"];
  const results = families.map((family) => runLearningTest(family));

  console.log("=== Recurrent Learning Correctness ===");
  console.table(
    results.map((result) => ({
      family: result.family,
      initialLoss: Number(result.initialLoss.toFixed(6)),
      finalLoss: Number(result.finalLoss.toFixed(6)),
      improvement: Number((result.initialLoss - result.finalLoss).toFixed(6)),
    }))
  );

  return results;
}
