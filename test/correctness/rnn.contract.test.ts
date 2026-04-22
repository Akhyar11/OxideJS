import assert from "node:assert/strict";
import { mkdtempSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";
import RNN from "../../src/layers/rnn";
import Sequential from "../../src/models/sequential";
import mj from "../../src/math";

function test(name: string, fn: () => void) {
  try {
    fn();
    console.log(`PASS ${name}`);
  } catch (error) {
    console.error(`FAIL ${name}`);
    throw error;
  }
}

function approxEqual(actual: number, expected: number, epsilon = 1e-6) {
  assert.ok(
    Math.abs(actual - expected) <= epsilon,
    `expected ${actual} to be within ${epsilon} of ${expected}`
  );
}

function assertMatrixShape(matrix: { _shape: [number, number] }, expected: [number, number]) {
  assert.deepEqual(matrix._shape, expected);
}

function createDeterministicRNN(config: Partial<ConstructorParameters<typeof RNN>[0]> = {}) {
  const rnn = new RNN({
    units: 1,
    hiddenUnits: 1,
    activation: "tanh",
    optimizer: "sgd",
    alpha: 0,
    ...config,
  });
  rnn.load({
    Wxh: [[1]],
    Whh: [[1]],
    bh: [[0]],
    hStateful: [[0]],
    clipGradient: false,
  });
  return rnn;
}

test("RNN forward shape with returnSequences=false", () => {
  const rnn = new RNN({
    units: 2,
    hiddenUnits: 3,
    returnSequences: false,
    optimizer: "sgd",
    alpha: 0,
  });
  const x = mj.matrix([
    [0.1, 0.2, 0.3],
    [0.4, 0.5, 0.6],
  ]);

  const out = rnn.forward(x);

  assertMatrixShape(out, [3, 1]);
  assert.deepEqual(rnn.inputShape, [2, 3]);
  assert.deepEqual(rnn.outputShape, [3, 1]);
});

test("RNN forward shape with returnSequences=true", () => {
  const rnn = new RNN({
    units: 2,
    hiddenUnits: 3,
    returnSequences: true,
    optimizer: "sgd",
    alpha: 0,
  });
  const x = mj.matrix([
    [0.1, 0.2, 0.3, 0.4],
    [0.5, 0.6, 0.7, 0.8],
  ]);

  const out = rnn.forward(x);

  assertMatrixShape(out, [3, 4]);
  assert.deepEqual(rnn.inputShape, [2, 4]);
  assert.deepEqual(rnn.outputShape, [3, 4]);
});

test("RNN backward returns dx with original single-sequence shape", () => {
  const rnn = new RNN({
    units: 2,
    hiddenUnits: 3,
    returnSequences: false,
    optimizer: "sgd",
    alpha: 0,
  });
  const x = mj.matrix([
    [0.5, -0.2, 0.1],
    [0.3, 0.4, -0.1],
  ]);

  rnn.forward(x);
  const dx = rnn.backward(mj.zeros([3, 1]), mj.zeros([3, 1]));

  assertMatrixShape(dx, [2, 3]);
});

test("RNN stateful carries state across calls and resetState clears it", () => {
  const rnn = createDeterministicRNN({ stateful: true });
  const first = rnn.forward(mj.matrix([[1]]));
  const firstValue = first.get(0, 0);
  const carried = rnn.forward(mj.matrix([[0]]));
  const carriedValue = carried.get(0, 0);

  approxEqual(firstValue, Math.tanh(1));
  approxEqual(carriedValue, Math.tanh(firstValue));
  approxEqual(rnn.getState().get(0, 0), carriedValue);

  rnn.resetState();
  approxEqual(rnn.getState().get(0, 0), 0);

  const afterReset = rnn.forward(mj.matrix([[0]]));
  approxEqual(afterReset.get(0, 0), 0);
});

test("Sequential save/load roundtrip preserves RNN weights and state", () => {
  const tempDir = mkdtempSync(join(tmpdir(), "ml-v2-rnn-"));
  const modelPath = join(tempDir, "rnn.json");

  try {
    const model = new Sequential({
      layers: [createDeterministicRNN({ stateful: true, returnSequences: false })],
    });
    model.forward(mj.matrix([[1]]));
    model.save(modelPath);

    const loaded = new Sequential();
    loaded.load(modelPath);

    const loadedLayer = loaded.layers[0] as RNN;
    assert.deepEqual(loadedLayer.Wxh._value, [[1]]);
    assert.deepEqual(loadedLayer.Whh._value, [[1]]);
    assert.deepEqual(loadedLayer.bh._value, [[0]]);
    approxEqual(loadedLayer.getState().get(0, 0), Math.tanh(1));
  } finally {
    rmSync(tempDir, { recursive: true, force: true });
  }
});

test("RNN returnState throws instead of being silently ignored", () => {
  const rnn = new RNN({
    units: 1,
    hiddenUnits: 1,
    returnState: true,
  });

  assert.throws(
    () => rnn.forward(mj.matrix([[1]])),
    /returnState=true is not supported yet/
  );
});

test("Sequential.fit rejects recurrent batchSize > 1 because generic batching is invalid for sequences", () => {
  const model = new Sequential({
    layers: [
      new RNN({
        units: 1,
        hiddenUnits: 1,
        status: "output",
      }),
    ],
  });
  const X = [mj.matrix([[1, 2]]), mj.matrix([[3, 4]])];
  const y = [mj.matrix([[0]]), mj.matrix([[0]])];

  assert.throws(
    () => model.fit(X, y, 1, { batchSize: 2, shuffle: false }),
    /batchSize=1/
  );
});

test("Sequential.fit rejects stateful recurrent training with shuffle=true", () => {
  const model = new Sequential({
    layers: [
      new RNN({
        units: 1,
        hiddenUnits: 1,
        stateful: true,
        status: "output",
      }),
    ],
  });
  const X = [mj.matrix([[1]]), mj.matrix([[2]])];
  const y = [mj.matrix([[0]]), mj.matrix([[0]])];

  assert.throws(
    () => model.fit(X, y, 1, { batchSize: 1, shuffle: true }),
    /stateful=true.*shuffle=true/
  );
});
