import assert from "node:assert/strict";
import mj from "../../src/math";
import Matrix from "../../src/matrix";
import Transformers from "../../src/models/transformers";
import { trimPaddingBatch } from "../../src/utils/trimPaddingBatch";

function test(name: string, fn: () => void) {
  try {
    fn();
    console.log(`PASS ${name}`);
  } catch (error) {
    console.error(`FAIL ${name}`);
    throw error;
  }
}

function assertClose(a: number, b: number, tol = 1e-4, label = "") {
  assert.ok(
    Math.abs(a - b) <= tol,
    `${label}: expected ${a} ≈ ${b} (diff=${Math.abs(a - b)}, tol=${tol})`
  );
}

// ---------------------------------------------------------------------------
// Helper: build a tiny deterministic model (dropoutRate=0, no grad clip)
// ---------------------------------------------------------------------------
function buildModel() {
  return new Transformers({
    units: 8,
    seqLen: 6,
    vocabSize: 16,
    heads: 2,
    numBlocks: 1,
    dropoutRate: 0,
    alpha: 1e-3,
    padTokenId: 0,
    clipGradient: false,
  });
}

// ---------------------------------------------------------------------------
// 1. trimPaddingBatch – right-padding
// ---------------------------------------------------------------------------
test("trimPaddingBatch right: trims trailing PAD rows", () => {
  // seqLen=6, batchSize=2; real tokens occupy positions 0-3, rest are PAD (0).
  const xArr = [
    [1, 2],  // pos 0 – real
    [3, 4],  // pos 1 – real
    [5, 6],  // pos 2 – real
    [7, 8],  // pos 3 – real
    [0, 0],  // pos 4 – PAD
    [0, 0],  // pos 5 – PAD
  ];
  const yArr = [
    [3, 4],
    [5, 6],
    [7, 8],
    [9, 10],
    [0, 0],
    [0, 0],
  ];
  const x = mj.matrix(xArr);
  const y = mj.matrix(yArr);

  const result = trimPaddingBatch(x, y, 0, "right");
  assert.equal(result.trimmed, true, "should be trimmed");
  assert.equal(result.effectiveSeqLen, 4, "effectiveSeqLen should be 4");
  assert.equal(result.positionOffset, 0, "positionOffset should be 0");
  assert.deepEqual([...result.x._shape], [4, 2], "x shape after trim");
  assert.deepEqual([...result.y._shape], [4, 2], "y shape after trim");
  // First row of trimmed x should match original pos 0.
  assert.equal(result.x._data[0], 1);
  assert.equal(result.x._data[1], 2);
});

// ---------------------------------------------------------------------------
// 2. trimPaddingBatch – left-padding
// ---------------------------------------------------------------------------
test("trimPaddingBatch left: trims leading PAD rows and returns correct offset", () => {
  const xArr = [
    [0, 0],  // pos 0 – PAD
    [0, 0],  // pos 1 – PAD
    [1, 2],  // pos 2 – real
    [3, 4],  // pos 3 – real
    [5, 6],  // pos 4 – real
    [7, 8],  // pos 5 – real
  ];
  const yArr = [
    [0, 0],
    [0, 0],
    [3, 4],
    [5, 6],
    [7, 8],
    [9, 10],
  ];
  const x = mj.matrix(xArr);
  const y = mj.matrix(yArr);

  const result = trimPaddingBatch(x, y, 0, "left");
  assert.equal(result.trimmed, true, "should be trimmed");
  assert.equal(result.effectiveSeqLen, 4, "effectiveSeqLen should be 4");
  assert.equal(result.positionOffset, 2, "positionOffset should be 2 (firstUsefulPos)");
  assert.deepEqual([...result.x._shape], [4, 2], "x shape after trim");
  // First row of trimmed x should be the original pos 2 row.
  assert.equal(result.x._data[0], 1);
  assert.equal(result.x._data[1], 2);
});

// ---------------------------------------------------------------------------
// 3. trimPaddingBatch – no-trim if no PAD present
// ---------------------------------------------------------------------------
test("trimPaddingBatch right: no-op if no PAD present", () => {
  const x = mj.matrix([[1, 2], [3, 4], [5, 6]]);
  const y = mj.matrix([[2, 3], [4, 5], [6, 7]]);
  const result = trimPaddingBatch(x, y, 0, "right");
  assert.equal(result.trimmed, false);
  assert.equal(result.effectiveSeqLen, 3);
  // Should be the same object references.
  assert.strictEqual(result.x, x);
  assert.strictEqual(result.y, y);
});

// ---------------------------------------------------------------------------
// 4. trimPaddingBatch – legacy Y=[1,batch] is NOT trimmed (caller must skip)
// ---------------------------------------------------------------------------
test("trimPaddingBatch: mismatched shapes -> returns original", () => {
  const x = mj.matrix([[1, 2], [3, 4], [5, 6]]);        // [3, 2]
  const y = mj.matrix([[7, 8]]);                          // [1, 2]
  // Shapes differ in seqLen so function returns original.
  const result = trimPaddingBatch(x, y, 0, "right");
  assert.equal(result.trimmed, false);
  assert.strictEqual(result.x, x);
  assert.strictEqual(result.y, y);
});

// ---------------------------------------------------------------------------
// 5. Right-pad trim: training works, loss is finite, effective trim occurs
// ---------------------------------------------------------------------------
test("right-pad: trimming produces finite loss and correctly reduces seqLen", () => {
  const model = buildModel();
  model.train();

  // seqLen=6; real tokens at pos 0..3, PAD at pos 4..5.
  const xArr = [
    [1, 2],  // pos 0 – real
    [3, 4],  // pos 1 – real
    [5, 6],  // pos 2 – real
    [7, 8],  // pos 3 – real
    [0, 0],  // pos 4 – PAD
    [0, 0],  // pos 5 – PAD
  ];
  const yArr = [
    [3, 4],
    [5, 6],
    [7, 8],
    [9, 10],
    [0, 0],
    [0, 0],
  ];
  const x = mj.matrix(xArr);
  const y = mj.matrix(yArr);

  // Trim: should reduce seqLen from 6 to 4.
  const trimResult = trimPaddingBatch(x, y, 0, "right");
  assert.equal(trimResult.trimmed, true, "should be trimmed");
  assert.equal(trimResult.effectiveSeqLen, 4, "effectiveSeqLen should be 4");
  assert.equal(trimResult.positionOffset, 0);

  // Forward + backward on trimmed batch.
  model.setPositionOffset(trimResult.positionOffset);
  model.forwardFullSequence(trimResult.x);
  model.backward(trimResult.y);
  model.resetPositionOffset();

  assert.ok(Number.isFinite(model.loss), `loss should be finite, got ${model.loss}`);
  assert.ok(model.loss > 0, "loss should be positive");
});

// ---------------------------------------------------------------------------
// 6. Left-pad trim: training works, loss is finite, offset is applied
// ---------------------------------------------------------------------------
test("left-pad: trimming with positionOffset produces finite loss", () => {
  const model = buildModel();
  model.train();

  const xArr = [
    [0, 0],  // pos 0 – PAD
    [0, 0],  // pos 1 – PAD
    [1, 2],  // pos 2 – real
    [3, 4],  // pos 3 – real
    [5, 6],  // pos 4 – real
    [7, 8],  // pos 5 – real
  ];
  const yArr = [
    [0, 0],
    [0, 0],
    [3, 4],
    [5, 6],
    [7, 8],
    [9, 10],
  ];
  const x = mj.matrix(xArr);
  const y = mj.matrix(yArr);

  const trimResult = trimPaddingBatch(x, y, 0, "left");
  assert.equal(trimResult.trimmed, true, "should be trimmed");
  assert.equal(trimResult.positionOffset, 2, "offset should be 2");
  assert.equal(trimResult.effectiveSeqLen, 4, "effectiveSeqLen should be 4");

  model.setPositionOffset(trimResult.positionOffset);
  model.forwardFullSequence(trimResult.x);
  model.backward(trimResult.y);
  model.resetPositionOffset();

  assert.ok(Number.isFinite(model.loss), `loss should be finite, got ${model.loss}`);
  assert.ok(model.loss > 0, "loss should be positive");

  // Verify positionOffset doesn't leak after reset.
  model.forwardFullSequence(x);
  assert.ok(Number.isFinite(model.loss));
});

// ---------------------------------------------------------------------------
// 7. Transformers.fit() with trimPadding=true doesn't crash and reduces loss
// ---------------------------------------------------------------------------
test("Transformers.fit() with trimPadding=true: runs without error", () => {
  const model = buildModel();

  const makeSeq = (tokens: number[], pad: number, seqLen: number): number[] => {
    const arr = Array(seqLen).fill(pad);
    for (let i = 0; i < tokens.length && i < seqLen; i++) arr[i] = tokens[i];
    return arr;
  };

  const seqLen = 6;
  const pad = 0;
  const xs: Matrix[] = [];
  const ys: Matrix[] = [];

  for (let i = 0; i < 6; i++) {
    const tok = [1, 2, 3, 4, 5].map(t => (t + i) % 15 + 1);
    const xSeq = makeSeq(tok, pad, seqLen);
    const ySeq = makeSeq([...tok.slice(1), pad], pad, seqLen);
    xs.push(Matrix.fromFlat(new Float32Array(xSeq), [seqLen, 1]));
    ys.push(Matrix.fromFlat(new Float32Array(ySeq), [seqLen, 1]));
  }

  let threw = false;
  try {
    model.fit(xs, ys, 2, { batchSize: 2, trimPadding: true, paddingSide: "right", shuffle: false });
  } catch (e) {
    threw = true;
    throw e;
  }
  assert.equal(threw, false, "fit should not throw");
});

// ---------------------------------------------------------------------------
// 8. Legacy Y=[1,batch] is NOT trimmed (fit should still work normally)
// ---------------------------------------------------------------------------
test("Transformers.fit() legacy Y=[1,batch] is not trimmed", () => {
  const model = buildModel();

  const seqLen = 6;
  const xs: Matrix[] = [];
  const ys: Matrix[] = [];

  for (let i = 0; i < 4; i++) {
    xs.push(Matrix.fromFlat(
      new Float32Array([0, 1, 2, 3, 4, 5].map(v => (v + i) % 15)),
      [seqLen, 1]
    ));
    // Legacy: target is only [1, 1] – one token per sample.
    ys.push(Matrix.fromFlat(new Float32Array([(i + 1) % 15]), [1, 1]));
  }

  let threw = false;
  try {
    model.fit(xs, ys, 2, { batchSize: 1, trimPadding: true, paddingSide: "right", shuffle: false });
  } catch (e) {
    threw = true;
    throw e;
  }
  assert.equal(threw, false, "legacy fit should not throw");
});

// ---------------------------------------------------------------------------
// 9. getPadTokenId / setPositionOffset / resetPositionOffset bridge methods
// ---------------------------------------------------------------------------
test("Transformers: getPadTokenId returns correct value", () => {
  const model = buildModel();
  assert.equal(model.getPadTokenId(), 0);
});

test("Transformers: setPositionOffset / resetPositionOffset round-trip", () => {
  const model = buildModel(); // seqLen=6, maxSeqLen=6
  // With x=[4,1], positionOffset=2: absolutePos range = 2..5, all < 6 (maxSeqLen).
  model.setPositionOffset(2);
  const x = Matrix.fromFlat(new Float32Array([1, 2, 3, 4]), [4, 1]);
  model.train();
  assert.doesNotThrow(() => model.forwardFullSequence(x));
  model.resetPositionOffset();
  assert.doesNotThrow(() => model.forwardFullSequence(x));
});
