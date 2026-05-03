import { unlinkSync } from "fs";
import { MemoryBank } from "../../src/layers";
import mj from "../../src/math";
import Matrix from "../../src/matrix";
import setLayers from "../../src/utils/setLayers";
import { isNativeAvailable, setForceDisableNative } from "../../src/math/rust_backend";

function assert(condition: boolean, message: string): void {
  if (!condition) throw new Error(message);
}

function assertShape(m: Matrix, rows: number, cols: number, message: string): void {
  assert(m._shape[0] === rows && m._shape[1] === cols, `${message}: expected [${rows},${cols}], got [${m._shape[0]},${m._shape[1]}]`);
}

function assertMatrixClose(a: Matrix, b: Matrix, tol: number, message: string): void {
  assert(a._shape[0] === b._shape[0] && a._shape[1] === b._shape[1], `${message}: shape mismatch`);
  for (let i = 0; i < a._data.length; i++) {
    if (Math.abs(a._data[i] - b._data[i]) > tol) {
      throw new Error(`${message}: mismatch at index ${i}, ${a._data[i]} vs ${b._data[i]}`);
    }
  }
}

function assertMatrixUnchanged(before: Matrix, after: Matrix, message: string): void {
  assert(before._data.length === after._data.length, `${message}: length mismatch`);
  for (let i = 0; i < before._data.length; i++) {
    if (Math.abs(before._data[i] - after._data[i]) > 1e-12) {
      throw new Error(`${message}: changed at index ${i}`);
    }
  }
}

function assertMatrixChanged(before: Matrix, after: Matrix, message: string): void {
  assert(before._data.length === after._data.length, `${message}: length mismatch`);
  let changed = false;
  for (let i = 0; i < before._data.length; i++) {
    if (Math.abs(before._data[i] - after._data[i]) > 1e-12) {
      changed = true;
      break;
    }
  }
  assert(changed, message);
}

function setKernelIdentityLike(layer: any): void {
  const units = layer.units;
  const dim = layer.memoryDim;

  layer.writeKeyKernel = mj.zeros([dim, units]);
  layer.writeValueKernel = mj.zeros([dim, units]);
  for (let i = 0; i < Math.min(dim, units); i++) {
    layer.writeKeyKernel._data[i * units + i] = 1;
    layer.writeValueKernel._data[i * units + i] = 1;
  }
}

function constantMatrix(rows: number, cols: number, v: number): Matrix {
  return mj.matrix(new Array(rows).fill(0).map(() => new Array(cols).fill(v)));
}

function baseInput(): Matrix {
  return mj.matrix([
    [0.2, 0.4],
    [0.1, -0.2],
    [0.3, 0.5],
  ]);
}

function cloneState(state: any): any {
  return JSON.parse(JSON.stringify(state));
}

function changedSlotIndex(before: any, after: any): number {
  for (let s = 0; s < before.memorySlots; s++) {
    let diff = false;
    for (let d = 0; d < before.memoryDim; d++) {
      if (Math.abs(before.memoryKeys[d][s] - after.memoryKeys[d][s]) > 1e-8) {
        diff = true;
        break;
      }
      if (Math.abs(before.memoryValues[d][s] - after.memoryValues[d][s]) > 1e-8) {
        diff = true;
        break;
      }
    }
    if (diff) return s;
  }
  return -1;
}

export function runMemoryBankCorrectnessSuite(): void {
  // 1) zero error no update
  {
    const layer = new MemoryBank({ units: 3, memorySlots: 4, mode: "project", writeThreshold: 1.0, optimizer: "sgd", alpha: 0.2 });
    const x = baseInput();
    const out = layer.forward(x);
    const q0 = (layer as any).queryKernel.clone();
    const n0 = (layer as any).needKernel.clone();
    const o0 = (layer as any).outputKernel.clone();
    const b0 = (layer as any).outputBias.clone();
    const wk0 = (layer as any).writeKeyKernel.clone();
    const wv0 = (layer as any).writeValueKernel.clone();
    const wg0 = (layer as any).writeGateKernel.clone();

    const err = mj.zeros(out._shape);
    layer.backward(mj.matrix([[]]), err);

    assertMatrixUnchanged(q0, (layer as any).queryKernel, "zero error should not update queryKernel");
    assertMatrixUnchanged(n0, (layer as any).needKernel, "zero error should not update needKernel");
    assertMatrixUnchanged(o0, (layer as any).outputKernel, "zero error should not update outputKernel");
    assertMatrixUnchanged(b0, (layer as any).outputBias, "zero error should not update outputBias");
    assertMatrixUnchanged(wk0, (layer as any).writeKeyKernel, "zero error should not update writeKeyKernel");
    assertMatrixUnchanged(wv0, (layer as any).writeValueKernel, "zero error should not update writeValueKernel");
    assertMatrixUnchanged(wg0, (layer as any).writeGateKernel, "zero error should not update writeGateKernel");
  }

  // 2) non-zero error updates read/output policy params
  {
    const layer = new MemoryBank({ units: 3, memorySlots: 4, mode: "project", writeThreshold: 0.0, optimizer: "sgd", alpha: 0.1 });
    layer.forward(mj.matrix([[0], [0], [0]])); // init
    layer.setMemoryState({
      memoryKeys: [
        [1, -1, 0, 0.5],
        [0, 0.2, -0.3, 0.1],
        [0.5, 0.5, 0.5, 0.5],
      ],
      memoryValues: [
        [0.3, -0.1, 0.8, 0.2],
        [0.4, 0.5, -0.6, 0.1],
        [0.7, -0.2, 0.2, 0.9],
      ],
      memoryFilled: [1, 1, 1, 1],
      memoryUsage: [1, 2, 3, 4],
      memoryAge: [1, 2, 3, 4],
      memoryStep: 5,
      units: 3,
      memoryDim: 3,
      memorySlots: 4,
    });

    const x = baseInput();
    const out = layer.forward(x);

    const q0 = (layer as any).queryKernel.clone();
    const n0 = (layer as any).needKernel.clone();
    const o0 = (layer as any).outputKernel.clone();
    const b0 = (layer as any).outputBias.clone();
    const wk0 = (layer as any).writeKeyKernel.clone();
    const wv0 = (layer as any).writeValueKernel.clone();
    const wg0 = (layer as any).writeGateKernel.clone();

    const err = constantMatrix(out._shape[0], out._shape[1], 1);
    const dx = layer.backward(mj.matrix([[]]), err);
    assertShape(dx, 3, 2, "dx shape for non-zero error");

    assertMatrixChanged(q0, (layer as any).queryKernel, "non-zero error should update queryKernel");
    assertMatrixChanged(n0, (layer as any).needKernel, "non-zero error should update needKernel");
    assertMatrixChanged(o0, (layer as any).outputKernel, "non-zero error should update outputKernel");
    assertMatrixChanged(b0, (layer as any).outputBias, "non-zero error should update outputBias");

    assertMatrixChanged(wk0, (layer as any).writeKeyKernel, "non-zero error with writes should update writeKeyKernel");
    assertMatrixChanged(wv0, (layer as any).writeValueKernel, "non-zero error with writes should update writeValueKernel");
    assertMatrixChanged(wg0, (layer as any).writeGateKernel, "non-zero error with writes should update writeGateKernel");
  }

  // 3) deterministic forward/write (no random projection inside forward)
  {
    const x = baseInput();
    const a = new MemoryBank({ units: 3, memorySlots: 3, mode: "project", writeThreshold: 0.0 });
    const _ = a.forward(x); // initialize

    const saveA = a.save();
    const initState = a.getMemoryState();
    a.resetMemory();
    a.setMemoryState(cloneState(initState));

    const outA = a.forward(x);
    const stateA = a.getMemoryState();

    const b = new MemoryBank({ units: 3, memorySlots: 3, mode: "project", writeThreshold: 0.0 });
    b.load(saveA);
    b.setMemoryState(cloneState(initState));

    const outB = b.forward(x);
    const stateB = b.getMemoryState();

    assertMatrixClose(outA, outB, 1e-7, "deterministic output with same weights/state/input");
    assert(JSON.stringify(stateA) === JSON.stringify(stateB), "deterministic memory write with same weights/state/input");
  }

  // 4) writeGate depends on input
  {
    const layer = new MemoryBank({ units: 3, memorySlots: 2, mode: "project", writeThreshold: 0.9 });
    layer.forward(mj.matrix([[0], [0], [0]])); // init
    layer.resetMemory();

    const wg = (layer as any).writeGateKernel as Matrix;
    wg._data.fill(0);
    wg._data[0] = 8; // depends on first input feature

    layer.forward(mj.matrix([[1], [0], [0]]));
    const wrotePositive = layer.hasMemory();

    layer.resetMemory();
    layer.forward(mj.matrix([[-1], [0], [0]]));
    const wroteNegative = layer.hasMemory();

    assert(wrotePositive !== wroteNegative, "write decision should differ for different inputs via writeGateKernel");
  }

  // 5) updateMode behavior
  {
    const oldState = {
      memoryKeys: [[1], [0]],
      memoryValues: [[1], [3]],
      memoryFilled: [1],
      memoryUsage: [5],
      memoryAge: [2],
      memoryStep: 10,
      units: 2,
      memoryDim: 2,
      memorySlots: 1,
    };

    const x = mj.matrix([[2], [4]]);

    const replace = new MemoryBank({ units: 2, memorySlots: 1, memoryDim: 2, outputUnits: 2, updateMode: "replace", writeThreshold: 0, mode: "project" });
    replace.setMemoryState(cloneState(oldState));
    setKernelIdentityLike(replace as any);
    replace.forward(x);
    const sReplace = replace.getMemoryState();

    const merge = new MemoryBank({ units: 2, memorySlots: 1, memoryDim: 2, outputUnits: 2, updateMode: "merge", writeThreshold: 0, mode: "project" });
    merge.setMemoryState(cloneState(oldState));
    setKernelIdentityLike(merge as any);
    merge.forward(x);
    const sMerge = merge.getMemoryState();

    const gated = new MemoryBank({ units: 2, memorySlots: 1, memoryDim: 2, outputUnits: 2, updateMode: "gated-merge", writeThreshold: 0, mode: "project" });
    gated.setMemoryState(cloneState(oldState));
    setKernelIdentityLike(gated as any);
    // gate=0.5
    ((gated as any).writeGateKernel as Matrix)._data.fill(0);
    gated.forward(x);
    const sGated = gated.getMemoryState();

    // replace value should equal new value [2,4]
    assert(Math.abs(sReplace.memoryValues[0][0] - 2) < 1e-6 && Math.abs(sReplace.memoryValues[1][0] - 4) < 1e-6, "replace should overwrite value exactly");
    assert(Math.abs(sMerge.memoryValues[0][0] - 2) > 1e-6 || Math.abs(sMerge.memoryValues[1][0] - 4) > 1e-6, "merge should not equal full replace");
    assert(Math.abs(sGated.memoryValues[0][0] - 2) > 1e-6 || Math.abs(sGated.memoryValues[1][0] - 4) > 1e-6, "gated-merge should not equal full replace");
  }

  // 6) writePolicy behavior
  {
    // empty-first fills empty before overwrite
    const emptyFirst = new MemoryBank({ units: 2, memorySlots: 3, writePolicy: "empty-first", writeThreshold: 0, mode: "project" });
    emptyFirst.forward(mj.matrix([[1, 2], [3, 4]]));
    const efState = emptyFirst.getMemoryState();
    assert(efState.memoryFilled[0] === 1 && efState.memoryFilled[1] === 1 && efState.memoryFilled[2] === 0, "empty-first should fill empty slots first");

    // least-used when full
    const leastUsed = new MemoryBank({ units: 2, memorySlots: 3, writePolicy: "least-used", updateMode: "replace", writeThreshold: 0, mode: "project" });
    leastUsed.forward(mj.matrix([[0], [0]])); // init
    leastUsed.setMemoryState({
      memoryKeys: [[0, 0, 0], [0, 0, 0]],
      memoryValues: [[1, 2, 3], [4, 5, 6]],
      memoryFilled: [1, 1, 1],
      memoryUsage: [10, 1, 5],
      memoryAge: [5, 6, 7],
      memoryStep: 8,
      units: 2,
      memoryDim: 2,
      memorySlots: 3,
    });
    const beforeLU = leastUsed.getMemoryState();
    leastUsed.forward(mj.matrix([[9], [9]]));
    const afterLU = leastUsed.getMemoryState();
    assert(changedSlotIndex(beforeLU, afterLU) === 1, "least-used should choose slot with smallest usage");

    // oldest chooses minimum age
    const oldest = new MemoryBank({ units: 2, memorySlots: 3, writePolicy: "oldest", updateMode: "replace", writeThreshold: 0, mode: "project" });
    oldest.forward(mj.matrix([[0], [0]]));
    oldest.setMemoryState({
      memoryKeys: [[0, 0, 0], [0, 0, 0]],
      memoryValues: [[1, 2, 3], [4, 5, 6]],
      memoryFilled: [1, 1, 1],
      memoryUsage: [5, 5, 5],
      memoryAge: [3, 1, 2],
      memoryStep: 8,
      units: 2,
      memoryDim: 2,
      memorySlots: 3,
    });
    const beforeOld = oldest.getMemoryState();
    oldest.forward(mj.matrix([[9], [9]]));
    const afterOld = oldest.getMemoryState();
    assert(changedSlotIndex(beforeOld, afterOld) === 1, "oldest should choose slot with smallest age");

    // least-relevant chooses lowest similarity
    const leastRel = new MemoryBank({ units: 2, memorySlots: 3, writePolicy: "least-relevant", similarity: "dot", updateMode: "replace", writeThreshold: 0, mode: "project" });
    leastRel.forward(mj.matrix([[0], [0]]));
    // make query ~= [1,0]
    ((leastRel as any).queryKernel as Matrix)._data.fill(0);
    ((leastRel as any).queryKernel as Matrix)._data[0] = 1;

    leastRel.setMemoryState({
      memoryKeys: [[1, 0.5, -1], [0, 0, 0]],
      memoryValues: [[1, 2, 3], [4, 5, 6]],
      memoryFilled: [1, 1, 1],
      memoryUsage: [1, 1, 1],
      memoryAge: [1, 1, 1],
      memoryStep: 2,
      units: 2,
      memoryDim: 2,
      memorySlots: 3,
    });
    const beforeLR = leastRel.getMemoryState();
    leastRel.forward(mj.matrix([[1], [0]]));
    const afterLR = leastRel.getMemoryState();
    assert(changedSlotIndex(beforeLR, afterLR) === 2, "least-relevant should choose slot with lowest similarity");
  }

  // 7) save/load model weights via setLayers
  {
    const layer = new MemoryBank({ units: 3, memorySlots: 3, mode: "project", writeThreshold: 0.1 });
    const x = baseInput();
    layer.forward(x);

    const memory = layer.getMemoryState();
    const saved = layer.save();
    const loaded = setLayers([saved])[0] as MemoryBank;
    loaded.setMemoryState(cloneState(memory));

    const outA = layer.forward(x);
    const outB = loaded.forward(x);
    assertMatrixClose(outA, outB, 1e-7, "save/load through setLayers should preserve output with same memory state");
  }

  // 8) runtime memory state is separate from save()
  {
    const layer = new MemoryBank({ units: 3, memorySlots: 2, mode: "project", writeThreshold: 0 });
    const x = baseInput();
    layer.forward(x);

    const saved = layer.save();
    assert((saved as any).memoryKeys === undefined, "save() must not include memoryKeys runtime state");
    assert((saved as any).memoryValues === undefined, "save() must not include memoryValues runtime state");

    const tmp = "/tmp/memory_bank_runtime_state_test.json";
    const before = layer.getMemoryState();
    layer.saveMemory(tmp);
    layer.resetMemory();
    layer.loadMemory(tmp);
    const after = layer.getMemoryState();
    unlinkSync(tmp);

    assert(JSON.stringify(before) === JSON.stringify(after), "saveMemory/loadMemory should restore runtime memory exactly");
  }

  // trainablePolicy=false: no parameter update but dx finite
  {
    const layer = new MemoryBank({ units: 3, memorySlots: 3, mode: "project", trainablePolicy: false, writeThreshold: 0, optimizer: "sgd", alpha: 0.1 });
    const out = layer.forward(baseInput());
    const q0 = (layer as any).queryKernel.clone();
    const n0 = (layer as any).needKernel.clone();
    const o0 = (layer as any).outputKernel.clone();
    const b0 = (layer as any).outputBias.clone();
    const wk0 = (layer as any).writeKeyKernel.clone();
    const wv0 = (layer as any).writeValueKernel.clone();
    const wg0 = (layer as any).writeGateKernel.clone();

    const dx = layer.backward(mj.matrix([[]]), constantMatrix(out._shape[0], out._shape[1], 1));
    assertShape(dx, 3, 2, "dx shape when trainablePolicy=false");
    for (const v of dx._data) assert(Number.isFinite(v), "dx must be finite when trainablePolicy=false");

    assertMatrixUnchanged(q0, (layer as any).queryKernel, "queryKernel should not change when trainablePolicy=false");
    assertMatrixUnchanged(n0, (layer as any).needKernel, "needKernel should not change when trainablePolicy=false");
    assertMatrixUnchanged(o0, (layer as any).outputKernel, "outputKernel should not change when trainablePolicy=false");
    assertMatrixUnchanged(b0, (layer as any).outputBias, "outputBias should not change when trainablePolicy=false");
    assertMatrixUnchanged(wk0, (layer as any).writeKeyKernel, "writeKeyKernel should not change when trainablePolicy=false");
    assertMatrixUnchanged(wv0, (layer as any).writeValueKernel, "writeValueKernel should not change when trainablePolicy=false");
    assertMatrixUnchanged(wg0, (layer as any).writeGateKernel, "writeGateKernel should not change when trainablePolicy=false");

    assert(layer.hasMemory(), "memory may still update during forward with trainablePolicy=false");
  }

  // Rust vs JS fallback correctness parity for MemoryBank
  if (isNativeAvailable()) {
    const x = baseInput();
    const base = new MemoryBank({ units: 3, memorySlots: 4, mode: "project", writeThreshold: 0.0, updateMode: "gated-merge", similarity: "cosine", optimizer: "sgd", alpha: 0.05 });
    base.forward(mj.matrix([[0], [0], [0]]));
    const saved = base.save();
    const memoryState = {
      memoryKeys: [
        [0.4, -0.2, 0.1, 0.7],
        [0.1, 0.3, -0.5, 0.2],
        [0.2, 0.4, 0.6, -0.1],
      ],
      memoryValues: [
        [0.5, -0.1, 0.2, 0.3],
        [0.6, 0.4, -0.2, 0.1],
        [0.7, -0.3, 0.8, 0.9],
      ],
      memoryFilled: [1, 1, 1, 1],
      memoryUsage: [2, 3, 4, 5],
      memoryAge: [1, 2, 3, 4],
      memoryStep: 6,
      units: 3,
      memoryDim: 3,
      memorySlots: 4,
    };

    const err = constantMatrix(3, 2, 1);

    setForceDisableNative(true);
    const jsLayer = new MemoryBank({ units: 3, memorySlots: 4, mode: "project", writeThreshold: 0.0, updateMode: "gated-merge", similarity: "cosine", optimizer: "sgd", alpha: 0.05 });
    jsLayer.load(saved);
    jsLayer.setMemoryState(cloneState(memoryState));
    const jsOut = jsLayer.forward(x);
    const jsDx = jsLayer.backward(mj.matrix([[]]), err);

    setForceDisableNative(false);
    const nativeLayer = new MemoryBank({ units: 3, memorySlots: 4, mode: "project", writeThreshold: 0.0, updateMode: "gated-merge", similarity: "cosine", optimizer: "sgd", alpha: 0.05 });
    nativeLayer.load(saved);
    nativeLayer.setMemoryState(cloneState(memoryState));
    const nativeOut = nativeLayer.forward(x);
    const nativeDx = nativeLayer.backward(mj.matrix([[]]), err);

    assertMatrixClose(jsOut, nativeOut, 1e-5, "MemoryBank forward parity between JS fallback and native-enabled path");
    assertMatrixClose(jsDx, nativeDx, 1e-5, "MemoryBank backward dx parity between JS fallback and native-enabled path");

    // restore default for subsequent tests/process
    setForceDisableNative(false);
  }
}

if (require.main === module) {
  runMemoryBankCorrectnessSuite();
}
