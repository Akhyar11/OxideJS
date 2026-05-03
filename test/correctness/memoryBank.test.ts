import { unlinkSync } from "fs";
import { MemoryBank } from "../../src/layers";
import mj from "../../src/math";
import Matrix from "../../src/matrix";
import setLayers from "../../src/utils/setLayers";

function assert(condition: boolean, message: string): void {
  if (!condition) throw new Error(message);
}

function assertClose(actual: number, expected: number, tol: number, message: string): void {
  if (Math.abs(actual - expected) > tol) {
    throw new Error(`${message}: expected ${expected}, got ${actual}`);
  }
}

function assertMatrixClose(a: Matrix, b: Matrix, tol: number, message: string): void {
  assert(a._shape[0] === b._shape[0] && a._shape[1] === b._shape[1], `${message}: shape mismatch`);
  for (let i = 0; i < a._data.length; i++) {
    if (Math.abs(a._data[i] - b._data[i]) > tol) {
      throw new Error(`${message}: mismatch at index ${i}, ${a._data[i]} vs ${b._data[i]}`);
    }
  }
}

function setIdentity(m: Matrix): void {
  m._data.fill(0);
  const rows = m._shape[0];
  const cols = m._shape[1];
  for (let i = 0; i < Math.min(rows, cols); i++) {
    m._data[i * cols + i] = 1;
  }
}

function cloneState(state: any): any {
  return JSON.parse(JSON.stringify(state));
}

function topSlot(layer: MemoryBank, x: Matrix): number {
  layer.forward(x);
  const trace = layer.getDebugTrace();
  return trace[0].readSlots[0]?.slot ?? -1;
}

function argmaxCol(m: Matrix, col = 0): number {
  let best = 0;
  let bestValue = -Infinity;
  for (let i = 0; i < m._shape[0]; i++) {
    const value = m._data[i * m._shape[1] + col];
    if (value > bestValue) {
      bestValue = value;
      best = i;
    }
  }
  return best;
}

function makeDeterministicLayer(extra: Partial<ConstructorParameters<typeof MemoryBank>[0]> = {}): MemoryBank {
  const layer = new MemoryBank({
    units: 3,
    memorySlots: 3,
    memoryDim: 3,
    outputUnits: 3,
    mode: "read-project",
    similarity: "cosine",
    readTopK: 1,
    updateMode: "replace",
    writePolicy: "empty-first",
    writeThreshold: 0.5,
    writeEnabled: true,
    forceNeedGate: 1,
    valueMode: "identity",
    writeKeyMode: "shared-query",
    writeGateMode: "always",
    optimizer: "sgd",
    alpha: 0.1,
    ...extra,
  });
  layer.forward(mj.matrix([[0], [0], [0]]));
  setIdentity((layer as any).queryKernel);
  setIdentity((layer as any).outputKernel);
  (layer as any).outputBias._data.fill(0);
  layer.resetMemory();
  return layer;
}

export function runMemoryBankCorrectnessSuite(): void {
  // 1) backward updates differentiable read/output params but not runtime state
  {
    const layer = new MemoryBank({
      units: 3,
      memorySlots: 2,
      memoryDim: 3,
      outputUnits: 3,
      mode: "project",
      similarity: "cosine",
      readTopK: 2,
      updateMode: "replace",
      writeThreshold: 2,
      optimizer: "sgd",
      alpha: 0.1,
    });

    layer.setMemoryState({
      memoryKeys: [
        [1, 0],
        [0, 1],
        [0, 0],
      ],
      memoryValues: [
        [0, 1],
        [1, 0],
        [0, 0],
      ],
      memoryFilled: [1, 1],
      memoryUsage: [1, 1],
      memoryAge: [1, 2],
      memoryStep: 2,
      units: 3,
      memoryDim: 3,
      memorySlots: 2,
    });

    const q0 = (layer as any).queryKernel.clone();
    const n0 = (layer as any).needKernel.clone();
    const o0 = (layer as any).outputKernel.clone();
    const b0 = (layer as any).outputBias.clone();

    const out = layer.forward(mj.matrix([[1], [0], [0]]));
    const stateAfterForward = cloneState(layer.getMemoryState());
    const dx = layer.backward(mj.matrix([[]]), mj.matrix([[1], [0], [-1]]));

    assert(dx._shape[0] === 3 && dx._shape[1] === 1, "backward dx shape should match input");
    assert(JSON.stringify(stateAfterForward) === JSON.stringify(layer.getMemoryState()), "optimizer step must not mutate memory state");

    let queryChanged = false;
    let needChanged = false;
    let outputChanged = false;
    let biasChanged = false;
    for (let i = 0; i < q0._data.length; i++) if (Math.abs(q0._data[i] - (layer as any).queryKernel._data[i]) > 1e-12) queryChanged = true;
    for (let i = 0; i < n0._data.length; i++) if (Math.abs(n0._data[i] - (layer as any).needKernel._data[i]) > 1e-12) needChanged = true;
    for (let i = 0; i < o0._data.length; i++) if (Math.abs(o0._data[i] - (layer as any).outputKernel._data[i]) > 1e-12) outputChanged = true;
    for (let i = 0; i < b0._data.length; i++) if (Math.abs(b0._data[i] - (layer as any).outputBias._data[i]) > 1e-12) biasChanged = true;

    assert(queryChanged, "backward should update queryKernel");
    assert(needChanged, "backward should update needKernel");
    assert(outputChanged, "backward should update outputKernel");
    assert(biasChanged, "backward should update outputBias");
    assert(out._shape[0] === 3 && out._shape[1] === 1, "project mode output shape should be [outputUnits, cols]");
  }

  // 2) empty slot write is full replace even under gated-merge
  {
    const layer = new MemoryBank({
      units: 2,
      memorySlots: 1,
      memoryDim: 2,
      outputUnits: 2,
      mode: "read-project",
      updateMode: "gated-merge",
      writeThreshold: 0.2,
      valueMode: "identity",
      writeKeyMode: "shared-query",
      writeGateMode: "learned",
      forceNeedGate: 0.3,
    });
    layer.forward(mj.matrix([[0], [0]]));
    setIdentity((layer as any).queryKernel);
    setIdentity((layer as any).outputKernel);
    (layer as any).outputBias._data.fill(0);
    (layer as any).writeGateKernel._data.fill(0);
    (layer as any).writeGateKernel._data[0] = 1;
    layer.resetMemory();

    layer.forward(mj.matrix([[1], [0]]));
    const info = layer.getLastWriteInfo();
    const state = layer.getMemoryState();

    assert(info !== null && info.writeGate > 0.2 && info.writeGate < 1, "learned gate should be above threshold but below 1");
    assertClose(state.memoryValues[0][0], 1, 1e-6, "empty slot write should store full new value on dim0");
    assertClose(state.memoryValues[1][0], 0, 1e-6, "empty slot write should store full new value on dim1");
  }

  // 3) shared-key deterministic write/read
  {
    const layer = makeDeterministicLayer();
    const items = [
      { x: mj.matrix([[1], [0], [0]]), slot: 0, value: [1, 0, 0] },
      { x: mj.matrix([[0], [1], [0]]), slot: 1, value: [0, 1, 0] },
      { x: mj.matrix([[0], [0], [1]]), slot: 2, value: [0, 0, 1] },
    ];

    for (const item of items) layer.forward(item.x);
    layer.freezeWrites();

    for (const item of items) {
      const slot = topSlot(layer, item.x);
      assert(slot === item.slot, `shared key write/read should retrieve slot ${item.slot}, got ${slot}`);
      const out = layer.forward(item.x);
      const predicted = Array.from(out.getCol(0));
      for (let i = 0; i < item.value.length; i++) {
        assertClose(predicted[i], item.value[i], 1e-6, `shared key read should reconstruct stored value at dim ${i}`);
      }
    }
  }

  // 4) active writes provide causal gain over frozen writes
  {
    const keys = [
      mj.matrix([[1], [0], [0]]),
      mj.matrix([[0], [1], [0]]),
      mj.matrix([[0], [0], [1]]),
    ];
    const labels = [0, 1, 2];

    const evaluate = (freezeWrites: boolean): number => {
      const layer = makeDeterministicLayer();
      if (freezeWrites) layer.freezeWrites();
      for (const key of keys) layer.forward(key);
      layer.freezeWrites();

      let correct = 0;
      for (let i = 0; i < keys.length; i++) {
        const pred = argmaxCol(layer.forward(keys[i]));
        if (pred === labels[i]) correct += 1;
      }
      return correct / keys.length;
    };

    const activeAcc = evaluate(false);
    const frozenAcc = evaluate(true);

    assert(activeAcc >= 0.99, `active writes should solve episodic task, got ${activeAcc}`);
    assert(frozenAcc <= 0.34, `frozen writes should stay near random, got ${frozenAcc}`);
    assert(activeAcc - frozenAcc >= 0.5, `active writes should beat frozen writes by a meaningful margin, got ${activeAcc - frozenAcc}`);
  }

  // 5) output path really uses memory values
  {
    const layer = makeDeterministicLayer();
    layer.setMemoryState({
      memoryKeys: [
        [1, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
      ],
      memoryValues: [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
      ],
      memoryFilled: [1, 0, 0],
      memoryUsage: [1, 0, 0],
      memoryAge: [1, 0, 0],
      memoryStep: 1,
      units: 3,
      memoryDim: 3,
      memorySlots: 3,
    });
    layer.freezeWrites();

    const query = mj.matrix([[1], [0], [0]]);
    const outA = layer.forward(query);

    layer.setMemoryState({
      memoryKeys: [
        [1, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
      ],
      memoryValues: [
        [0, 0, 0],
        [1, 0, 0],
        [0, 0, 0],
      ],
      memoryFilled: [1, 0, 0],
      memoryUsage: [1, 0, 0],
      memoryAge: [1, 0, 0],
      memoryStep: 1,
      units: 3,
      memoryDim: 3,
      memorySlots: 3,
    });
    const outB = layer.forward(query);

    assert(argmaxCol(outA) !== argmaxCol(outB), "changing memoryValues should change prediction when mode='read-project'");
  }

  // 6) save/load roundtrip preserves config, state, and output
  {
    const layer = makeDeterministicLayer({
      memorySlots: 2,
      writeThreshold: 0.25,
      writeGateMode: "threshold",
    });
    layer.forward(mj.matrix([[1], [0], [0]]));
    layer.forward(mj.matrix([[0], [1], [0]]));
    layer.freezeWrites();

    const probe = mj.matrix([[1], [0], [0]]);
    const expected = layer.forward(probe);
    const saved = layer.save();

    const [loaded] = setLayers([saved]) as MemoryBank[];
    loaded.freezeWrites();
    const loadedState = loaded.getMemoryState();
    const actual = loaded.forward(probe);
    const savedState = layer.getMemoryState();
    const loadedSaved = loaded.save();

    assert(saved.config.forceNeedGate === loadedSaved.config.forceNeedGate, "save/load should preserve forceNeedGate");
    assert(saved.config.valueMode === loadedSaved.config.valueMode, "save/load should preserve valueMode");
    assert(saved.config.writeKeyMode === loadedSaved.config.writeKeyMode, "save/load should preserve writeKeyMode");
    assert(saved.config.writeGateMode === loadedSaved.config.writeGateMode, "save/load should preserve writeGateMode");
    assert(JSON.stringify(savedState) === JSON.stringify(loadedState), "save/load should preserve runtime memory state");
    assertMatrixClose(expected, actual, 1e-6, "save/load should preserve query output");
  }

  // 7) saveMemory/loadMemory remains functional with lazy fs access
  {
    const path = "/tmp/ml-v1-memory-bank-state.json";
    const layer = makeDeterministicLayer({ memorySlots: 2 });
    layer.forward(mj.matrix([[1], [0], [0]]));
    layer.saveMemory(path);

    const restored = makeDeterministicLayer({ memorySlots: 2 });
    restored.loadMemory(path);
    assert(JSON.stringify(layer.getMemoryState()) === JSON.stringify(restored.getMemoryState()), "saveMemory/loadMemory should roundtrip runtime state");

    try {
      unlinkSync(path);
    } catch {}
  }

  // 8) separate-project auxiliary key training improves alignment
  {
    const layer = new MemoryBank({
      units: 2,
      memorySlots: 2,
      memoryDim: 2,
      outputUnits: 2,
      mode: "read-project",
      similarity: "cosine",
      readTopK: 1,
      updateMode: "replace",
      writePolicy: "empty-first",
      writeThreshold: 0.5,
      writeEnabled: true,
      forceNeedGate: 1,
      valueMode: "identity",
      writeKeyMode: "separate-project",
      writeGateMode: "always",
      optimizer: "sgd",
      alpha: 0.3,
    });
    layer.forward(mj.matrix([[0], [0]]));
    setIdentity((layer as any).queryKernel);
    setIdentity((layer as any).outputKernel);
    (layer as any).outputBias._data.fill(0);
    (layer as any).writeKeyKernel._data.set([0, 1, 1, 0]);
    layer.resetMemory();

    const dataset = [
      { x: mj.matrix([[1], [0]]), expectedSlot: 0 },
      { x: mj.matrix([[0], [1]]), expectedSlot: 1 },
    ];

    const accuracy = (): number => {
      layer.resetMemory();
      for (const item of dataset) layer.forward(item.x);
      layer.freezeWrites();
      let correct = 0;
      for (const item of dataset) {
        if (topSlot(layer, item.x) === item.expectedSlot) correct += 1;
      }
      layer.unfreezeWrites();
      return correct / dataset.length;
    };

    const before = accuracy();
    for (let epoch = 0; epoch < 8; epoch++) {
      layer.resetMemory();
      for (const item of dataset) {
        layer.forward(item.x);
        const targetKey = layer.getQueryVectorForInput(item.x, true);
        const loss = layer.trainLastWriteKey(targetKey);
        assert(loss !== null, "trainLastWriteKey should be available in separate-project mode");
      }
    }
    const after = accuracy();

    assert(before <= 0.5, `separate-project alignment should start poor, got ${before}`);
    assert(after >= 0.99, `trainLastWriteKey should improve retrieval alignment, got ${after}`);
  }
}

if (require.main === module) {
  runMemoryBankCorrectnessSuite();
  console.log("[PASS] memoryBank.test: all tests passed");
}
