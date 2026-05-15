import { memoryBankSimilarityScoresNative, memoryBankUpdateNative, isNativeAvailable } from "@oxide-js/core";

function assertClose(a: number, b: number, tol: number = 1e-5) {
  if (Math.abs(a - b) > tol) {
    throw new Error(`Expected ${b}, got ${a}`);
  }
}

function assertArrayClose(a: Float32Array, b: Float32Array, tol: number = 1e-5) {
  if (a.length !== b.length) throw new Error(`Length mismatch: ${a.length} vs ${b.length}`);
  for (let i = 0; i < a.length; i++) {
    assertClose(a[i], b[i], tol);
  }
}

function l2Norm(v: Float32Array): number {
  let s = 0;
  for (let i = 0; i < v.length; i++) s += v[i] * v[i];
  return Math.sqrt(s);
}

function vectorDot(a: Float32Array, b: Float32Array): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

export function runMemoryBankNativeTest() {
  if (!isNativeAvailable()) {
    console.log("Native not available, skipping MemoryBank native test");
    return;
  }

  const units = 4;
  const slots = 3;

  // Test similarity
  const query = new Float32Array([0.1, 0.2, 0.3, 0.4]);
  const keys = new Float32Array([
    // slot 0
    1, 0, 0, 0,
    // slot 1
    0, 1, 0, 0,
    // slot 2
    0.1, 0.2, 0.3, 0.4
  ]);

  // Keys in JS are stored as [units, memorySlots] where index is i * memorySlots + slot
  const jsKeys = new Float32Array(units * slots);
  for (let s = 0; s < slots; s++) {
    for (let i = 0; i < units; i++) {
      jsKeys[i * slots + s] = keys[s * units + i];
    }
  }

  // 1. Cosine similarity
  const scoresCosine = new Float32Array(slots);
  memoryBankSimilarityScoresNative(query, jsKeys, units, slots, "cosine", scoresCosine);

  const expectedCosine = new Float32Array(slots);
  for (let s = 0; s < slots; s++) {
    const k = new Float32Array(units);
    for (let i = 0; i < units; i++) k[i] = jsKeys[i * slots + s];
    const nq = l2Norm(query);
    const nk = l2Norm(k);
    expectedCosine[s] = (nq > 1e-12 && nk > 1e-12) ? vectorDot(query, k) / (nq * nk) : 0;
  }

  assertArrayClose(scoresCosine, expectedCosine, 1e-5);
  console.log("Cosine similarity passed");

  // 2. Dot similarity
  const scoresDot = new Float32Array(slots);
  memoryBankSimilarityScoresNative(query, jsKeys, units, slots, "dot", scoresDot);

  const expectedDot = new Float32Array(slots);
  for (let s = 0; s < slots; s++) {
    const k = new Float32Array(units);
    for (let i = 0; i < units; i++) k[i] = jsKeys[i * slots + s];
    expectedDot[s] = vectorDot(query, k) / Math.sqrt(units);
  }

  assertArrayClose(scoresDot, expectedDot, 1e-5);
  console.log("Dot similarity passed");

  // 3. Update
  const keysToUpdate = new Float32Array(jsKeys);
  const valuesToUpdate = new Float32Array(jsKeys); // dummy
  const newKey = new Float32Array([0.5, 0.5, 0.5, 0.5]);
  const newValue = new Float32Array([0.9, 0.9, 0.9, 0.9]);
  const targetSlot = 1;
  const gate = new Float32Array([0.5, 0.5, 0.5, 0.5]); 
  
  memoryBankUpdateNative(keysToUpdate, valuesToUpdate, newKey, newValue, targetSlot, gate, units, slots);

  const expectedKeys = new Float32Array(jsKeys);
  const expectedValues = new Float32Array(jsKeys);
  for (let i = 0; i < units; i++) {
    const idx = i * slots + targetSlot;
    const gateVal = gate[i];
    expectedKeys[idx] = (1 - gateVal) * expectedKeys[idx] + gateVal * newKey[i];
    expectedValues[idx] = (1 - gateVal) * expectedValues[idx] + gateVal * newValue[i];
  }

  assertArrayClose(keysToUpdate, expectedKeys, 1e-5);
  assertArrayClose(valuesToUpdate, expectedValues, 1e-5);
  console.log("Update passed");
}

runMemoryBankNativeTest();
