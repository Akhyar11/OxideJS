import { Optimzier, OptimzierType, StatusLayer, matrix2d } from "../@types/type";
import mj from "../math";
import Matrix from "../matrix";
import { dotProductNative, isNativeAvailable } from "../math/rust_backend";
import setOptimizer from "../utils/setOptimizer";

type Vec = Float32Array<ArrayBufferLike>;

export type MemoryBankMode = "project" | "concat";
export type MemorySimilarity = "cosine" | "dot";
export type MemoryPersistence = "session" | "manual";

export interface MemoryBankDebugReadSlot {
  slot: number;
  score: number;
  attn: number;
}

export interface MemoryBankDebugTrace {
  column: number;
  readSlots: MemoryBankDebugReadSlot[];
  need: number;
  readNorm: number;
  contextNorm: number;
  writeCommitted: boolean;
  writeSlot: number;
  memoryFilled: number[];
  memoryUsage: number[];
  memoryAge: number[];
}

export interface MemoryBankConfig {
  units?: number;
  memorySlots: number;
  outputUnits?: number;
  mode?: MemoryBankMode;
  similarity?: MemorySimilarity;
  readTopK?: number;
  persistence?: MemoryPersistence;
  resetOnInit?: boolean;
  writeEnabled?: boolean;
  alpha?: number;
  optimizer?: Optimzier;
  clipGradient?: number | boolean;
  status?: StatusLayer;
}

export interface MemoryBankState {
  memoryKeys: number[][];
  memoryValues: number[][];
  memoryFilled: number[];
  memoryUsage: number[];
  memoryAge: number[];
  memoryStep: number;
  units: number;
  memoryDim: number;
  memorySlots: number;
}

export interface MemoryBankSequenceConfig {
  maxHistorySteps?: number;
}

interface ReadSlotCache {
  slot: number;
  attn: number;
  score: number;
  key: Vec;
  value: Vec;
}

interface ForwardCacheItem {
  xCol: Vec;
  qRaw: Vec;
  q: Vec;
  read: Vec;
  need: number;
  needInput: Vec | null;
  context: Vec;
  combined: Vec;
  readSlots: ReadSlotCache[];
  writeCommitted: boolean;
  writeSlot: number;
  newKeyRaw: Vec;
  newKey: Vec;
  newValue: Vec;
  writeSlotWasFilled: boolean;
}

type MemoryBankSaveData = {
  name: string;
  status: StatusLayer;
  config: {
    mode: MemoryBankMode;
    similarity: MemorySimilarity;
    readTopK: number;
    persistence: MemoryPersistence;
    resetOnInit: boolean;
    writeEnabled: boolean;
  };
  dimensions: {
    units: number;
    memorySlots: number;
    outputUnits: number;
  };
  trainableParams: {
    queryKernel: matrix2d;
    needKernel?: matrix2d;
    outputKernel?: matrix2d;
    outputBias?: matrix2d;
  };
  memoryState: MemoryBankState;
  optimizerState: {
    alpha: number;
    optimizer: Optimzier;
    clipGradient: number | boolean;
    status: StatusLayer;
  };
  units: number;
  memorySlots: number;
  memoryDim: number;
  outputUnits: number;
  mode: MemoryBankMode;
  similarity: MemorySimilarity;
  readTopK: number;
  persistence: MemoryPersistence;
  resetOnInit: boolean;
  writeEnabled: boolean;
  alpha: number;
  optimizer: Optimzier;
  clipGradient: number | boolean;
  queryKernel: matrix2d;
  needKernel?: matrix2d;
  outputKernel?: matrix2d;
  outputBias?: matrix2d;
  memoryKeys: number[][];
  memoryValues: number[][];
  memoryFilled: number[];
  memoryUsage: number[];
  memoryAge: number[];
  memoryStep: number;
};

export default class MemoryBank {
  name = "memory bank layer";

  units!: number;
  memoryDim!: number;
  memorySlots: number;
  outputUnits!: number;

  mode: MemoryBankMode;
  similarity: MemorySimilarity;
  readTopK: number;
  persistence: MemoryPersistence;
  resetOnInit: boolean;
  writeEnabled: boolean;
  alpha: number;
  optimizerName: Optimzier;
  clipGradient: number | boolean;
  status: StatusLayer;

  queryKernel!: Matrix;
  needKernel?: Matrix;
  outputKernel?: Matrix;
  outputBias?: Matrix;

  private optimizerQuery!: OptimzierType;
  private optimizerNeed?: OptimzierType;
  private optimizerOutput?: OptimzierType;
  private optimizerOutputBias?: OptimzierType;

  memoryKeys!: Matrix;
  memoryValues!: Matrix;
  memoryFilled!: Uint8Array;
  memoryUsage!: Float32Array;
  memoryAge!: Float32Array;
  memoryStep = 0;

  inputShape: [number, number] = [0, 1];
  outputShape: [number, number] = [0, 1];
  params = 0;

  private initialized = false;
  private writeFrozen = false;
  private cache: ForwardCacheItem[] = [];
  private debugTrace: MemoryBankDebugTrace[] = [];
  private configuredOutputUnits?: number;
  private sequenceActive = false;
  private sequenceMaxHistorySteps: number | null = null;
  private sequenceHistory: ForwardCacheItem[] = [];

  private lastWriteInfo: {
    committed: boolean;
    slot: number;
    newKeyRaw: Vec;
    newKey: Vec;
    newValue: Vec;
  } | null = null;

  constructor(cfg: MemoryBankConfig) {
    this.assertPositiveInt(cfg.memorySlots, "memorySlots");
    if (cfg.units !== undefined) this.assertPositiveInt(cfg.units, "units");
    if (cfg.outputUnits !== undefined) this.assertPositiveInt(cfg.outputUnits, "outputUnits");

    this.memorySlots = cfg.memorySlots;
    this.mode = cfg.mode ?? "project";
    this.similarity = cfg.similarity ?? "cosine";
    this.readTopK = cfg.readTopK ?? Math.min(4, this.memorySlots);
    this.persistence = cfg.persistence ?? "session";
    this.resetOnInit = cfg.resetOnInit ?? true;
    this.writeEnabled = cfg.writeEnabled ?? true;
    this.alpha = cfg.alpha ?? 0.01;
    this.optimizerName = cfg.optimizer ?? "adam";
    this.clipGradient = cfg.clipGradient ?? 5.0;
    this.status = cfg.status ?? "train";
    this.configuredOutputUnits = cfg.outputUnits;

    if (this.mode !== "project" && this.mode !== "concat") {
      throw new Error(`MemoryBank: unsupported mode '${this.mode}', supported modes are 'project' and 'concat'`);
    }
    if (this.readTopK <= 0 || this.readTopK > this.memorySlots) {
      throw new Error("MemoryBank: readTopK must be in [1, memorySlots]");
    }

    if (cfg.units !== undefined) {
      this.init(cfg.units, cfg.outputUnits ?? cfg.units);
    }
  }

  private assertPositiveInt(v: number, name: string): void {
    if (!Number.isInteger(v) || v <= 0) throw new Error(`MemoryBank: ${name} must be positive integer`);
  }

  private sigmoid(x: number): number {
    return 1 / (1 + Math.exp(-x));
  }

  private vectorDot(a: Vec, b: Vec): number {
    let s = 0;
    for (let i = 0; i < a.length; i++) s += a[i] * b[i];
    return s;
  }

  private l2Norm(v: Vec): number {
    let s = 0;
    for (let i = 0; i < v.length; i++) s += v[i] * v[i];
    return Math.sqrt(s);
  }

  private normalizeSafe(v: Vec): Vec {
    const out = new Float32Array(v.length);
    const n = this.l2Norm(v);
    if (!Number.isFinite(n) || n <= 1e-12) return out;
    const inv = 1 / n;
    for (let i = 0; i < v.length; i++) out[i] = v[i] * inv;
    return out;
  }

  private normalizeBackward(raw: Vec, gradOut: Vec): Vec {
    const out = new Float32Array(raw.length);
    const n = this.l2Norm(raw);
    if (!Number.isFinite(n) || n <= 1e-12) return out;
    const dot = this.vectorDot(raw, gradOut);
    const invN = 1 / n;
    const invN3 = invN * invN * invN;
    for (let i = 0; i < raw.length; i++) out[i] = gradOut[i] * invN - raw[i] * dot * invN3;
    return out;
  }

  private matVecMul(weight: Matrix, x: Vec): Vec {
    const [rows, cols] = weight._shape;
    if (cols !== x.length) {
      throw new Error(`MemoryBank: matVecMul shape mismatch [${rows},${cols}] x [${x.length}]`);
    }
    if (isNativeAvailable()) {
      const out = new Float32Array(rows);
      dotProductNative(weight._data, rows, cols, x, cols, 1, false, false, out);
      return out;
    }
    const out = new Float32Array(rows);
    for (let r = 0; r < rows; r++) {
      let sum = 0;
      const offset = r * cols;
      for (let c = 0; c < cols; c++) sum += weight._data[offset + c] * x[c];
      out[r] = sum;
    }
    return out;
  }

  private matTVecMul(weight: Matrix, x: Vec): Vec {
    const [rows, cols] = weight._shape;
    if (rows !== x.length) {
      throw new Error(`MemoryBank: matTVecMul shape mismatch [${rows},${cols}]^T x [${x.length}]`);
    }
    if (isNativeAvailable()) {
      const out = new Float32Array(cols);
      dotProductNative(weight._data, rows, cols, x, rows, 1, true, false, out);
      return out;
    }
    const out = new Float32Array(cols);
    for (let c = 0; c < cols; c++) {
      let sum = 0;
      for (let r = 0; r < rows; r++) sum += weight._data[r * cols + c] * x[r];
      out[c] = sum;
    }
    return out;
  }

  private addOuter(grad: Matrix, a: Vec, b: Vec, scale = 1): void {
    const cols = grad._shape[1];
    for (let i = 0; i < a.length; i++) {
      const ai = a[i] * scale;
      const offset = i * cols;
      for (let j = 0; j < b.length; j++) grad._data[offset + j] += ai * b[j];
    }
  }

  private cosineGradWrtQ(q: Vec, key: Vec): Vec {
    const out = new Float32Array(q.length);
    const nq = this.l2Norm(q);
    const nk = this.l2Norm(key);
    if (!Number.isFinite(nq) || !Number.isFinite(nk) || nq <= 1e-12 || nk <= 1e-12) return out;
    const dot = this.vectorDot(q, key);
    const invNqNk = 1 / (nq * nk);
    const coeffQ = dot / (nq * nq * nq * nk);
    for (let i = 0; i < q.length; i++) out[i] = key[i] * invNqNk - q[i] * coeffQ;
    return out;
  }

  private cosineGradWrtKey(q: Vec, key: Vec): Vec {
    const out = new Float32Array(key.length);
    const nq = this.l2Norm(q);
    const nk = this.l2Norm(key);
    if (!Number.isFinite(nq) || !Number.isFinite(nk) || nq <= 1e-12 || nk <= 1e-12) return out;
    const dot = this.vectorDot(q, key);
    const invNqNk = 1 / (nq * nk);
    const coeffK = dot / (nq * nk * nk * nk);
    for (let i = 0; i < key.length; i++) out[i] = q[i] * invNqNk - key[i] * coeffK;
    return out;
  }

  private addStateGrad(flat: Float32Array, slot: number, grad: Vec, scale = 1): void {
    for (let i = 0; i < this.units; i++) flat[i * this.memorySlots + slot] += grad[i] * scale;
  }

  private getStateGrad(flat: Float32Array, slot: number): Vec {
    const out = new Float32Array(this.units);
    for (let i = 0; i < this.units; i++) out[i] = flat[i * this.memorySlots + slot];
    return out;
  }

  private softmax(scores: number[]): number[] {
    if (scores.length === 0) return [];
    let maxScore = -Infinity;
    for (const score of scores) if (score > maxScore) maxScore = score;
    let sum = 0;
    const exps = new Array(scores.length);
    for (let i = 0; i < scores.length; i++) {
      const e = Math.exp(scores[i] - maxScore);
      exps[i] = e;
      sum += e;
    }
    if (!Number.isFinite(sum) || sum <= 0) return new Array(scores.length).fill(1 / scores.length);
    for (let i = 0; i < exps.length; i++) exps[i] /= sum;
    return exps;
  }

  private similarityScore(q: Vec, key: Vec): number {
    if (this.similarity === "dot") return this.vectorDot(q, key) / Math.sqrt(this.units);
    return this.vectorDot(q, key);
  }

  private ensureInitializedFromInput(rows: number): void {
    if (this.initialized) return;
    this.init(rows, this.configuredOutputUnits ?? rows);
  }

  private init(units: number, outputUnits: number): void {
    if (this.initialized) return;

    this.units = units;
    this.memoryDim = units;
    this.outputUnits = outputUnits;

    this.queryKernel = mj.xavier([this.units, this.units]);
    this.optimizerQuery = setOptimizer(this.optimizerName, this.queryKernel._shape, 1e-5);

    if (this.mode === "project") {
      this.needKernel = mj.xavier([1, this.units + this.units]);
      this.outputKernel = mj.xavier([this.outputUnits, this.units + this.units]);
      this.outputBias = mj.zeros([this.outputUnits, 1]);
      this.optimizerNeed = setOptimizer(this.optimizerName, this.needKernel._shape, 1e-5);
      this.optimizerOutput = setOptimizer(this.optimizerName, this.outputKernel._shape, 1e-5);
      this.optimizerOutputBias = setOptimizer(this.optimizerName, this.outputBias._shape, 1e-5);
    }

    this.memoryKeys = mj.zeros([this.units, this.memorySlots]);
    this.memoryValues = mj.zeros([this.units, this.memorySlots]);
    this.memoryFilled = new Uint8Array(this.memorySlots);
    this.memoryUsage = new Float32Array(this.memorySlots);
    this.memoryAge = new Float32Array(this.memorySlots);
    this.memoryStep = 0;

    this.inputShape = [this.units, 1];
    this.outputShape = this.mode === "concat" ? [this.units + this.units, 1] : [this.outputUnits, 1];

    this.params =
      this.queryKernel._data.length +
      (this.needKernel ? this.needKernel._data.length : 0) +
      (this.outputKernel ? this.outputKernel._data.length : 0) +
      (this.outputBias ? this.outputBias._data.length : 0);

    this.initialized = true;
    if (this.resetOnInit) this.resetMemory();
  }

  private getMemoryColumn(m: Matrix, col: number): Vec {
    const out = new Float32Array(this.units);
    for (let i = 0; i < this.units; i++) out[i] = m._data[i * this.memorySlots + col];
    return out;
  }

  private setMemoryColumn(m: Matrix, col: number, v: Vec): void {
    for (let i = 0; i < this.units; i++) m._data[i * this.memorySlots + col] = v[i];
  }

  private pickWriteSlot(newKey: Vec): number {
    for (let slot = 0; slot < this.memorySlots; slot++) {
      if (!this.memoryFilled[slot]) return slot;
    }

    let best = 0;
    let minScore = Infinity;
    for (let slot = 0; slot < this.memorySlots; slot++) {
      const key = this.getMemoryColumn(this.memoryKeys, slot);
      const score = this.similarityScore(newKey, key);
      if (score < minScore) {
        minScore = score;
        best = slot;
      }
    }
    return best;
  }

  private writeSlot(slot: number, newKey: Vec, newValue: Vec): void {
    this.setMemoryColumn(this.memoryKeys, slot, newKey);
    this.setMemoryColumn(this.memoryValues, slot, newValue);
    this.memoryFilled[slot] = 1;
    this.memoryUsage[slot] += 1;
    this.memoryAge[slot] = this.memoryStep;
  }

  private maybeClip(...grads: (Matrix | undefined)[]): void {
    if (this.clipGradient === false) return;
    const limit = typeof this.clipGradient === "number" ? this.clipGradient : 5.0;
    for (const grad of grads) if (grad) mj.clipGradients(grad, limit);
  }

  private trimSequenceHistoryIfNeeded(): void {
    if (this.sequenceMaxHistorySteps === null) return;
    if (this.sequenceHistory.length <= this.sequenceMaxHistorySteps) return;
    this.sequenceHistory.splice(0, this.sequenceHistory.length - this.sequenceMaxHistorySteps);
  }

  private resetInitializationState(): void {
    this.initialized = false;
    this.writeFrozen = false;
    this.sequenceActive = false;
    this.sequenceMaxHistorySteps = null;
    this.queryKernel = undefined as unknown as Matrix;
    this.needKernel = undefined;
    this.outputKernel = undefined;
    this.outputBias = undefined;
    this.optimizerQuery = undefined as unknown as OptimzierType;
    this.optimizerNeed = undefined;
    this.optimizerOutput = undefined;
    this.optimizerOutputBias = undefined;
    this.memoryKeys = undefined as unknown as Matrix;
    this.memoryValues = undefined as unknown as Matrix;
    this.memoryFilled = new Uint8Array(0);
    this.memoryUsage = new Float32Array(0);
    this.memoryAge = new Float32Array(0);
    this.memoryStep = 0;
    this.inputShape = [0, 1];
    this.outputShape = [0, 1];
    this.params = 0;
    this.cache = [];
    this.debugTrace = [];
    this.lastWriteInfo = null;
    this.sequenceHistory = [];
  }

  getDebugTrace(): MemoryBankDebugTrace[] {
    return JSON.parse(JSON.stringify(this.debugTrace));
  }

  clearDebugTrace(): void {
    this.debugTrace = [];
  }

  getLastWriteInfo(): { committed: boolean; slot: number; newKey: number[]; newValue: number[] } | null {
    if (!this.lastWriteInfo || !this.lastWriteInfo.committed) return null;
    return {
      committed: true,
      slot: this.lastWriteInfo.slot,
      newKey: Array.from(this.lastWriteInfo.newKey),
      newValue: Array.from(this.lastWriteInfo.newValue),
    };
  }

  getLastReadValueMatrix(): Matrix | null {
    if (this.cache.length === 0) return null;
    const item = this.cache[this.cache.length - 1];
    const out = mj.zeros([this.units, 1]);
    out.setCol(0, item.read);
    return out;
  }

  getLastContextMatrix(): Matrix | null {
    if (this.cache.length === 0) return null;
    const item = this.cache[this.cache.length - 1];
    const out = mj.zeros([this.units, 1]);
    out.setCol(0, item.context);
    return out;
  }

  getLastCombinedMatrix(): Matrix | null {
    if (this.cache.length === 0) return null;
    const item = this.cache[this.cache.length - 1];
    const out = mj.zeros([this.units + this.units, 1]);
    out.setCol(0, item.combined);
    return out;
  }

  getLastWriteValueMatrix(): Matrix | null {
    if (!this.lastWriteInfo || !this.lastWriteInfo.committed) return null;
    const out = mj.zeros([this.units, 1]);
    out.setCol(0, this.lastWriteInfo.newValue);
    return out;
  }

  getQueryVectorForInput(x: Matrix, normalize = true): Matrix {
    if (!this.initialized) this.ensureInitializedFromInput(x._shape[0]);
    if (x._shape[0] !== this.units || x._shape[1] !== 1) {
      throw new Error(`MemoryBank.getQueryVectorForInput: expected [${this.units}, 1], got [${x._shape[0]}, ${x._shape[1]}]`);
    }
    const qRaw = this.matVecMul(this.queryKernel, x.getCol(0));
    const q = normalize && this.similarity === "cosine" ? this.normalizeSafe(qRaw) : qRaw;
    const out = mj.zeros([this.units, 1]);
    out.setCol(0, q);
    return out;
  }

  writeMemoryForDebug(keyVector: number[], valueVector: number[], slot?: number): void {
    if (!this.initialized) throw new Error("MemoryBank.writeMemoryForDebug: layer not initialized");
    if (keyVector.length !== this.units || valueVector.length !== this.units) {
      throw new Error(`MemoryBank.writeMemoryForDebug: vectors must be length ${this.units}`);
    }
    let targetSlot = slot;
    if (targetSlot === undefined) targetSlot = this.pickWriteSlot(Float32Array.from(keyVector));
    if (targetSlot < 0 || targetSlot >= this.memorySlots) {
      throw new Error(`MemoryBank.writeMemoryForDebug: slot ${targetSlot} out of range`);
    }
    const key = Float32Array.from(keyVector);
    const normalizedKey = this.similarity === "cosine" ? this.normalizeSafe(key) : key;
    this.writeSlot(targetSlot, normalizedKey, Float32Array.from(valueVector));
  }

  forward(x: Matrix): Matrix {
    const [rows, cols] = x._shape;
    this.ensureInitializedFromInput(rows);
    if (rows !== this.units) throw new Error(`MemoryBank: input rows ${rows} does not match units ${this.units}`);

    const out = this.mode === "concat" ? mj.zeros([this.units + this.units, cols]) : mj.zeros([this.outputUnits, cols]);
    this.cache = [];
    this.debugTrace = [];
    this.lastWriteInfo = null;

    for (let c = 0; c < cols; c++) {
      const xCol = x.getCol(c);
      const qRaw = this.matVecMul(this.queryKernel, xCol);
      const q = this.similarity === "cosine" ? this.normalizeSafe(qRaw) : qRaw;

      const scored: Array<{ slot: number; score: number; key: Vec }> = [];
      for (let slot = 0; slot < this.memorySlots; slot++) {
        if (!this.memoryFilled[slot]) continue;
        const key = this.getMemoryColumn(this.memoryKeys, slot);
        scored.push({ slot, score: this.similarityScore(q, key), key });
      }
      scored.sort((a, b) => b.score - a.score);

      const top = scored.slice(0, Math.min(this.readTopK, scored.length));
      const attn = this.softmax(top.map((item) => item.score));
      const read = new Float32Array(this.units);
      const readSlots: ReadSlotCache[] = [];
      for (let i = 0; i < top.length; i++) {
        const value = this.getMemoryColumn(this.memoryValues, top[i].slot);
        for (let d = 0; d < this.units; d++) read[d] += attn[i] * value[d];
        readSlots.push({ slot: top[i].slot, score: top[i].score, attn: attn[i], key: top[i].key, value });
      }

      let need = 1;
      let needInput: Vec | null = null;
      const context = new Float32Array(this.units);
      if (this.mode === "project") {
        needInput = new Float32Array(this.units + this.units);
        needInput.set(xCol, 0);
        needInput.set(read, this.units);
        need = this.sigmoid(this.matVecMul(this.needKernel!, needInput)[0]);
        for (let i = 0; i < this.units; i++) context[i] = need * read[i];
      } else {
        context.set(read);
      }

      const combined = new Float32Array(this.units + this.units);
      combined.set(xCol, 0);
      combined.set(context, this.units);

      if (this.mode === "project") {
        const projected = this.matVecMul(this.outputKernel!, combined);
        for (let r = 0; r < this.outputUnits; r++) out._data[r * cols + c] = projected[r] + this.outputBias!._data[r];
      } else {
        for (let r = 0; r < this.units + this.units; r++) out._data[r * cols + c] = combined[r];
      }

      let writeCommitted = false;
      let writeSlot = -1;
      let newKeyRaw: Vec = new Float32Array(this.units);
      let newKey: Vec = new Float32Array(this.units);
      let newValue: Vec = new Float32Array(this.units);
      let writeSlotWasFilled = false;

      if (this.writeEnabled && !this.writeFrozen) {
        newKeyRaw = this.matVecMul(this.queryKernel, xCol);
        newKey = this.similarity === "cosine" ? this.normalizeSafe(newKeyRaw) : newKeyRaw;
        newValue = new Float32Array(xCol);
        writeSlot = this.pickWriteSlot(newKey);
        writeSlotWasFilled = this.memoryFilled[writeSlot] === 1;
        this.writeSlot(writeSlot, newKey, newValue);
        writeCommitted = true;
        this.lastWriteInfo = {
          committed: true,
          slot: writeSlot,
          newKeyRaw: new Float32Array(newKeyRaw),
          newKey: new Float32Array(newKey),
          newValue: new Float32Array(newValue),
        };
      }

      this.memoryStep += 1;
      this.debugTrace.push({
        column: c,
        readSlots: readSlots.map((slot) => ({ slot: slot.slot, score: slot.score, attn: slot.attn })),
        need,
        readNorm: this.l2Norm(read),
        contextNorm: this.l2Norm(context),
        writeCommitted,
        writeSlot,
        memoryFilled: Array.from(this.memoryFilled),
        memoryUsage: Array.from(this.memoryUsage),
        memoryAge: Array.from(this.memoryAge),
      });

      this.cache.push({
        xCol,
        qRaw,
        q,
        read,
        need,
        needInput,
        context,
        combined,
        readSlots,
        writeCommitted,
        writeSlot,
        newKeyRaw,
        newKey,
        newValue,
        writeSlotWasFilled,
      });
    }

    if (this.sequenceActive && this.cache.length > 0) {
      this.sequenceHistory.push(...this.cache);
      this.trimSequenceHistoryIfNeeded();
    }

    return out;
  }

  private backwardThroughCaches(caches: ForwardCacheItem[], err: Matrix, callerName: string): Matrix {
    if (!this.initialized) throw new Error(`${callerName} called before forward initialization`);
    if (caches.length !== err._shape[1]) {
      throw new Error(`${callerName}: cache length mismatch, expected ${caches.length} columns, got ${err._shape[1]}`);
    }

    const expectedRows = this.mode === "concat" ? this.units + this.units : this.outputUnits;
    if (err._shape[0] !== expectedRows) {
      throw new Error(`${callerName}: err rows must be ${expectedRows}, got ${err._shape[0]}`);
    }

    const dx = mj.zeros([this.units, err._shape[1]]);
    const gQuery = mj.zeros(this.queryKernel._shape);
    const gNeed = this.needKernel ? mj.zeros(this.needKernel._shape) : undefined;
    const gOutput = this.outputKernel ? mj.zeros(this.outputKernel._shape) : undefined;
    const gOutputBias = this.outputBias ? mj.zeros(this.outputBias._shape) : undefined;

    let futureKeyStateGrad = new Float32Array(this.units * this.memorySlots);
    let futureValueStateGrad = new Float32Array(this.units * this.memorySlots);

    for (let c = err._shape[1] - 1; c >= 0; c--) {
      const cache = caches[c];
      const dxDirect = new Float32Array(this.units);
      const dRead = new Float32Array(this.units);
      const dContext = new Float32Array(this.units);
      const dPreKeys = new Float32Array(futureKeyStateGrad);
      const dPreValues = new Float32Array(futureValueStateGrad);

      if (this.mode === "project") {
        const e = err.getCol(c);
        this.addOuter(gOutput!, e, cache.combined);
        for (let i = 0; i < e.length; i++) gOutputBias!._data[i] += e[i];
        const dCombined = this.matTVecMul(this.outputKernel!, e);
        for (let i = 0; i < this.units; i++) dxDirect[i] += dCombined[i];
        for (let i = 0; i < this.units; i++) dContext[i] += dCombined[this.units + i];
      } else {
        const e = err.getCol(c);
        for (let i = 0; i < this.units; i++) dxDirect[i] += e[i];
        for (let i = 0; i < this.units; i++) dRead[i] += e[this.units + i];
      }

      if (cache.writeCommitted) {
        const slot = cache.writeSlot;
        const dPostKeySlot = this.getStateGrad(futureKeyStateGrad, slot);
        const dPostValueSlot = this.getStateGrad(futureValueStateGrad, slot);
        for (let i = 0; i < this.units; i++) {
          dPreKeys[i * this.memorySlots + slot] = 0;
          dPreValues[i * this.memorySlots + slot] = 0;
          dxDirect[i] += dPostValueSlot[i];
        }

        const dNewKeyRaw = this.similarity === "cosine"
          ? this.normalizeBackward(cache.newKeyRaw, dPostKeySlot)
          : dPostKeySlot;
        this.addOuter(gQuery, dNewKeyRaw, cache.xCol);
        const dxWriteKey = this.matTVecMul(this.queryKernel, dNewKeyRaw);
        for (let i = 0; i < this.units; i++) dxDirect[i] += dxWriteKey[i];
      }

      if (this.mode === "project") {
        let dNeed = 0;
        for (let i = 0; i < this.units; i++) {
          dNeed += dContext[i] * cache.read[i];
          dRead[i] += dContext[i] * cache.need;
        }
        const dNeedPre = dNeed * cache.need * (1 - cache.need);
        this.addOuter(gNeed!, new Float32Array([dNeedPre]), cache.needInput!);
        const dNeedInput = this.matTVecMul(this.needKernel!, new Float32Array([dNeedPre]));
        for (let i = 0; i < this.units; i++) dxDirect[i] += dNeedInput[i];
        for (let i = 0; i < this.units; i++) dRead[i] += dNeedInput[this.units + i];
      }

      const dQuery = new Float32Array(this.units);
      if (cache.readSlots.length > 0) {
        const dAttn = new Array<number>(cache.readSlots.length).fill(0);
        for (let i = 0; i < cache.readSlots.length; i++) dAttn[i] = this.vectorDot(dRead, cache.readSlots[i].value);

        let weighted = 0;
        for (let i = 0; i < cache.readSlots.length; i++) weighted += cache.readSlots[i].attn * dAttn[i];

        for (let i = 0; i < cache.readSlots.length; i++) {
          this.addStateGrad(dPreValues, cache.readSlots[i].slot, dRead, cache.readSlots[i].attn);
          const dScore = cache.readSlots[i].attn * (dAttn[i] - weighted);
          if (this.similarity === "dot") {
            const scale = dScore / Math.sqrt(this.units);
            const dKey = new Float32Array(this.units);
            for (let d = 0; d < this.units; d++) {
              dQuery[d] += scale * cache.readSlots[i].key[d];
              dKey[d] = scale * cache.q[d];
            }
            this.addStateGrad(dPreKeys, cache.readSlots[i].slot, dKey);
          } else {
            const gradCosQ = this.cosineGradWrtQ(cache.q, cache.readSlots[i].key);
            const gradCosKey = this.cosineGradWrtKey(cache.q, cache.readSlots[i].key);
            for (let d = 0; d < this.units; d++) {
              dQuery[d] += dScore * gradCosQ[d];
              gradCosKey[d] *= dScore;
            }
            this.addStateGrad(dPreKeys, cache.readSlots[i].slot, gradCosKey);
          }
        }
      }

      const gradQueryRaw = this.similarity === "cosine" ? this.normalizeBackward(cache.qRaw, dQuery) : dQuery;
      this.addOuter(gQuery, gradQueryRaw, cache.xCol);
      const dxQuery = this.matTVecMul(this.queryKernel, gradQueryRaw);

      for (let i = 0; i < this.units; i++) dx._data[i * err._shape[1] + c] = dxDirect[i] + dxQuery[i];
      futureKeyStateGrad = dPreKeys;
      futureValueStateGrad = dPreValues;
    }

    this.maybeClip(gQuery, gNeed, gOutput, gOutputBias);
    this.queryKernel.subInPlace(this.optimizerQuery.calculate(gQuery, this.alpha));
    if (gNeed && this.needKernel && this.optimizerNeed) {
      this.needKernel.subInPlace(this.optimizerNeed.calculate(gNeed, this.alpha));
    }
    if (gOutput && this.outputKernel && this.optimizerOutput) {
      this.outputKernel.subInPlace(this.optimizerOutput.calculate(gOutput, this.alpha));
    }
    if (gOutputBias && this.outputBias && this.optimizerOutputBias) {
      this.outputBias.subInPlace(this.optimizerOutputBias.calculate(gOutputBias, this.alpha));
    }

    return dx;
  }

  backward(_y: Matrix, err: Matrix): Matrix {
    return this.backwardThroughCaches(this.cache, err, "MemoryBank.backward");
  }

  backwardSequence(err: Matrix): Matrix {
    if (!this.sequenceActive) {
      throw new Error("MemoryBank.backwardSequence: sequence mode is not active, call beginSequence() first");
    }
    return this.backwardThroughCaches(this.sequenceHistory, err, "MemoryBank.backwardSequence");
  }

  beginSequence(cfg: MemoryBankSequenceConfig = {}): this {
    if (cfg.maxHistorySteps !== undefined) {
      if (!Number.isInteger(cfg.maxHistorySteps) || cfg.maxHistorySteps <= 0) {
        throw new Error(`MemoryBank.beginSequence: maxHistorySteps must be positive integer, got ${cfg.maxHistorySteps}`);
      }
      this.sequenceMaxHistorySteps = cfg.maxHistorySteps;
    } else {
      this.sequenceMaxHistorySteps = null;
    }
    this.sequenceHistory = [];
    this.sequenceActive = true;
    return this;
  }

  detachSequence(): this {
    this.sequenceHistory = [];
    return this;
  }

  clearSequenceHistory(): this {
    this.sequenceHistory = [];
    return this;
  }

  endSequence(): this {
    this.sequenceHistory = [];
    this.sequenceActive = false;
    this.sequenceMaxHistorySteps = null;
    return this;
  }

  isSequenceActive(): boolean {
    return this.sequenceActive;
  }

  getSequenceLength(): number {
    return this.sequenceHistory.length;
  }

  compile(cfg: { alpha?: number; optimizer?: Optimzier; clipGradient?: number | boolean }): void {
    if (cfg.alpha !== undefined) this.alpha = cfg.alpha;
    if (cfg.clipGradient !== undefined) this.clipGradient = cfg.clipGradient;
    if (cfg.optimizer !== undefined) this.optimizerName = cfg.optimizer;
    if (!this.initialized) return;

    this.optimizerQuery = setOptimizer(this.optimizerName, this.queryKernel._shape, 1e-5);
    if (this.needKernel) this.optimizerNeed = setOptimizer(this.optimizerName, this.needKernel._shape, 1e-5);
    if (this.outputKernel) this.optimizerOutput = setOptimizer(this.optimizerName, this.outputKernel._shape, 1e-5);
    if (this.outputBias) this.optimizerOutputBias = setOptimizer(this.optimizerName, this.outputBias._shape, 1e-5);
  }

  private toMatrix2D(data: matrix2d, rows: number, cols: number, name: string): Matrix {
    if (!Array.isArray(data) || data.length !== rows) throw new Error(`MemoryBank.load: invalid ${name} rows`);
    for (let r = 0; r < rows; r++) {
      if (!Array.isArray(data[r]) || data[r].length !== cols) throw new Error(`MemoryBank.load: invalid ${name} cols`);
      for (let c = 0; c < cols; c++) {
        if (!Number.isFinite(data[r][c])) throw new Error(`MemoryBank.load: non-finite value in ${name}`);
      }
    }
    return new Matrix({ array: data });
  }

  save(): MemoryBankSaveData {
    const memoryState = this.getMemoryState();
    const outputKernel = this.outputKernel?._value;
    const outputBias = this.outputBias?._value;
    const needKernel = this.needKernel?._value;

    return {
      name: this.name,
      status: this.status,
      config: {
        mode: this.mode,
        similarity: this.similarity,
        readTopK: this.readTopK,
        persistence: this.persistence,
        resetOnInit: this.resetOnInit,
        writeEnabled: this.writeEnabled,
      },
      dimensions: {
        units: this.units,
        memorySlots: this.memorySlots,
        outputUnits: this.outputUnits,
      },
      trainableParams: {
        queryKernel: this.queryKernel._value,
        needKernel,
        outputKernel,
        outputBias,
      },
      memoryState,
      optimizerState: {
        alpha: this.alpha,
        optimizer: this.optimizerName,
        clipGradient: this.clipGradient,
        status: this.status,
      },
      units: this.units,
      memorySlots: this.memorySlots,
      memoryDim: this.memoryDim,
      outputUnits: this.outputUnits,
      mode: this.mode,
      similarity: this.similarity,
      readTopK: this.readTopK,
      persistence: this.persistence,
      resetOnInit: this.resetOnInit,
      writeEnabled: this.writeEnabled,
      alpha: this.alpha,
      optimizer: this.optimizerName,
      clipGradient: this.clipGradient,
      queryKernel: this.queryKernel._value,
      needKernel,
      outputKernel,
      outputBias,
      memoryKeys: memoryState.memoryKeys,
      memoryValues: memoryState.memoryValues,
      memoryFilled: memoryState.memoryFilled,
      memoryUsage: memoryState.memoryUsage,
      memoryAge: memoryState.memoryAge,
      memoryStep: memoryState.memoryStep,
    };
  }

  load(data: any): void {
    const config = data.config ?? {};
    const dimensions = data.dimensions ?? {};
    const trainableParams = data.trainableParams ?? {};
    const optimizerState = data.optimizerState ?? {};
    const memoryStateData = data.memoryState ?? (
      data.memoryKeys
        ? {
            memoryKeys: data.memoryKeys,
            memoryValues: data.memoryValues,
            memoryFilled: data.memoryFilled,
            memoryUsage: data.memoryUsage,
            memoryAge: data.memoryAge,
            memoryStep: data.memoryStep,
            units: data.units,
            memoryDim: data.memoryDim ?? data.units,
            memorySlots: data.memorySlots,
          }
        : null
    );

    this.mode = config.mode ?? data.mode ?? this.mode;
    if (this.mode !== "project" && this.mode !== "concat") this.mode = "project";
    this.similarity = config.similarity ?? data.similarity ?? this.similarity;
    this.readTopK = config.readTopK ?? data.readTopK ?? this.readTopK;
    this.persistence = config.persistence ?? data.persistence ?? this.persistence;
    this.resetOnInit = config.resetOnInit ?? data.resetOnInit ?? this.resetOnInit;
    this.writeEnabled = config.writeEnabled ?? data.writeEnabled ?? this.writeEnabled;
    this.alpha = optimizerState.alpha ?? data.alpha ?? this.alpha;
    this.optimizerName = optimizerState.optimizer ?? data.optimizer ?? this.optimizerName;
    this.clipGradient = optimizerState.clipGradient ?? data.clipGradient ?? this.clipGradient;
    this.status = optimizerState.status ?? data.status ?? this.status;

    const units = dimensions.units ?? data.units;
    const memorySlots = dimensions.memorySlots ?? data.memorySlots ?? this.memorySlots;
    const outputUnits = dimensions.outputUnits ?? data.outputUnits ?? units;
    this.assertPositiveInt(units, "units");
    this.assertPositiveInt(memorySlots, "memorySlots");
    this.assertPositiveInt(outputUnits, "outputUnits");
    this.assertPositiveInt(this.readTopK, "readTopK");
    if (this.readTopK > memorySlots) {
      throw new Error(`MemoryBank.load: readTopK must be in [1, memorySlots], got ${this.readTopK} for memorySlots=${memorySlots}`);
    }
    if (this.similarity !== "cosine" && this.similarity !== "dot") {
      throw new Error(`MemoryBank.load: unsupported similarity '${this.similarity}'`);
    }
    this.memorySlots = memorySlots;
    this.configuredOutputUnits = outputUnits;
    this.resetInitializationState();
    this.init(units, outputUnits);

    const queryKernel = trainableParams.queryKernel ?? data.queryKernel;
    const needKernel = trainableParams.needKernel ?? data.needKernel;
    const outputKernel = trainableParams.outputKernel ?? data.outputKernel;
    const outputBias = trainableParams.outputBias ?? data.outputBias;

    if (queryKernel) this.queryKernel = this.toMatrix2D(queryKernel, this.units, this.units, "queryKernel");
    if (this.needKernel && needKernel) {
      this.needKernel = this.toMatrix2D(needKernel, 1, this.units + this.units, "needKernel");
    }
    if (this.outputKernel && outputKernel) {
      this.outputKernel = this.toMatrix2D(outputKernel, this.outputUnits, this.units + this.units, "outputKernel");
    }
    if (this.outputBias && outputBias) {
      this.outputBias = this.toMatrix2D(outputBias, this.outputUnits, 1, "outputBias");
    }

    if (memoryStateData) this.setMemoryState(memoryStateData as MemoryBankState);
    this.compile({ optimizer: this.optimizerName, alpha: this.alpha, clipGradient: this.clipGradient });
  }

  resetMemory(): void {
    if (!this.initialized) return;
    this.memoryKeys = mj.zeros([this.units, this.memorySlots]);
    this.memoryValues = mj.zeros([this.units, this.memorySlots]);
    this.memoryFilled = new Uint8Array(this.memorySlots);
    this.memoryUsage = new Float32Array(this.memorySlots);
    this.memoryAge = new Float32Array(this.memorySlots);
    this.memoryStep = 0;
  }

  clearMemory(): void {
    this.resetMemory();
  }

  hasMemory(): boolean {
    if (!this.initialized) return false;
    for (let i = 0; i < this.memoryFilled.length; i++) if (this.memoryFilled[i] === 1) return true;
    return false;
  }

  getMemoryState(): MemoryBankState {
    if (!this.initialized) throw new Error("MemoryBank.getMemoryState: layer is not initialized yet");
    return {
      memoryKeys: this.memoryKeys._value,
      memoryValues: this.memoryValues._value,
      memoryFilled: Array.from(this.memoryFilled),
      memoryUsage: Array.from(this.memoryUsage),
      memoryAge: Array.from(this.memoryAge),
      memoryStep: this.memoryStep,
      units: this.units,
      memoryDim: this.units,
      memorySlots: this.memorySlots,
    };
  }

  setMemoryState(state: MemoryBankState): void {
    if (!state || typeof state !== "object") throw new Error("MemoryBank.setMemoryState: invalid state object");
    if (!this.initialized) this.init(state.units, this.configuredOutputUnits ?? state.units);
    if (state.units !== this.units || state.memorySlots !== this.memorySlots) {
      throw new Error("MemoryBank.setMemoryState: dimensions mismatch with current layer configuration");
    }
    if (state.memoryDim !== undefined && state.memoryDim !== this.units) {
      throw new Error("MemoryBank.setMemoryState: memoryDim must match units in the simplified MemoryBank");
    }

    this.memoryKeys = this.toMatrix2D(state.memoryKeys, this.units, this.memorySlots, "memoryKeys");
    this.memoryValues = this.toMatrix2D(state.memoryValues, this.units, this.memorySlots, "memoryValues");
    if (state.memoryFilled.length !== this.memorySlots || state.memoryUsage.length !== this.memorySlots || state.memoryAge.length !== this.memorySlots) {
      throw new Error("MemoryBank.setMemoryState: invalid vector lengths");
    }
    for (let i = 0; i < this.memorySlots; i++) {
      if (state.memoryFilled[i] !== 0 && state.memoryFilled[i] !== 1) {
        throw new Error("MemoryBank.setMemoryState: memoryFilled values must be 0 or 1");
      }
      if (!Number.isFinite(state.memoryUsage[i]) || !Number.isFinite(state.memoryAge[i])) {
        throw new Error("MemoryBank.setMemoryState: memoryUsage/memoryAge must be finite");
      }
    }
    this.memoryFilled = Uint8Array.from(state.memoryFilled);
    this.memoryUsage = Float32Array.from(state.memoryUsage);
    this.memoryAge = Float32Array.from(state.memoryAge);
    this.memoryStep = state.memoryStep;
  }

  saveMemory(path: string): void {
    const fs = require("fs") as typeof import("fs");
    fs.writeFileSync(path, JSON.stringify(this.getMemoryState()), "utf-8");
  }

  loadMemory(path: string): void {
    const fs = require("fs") as typeof import("fs");
    const raw = fs.readFileSync(path, "utf-8");
    this.setMemoryState(JSON.parse(raw) as MemoryBankState);
  }

  freezeWrites(): this {
    this.writeFrozen = true;
    return this;
  }

  unfreezeWrites(): this {
    this.writeFrozen = false;
    return this;
  }

  setWriteFrozen(value: boolean): this {
    this.writeFrozen = value;
    return this;
  }

  enableWrites(): this {
    this.writeEnabled = true;
    return this.unfreezeWrites();
  }

  disableWrites(): this {
    this.writeEnabled = false;
    return this.freezeWrites();
  }

  dispose(): void {
    this.cache = [];
    this.debugTrace = [];
    this.lastWriteInfo = null;
    this.sequenceHistory = [];
  }
}
