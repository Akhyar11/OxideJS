import fs from "fs";
import path from "path";

import { BPETokenizer } from "../../src/tokenizer";
import { Dense, Embedding, LSTM, MemoryBank } from "../../src/layers";
import mj from "../../src/math";
import Matrix from "../../src/matrix";

export type DialogueRole = "system" | "user" | "assistant";

export type DialogueTurn = {
  role: DialogueRole;
  text: string;
};

export type DialogueProbe = {
  question: string;
  answer: string;
  not_answer?: string;
};

export type DialogueEpisode = {
  episode_id: string;
  split: string;
  category: string;
  reset_before: boolean;
  turns: DialogueTurn[];
  expected_memory?: Record<string, string>;
  probes?: DialogueProbe[];
};

export type DialogueDatasetFile = {
  name: string;
  version: string;
  language: string;
  created_at: string;
  split: string;
  episode_count: number;
  episodes: DialogueEpisode[];
};

export type EncodedTurn = {
  tokenIds: number[];
  validLength: number;
};

export type DialogueExperimentArtifacts = {
  format: "memorybank-dialogue-experiment-v3-single-embedding";
  createdAt: string;
  config: {
    vocabSize: number;
    minFrequency: number;
    maxTurnTokens: number;
    maxResponseTokens: number;
    embeddingDim: number;
    decoderHiddenUnits: number;
    memorySlots: number;
    memoryMode: "project" | "concat";
    optimizer: string;
    alpha: number;
    clipGradient: number | boolean;
    preTokenizer: "unicode-grapheme";
  };
  metrics: {
    trainExactMatch: number;
    validationExactMatch: number;
    testExactMatch: number;
    trainTokenAccuracy: number;
    validationTokenAccuracy: number;
    testTokenAccuracy: number;
  };
  layers: {
    embedding: ReturnType<Embedding["save"]>;
    memory: ReturnType<MemoryBank["save"]>;
    contextProject: ReturnType<Dense["save"]>;
    decoderLstm: ReturnType<LSTM["save"]>;
    decoderOutput: ReturnType<Dense["save"]>;
  };
};

export type DecoderEpisodeResult = {
  avgLoss: number;
  errContexts: Matrix;
};

export const DATASET_DIR = __dirname;
export const TRAIN_PATH = path.join(DATASET_DIR, "memory_bank_train_1000.json");
export const VALIDATION_PATH = path.join(DATASET_DIR, "memory_bank_validation_100.json");
export const TEST_PATH = path.join(DATASET_DIR, "memory_bank_test_200.json");
export const ARTIFACT_DIR = path.join(DATASET_DIR, "artifacts");
export const MODEL_PATH = path.join(ARTIFACT_DIR, "dialogue_memory_model.json");
export const TOKENIZER_PATH = path.join(ARTIFACT_DIR, "dialogue_tokenizer.json");
export const METRICS_PATH = path.join(ARTIFACT_DIR, "dialogue_metrics.json");

type MaskedLossResult = {
  avgLoss: number;
  errLogits: Matrix;
};

export class DialogueMemoryModel {
  private training = true;

  constructor(
    public embedding: Embedding,
    public memory: MemoryBank,
    public contextProject: Dense,
    public decoderLstm: LSTM,
    public decoderOutput: Dense
  ) {
    this.memory.train();
  }

  compile(config: { alpha?: number; optimizer?: string; clipGradient?: number | boolean }): void {
    this.embedding.compile({
      alpha: config.alpha,
      optimizer: config.optimizer as any,
    });
    this.memory.compile({
      alpha: config.alpha,
      optimizer: config.optimizer as any,
      clipGradient: config.clipGradient,
    });
    this.contextProject.compile({
      alpha: config.alpha,
      optimizer: config.optimizer as any,
      clipGradient: config.clipGradient,
    });
    this.decoderLstm.compile({
      alpha: config.alpha,
      optimizer: config.optimizer as any,
      clipGradient: config.clipGradient,
    });
    this.decoderOutput.compile({
      alpha: config.alpha,
      optimizer: config.optimizer as any,
      error: "softmaxCrossEntropy",
      clipGradient: config.clipGradient,
    });
  }

  resetMemory(): void {
    this.memory.resetMemory();
  }

  train(): this {
    this.training = true;
    this.memory.train();
    return this;
  }

  eval(): this {
    this.training = false;
    this.memory.eval();
    return this;
  }

  freezeWrites(): void {
    this.memory.freezeWrites();
  }

  enableWrites(): void {
    this.memory.unfreezeWrites();
  }

  forwardTurn(encodedTurn: EncodedTurn): { contextSequence: Matrix; context: Matrix } {
    const tokenMatrix = tokenIdsToColumnMatrix(encodedTurn.tokenIds);
    const emb = this.embedding.forward(tokenMatrix);
    const contextSequence = this.memory.forward(emb);
    const context = columnAsMatrix(contextSequence, encodedTurn.validLength - 1);
    return { contextSequence, context };
  }

  decodeGreedy(
    context: Matrix,
    maxResponseTokens: number,
    bosId: number,
    eosId: number,
    padId: number
  ): number[] {
    const predicted: number[] = [];
    const decoderInputIds: number[] = [bosId];

    for (let step = 0; step < maxResponseTokens; step++) {
      const inputMatrix = tokenIdsToColumnMatrix(decoderInputIds);
      const emb = this.embedding.forward(inputMatrix);
      const contextSeed = this.contextProject.forward(context);
      addSeedToFirstStep(emb, contextSeed, 1);
      const hidden = this.decoderLstm.forward(emb);
      const logits = this.decoderOutput.forward(hidden);
      const nextId = argmaxFromColumn(logits, decoderInputIds.length - 1);
      if (nextId === eosId || nextId === padId) break;
      predicted.push(nextId);
      decoderInputIds.push(nextId);
    }

    return predicted;
  }

  trainEpisodeDecoder(
    contexts: Matrix,
    targetIdsBySample: number[][],
    maxResponseTokens: number,
    bosId: number,
    padId: number
  ): DecoderEpisodeResult {
    const batchSize = targetIdsBySample.length;
    if (batchSize === 0) {
      return {
        avgLoss: 0,
        errContexts: mj.zeros([contexts._shape[0], 0]),
      };
    }

    const inputIdMatrix = mj.zeros([maxResponseTokens, batchSize]);
    for (let t = 0; t < maxResponseTokens; t++) {
      for (let b = 0; b < batchSize; b++) {
        const targets = targetIdsBySample[b]!;
        const targetId = targets[t] ?? padId;
        const decoderInputId = t === 0 ? bosId : (targets[t - 1] ?? padId);
        inputIdMatrix._data[t * batchSize + b] = decoderInputId;
        if (targetId === padId && t > 0) {
          inputIdMatrix._data[t * batchSize + b] = padId;
        }
      }
    }

    const decoderEmb = this.embedding.forward(inputIdMatrix);
    const contextSeed = this.contextProject.forward(contexts);
    addSeedToFirstStep(decoderEmb, contextSeed, batchSize);

    this.decoderLstm.resetLoss();
    this.decoderOutput.resetLoss();

    const hidden = this.decoderLstm.forwardBatch(decoderEmb, batchSize);
    const logits = this.decoderOutput.forward(hidden);
    const masked = maskedSoftmaxCrossEntropyFromTargets(
      logits,
      targetIdsBySample,
      maxResponseTokens,
      batchSize,
      padId
    );

    const errHidden = this.decoderOutput.backward(mj.matrix([]), masked.errLogits);
    const errDecoderInput = this.decoderLstm.backwardBatch(mj.matrix([]), errHidden, batchSize);
    this.embedding.backward(mj.matrix([]), errDecoderInput);

    const errSeed = mj.zeros([this.embedding.embeddingDim, batchSize]);
    for (let b = 0; b < batchSize; b++) {
      copyColumn(errDecoderInput, b, errSeed, b);
    }
    const errContexts = this.contextProject.backward(mj.matrix([]), errSeed);

    return {
      avgLoss: masked.avgLoss,
      errContexts,
    };
  }

  printSummary(): void {
    console.log("Model Summary:");
    console.log(`- SharedEmbedding: vocabSize=${this.embedding.vocabSize}, embeddingDim=${this.embedding.embeddingDim}`);
    console.log(
      `- MemoryBank: units=${this.memory.units}, memorySlots=${this.memory.memorySlots}, outputUnits=${this.memory.outputUnits}, mode=${this.memory.mode}`
    );
    console.log(`- ContextProject: units=${this.contextProject.units}, outputUnits=${this.contextProject.outputUnits}`);
    console.log(
      `- DecoderLSTM: units=${this.decoderLstm.units}, hiddenUnits=${this.decoderLstm.hiddenUnits}, returnSequences=${this.decoderLstm.returnSequences}`
    );
    console.log(`- DecoderOutput: units=${this.decoderOutput.units}, outputUnits=${this.decoderOutput.outputUnits}`);
  }
}

export function ensureArtifactDir(): void {
  fs.mkdirSync(ARTIFACT_DIR, { recursive: true });
}

export function loadDataset(filePath: string): DialogueDatasetFile {
  return JSON.parse(fs.readFileSync(filePath, "utf-8")) as DialogueDatasetFile;
}

export function collectCorpus(episodes: DialogueEpisode[]): string[] {
  const texts: string[] = [];
  for (const episode of episodes) {
    for (const turn of episode.turns) {
      texts.push(turn.text);
    }
  }
  return texts;
}

export function buildTokenizer(texts: string[], vocabSize: number, minFrequency: number): BPETokenizer {
  const tokenizer = BPETokenizer.load(TOKENIZER_PATH)


  return tokenizer;
}

export function getNextAssistantTurn(turns: DialogueTurn[], index: number): DialogueTurn | null {
  for (let i = index + 1; i < turns.length; i++) {
    const turn = turns[i]!;
    if (turn.role === "assistant") return turn;
    if (turn.role === "user") return null;
  }
  return null;
}

export function containsResetMarker(text: string): boolean {
  return /<MEMORY_RESET>/i.test(text);
}

export function looksLikeQuestion(text: string): boolean {
  const normalized = text.trim().toLowerCase();
  if (normalized.includes("?")) return true;
  return /^(apa|siapa|berapa|di mana|dimana|kapan|bagaimana|minuman seperti apa|warna apa|nama apa)/.test(normalized);
}

export function cleanResetMarker(text: string): string {
  return text.replace(/<MEMORY_RESET>/gi, "").trim();
}

export function encodeTurn(tokenizer: BPETokenizer, text: string, maxTurnTokens: number): EncodedTurn {
  return encodeTurnInternal(tokenizer, text, maxTurnTokens, false);
}

export function encodeTurnForTraining(tokenizer: BPETokenizer, text: string, maxTurnTokens: number): EncodedTurn {
  return encodeTurnInternal(tokenizer, text, maxTurnTokens, true);
}

function encodeTurnInternal(
  tokenizer: BPETokenizer,
  text: string,
  maxTurnTokens: number,
  training: boolean
): EncodedTurn {
  const rawIds = training ? tokenizer.encodeForTraining(text) : tokenizer.encode(text);
  const unkId = tokenizer.getTokenId("<UNK>");
  const tokenIds = rawIds.length > 0 ? rawIds.slice(0, maxTurnTokens) : [unkId ?? tokenizer.getPadId()];
  return {
    tokenIds,
    validLength: tokenIds.length,
  };
}

export function encodeResponseTarget(
  tokenizer: BPETokenizer,
  text: string,
  maxResponseTokens: number
): number[] {
  const eosId = tokenizer.getTokenId("<EOS>");
  const padId = tokenizer.getPadId();
  if (eosId === undefined) {
    throw new Error("Tokenizer tidak memiliki token <EOS>.");
  }
  const ids = [...tokenizer.encode(text), eosId];
  if (ids.length >= maxResponseTokens) {
    return ids.slice(0, maxResponseTokens);
  }
  return [...ids, ...Array(maxResponseTokens - ids.length).fill(padId)];
}

export function argmaxFromMatrix(matrix: Matrix): number {
  return argmaxFromColumn(matrix, 0);
}

export function argmaxFromColumn(matrix: Matrix, column: number): number {
  const [, cols] = matrix._shape;
  if (column < 0 || column >= cols) {
    throw new Error(`argmaxFromColumn: column ${column} di luar range 0..${cols - 1}`);
  }
  let maxIndex = 0;
  let maxValue = matrix._data[column] ?? Number.NEGATIVE_INFINITY;
  for (let row = 1; row < matrix._shape[0]; row++) {
    const value = matrix._data[row * cols + column] ?? Number.NEGATIVE_INFINITY;
    if (value > maxValue) {
      maxValue = value;
      maxIndex = row;
    }
  }
  return maxIndex;
}

export function decodeResponse(tokenizer: BPETokenizer, predictedIds: number[]): string {
  const eosId = tokenizer.getTokenId("<EOS>");
  const padId = tokenizer.getPadId();
  const trimmed: number[] = [];
  for (const id of predictedIds) {
    if (id === eosId || id === padId) break;
    trimmed.push(id);
  }
  return tokenizer.decode(trimmed).trim();
}

export function normalizeText(text: string): string {
  return text.toLowerCase().replace(/\s+/g, " ").trim();
}

export function createModel(config: {
  vocabSize: number;
  maxTurnTokens: number;
  maxResponseTokens: number;
  embeddingDim: number;
  decoderHiddenUnits: number;
  memorySlots: number;
  memoryMode: "project" | "concat";
  alpha: number;
  optimizer: string;
  clipGradient: number | boolean;
  padTokenId: number;
}): DialogueMemoryModel {
  const embedding = new Embedding({
    vocabSize: config.vocabSize,
    embeddingDim: config.embeddingDim,
    alpha: config.alpha,
    optimizer: config.optimizer as any,
    padTokenId: config.padTokenId,
    trainable: true,
  });

  const memory = new MemoryBank({
    units: config.embeddingDim,
    memorySlots: config.memorySlots,
    outputUnits: config.memoryMode === "concat" ? config.embeddingDim * 2 : config.embeddingDim,
    mode: config.memoryMode,
    similarity: "cosine",
    readTopK: Math.min(4, config.memorySlots),
    alpha: config.alpha,
    optimizer: config.optimizer as any,
    clipGradient: config.clipGradient,
    writeEnabled: true,
    persistence: "session",
  });

  const contextUnits = config.memoryMode === "concat" ? config.embeddingDim * 2 : config.embeddingDim;
  const contextProject = new Dense({
    units: contextUnits,
    outputUnits: config.embeddingDim,
    activation: "linear",
    optimizer: config.optimizer as any,
    alpha: config.alpha,
    clipGradient: config.clipGradient,
  });

  const decoderLstm = new LSTM({
    units: config.embeddingDim,
    hiddenUnits: config.decoderHiddenUnits,
    returnSequences: true,
    stateful: false,
    optimizer: config.optimizer as any,
    alpha: config.alpha,
    clipGradient: config.clipGradient,
  });

  const decoderOutput = new Dense({
    units: config.decoderHiddenUnits,
    outputUnits: config.vocabSize,
    activation: "linear",
    optimizer: config.optimizer as any,
    alpha: config.alpha,
    loss: "softmaxCrossEntropy",
    clipGradient: config.clipGradient,
    status: "output",
  });

  const model = new DialogueMemoryModel(
    embedding,
    memory,
    contextProject,
    decoderLstm,
    decoderOutput
  );
  model.printSummary();
  return model;
}

export function saveArtifacts(
  artifacts: DialogueExperimentArtifacts,
  tokenizer: BPETokenizer
): void {
  ensureArtifactDir();
  fs.writeFileSync(MODEL_PATH, JSON.stringify(artifacts, null, 2), "utf-8");
  fs.writeFileSync(METRICS_PATH, JSON.stringify(artifacts.metrics, null, 2), "utf-8");
  tokenizer.save(TOKENIZER_PATH);
}

export function loadArtifacts(): { artifacts: DialogueExperimentArtifacts; tokenizer: BPETokenizer } {
  if (!fs.existsSync(MODEL_PATH)) {
    throw new Error(`Model artifact belum ada: ${MODEL_PATH}. Jalankan training lebih dulu.`);
  }
  if (!fs.existsSync(TOKENIZER_PATH)) {
    throw new Error(`Tokenizer artifact belum ada: ${TOKENIZER_PATH}. Jalankan training lebih dulu.`);
  }
  const artifacts = JSON.parse(fs.readFileSync(MODEL_PATH, "utf-8")) as DialogueExperimentArtifacts;
  const tokenizer = BPETokenizer.load(TOKENIZER_PATH);
  return { artifacts, tokenizer };
}

export function tokenIdsToColumnMatrix(tokenIds: number[]): Matrix {
  return Matrix.fromFlat(Float32Array.from(tokenIds), [tokenIds.length, 1]);
}

export function concatenateTokenIds(turns: EncodedTurn[]): { ids: number[]; ranges: Array<{ start: number; length: number }> } {
  const ids: number[] = [];
  const ranges: Array<{ start: number; length: number }> = [];
  let cursor = 0;
  for (const turn of turns) {
    ranges.push({ start: cursor, length: turn.tokenIds.length });
    ids.push(...turn.tokenIds);
    cursor += turn.tokenIds.length;
  }
  return { ids, ranges };
}

export function sliceColumns(matrix: Matrix, start: number, length: number): Matrix {
  const [rows, cols] = matrix._shape;
  if (start < 0 || length < 1 || start + length > cols) {
    throw new Error(`sliceColumns: invalid range start=${start}, length=${length}, cols=${cols}`);
  }
  const out = mj.zeros([rows, length]);
  for (let row = 0; row < rows; row++) {
    const srcOffset = row * cols + start;
    const dstOffset = row * length;
    out._data.set(matrix._data.subarray(srcOffset, srcOffset + length), dstOffset);
  }
  return out;
}

export function columnAsMatrix(matrix: Matrix, column: number): Matrix {
  return Matrix.fromFlat(matrix.getCol(column), [matrix._shape[0], 1]);
}

export function copyColumn(source: Matrix, sourceColumn: number, target: Matrix, targetColumn: number): void {
  const sourceCols = source._shape[1];
  const targetCols = target._shape[1];
  for (let row = 0; row < source._shape[0]; row++) {
    target._data[row * targetCols + targetColumn] = source._data[row * sourceCols + sourceColumn] ?? 0;
  }
}

export function addSeedToFirstStep(emb: Matrix, seed: Matrix, batchSize: number): void {
  const seedData = seed._data;
  const embData = emb._data;
  const dim = seed._shape[0];
  const totalCols = emb._shape[1];
  for (let b = 0; b < batchSize; b++) {
    for (let d = 0; d < dim; d++) {
      embData[d * totalCols + b] += seedData[d * batchSize + b];
    }
  }
}

function maskedSoftmaxCrossEntropyFromTargets(
  logits: Matrix,
  targetIdsBySample: number[][],
  maxResponseTokens: number,
  batchSize: number,
  padId: number
): MaskedLossResult {
  const [vocabSize, totalCols] = logits._shape;
  if (totalCols !== maxResponseTokens * batchSize) {
    throw new Error(
      `maskedSoftmaxCrossEntropyFromTargets: logits cols=${totalCols} tidak cocok dengan maxResponseTokens * batchSize=${maxResponseTokens * batchSize}`
    );
  }

  const errLogits = mj.zeros([vocabSize, totalCols]);
  let totalLoss = 0;
  let validTokenCount = 0;

  for (let b = 0; b < batchSize; b++) {
    for (let t = 0; t < maxResponseTokens; t++) {
      const targetId = targetIdsBySample[b]![t] ?? padId;
      if (targetId === padId) {
        continue;
      }

      const col = t * batchSize + b;
      let maxLogit = Number.NEGATIVE_INFINITY;
      for (let v = 0; v < vocabSize; v++) {
        const value = logits._data[v * totalCols + col] ?? Number.NEGATIVE_INFINITY;
        if (value > maxLogit) maxLogit = value;
      }

      let sumExp = 0;
      for (let v = 0; v < vocabSize; v++) {
        sumExp += Math.exp((logits._data[v * totalCols + col] ?? 0) - maxLogit);
      }

      const safeDenom = sumExp > 0 ? sumExp : 1;
      for (let v = 0; v < vocabSize; v++) {
        const prob = Math.exp((logits._data[v * totalCols + col] ?? 0) - maxLogit) / safeDenom;
        errLogits._data[v * totalCols + col] = prob;
      }

      const targetProb = Math.max(errLogits._data[targetId * totalCols + col] ?? 1e-12, 1e-12);
      totalLoss += -Math.log(targetProb);
      errLogits._data[targetId * totalCols + col] -= 1;
      validTokenCount++;
    }
  }

  if (validTokenCount > 0) {
    const scale = 1 / validTokenCount;
    for (let i = 0; i < errLogits._data.length; i++) {
      errLogits._data[i] *= scale;
    }
  }

  return {
    avgLoss: validTokenCount > 0 ? totalLoss / validTokenCount : 0,
    errLogits,
  };
}
