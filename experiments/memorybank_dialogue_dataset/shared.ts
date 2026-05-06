import fs from "fs";
import path from "path";

import { BPETokenizer } from "../../src/tokenizer";
import { AttentionPooling, Dense, Embedding, MemoryBank } from "../../src/layers";
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
  x: Matrix;
  validLength: number;
  tokenIds: number[];
};

export type DialogueExperimentArtifacts = {
  format: "memorybank-dialogue-experiment-v1";
  createdAt: string;
  config: {
    vocabSize: number;
    minFrequency: number;
    maxTurnTokens: number;
    maxResponseTokens: number;
    embeddingDim: number;
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
    pooling: ReturnType<AttentionPooling["save"]>;
    memory: ReturnType<MemoryBank["save"]>;
    decoder: Array<ReturnType<Dense["save"]>>;
  };
};

export const DATASET_DIR = __dirname;
export const TRAIN_PATH = path.join(DATASET_DIR, "memory_bank_train_1000.json");
export const VALIDATION_PATH = path.join(DATASET_DIR, "memory_bank_validation_100.json");
export const TEST_PATH = path.join(DATASET_DIR, "memory_bank_test_200.json");
export const ARTIFACT_DIR = path.join(DATASET_DIR, "artifacts");
export const MODEL_PATH = path.join(ARTIFACT_DIR, "dialogue_memory_model.json");
export const TOKENIZER_PATH = path.join(ARTIFACT_DIR, "dialogue_tokenizer.json");
export const METRICS_PATH = path.join(ARTIFACT_DIR, "dialogue_metrics.json");

export class MultiTokenDecoder {
  heads: Dense[];
  maxTokens: number;
  vocabSize: number;
  contextUnits: number;

  constructor({
    contextUnits,
    vocabSize,
    maxTokens,
    alpha,
    optimizer,
    clipGradient,
  }: {
    contextUnits: number;
    vocabSize: number;
    maxTokens: number;
    alpha: number;
    optimizer: string;
    clipGradient: number | boolean;
  }) {
    this.contextUnits = contextUnits;
    this.vocabSize = vocabSize;
    this.maxTokens = maxTokens;
    this.heads = Array.from({ length: maxTokens }, () =>
      new Dense({
        units: contextUnits,
        outputUnits: vocabSize,
        activation: "linear",
        optimizer: optimizer as any,
        alpha,
        loss: "softmaxCrossEntropy",
        clipGradient,
        status: "output",
      })
    );
  }

  compile(config: { alpha?: number; optimizer?: string; clipGradient?: number | boolean }): void {
    for (const head of this.heads) {
      head.compile({
        alpha: config.alpha,
        optimizer: config.optimizer as any,
        clipGradient: config.clipGradient,
        error: "softmaxCrossEntropy",
      });
    }
  }

  forward(x: Matrix): Matrix[] {
    return this.heads.map((head) => head.forward(x));
  }

  backward(targetIds: number[]): { err: Matrix; avgLoss: number } {
    const sumErr = mj.zeros([this.contextUnits, 1]);
    let totalLoss = 0;

    for (let i = 0; i < this.maxTokens; i++) {
      const targetId = targetIds[i] ?? targetIds[targetIds.length - 1] ?? 0;
      const err = this.heads[i]!.backward(mj.matrix([[targetId]]), mj.matrix([]));
      totalLoss += this.heads[i]!.loss ?? 0;

      for (let j = 0; j < sumErr._data.length; j++) {
        sumErr._data[j] += err._data[j];
      }
    }

    const scale = this.maxTokens > 0 ? 1 / this.maxTokens : 1;
    for (let j = 0; j < sumErr._data.length; j++) {
      sumErr._data[j] *= scale;
    }

    return {
      err: sumErr,
      avgLoss: this.maxTokens > 0 ? totalLoss / this.maxTokens : 0,
    };
  }

  save(): Array<ReturnType<Dense["save"]>> {
    return this.heads.map((head) => head.save());
  }

  load(data: Array<ReturnType<Dense["save"]>>): void {
    if (!Array.isArray(data) || data.length !== this.heads.length) {
      throw new Error(`MultiTokenDecoder.load: expected ${this.heads.length} decoder heads`);
    }
    for (let i = 0; i < this.heads.length; i++) {
      const head = this.heads[i]!;
      const saved = data[i]!;
      head.load(saved.weight, saved.bias, saved.clipGradient);
      head.compile({
        alpha: head.alpha,
        optimizer: saved.optimizer as any,
        error: "softmaxCrossEntropy",
        clipGradient: saved.clipGradient,
      });
    }
  }
}

export class DialogueMemoryModel {
  private training = true;

  constructor(
    public embedding: Embedding,
    public pooling: AttentionPooling,
    public memory: MemoryBank,
    public decoder: MultiTokenDecoder
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
    this.decoder.compile(config);
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

  forwardTurn(encodedTurn: EncodedTurn): { context: Matrix; logits: Matrix[] } {
    this.pooling.setValidLength(encodedTurn.validLength);
    const emb = this.embedding.forward(encodedTurn.x);
    const pooled = this.pooling.forward(emb);
    const context = this.memory.forward(pooled);
    const logits = this.decoder.forward(context);
    return { context, logits };
  }

  backwardResponse(err: Matrix, target: Matrix): number {
    const errMemory = this.memory.backward(target, err);
    const errPool = this.pooling.backward(target, errMemory);
    this.embedding.backward(target, errPool);
    return 0;
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
  const tokenizer = new BPETokenizer({
    vocabSize,
    minFrequency,
    preTokenizer: "unicode-grapheme",
  });
  tokenizer.train(texts);
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
  const rawIds = tokenizer.encode(text);
  const validLength = Math.max(1, Math.min(rawIds.length, maxTurnTokens));
  const tokenIds = tokenizer.padSequence(rawIds, maxTurnTokens);
  return {
    x: Matrix.fromFlat(Float32Array.from(tokenIds), [maxTurnTokens, 1]),
    validLength,
    tokenIds,
  };
}

export function encodeTurnForTraining(tokenizer: BPETokenizer, text: string, maxTurnTokens: number): EncodedTurn {
  const rawIds = tokenizer.encodeForTraining(text);
  const validLength = Math.max(1, Math.min(rawIds.length, maxTurnTokens));
  const tokenIds = tokenizer.padSequence(rawIds, maxTurnTokens);
  return {
    x: Matrix.fromFlat(Float32Array.from(tokenIds), [maxTurnTokens, 1]),
    validLength,
    tokenIds,
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

export function encodeResponseTargetForTraining(
  tokenizer: BPETokenizer,
  text: string,
  maxResponseTokens: number
): number[] {
  const eosId = tokenizer.getTokenId("<EOS>");
  const padId = tokenizer.getPadId();
  if (eosId === undefined) {
    throw new Error("Tokenizer tidak memiliki token <EOS>.");
  }
  const ids = [...tokenizer.encodeForTraining(text), eosId];
  if (ids.length >= maxResponseTokens) {
    return ids.slice(0, maxResponseTokens);
  }
  return [...ids, ...Array(maxResponseTokens - ids.length).fill(padId)];
}

export function argmaxFromMatrix(matrix: Matrix): number {
  let maxIndex = 0;
  let maxValue = matrix._data[0] ?? Number.NEGATIVE_INFINITY;
  for (let i = 1; i < matrix._data.length; i++) {
    if (matrix._data[i]! > maxValue) {
      maxValue = matrix._data[i]!;
      maxIndex = i;
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

  const pooling = new AttentionPooling({
    units: config.embeddingDim,
    maxTokens: config.maxTurnTokens,
    alpha: config.alpha,
    optimizer: config.optimizer as any,
    clipGradient: config.clipGradient,
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

  const decoderUnits = config.memoryMode === "concat" ? config.embeddingDim * 2 : config.embeddingDim;
  const decoder = new MultiTokenDecoder({
    contextUnits: decoderUnits,
    vocabSize: config.vocabSize,
    maxTokens: config.maxResponseTokens,
    alpha: config.alpha,
    optimizer: config.optimizer,
    clipGradient: config.clipGradient,
  });

  return new DialogueMemoryModel(embedding, pooling, memory, decoder);
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
