import readline from "readline";

import {
  createModel,
  decodeResponse,
  encodeTurn,
  EncodedTurn,
  loadArtifacts,
  looksLikeQuestion,
} from "./shared";
import { argmaxFromMatrix } from "./shared";

function formatPct(value: number): string {
  return `${(value * 100).toFixed(2)}%`;
}

function cosineSimilarity(a: number[], b: number[]): number {
  let dot = 0;
  let normA = 0;
  let normB = 0;
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) {
    const av = a[i] ?? 0;
    const bv = b[i] ?? 0;
    dot += av * bv;
    normA += av * av;
    normB += bv * bv;
  }
  if (normA <= 0 || normB <= 0) return 0;
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

function buildSlotRanking(model: ReturnType<typeof createModel>, encodedTurn: EncodedTurn) {
  model.pooling.setValidLength(encodedTurn.validLength);
  const emb = model.embedding.forward(encodedTurn.x);
  const pooled = model.pooling.forward(emb);
  const query = model.memory.getQueryVectorForInput(pooled);
  const state = model.memory.getMemoryState();
  const queryVec = Array.from(query._data);

  const ranking: Array<{
    slot: number;
    similarity: number;
    usage: number;
    age: number;
    filled: number;
  }> = [];

  for (let slot = 0; slot < state.memorySlots; slot++) {
    const filled = state.memoryFilled[slot] ?? 0;
    if (!filled) continue;
    ranking.push({
      slot,
      similarity: cosineSimilarity(queryVec, state.memoryKeys[slot] ?? []),
      usage: state.memoryUsage[slot] ?? 0,
      age: state.memoryAge[slot] ?? 0,
      filled,
    });
  }

  ranking.sort((a, b) => b.similarity - a.similarity);
  return ranking;
}

function printMemoryMonitor(model: ReturnType<typeof createModel>, encodedTurn: EncodedTurn): void {
  const trace = model.memory.getDebugTrace();
  const state = model.memory.getMemoryState();
  const last = trace[trace.length - 1];
  const ranking = buildSlotRanking(model, encodedTurn);

  if (!last) {
    console.log("[memory] trace kosong");
    return;
  }

  const filled = state.memoryFilled.reduce((sum, value) => sum + value, 0);
  const topReads = last.readSlots
    .slice(0, 3)
    .map((slot) => `slot=${slot.slot} score=${slot.score.toFixed(4)} attn=${formatPct(slot.attn)}`)
    .join(" | ");

  console.log(`[memory] need=${last.need.toFixed(4)} writeGate=${last.writeGate.toFixed(4)} filled=${filled}/${state.memorySlots}`);
  console.log(`[memory] readTopK: ${topReads || "-"}`);
  console.log(
    `[memory] overwriteCandidate: ${
      ranking[0]
        ? `slot=${ranking[0].slot} sim=${ranking[0].similarity.toFixed(4)} usage=${ranking[0].usage.toFixed(4)} age=${ranking[0].age.toFixed(0)}`
        : "-"
    }`
  );

  if (last.writeCommitted) {
    console.log(`[memory] write: committed slot=${last.writeSlot}`);
  } else {
    console.log("[memory] write: skipped");
  }

  console.log(
    `[memory] usage(slot=${last.writeSlot >= 0 ? last.writeSlot : 0})=` +
      `${(state.memoryUsage[last.writeSlot >= 0 ? last.writeSlot : 0] ?? 0).toFixed(4)} age=` +
      `${(state.memoryAge[last.writeSlot >= 0 ? last.writeSlot : 0] ?? 0).toFixed(0)}`
  );

  if (ranking.length === 0) {
    console.log("[memory] activeSlots: -");
    return;
  }

  console.log("[memory] activeSlots:");
  for (const item of ranking) {
    console.log(
      `  slot=${item.slot} sim=${item.similarity.toFixed(4)} usage=${item.usage.toFixed(4)} age=${item.age.toFixed(0)}`
    );
  }
}

function main(): void {
  const { artifacts, tokenizer } = loadArtifacts();
  const model = createModel({
    vocabSize: tokenizer.getVocabularyCapacity(),
    maxTurnTokens: artifacts.config.maxTurnTokens,
    maxResponseTokens: artifacts.config.maxResponseTokens,
    embeddingDim: artifacts.config.embeddingDim,
    memorySlots: artifacts.config.memorySlots,
    memoryMode: artifacts.config.memoryMode,
    alpha: artifacts.config.alpha,
    optimizer: artifacts.config.optimizer,
    clipGradient: artifacts.config.clipGradient,
    padTokenId: tokenizer.getPadId(),
  });

  model.embedding.load(artifacts.layers.embedding);
  model.pooling.load(artifacts.layers.pooling);
  model.memory.load(artifacts.layers.memory);
  model.decoder.load(artifacts.layers.decoder);
  model.eval();
  model.resetMemory();
  model.memory.clearDebugTrace();

  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  let debugEnabled = true;

  console.log("Perintah: `/reset` untuk reset memori, `/debug on`, `/debug off`, `exit` untuk keluar.");

  const ask = (): void => {
    rl.question("\nAnda: ", (input) => {
      const text = input.trim();

      if (text === "exit") {
        rl.close();
        return;
      }

      if (text === "/reset") {
        model.resetMemory();
        model.memory.clearDebugTrace();
        console.log("Model: Memori episode direset.");
        ask();
        return;
      }

      if (text === "/debug on") {
        debugEnabled = true;
        console.log("Model: Monitor memory aktif.");
        ask();
        return;
      }

      if (text === "/debug off") {
        debugEnabled = false;
        console.log("Model: Monitor memory nonaktif.");
        ask();
        return;
      }

      const encodedTurn = encodeTurn(tokenizer, text, artifacts.config.maxTurnTokens);
      if (looksLikeQuestion(text)) model.freezeWrites();
      else model.enableWrites();

      const { logits } = model.forwardTurn(encodedTurn);
      const predictedIds = logits.map((matrix) => argmaxFromMatrix(matrix));
      let response = decodeResponse(tokenizer, predictedIds);

      if (!response) {
        response = looksLikeQuestion(text)
          ? "Aku belum yakin dengan jawabannya."
          : "Baik, aku catat.";
      }

      console.log(`Model: ${response}`);
      if (debugEnabled) {
        printMemoryMonitor(model, encodedTurn);
      }
      ask();
    });
  };

  ask();
}

main();
