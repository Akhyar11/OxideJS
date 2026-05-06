import readline from "readline";

import {
  createModel,
  decodeResponse,
  encodeTurn,
  EncodedTurn,
  loadArtifacts,
  looksLikeQuestion,
} from "./shared";

function printMemoryMonitor(model: ReturnType<typeof createModel>, encodedTurn: EncodedTurn): void {
  const trace = model.memory.getDebugTrace();
  const state = model.memory.getMemoryState();
  const last = trace[trace.length - 1];

  if (!last) {
    console.log("[memory] trace kosong");
    return;
  }

  const filled = state.memoryFilled.reduce((sum, value) => sum + value, 0);
  const topReads = last.readSlots
    .slice(0, 3)
    .map((slot) => `slot=${slot.slot} score=${slot.score.toFixed(4)} attn=${(slot.attn * 100).toFixed(2)}%`)
    .join(" | ");

  console.log(
    `[memory] tokens=${encodedTurn.validLength} need=${last.need.toFixed(4)} writeGate=${last.writeGate.toFixed(4)} filled=${filled}/${state.memorySlots}`
  );
  console.log(`[memory] readTopK: ${topReads || "-"}`);

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

  const activeSlots = state.memoryFilled
    .map((filledValue, slot) => ({
      slot,
      filledValue,
      usage: state.memoryUsage[slot] ?? 0,
      age: state.memoryAge[slot] ?? 0,
    }))
    .filter((item) => item.filledValue);

  if (activeSlots.length === 0) {
    console.log("[memory] activeSlots: -");
    return;
  }

  console.log("[memory] activeSlots:");
  for (const item of activeSlots) {
    console.log(`  slot=${item.slot} usage=${item.usage.toFixed(4)} age=${item.age.toFixed(0)}`);
  }
}

function main(): void {
  const { artifacts, tokenizer } = loadArtifacts();
  const model = createModel({
    vocabSize: tokenizer.getVocabularyCapacity(),
    maxTurnTokens: artifacts.config.maxTurnTokens,
    maxResponseTokens: artifacts.config.maxResponseTokens,
    embeddingDim: artifacts.config.embeddingDim,
    decoderHiddenUnits: artifacts.config.decoderHiddenUnits,
    memorySlots: artifacts.config.memorySlots,
    memoryMode: artifacts.config.memoryMode,
    alpha: artifacts.config.alpha,
    optimizer: artifacts.config.optimizer,
    clipGradient: artifacts.config.clipGradient,
    padTokenId: tokenizer.getPadId(),
  });

  model.embedding.load(artifacts.layers.embedding);
  model.memory.load(artifacts.layers.memory);
  model.contextProject.load(
    artifacts.layers.contextProject.weight,
    artifacts.layers.contextProject.bias,
    artifacts.layers.contextProject.clipGradient
  );
  model.decoderLstm.load(artifacts.layers.decoderLstm as any);
  model.decoderOutput.load(
    artifacts.layers.decoderOutput.weight,
    artifacts.layers.decoderOutput.bias,
    artifacts.layers.decoderOutput.clipGradient
  );
  model.eval();
  model.resetMemory();
  model.memory.clearDebugTrace();

  const bosId = tokenizer.getTokenId("<BOS>");
  const eosId = tokenizer.getTokenId("<EOS>");
  const padId = tokenizer.getPadId();
  if (bosId === undefined || eosId === undefined) {
    throw new Error("Tokenizer wajib memiliki token <BOS> dan <EOS>.");
  }

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

      const { context } = model.forwardTurn(encodedTurn);
      const predictedIds = model.decodeGreedy(context, artifacts.config.maxResponseTokens, bosId, eosId, padId);
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
