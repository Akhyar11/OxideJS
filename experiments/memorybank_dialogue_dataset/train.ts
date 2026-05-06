import {
  ARTIFACT_DIR,
  DialogueEpisode,
  DialogueExperimentArtifacts,
  DialogueMemoryModel,
  TEST_PATH,
  TOKENIZER_PATH,
  TRAIN_PATH,
  VALIDATION_PATH,
  argmaxFromMatrix,
  buildTokenizer,
  collectCorpus,
  containsResetMarker,
  createModel,
  decodeResponse,
  encodeResponseTarget,
  encodeTurn,
  getNextAssistantTurn,
  loadDataset,
  looksLikeQuestion,
  normalizeText,
  saveArtifacts,
} from "./shared";
import mj from "../../src/math";

const EPOCHS = Number(process.env.EPOCHS ?? 30);
const ALPHA = Number(process.env.ALPHA ?? 0.003);
const OPTIMIZER = process.env.OPTIMIZER ?? "adam";
const CLIP_GRADIENT = Number(process.env.CLIP_GRADIENT ?? 5);
const VOCAB_SIZE = Number(process.env.VOCAB_SIZE ?? 128);
const MIN_FREQUENCY = Number(process.env.MIN_FREQUENCY ?? 2);
const MAX_TURN_TOKENS = Number(process.env.MAX_TURN_TOKENS ?? 20);
const MAX_RESPONSE_TOKENS = Number(process.env.MAX_RESPONSE_TOKENS ?? 24);
const EMBEDDING_DIM = Number(process.env.EMBEDDING_DIM ?? 96);
const MEMORY_SLOTS = Number(process.env.MEMORY_SLOTS ?? 24);
const MEMORY_MODE = (process.env.MEMORY_MODE ?? "project") as "project" | "concat";
const TRAIN_LIMIT = Number(process.env.TRAIN_LIMIT ?? 0);
const VALIDATION_LIMIT = Number(process.env.VALIDATION_LIMIT ?? 0);
const TEST_LIMIT = Number(process.env.TEST_LIMIT ?? 0);

type SplitMetrics = {
  avgLoss: number;
  exactMatch: number;
  tokenAccuracy: number;
  totalSamples: number;
};

function takeLimit<T>(arr: T[], limit: number): T[] {
  if (!limit || limit <= 0) return arr;
  return arr.slice(0, Math.min(limit, arr.length));
}

function shuffleInPlace<T>(arr: T[]): void {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
}

function decodePrediction(logits: ReturnType<DialogueMemoryModel["forwardTurn"]>["logits"]): number[] {
  return logits.map((matrix) => argmaxFromMatrix(matrix));
}

function runEpoch(
  model: DialogueMemoryModel,
  episodes: DialogueEpisode[],
  tokenizer: ReturnType<typeof buildTokenizer>,
  training: boolean
): SplitMetrics {
  let totalLoss = 0;
  let totalSamples = 0;
  let exactMatches = 0;
  let correctTokens = 0;
  let totalTokens = 0;

  if (training) model.train();
  else model.eval();

  for (const episode of episodes) {
    model.resetMemory();

    for (let i = 0; i < episode.turns.length; i++) {
      const turn = episode.turns[i]!;

      if (turn.role === "system") {
        if (containsResetMarker(turn.text)) model.resetMemory();
        continue;
      }

      if (turn.role !== "user") continue;

      const assistantTurn = getNextAssistantTurn(episode.turns, i);
      if (!assistantTurn) continue;

      const encodedTurn = encodeTurn(tokenizer, turn.text, MAX_TURN_TOKENS);
      const targetIds = encodeResponseTarget(tokenizer, assistantTurn.text, MAX_RESPONSE_TOKENS);
      const isQuestion = looksLikeQuestion(turn.text);

      if (isQuestion) model.freezeWrites();
      else model.enableWrites();

      const { logits } = model.forwardTurn(encodedTurn);
      const predictedIds = decodePrediction(logits);
      const predictedText = decodeResponse(tokenizer, predictedIds);
      const targetText = decodeResponse(tokenizer, targetIds);

      if (normalizeText(predictedText) === normalizeText(targetText)) {
        exactMatches++;
      }

      for (let pos = 0; pos < targetIds.length; pos++) {
        if (predictedIds[pos] === targetIds[pos]) correctTokens++;
        totalTokens++;
      }

      if (training) {
        const { err, avgLoss } = model.decoder.backward(targetIds);
        totalLoss += avgLoss;
        model.backwardResponse(err, mj.matrix([[0]]));
      }

      totalSamples++;
    }
  }

  return {
    avgLoss: totalSamples > 0 ? totalLoss / totalSamples : 0,
    exactMatch: totalSamples > 0 ? exactMatches / totalSamples : 0,
    tokenAccuracy: totalTokens > 0 ? correctTokens / totalTokens : 0,
    totalSamples,
  };
}

function main(): void {
  const trainEpisodes = takeLimit(loadDataset(TRAIN_PATH).episodes, TRAIN_LIMIT);
  const validationEpisodes = takeLimit(loadDataset(VALIDATION_PATH).episodes, VALIDATION_LIMIT);
  const testEpisodes = takeLimit(loadDataset(TEST_PATH).episodes, TEST_LIMIT);

  const corpus = collectCorpus(trainEpisodes);

  const tokenizer = buildTokenizer(corpus, VOCAB_SIZE, MIN_FREQUENCY);
  const vocabCapacity = tokenizer.getVocabularyCapacity();
  const padTokenId = tokenizer.getPadId();

  const model = createModel({
    vocabSize: vocabCapacity,
    maxTurnTokens: MAX_TURN_TOKENS,
    maxResponseTokens: MAX_RESPONSE_TOKENS,
    embeddingDim: EMBEDDING_DIM,
    memorySlots: MEMORY_SLOTS,
    memoryMode: MEMORY_MODE,
    alpha: ALPHA,
    optimizer: OPTIMIZER,
    clipGradient: CLIP_GRADIENT,
    padTokenId,
  });

  model.compile({
    alpha: ALPHA,
    optimizer: OPTIMIZER,
    clipGradient: CLIP_GRADIENT,
  });

  let bestValidation = Number.NEGATIVE_INFINITY;
  let bestArtifacts: DialogueExperimentArtifacts | null = null;

  console.log(`[dialogue-train] vocabCapacity=${vocabCapacity} tokenizer=${TOKENIZER_PATH}`);
  console.log(`[dialogue-train] train=${trainEpisodes.length} val=${validationEpisodes.length} test=${testEpisodes.length}`);
  console.log(`[dialogue-train] artifactDir=${ARTIFACT_DIR}`);

  for (let epoch = 1; epoch <= EPOCHS; epoch++) {
    shuffleInPlace(trainEpisodes);
    const trainMetrics = runEpoch(model, trainEpisodes, tokenizer, true);
    const validationMetrics = runEpoch(model, validationEpisodes, tokenizer, false);

    console.log(
      [
        `epoch=${epoch}/${EPOCHS}`,
        `trainLoss=${trainMetrics.avgLoss.toFixed(4)}`,
        `trainExact=${(trainMetrics.exactMatch * 100).toFixed(2)}%`,
        `trainTok=${(trainMetrics.tokenAccuracy * 100).toFixed(2)}%`,
        `valExact=${(validationMetrics.exactMatch * 100).toFixed(2)}%`,
        `valTok=${(validationMetrics.tokenAccuracy * 100).toFixed(2)}%`,
      ].join(" | ")
    );

    if (validationMetrics.exactMatch >= bestValidation) {
      bestValidation = validationMetrics.exactMatch;
      const testMetrics = runEpoch(model, testEpisodes, tokenizer, false);
      bestArtifacts = {
        format: "memorybank-dialogue-experiment-v1",
        createdAt: new Date().toISOString(),
        config: {
          vocabSize: VOCAB_SIZE,
          minFrequency: MIN_FREQUENCY,
          maxTurnTokens: MAX_TURN_TOKENS,
          maxResponseTokens: MAX_RESPONSE_TOKENS,
          embeddingDim: EMBEDDING_DIM,
          memorySlots: MEMORY_SLOTS,
          memoryMode: MEMORY_MODE,
          optimizer: OPTIMIZER,
          alpha: ALPHA,
          clipGradient: CLIP_GRADIENT,
          preTokenizer: "unicode-grapheme",
        },
        metrics: {
          trainExactMatch: trainMetrics.exactMatch,
          validationExactMatch: validationMetrics.exactMatch,
          testExactMatch: testMetrics.exactMatch,
          trainTokenAccuracy: trainMetrics.tokenAccuracy,
          validationTokenAccuracy: validationMetrics.tokenAccuracy,
          testTokenAccuracy: testMetrics.tokenAccuracy,
        },
        layers: {
          embedding: model.embedding.save(),
          pooling: model.pooling.save(),
          memory: model.memory.save(),
          decoder: model.decoder.save(),
        },
      };
      saveArtifacts(bestArtifacts, tokenizer);
      console.log(
        `[dialogue-train] saved best checkpoint | valExact=${(validationMetrics.exactMatch * 100).toFixed(2)}% | testExact=${(testMetrics.exactMatch * 100).toFixed(2)}%`
      );
    }
  }

  if (!bestArtifacts) {
    throw new Error("Training selesai tanpa checkpoint.");
  }

  console.log("[dialogue-train] selesai.");
  console.log(JSON.stringify(bestArtifacts.metrics, null, 2));
}

main();
