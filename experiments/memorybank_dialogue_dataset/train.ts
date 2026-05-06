import {
  ARTIFACT_DIR,
  DialogueEpisode,
  DialogueExperimentArtifacts,
  DialogueMemoryModel,
  TEST_PATH,
  TOKENIZER_PATH,
  TRAIN_PATH,
  VALIDATION_PATH,
  buildTokenizer,
  collectCorpus,
  concatenateTokenIds,
  containsResetMarker,
  createModel,
  decodeResponse,
  encodeResponseTarget,
  encodeTurn,
  encodeTurnForTraining,
  getNextAssistantTurn,
  loadDataset,
  looksLikeQuestion,
  normalizeText,
  saveArtifacts,
  sliceColumns,
  tokenIdsToColumnMatrix,
  copyColumn,
} from "./shared";
import Matrix from "../../src/matrix";
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
const DECODER_HIDDEN_UNITS = Number(process.env.DECODER_HIDDEN_UNITS ?? EMBEDDING_DIM);
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

type EpisodeSample = {
  encodedTurn: ReturnType<typeof encodeTurn>;
  targetIds: number[];
  targetText: string;
  isQuestion: boolean;
  memoryStepIndex: number;
  context: Matrix;
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

function createEmptyMatrix(rows: number, cols: number): Matrix {
  return mj.zeros([rows, cols]);
}

function prepareEpisodeSamples(
  episode: DialogueEpisode,
  tokenizer: ReturnType<typeof buildTokenizer>,
  training: boolean
): EpisodeSample[] {
  const samples: EpisodeSample[] = [];

  for (let i = 0; i < episode.turns.length; i++) {
    const turn = episode.turns[i]!;
    if (turn.role !== "user") continue;

    const assistantTurn = getNextAssistantTurn(episode.turns, i);
    if (!assistantTurn) continue;

    const encodedTurn = training
      ? encodeTurnForTraining(tokenizer, turn.text, MAX_TURN_TOKENS)
      : encodeTurn(tokenizer, turn.text, MAX_TURN_TOKENS);
    const targetIds = encodeResponseTarget(tokenizer, assistantTurn.text, MAX_RESPONSE_TOKENS);

    samples.push({
      encodedTurn,
      targetIds,
      targetText: assistantTurn.text,
      isQuestion: looksLikeQuestion(turn.text),
      memoryStepIndex: -1,
      context: mj.matrix([]),
    });
  }

  return samples;
}

function buildContextBatch(samples: EpisodeSample[]): Matrix {
  const rows = samples[0]!.context._shape[0];
  const batch = createEmptyMatrix(rows, samples.length);
  for (let i = 0; i < samples.length; i++) {
    copyColumn(samples[i]!.context, 0, batch, i);
  }
  return batch;
}

function runEpisode(
  model: DialogueMemoryModel,
  episode: DialogueEpisode,
  tokenizer: ReturnType<typeof buildTokenizer>,
  training: boolean,
  bosId: number,
  eosId: number,
  padId: number
): { totalLoss: number; totalSamples: number; exactMatches: number; correctTokens: number; totalTokens: number } {
  const samples = prepareEpisodeSamples(episode, tokenizer, training);
  if (samples.length === 0) {
    return { totalLoss: 0, totalSamples: 0, exactMatches: 0, correctTokens: 0, totalTokens: 0 };
  }

  const concatenated = concatenateTokenIds(samples.map((sample) => sample.encodedTurn));
  const encoderInput = tokenIdsToColumnMatrix(concatenated.ids);
  const encoderEmbeddings = model.embedding.forward(encoderInput);

  model.resetMemory();
  if (episode.turns.some((turn) => turn.role === "system" && containsResetMarker(turn.text))) {
    model.resetMemory();
  }

  if (training) {
    model.memory.beginSequence();
  }

  let exactMatches = 0;
  let correctTokens = 0;
  let totalTokens = 0;

  for (let i = 0; i < samples.length; i++) {
    const sample = samples[i]!;
    const range = concatenated.ranges[i]!;
    const turnEmb = sliceColumns(encoderEmbeddings, range.start, range.length);

    if (sample.isQuestion) model.freezeWrites();
    else model.enableWrites();

    const contextSequence = model.memory.forward(turnEmb);
    sample.context = Matrix.fromFlat(contextSequence.getCol(range.length - 1), [contextSequence._shape[0], 1]);
    sample.memoryStepIndex = range.start + range.length - 1;

    const predictedIds = model.decodeGreedy(sample.context, MAX_RESPONSE_TOKENS, bosId, eosId, padId);
    const predictedText = decodeResponse(tokenizer, predictedIds);
    if (normalizeText(predictedText) === normalizeText(sample.targetText)) {
      exactMatches++;
    }

    const paddedPrediction = tokenizer.padSequence([...predictedIds, eosId], MAX_RESPONSE_TOKENS);
    for (let t = 0; t < MAX_RESPONSE_TOKENS; t++) {
      if (sample.targetIds[t] === padId) continue;
      if (paddedPrediction[t] === sample.targetIds[t]) correctTokens++;
      totalTokens++;
    }
  }

  let totalLoss = 0;
  if (training) {
    const contextBatch = buildContextBatch(samples);
    const decoderResult = model.trainEpisodeDecoder(
      contextBatch,
      samples.map((sample) => sample.targetIds),
      MAX_RESPONSE_TOKENS,
      bosId,
      padId
    );
    totalLoss = decoderResult.avgLoss * samples.length;

    const memoryErr = createEmptyMatrix(model.memory.outputUnits, concatenated.ids.length);
    for (let i = 0; i < samples.length; i++) {
      const sample = samples[i]!;
      copyColumn(decoderResult.errContexts, i, memoryErr, sample.memoryStepIndex);
    }
    const encoderErr = model.memory.backwardSequence(memoryErr);
    model.embedding.forward(encoderInput);
    model.embedding.backward(mj.matrix([]), encoderErr);
    model.memory.endSequence();
  }

  return {
    totalLoss,
    totalSamples: samples.length,
    exactMatches,
    correctTokens,
    totalTokens,
  };
}

function runEpoch(
  model: DialogueMemoryModel,
  episodes: DialogueEpisode[],
  tokenizer: ReturnType<typeof buildTokenizer>,
  training: boolean,
  bosId: number,
  eosId: number,
  padId: number
): SplitMetrics {
  let totalLoss = 0;
  let totalSamples = 0;
  let exactMatches = 0;
  let correctTokens = 0;
  let totalTokens = 0;

  if (training) model.train();
  else model.eval();

  for (const episode of episodes) {
    const metrics = runEpisode(model, episode, tokenizer, training, bosId, eosId, padId);
    totalLoss += metrics.totalLoss;
    totalSamples += metrics.totalSamples;
    exactMatches += metrics.exactMatches;
    correctTokens += metrics.correctTokens;
    totalTokens += metrics.totalTokens;
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
  const bosId = tokenizer.getTokenId("<BOS>");
  const eosId = tokenizer.getTokenId("<EOS>");

  if (bosId === undefined || eosId === undefined) {
    throw new Error("Tokenizer wajib memiliki token <BOS> dan <EOS>.");
  }

  const model = createModel({
    vocabSize: vocabCapacity,
    maxTurnTokens: MAX_TURN_TOKENS,
    maxResponseTokens: MAX_RESPONSE_TOKENS,
    embeddingDim: EMBEDDING_DIM,
    decoderHiddenUnits: DECODER_HIDDEN_UNITS,
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
    const trainMetrics = runEpoch(model, trainEpisodes, tokenizer, true, bosId, eosId, padTokenId);
    const validationMetrics = runEpoch(model, validationEpisodes, tokenizer, false, bosId, eosId, padTokenId);

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
      const testMetrics = runEpoch(model, testEpisodes, tokenizer, false, bosId, eosId, padTokenId);
      bestArtifacts = {
        format: "memorybank-dialogue-experiment-v3-single-embedding",
        createdAt: new Date().toISOString(),
        config: {
          vocabSize: VOCAB_SIZE,
          minFrequency: MIN_FREQUENCY,
          maxTurnTokens: MAX_TURN_TOKENS,
          maxResponseTokens: MAX_RESPONSE_TOKENS,
          embeddingDim: EMBEDDING_DIM,
          decoderHiddenUnits: DECODER_HIDDEN_UNITS,
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
          memory: model.memory.save(),
          contextProject: model.contextProject.save(),
          decoderLstm: model.decoderLstm.save(),
          decoderOutput: model.decoderOutput.save(),
        },
      };
      saveArtifacts(bestArtifacts, tokenizer);
      console.log(
        `[dialogue-train] saved best checkpoint | valExact=${(validationMetrics.exactMatch * 100).toFixed(2)}% | testExact=${(testMetrics.exactMatch * 100).toFixed(2)}%`
      );
    }
  }

  if (bestArtifacts) {
    console.log("[dialogue-train] selesai.");
    console.log(JSON.stringify(bestArtifacts.metrics, null, 2));
  }
}

main();
