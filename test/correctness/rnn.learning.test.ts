import { Dense, GRU, LSTM, RNN } from "../../src/layers";
import mj from "../../src/math";
import Matrix from "../../src/matrix";
import { Sequential } from "../../src/models";

type RecurrentFamily = "rnn" | "lstm" | "gru";

type LearningResult = {
  family: RecurrentFamily;
  initialLoss: number;
  finalLoss: number;
  history: number[];
};

function assert(condition: boolean, message: string): void {
  if (!condition) {
    throw new Error(message);
  }
}

function buildSequence(bits: number[]): Matrix {
  const rows = [
    bits.map((bit) => (bit === 0 ? 1 : 0)),
    bits.map((bit) => (bit === 1 ? 1 : 0)),
  ];
  return mj.matrix(rows);
}

function createLearningDataset(): { X: Matrix[]; y: Matrix[] } {
  const patterns = [
    [0, 0, 0, 0],
    [0, 1, 0, 0],
    [1, 0, 1, 0],
    [1, 1, 0, 0],
    [0, 0, 1, 1],
    [0, 1, 1, 1],
    [1, 0, 0, 1],
    [1, 1, 1, 1],
  ];

  const X: Matrix[] = [];
  const y: Matrix[] = [];

  for (let repeat = 0; repeat < 8; repeat++) {
    for (const pattern of patterns) {
      X.push(buildSequence(pattern));
      y.push(mj.matrix([[pattern[pattern.length - 1]]]));
    }
  }

  return { X, y };
}

function createModel(family: RecurrentFamily): Sequential {
  const recurrentLayer =
    family === "rnn"
      ? new RNN({
          units: 2,
          hiddenUnits: 8,
          activation: "tanh",
          returnSequences: false,
          status: "input",
        })
      : family === "lstm"
        ? new LSTM({
            units: 2,
            hiddenUnits: 8,
            returnSequences: false,
            status: "input",
          })
        : new GRU({
            units: 2,
            hiddenUnits: 8,
            returnSequences: false,
            status: "input",
          });

  const model = new Sequential({
    layers: [
      recurrentLayer,
      new Dense({
        units: 8,
        outputUnits: 2,
        activation: "linear",
        status: "output",
        loss: "softmaxCrossEntropy",
      }),
    ],
  });

  model.compile({
    alpha: 0.01,
    optimizer: "adam",
    error: "softmaxCrossEntropy",
  });

  return model;
}

function runLearningTest(family: RecurrentFamily): LearningResult {
  const { X, y } = createLearningDataset();
  const model = createModel(family);
  const history: number[] = [];

  model.fit(X, y, 10, {
    batchSize: 1,
    shuffle: false,
    verbose: false,
    onEpochEnd: (_epoch, loss) => {
      history.push(loss);
    },
  });

  assert(history.length === 10, `${family}: expected 10 loss entries, got ${history.length}`);

  const initialLoss = history[0];
  const finalLoss = history[history.length - 1];
  const bestLoss = Math.min(...history);

  assert(Number.isFinite(initialLoss), `${family}: initial loss must be finite`);
  assert(Number.isFinite(finalLoss), `${family}: final loss must be finite`);
  assert(
    finalLoss < initialLoss,
    `${family}: expected final loss to be lower than initial loss (${finalLoss} >= ${initialLoss})`
  );
  assert(
    bestLoss <= initialLoss - 0.01,
    `${family}: expected at least one meaningful loss improvement, history=${history.map((v) => v.toFixed(6)).join(", ")}`
  );

  return {
    family,
    initialLoss,
    finalLoss,
    history,
  };
}

export function runRecurrentLearningCorrectnessSuite(): LearningResult[] {
  const families: RecurrentFamily[] = ["rnn", "lstm", "gru"];
  const results = families.map((family) => runLearningTest(family));

  console.log("=== Recurrent Learning Correctness ===");
  console.table(
    results.map((result) => ({
      family: result.family,
      initialLoss: Number(result.initialLoss.toFixed(6)),
      finalLoss: Number(result.finalLoss.toFixed(6)),
      improvement: Number((result.initialLoss - result.finalLoss).toFixed(6)),
    }))
  );

  return results;
}
