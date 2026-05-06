/**
 * @oxidejs/core: Foundation of OxideJS
 */

// === Matrix & Math ===
export { default as Matrix } from "./matrix";
export { default as mj } from "./math";

// === Activation Functions ===
export {
  default as linear,
  sigmoid,
  tanh,
  relu,
  lRelu,
  softmax,
  softmaxOnly,
  softmaxInto,
  softmaxBackward,
  softmaxBackwardInto,
  softmaxGradient,
} from "./activation";

// === Cost Functions ===
export {
  MeanSquerError,
  CategoricalCrossEntropy,
  BinaryCrossEntropy,
  SoftmaxCrossEntropy,
} from "./cost";

export {
  SGD,
  Adam,
  NAG,
  AdaGrad,
  Momentum,
} from "./optimizer";

// === Tokenizer ===
export {
  BPETokenizer,
  charPreTokenizer,
  unicodeGraphemePreTokenizer,
  unicodeWordPreTokenizer,
  whitespacePreTokenizer,
  scriptAwarePreTokenizer,
} from "./tokenizer";
export type { BPEConfig, BPETrainingEncodeOptions, BPEVocabData, BPETokenizerOptions, BuiltInPreTokenizer, PreTokenizer } from "./tokenizer";

// === Utils ===
export {
  setActivation,
  setLoss,
  setOptimizer,
  cosineSimilarity,
  shuffleInPlace,
  splitTrainValidation,
  formatLoss,
  formatProgressBar,
  formatTime,
  trimPaddingBatch,
} from "./utils";

// === Native Backend ===
export * from "./math/rust_backend";

// === Types ===
export * from "./@types/type";
export * from "./@types/fitConfig";
