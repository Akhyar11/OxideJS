# OxideJS (formerly ML-V1)

> **Historical Note:** This project was originally published as **ML-V1**. Version [v2.3.0](https://github.com/Akhyar11/ML-V1/releases/tag/v2.3.0) represents the stable research artifact for recurrent network stability evaluations.

> A TypeScript + Rust Native machine learning library — Matrix operations, neural network layers, Transformer models, and a BPE tokenizer, all in one package.

[![npm version](https://img.shields.io/npm/v/@akhyar11/oxidejs?style=flat-square)](https://www.npmjs.com/package/@akhyar11/oxidejs)
[![License: ISC](https://img.shields.io/badge/License-ISC-blue?style=flat-square)](https://opensource.org/licenses/ISC)
[![Node.js >=18](https://img.shields.io/badge/node-%3E%3D18-brightgreen?style=flat-square)](https://nodejs.org)

---

## What is OxideJS?

**OxideJS** is a low-to-mid-level machine learning library built with TypeScript and accelerated by a Rust native backend (via [napi-rs](https://napi.rs/)). It gives you full control over every detail of the training loop — shapes, parameter updates, and custom architectures — without depending on a large ML framework.

**Why OxideJS?**
- Full manual control over training loops, tensor shapes, and parameter updates.
- A research playground for custom model architectures.
- The productivity of TypeScript combined with Rust performance on hot paths.
- Graceful fallback to pure JavaScript when the native backend is unavailable.

---

## Features

- **`Matrix`** — flat `Float32Array`-backed tensor with zero-copy hot-path access via `_data`.
- **Math primitives** — `dotProduct`, `add`, `sub`, `sumAxis`, `clipGradients`, and more; automatically dispatched to Rust or JS.
- **Layers** — `Dense`, `Embedding`, `RNN`, `LSTM`, `GRU`, `SelfAttention`, `MultiHeadAttention`, `LayerNormalization`, `Dropout`, `PositionalEncoding`, `Flatten`, `Convolution`, `MemoryBank`.
- **Models** — `Sequential`, `Transformers` (causal LM), `RecurrentModel` (RNN/LSTM/GRU many-to-one and aligned many-to-many), `DimentionalityReduction`.
- **BPE Tokenizer** — train, incremental update, Unicode-aware pre-tokenization, encode/decode with special tokens, padding, and JSON save/load.
- **Rust-accelerated ops** — dot-product, activations, LayerNorm, embedding lookup, attention, and optimizer updates; auto-fallback to JS when unavailable.
- **Dynamic padding trim** (`trimPadding`) — reduces effective sequence length per batch, cutting attention cost from O(seqLen²) to O(effectiveSeqLen²).

---

## Installation

```bash
npm install @akhyar11/oxidejs
```

### Prerequisites for Native Acceleration

The library works out of the box with a pure JavaScript fallback. For up to **10× faster** matrix operations, install the Rust toolchain so the native addon can be compiled automatically on `npm install`:

1. **Rust Toolchain** — install via [rustup.rs](https://rustup.rs/).
2. **C/C++ Build Tools** — required by the native binding compiler (e.g. `build-essential` on Linux, Xcode CLI on macOS, MSVC on Windows).

> **Note:** If Rust is not installed, a warning is printed and the library falls back to pure JavaScript automatically. Performance will be noticeably slower for large models.

---

## Building from Source

OxideJS uses a monorepo structure with NPM workspaces. To build everything:

```bash
# Install dependencies for all packages
npm install

# Build the native Rust kernels for @oxidejs/core
npm run build:rust

# Build all TypeScript packages (core, layers, models)
npm run build
```

---

## Rust Native Backend

The native backend is loaded by `src/math/rust_backend.ts`. You can check whether it is active at runtime:

```ts
import { isNativeAvailable } from "@akhyar11/oxidejs";
console.log("Native active:", isNativeAvailable());
```

To force JavaScript-only execution (useful for debugging or regression comparisons):

```bash
ML_DISABLE_NATIVE=1 node your-script.js
```

---

## Quick Start

Train a simple XOR classifier in a few lines:

```ts
import { Dense, mj, Sequential } from "@akhyar11/oxidejs";

const model = new Sequential({
  layers: [
    new Dense({ units: 2, outputUnits: 4, activation: "relu", status: "input" }),
    new Dense({ units: 4, outputUnits: 1, activation: "sigmoid", status: "output", loss: "mse" }),
  ],
});

model.compile({ alpha: 0.01, optimizer: "adam", error: "mse" });

const X = [mj.matrix([[0], [0]]), mj.matrix([[0], [1]]), mj.matrix([[1], [0]]), mj.matrix([[1], [1]])];
const Y = [mj.matrix([[0]]), mj.matrix([[1]]), mj.matrix([[1]]), mj.matrix([[0]])];

const result = model.fit(X, Y, 200, {
  batchSize: 4,
  validationSplit: 0.25,
  earlyStoppingPatience: 10,
  verbose: true,
  onEpochEnd: (epoch, loss, valLoss) => {
    console.log(`epoch=${epoch} loss=${loss} valLoss=${valLoss}`);
  },
});
console.log("best", result.bestEpoch, result.bestLoss);
const pred = model.predict(mj.matrix([[1], [0]]));
pred.print();
```

A legacy callback overload is also supported for backward compatibility:

```ts
model.fit(X, Y, 200, (loss) => console.log("loss", loss));
```

Model split:
- `Sequential` is for generic feed-forward / per-sample supervised learning.
- `Transformers` has its own token-weighted `fit()` for causal LM.
- `RecurrentModel` is the high-level entry point for stacked `RNN` / `LSTM` / `GRU` sequence training, including `many-to-one` and aligned `many-to-many`.
  - For many-to-one, `pooling: "last" | "mean" | "max"` is available. `"mean"` / `"max"` support PAD masking via `padTokenId` when an `Embedding` layer is used.
  - `LSTM` initializes the forget gate bias to `1` by default on new layers.
  - `AdaptiveMemoryRNN` trains `Wxh`, `Whh`, `bh`, `Wq`, `Wm`, `Wg`, and `bg`; memory slots are treated as dynamic state rather than optimizer-trained global parameters, and the native backward path matches the JS reference.

---

## Examples

### Matrix & Math Operations

```ts
import { mj } from "@akhyar11/oxidejs";

const a = mj.matrix([[1, 2], [3, 4]]);
const b = mj.matrix([[5, 6], [7, 8]]);
const c = mj.dotProduct(a, b);
const d = mj.add(c, 1);
console.log(c._shape, d._shape);
```

### BPE Tokenizer

```ts
import { BPETokenizer } from "@akhyar11/oxidejs";

const tokenizer = new BPETokenizer({ vocabSize: 120, minFrequency: 2 });
tokenizer.train(["hello world", "hello there"]);
const ids = tokenizer.encodeWithSpecial("hello world");
const padded = tokenizer.padSequence(ids, 12);
console.log(ids, padded, tokenizer.decode(ids));
```

### Unicode and Multilingual Tokenization

OxideJS supports custom and built-in pre-tokenizers for non-Latin text. The default is still `"char"` for backward compatibility; use `"unicode-grapheme"` or `"script-aware"` for multilingual corpora.

Supported modes:
- `char`
- `unicode-grapheme`
- `unicode-word`
- `whitespace`
- `script-aware`

```ts
import { BPETokenizer } from "@akhyar11/oxidejs";

const tokenizer = new BPETokenizer({
  vocabSize: 1000,
  preTokenizer: "script-aware"
});

tokenizer.train([
  "hello world",
  "مرحبا بالعالم",
  "こんにちは世界",
  "你好世界",
  "ภาษาไทย",
  "한국어테스트",
  "ꦱꦺꦴꦥꦺꦴ",
  "x² + y² = z²",
  "hello ꦱꦺꦴꦥꦺꦴ 😊 你好"
]);
```

BPE alone is not enough for every writing system. Pre-tokenization is important for scripts without spaces, combining marks, emoji sequences, and mixed text. `script-aware` is a general built-in mode; for language-specific behavior, pass a custom `(text: string) => string[]` pre-tokenizer. `Intl.Segmenter` improves grapheme and word segmentation when the runtime supports it. Fallback behavior is deterministic but may be less linguistically accurate.

### Transformer Causal LM — Training

```ts
import { mj, Transformers } from "@akhyar11/oxidejs";

const model = new Transformers({ units: 64, seqLen: 8, vocabSize: 500, heads: 8, alpha: 0.001, padTokenId: 0 });
model.fillEmbeddingWeight("./pretrained-embedding.json");
model.compile({ alpha: 0.001, optimizer: "adam", error: "softmaxCrossEntropy" });
model.train();

const x = mj.matrix([[0], [0], [10], [20], [30], [40], [50], [60]]); // shape [seqLen, 1]
const y = mj.matrix([[0], [10], [20], [30], [40], [50], [60], [0]]); // shifted targets [seqLen, 1]

const logits = model.forward(x); // shape [vocabSize, seqLen * batch]
model.backward(y);
console.log("shape", logits._shape, "loss", model.loss);
```

### Transformer — Generation / Inference

```ts
import { mj, Transformers } from "@akhyar11/oxidejs";

const model = new Transformers({
  units: 64,
  seqLen: 8,
  vocabSize: 500,
  heads: 8,
  alpha: 0.001,
  padTokenId: 0,
  predictMode: "next-token",
});
model.eval();

const x = mj.matrix([[0], [0], [10], [20], [30], [40], [50], [60]]);

### RecurrentModel — Many-to-One

```ts
import { RecurrentModel } from "@akhyar11/oxidejs";

const model = new RecurrentModel({
  kind: "lstm",
  vocabSize: 1000,
  embeddingDim: 32,
  embeddingTrainable: false,
  hiddenSizes: [64, 64],
  outputSize: 5,
  seqLen: 20,
  mode: "many-to-one",
  loss: "softmaxCrossEntropy",
});
```
const nextTokenLogits = model.predict(x); // shape [vocabSize, batch]
model.setPredictMode("full-sequence");
const fullSequenceLogits = model.predict(x); // shape [vocabSize, seqLen * batch]
```

---

## API Overview

### Models

| Model | Description |
|---|---|
| `Sequential` | Generic layer stack (Dense, Embedding, Attention, CNN, etc.). |
| `Transformers` | Multi-block causal language model. Supports `numBlocks >= 1`, full-sequence training, and configurable `predictMode` (`"next-token"` / `"full-sequence"`). |
| `DimentionalityReduction` | Extends `Sequential` with an encoder/decoder split via the `outputReduction` layer status. |

### Layers

| Layer | Description |
|---|---|
| `Dense` | Fully-connected layer with activation, optimizer, and loss handling. |
| `Embedding` | Token-ID-to-vector lookup with `resize()`, `trainable`, and pretrained `fillWeight()` support. |
| `LayerNormalization` | Per-column/token normalization. |
| `Dropout` | Active only during training mode. |
| `PositionalEncoding` | Fixed sinusoidal positional encoding. |
| `MultiHeadAttention` / `SelfAttention` | Causal attention mask with padding support. |
| `RNN` / `LSTM` / `GRU` | Recurrent sequence modeling with BPTT, gradient clipping, save/load, and stateful mode. `returnSequences` is supported; `returnState` is not yet supported and will throw explicitly. |
| `Flatten` / `Convolution` | Standard CNN building blocks. |

### Tokenizer

`BPETokenizer` supports:

| Method | Description |
|---|---|
| `train(corpus)` | Initial BPE training on a string array. |
| `update(corpus)` | Incremental vocabulary update without retraining from scratch. |
| `encode(text)` / `encodeWithSpecial(text)` | Encode text to token IDs, with or without special tokens. |
| `decode(ids)` | Convert token IDs back to text. |
| `padSequence(ids, length)` | Pad or truncate a sequence to a fixed length. |
| `save(path)` / `load(path)` | Persist and restore the tokenizer as a JSON file. |

Tokenizer options:

```ts
type PreTokenizer = (text: string) => string[];

type BuiltInPreTokenizer =
  | "char"
  | "unicode-grapheme"
  | "unicode-word"
  | "whitespace"
  | "script-aware";

type BPETokenizerOptions = {
  vocabSize?: number;
  minFrequency?: number;
  preTokenizer?: BuiltInPreTokenizer | PreTokenizer;
};
```

Built-in pre-tokenizer names are saved in tokenizer JSON files. Custom pre-tokenizer functions are not serialized; saved metadata records `"custom"`, and the same function must be passed again to `BPETokenizer.load(path, { preTokenizer })`.

---

## Core Concepts

- **Shape convention:** most layers use `[rows, cols]`; batched Transformer inputs use column-sequence layout `[seqLen, batchSize]`.
- **Recurrent convention:** recurrent layers expect a single sequence sample with shape `[features, seqLen]`. The generic `Sequential.fit()` does not batch recurrent sequences yet — use `batchSize: 1`.
- **Sparse classification targets:** use `softmaxCrossEntropy` with a dense output layer and a target of shape `[1, batch]` containing class indices.
- **Training / eval mode:** call `model.train()` before training and `model.eval()` before inference. Layers like `Dropout` respect this flag.

---

## Training Workflow

1. Prepare data as `Matrix` inputs and targets.
2. Build your model and add layers.
3. Call `model.compile({ alpha, optimizer, error })`.
4. Run `model.fit()` (high-level) or loop `forward()` → `backward()` manually.
5. Save the model and tokenizer with `save()`.

## Inference Workflow

1. Load the model and tokenizer.
2. Convert input text to token IDs and pad to `seqLen`.
3. Call `model.predict()` (respects `predictMode` for `Transformers`) or `model.forward()`.
4. Extract the argmax or raw logits as required by your task.
5. Decode token IDs back to text for NLP tasks.

---

## Performance Notes

- The Rust backend accelerates dot-product, activations, LayerNorm, embedding lookup, attention, and optimizer hot paths.
- `Matrix` uses `Float32Array` to minimize allocation overhead. Use `_data` directly in hot paths.
- Several layers use pre-allocated output buffers to reduce garbage collection pressure.
- **Dynamic padding trim** (`trimPadding: true`, the default) reduces `effectiveSeqLen` per batch, cutting attention cost from O(seqLen²) to O(effectiveSeqLen²) and output projection cost from `vocabSize × seqLen × batch` to `vocabSize × effectiveSeqLen × batch`.

---

## Dynamic Padding Trim (v2.2.0+)

When training a Transformer on long-context sequences (e.g. `seqLen=1024`), enable `trimPadding` to avoid paying the full quadratic attention cost on padding tokens:

```ts
import { Transformers } from "@akhyar11/oxidejs";

const model = new Transformers({
  units: 64,
  seqLen: 1024,
  vocabSize: 5000,
  heads: 8,
  numBlocks: 2,
  padTokenId: 0
});

// Right-padding (recommended for new datasets)
model.fit(trainX, trainY, 80, {
  batchSize: 8,
  trimPadding: true,
  paddingSide: "right",
  shuffle: true
});

// Left-padding (for datasets already padded on the left)
model.fit(trainX, trainY, 80, {
  batchSize: 8,
  trimPadding: true,
  paddingSide: "left",
  shuffle: true
});
```

**Options:**
- `trimPadding: true` *(default)* — enabled automatically.
- `paddingSide: "right"` *(default)* — trailing PAD tokens are trimmed; `positionOffset` is 0.
- `paddingSide: "left"` — leading PAD tokens are trimmed; `positionOffset` is adjusted so that positional encodings for real tokens remain unchanged.
- `trimPadding: false` — disables the feature entirely.
- Only applies to full-sequence targets with shape `[seqLen, batch]`. Legacy targets with shape `[1, batch]` are not trimmed.

---

## Best Practices

- Use `softmaxCrossEntropy` for sparse token classification tasks.
- Keep `seqLen` consistent between your preprocessing pipeline and the model constructor.
- Set `padTokenId` in both the tokenizer and the model's `Embedding` layer.
- For `Transformers`, prepare shifted next-token targets with shape `[seqLen, batch]` and fill invalid positions with `padTokenId`.
- Call `model.train()` before training and `model.eval()` before inference.
- For Transformer inference, use `model.predict()` as the primary entry point and set `predictMode` to `"next-token"` or `"full-sequence"` as needed.
- For stateful recurrent models, avoid `shuffle: true` and `validationSplit > 0` in the generic `Sequential.fit()` loop.
- Start debugging with `ML_DISABLE_NATIVE=1` when comparing JS vs. native behavior.
- If loss does not decrease, verify tensor shapes at every layer boundary.

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `Native backend not available` | Run `npm run build:rust`, or verify that the `.node` binary matches your current platform. |
| Shape mismatch in dot product | Check that dimensions satisfy `[aRows × aCols] · [bRows × bCols]` where `aCols === bRows`. |
| Loss is `NaN` or `Inf` | Reduce the learning rate `alpha`, verify target format, and check for out-of-range token IDs in the embedding. |

---

## Project Structure

```text
packages/
  core/              ← Matrix operations & Rust native acceleration
    src/             ← TypeScript wrappers
    src-rust/        ← Rust native ops (napi-rs)
  layers/            ← Neural network layer implementations
    src/
  models/            ← High-level model compositions
    src/
test/                ← E2E Correctness & Benchmark suite
dataset/             ← Pre-processed datasets for research
docs/                ← In-depth guides and API reference
```

---

## Architecture Overview

| Package | Role |
|---|---|
| `@oxidejs/core` | Core `Matrix` data structure, numeric primitives, and modular Rust backend. |
| `@oxidejs/layers` | Reusable NN layers (Dense, Attention, RNN, etc.) with native acceleration support. |
| `@oxidejs/models` | High-level compositions like `Sequential` and `Transformers`. |
| `oxide-native` | NAPI-RS binding linking TypeScript to the optimized Rust kernels. |

---

## Benchmark & Testing

- Full test + benchmark entry point: [`test/index.ts`](./test/index.ts) — run with `npm test`.
- Correctness suite: [`test/correctness/index.ts`](./test/correctness/index.ts).
- Synthetic benchmark suite: [`test/benchmark/index.ts`](./test/benchmark/index.ts).
- Recurrent model benchmarks: [`test/benchmark/testFamilyRnn.test.ts`](./test/benchmark/testFamilyRnn.test.ts).
- Transformer mode benchmarks: [`test/benchmark/testFamilyTransformers.test.ts`](./test/benchmark/testFamilyTransformers.test.ts).
- Benchmark history: [`docs/benchmark-sintetis/README.md`](./docs/benchmark-sintetis/README.md).
- Correctness history: [`docs/correctness/README.md`](./docs/correctness/README.md).

---

## 📖 Documentation

For in-depth guides, see the official documentation:

1. **[Overview & Philosophy](docs/GUIDE-LINE/01-overview.md)** — Introduction to the library design and system architecture.
2. **[Installation & Setup](docs/GUIDE-LINE/02-installation.md)** — How to install and enable Rust native acceleration.
3. **[Practical Tutorial](docs/GUIDE-LINE/03-tutorial.md)** — Step-by-step guide to building a logic bot and a generative (GPT-style) bot.
4. **[Full API Reference](docs/api/README.md)** — Technical documentation for Matrix, Math, Layers, Tokenizer, Optimizers, and related APIs.

---

## Versioning

This project follows `MAJOR.MINOR.PATCH` semantic versioning. The current version is **`2.3.1`**.

- **MAJOR** — breaking changes or major architectural shifts.
- **MINOR** — new backward-compatible features or improvements.
- **PATCH** — bug fixes, small optimizations, or minor internal changes.

**Recent changelog:**

| Version | Summary |
|---|---|
| `2.3.1` | **Modularization Milestone**: Monorepo split (`@oxidejs/core`, `@oxidejs/layers`, `@oxidejs/models`), Modular Rust kernels, and ESM-first test suite. |
| `2.3.0` | Initial Monorepo structure and decoupled Layer Registry. |
| `2.2.8` | Full Native Optimizer support (Adam, SGD, AdaGrad, Momentum, NAG) and Sparse Embedding native backend. |
| `2.2.7` | Unicode-aware BPE pre-tokenizers and multilingual tokenizer documentation. |
| `2.2.5` | Hot-path optimizations for training/validation, embedding lookup, and BPE tokenizer. |
| `2.2.4` | `Transformers.predictMode` API ergonomics, docs sync, and correctness suite refactor. |
| `2.2.3` | Training/inference hot-path optimizations and updated correctness learning snapshots. |
| `2.2.2` | Combined root suite, family model benchmarks, and correctness learning snapshots. |
| `2.2.0` | Dynamic padding trim + positional encoding offset. |
| `2.0.2` | Transformer projector optimizations with no API changes. |

---

## Development & Contributing

```bash
npm install          # install dependencies
npm run build:rust   # compile the Rust native addon
npm test             # run the correctness suite + synthetic benchmark
```

Type-check only (no emit):

```bash
npx tsc --noEmit
```

---

## Roadmap

- Stabilize public API entry points (currently imported directly from `src/*`).
- Add deterministic floating-point tests.
- Clean up scripts that reference non-existent project folders.
- Add dataset recipe documentation and benchmark workflow guides.

---

## License & Credits

- **License:** ISC — see [`package.json`](./package.json).
- **Native backend:** [`napi-rs`](https://napi.rs/), [`matrixmultiply`](https://crates.io/crates/matrixmultiply), [`rayon`](https://crates.io/crates/rayon).
- **Issues & feature requests:** use the [GitHub issue tracker](https://github.com/Akhyar11/OxideJS/issues).
- **Support the project:**
  [![Saweria](https://img.shields.io/badge/Saweria-Support-orange?style=for-the-badge&logo=saweria)](https://saweria.co/akhyaruhui)
