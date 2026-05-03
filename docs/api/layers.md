# Layers

Neural network layer implementations in ML-V1.

## Import

```ts
import {
  Dense,
  Convolution,
  Activation,
  CompileDenseLayers,
  SelfAttention,
  MultiHeadAttention,
  Embedding,
  Flatten,
  PositionalEncoding,
  LayerNormalization,
  Dropout,
  RNN,
  LSTM,
  GRU,
  AdaptiveMemoryRNN
} from "@akhyar11/ml-v1"
```

## Overview

All layers share a common interface: `forward(input)` for the forward pass and `backward(grad)` for the backward pass. Layers are composable inside `Sequential`, `Transformers`, or `DimentionalityReduction` models, or they can be used standalone in a manual training loop.

---

## API Reference

### `Dense`

Fully-connected layer where every input is connected to every output.

#### `constructor(config)`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `units` | `number` | — | Number of input neurons |
| `outputUnits` | `number` | — | Number of output neurons |
| `activation` | `ActivationType` | `"linear"` | Activation function name |
| `alpha` | `number` | — | Layer-specific learning rate |
| `loss` | `Cost` | — | Loss function name for the output layer |
| `optimizer` | `Optimzier` | — | Optimization algorithm name |
| `status` | `StatusLayer` | — | Layer role in the model graph |
| `clipGradient` | `number \| boolean` | `5.0` | Gradient clipping limit for this layer |

```ts
import { Dense } from "@akhyar11/ml-v1"

const layer = new Dense({
  units: 128,
  outputUnits: 64,
  activation: "relu",
  optimizer: "adam"
});
```

> [!IMPORTANT]
> Combining `activation: "softmax"` with `loss: "softmaxCrossEntropy"` will throw an error — softmax would be applied twice.

---

### `Embedding`

Transforms integer token IDs into dense vectors. Required for NLP tasks.

#### `constructor(config)`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `vocabSize` | `number` | — | Total vocabulary size |
| `embeddingDim` | `number` | — | Vector dimension per token |
| `alpha` | `number` | `0.01` | Layer-specific learning rate |
| `status` | `StatusLayer` | `"input"` | Layer role in the model graph |
| `optimizer` | `Optimzier` | `"adam"` | Optimizer for the embedding table |
| `padTokenId` | `number \| null` | `null` | PAD token ID to skip during backward pass |
| `trainable` | `boolean` | `true` | Freeze weight updates when set to `false` |

```ts
import { Embedding } from "@akhyar11/ml-v1"

const embed = new Embedding({
  vocabSize: 5000,
  embeddingDim: 128,
  trainable: false,
});

embed.fillWeight("./pretrained-embedding.json");
```

Notes:
- `trainable: false` keeps `forward()` active but prevents `backward()` from updating the embedding table.
- `fillWeight()` only replaces the weight matrix. It does not change `trainable`, `alpha`, `optimizer`, `status`, `padTokenId`, `vocabSize`, or `embeddingDim`.
- The incoming weight shape must exactly match `[embeddingDim, vocabSize]`.
- JSON sources for `fillWeight()` must be either an Embedding-layer artifact or a model artifact whose first layer is the Embedding layer.

---

### `MultiHeadAttention`

Causal multi-head self-attention — the core of the Transformer architecture. Allows the model to simultaneously focus on different parts of the input.

#### `constructor(config)`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `units` | `number` | — | Internal dimension (must be divisible by `heads`) |
| `heads` | `number` | — | Number of parallel attention heads |
| `seqLen` | `number` | — | Maximum input sequence length |
| `alpha` | `number` | — | Layer-specific learning rate |
| `status` | `StatusLayer` | — | Layer role in the model graph |
| `clipGradient` | `number` | `5.0` | Gradient clipping limit |

```ts
import { MultiHeadAttention } from "@akhyar11/ml-v1"

const attention = new MultiHeadAttention({
  units: 512,
  heads: 8,
  seqLen: 128
});
```

---

### `SelfAttention`

Basic single-head self-attention mechanism. For most use cases prefer `MultiHeadAttention`.

#### `constructor(config)`

| Parameter | Type | Description |
|---|---|---|
| `units` | `number` | Internal dimension |
| `alpha` | `number` | Learning rate |
| `clipGradient` | `number` | Gradient clipping limit |

---

### `LayerNormalization`

Per-column normalization that stabilizes the distribution of values within the network.

#### `constructor(config)`

| Parameter | Type | Description |
|---|---|---|
| `units` | `number` | Number of features to normalize |
| `clipGradient` | `number` | Gradient clipping limit |

```ts
import { LayerNormalization } from "@akhyar11/ml-v1"

const norm = new LayerNormalization({ units: 128 });
```

---

### `Dropout`

Randomly deactivates a fraction of neurons during training to prevent overfitting. Inactive during evaluation mode.

#### `constructor(config)`

| Parameter | Type | Description |
|---|---|---|
| `rate` | `number` | Fraction of neurons to drop (0–1) |

```ts
import { Dropout } from "@akhyar11/ml-v1"

const drop = new Dropout({ rate: 0.1 });
```

---

### `PositionalEncoding`

Injects fixed sinusoidal positional information into the token embeddings. Used in Transformer architectures.

---

### `Flatten`

Flattens a matrix to a single row. Typically placed before a `Dense` layer in a CNN pipeline.

---

### `Convolution`

2D convolution layer for spatial feature extraction.

#### `constructor(config)`

| Parameter | Type | Description |
|---|---|---|
| `kernelSize` | `number` | Size of the convolution kernel |
| `inputShape` | `[number, number]` | Shape of the input matrix |
| `activation` | `ActivationType` | Activation function name |
| `clipGradient` | `number` | Gradient clipping limit |

---

### `Activation`

Standalone activation layer. Wraps any named activation function as a layer.

---

### `CompileDenseLayers`

Utility helper that compiles a list of dense layers with shared optimizer and learning-rate settings. Used internally by `Sequential.compile()`.

---

### Recurrent Layers: `RNN`, `LSTM`, `GRU`
 
Recurrent layers process sequential data. They support both **single-sample (features × seqLen)** and **batched (features × (seqLen * batchSize))** input layouts.

#### General Conventions

- **`returnSequences: false`** — output shape `[hiddenUnits, 1]` (last time step).
- **`returnSequences: true`** — output shape `[hiddenUnits, seqLen]` (all time steps).
- **`stateful: true`** — hidden state is carried over across `forward()` calls until `resetState()` is called.
- **`returnState`** — currently **not supported** across the entire recurrent family; will throw an explicit error.
- `getState()` / `resetState()` — available on all recurrent layers.
- `save()` / `load()` — persist weights, configuration, and stateful hidden states.

#### Batched Training Support (v2.3.0+)

Starting from version **2.3.0**, the entire recurrent family supports high-performance batched training via the native Rust backend. 
- Use **`batchSize > 1`** in `Sequential.fit()` to benefit from parallel sequence processing and "Hyper-Speed" native kernels.
- **RESTRICTION**: If **`stateful: true`**, the layer is still restricted to **`batchSize: 1`** due to persistent hidden state dependencies across batches.

---

#### `RNN`

Basic recurrent layer with one hidden state and Backpropagation Through Time (BPTT).

| Parameter | Type | Default | Description |
|---|---|---|---|
| `units` | `number` | — | Input features per time step |
| `hiddenUnits` | `number` | — | Hidden state size |
| `activation` | `string` | `"tanh"` | Recurrent activation |
| `returnSequences` | `boolean` | `false` | Return output for every time step |
| `stateful` | `boolean` | `false` | Maintain hidden state across calls |
| `optimizer` | `Optimzier` | — | Optimizer name |
| `clipGradient` | `number` | `5.0` | Gradient clipping limit |

```ts
import { RNN, mj } from "@akhyar11/ml-v1"

const layer = new RNN({
  units: 8,
  hiddenUnits: 16,
  activation: "tanh",
  returnSequences: true,
  stateful: false,
});

const x = mj.matrix([
  [1, 2, 3],
  [0, 1, 0],
  [1, 0, 1],
  [0, 0, 1],
  [1, 1, 1],
  [0, 1, 1],
  [1, 0, 0],
  [0, 0, 0],
]); // [8, 3]

const out = layer.forward(x); // [16, 3] with returnSequences=true
```

---

#### `LSTM`

Recurrent layer with cell state and input/forget/output gates for longer-range dependencies.

| Parameter | Type | Description |
|---|---|---|
| `units` | `number` | Input features per time step |
| `hiddenUnits` | `number` | Hidden and cell state size |
| `forgetBias` | `number` | Initial forget-gate bias for new layers. Default `1` |
| `returnSequences` | `boolean` | Return all time steps |
| `stateful` | `boolean` | Carry state across calls |
| `optimizer` | `Optimzier` | Optimizer name |
| `clipGradient` | `number` | Gradient clipping limit |

`getState()` returns `{ h, c }`.

Notes:
- New `LSTM` layers initialize `bf` to `1` by default for better long-range retention.
- `forgetBias` only affects fresh initialization. `load()` always preserves serialized `bf` from saved weights.

```ts
import { LSTM, mj } from "@akhyar11/ml-v1"

const layer = new LSTM({
  units: 8,
  hiddenUnits: 32,
  returnSequences: false,
  stateful: true,
});

const out = layer.forward(
  mj.matrix([
    [1, 2, 3, 4],
    [0, 1, 0, 1],
    [1, 0, 1, 0],
    [0, 0, 1, 1],
    [1, 1, 1, 1],
    [0, 1, 1, 0],
    [1, 0, 0, 1],
    [0, 0, 0, 1],
  ])
); // [32, 1]

layer.resetState();
```

---

#### `GRU`

Recurrent layer with update/reset gates. Also supports **bidirectional** mode.

| Parameter | Type | Description |
|---|---|---|
| `units` | `number` | Input features per time step |
| `hiddenUnits` | `number` | Hidden state size per direction |
| `bidirectional` | `boolean` | Run forward and backward and concatenate outputs |
| `returnSequences` | `boolean` | Return all time steps |
| `stateful` | `boolean` | Carry state across calls |
| `optimizer` | `Optimzier` | Optimizer name |
| `clipGradient` | `number` | Gradient clipping limit |

`getState()` returns `{ forward, backward? }`.

```ts
import { GRU, mj } from "@akhyar11/ml-v1"

const layer = new GRU({
  units: 8,
  hiddenUnits: 16,
  bidirectional: true,
  returnSequences: true,
});

const out = layer.forward(
  mj.matrix([
    [1, 2, 3],
    [0, 1, 0],
    [1, 0, 1],
    [0, 0, 1],
    [1, 1, 1],
    [0, 1, 1],
    [1, 0, 0],
    [0, 0, 0],
  ])
); // [32, 3]  — 16 forward + 16 backward
```

---

#### `AdaptiveMemoryRNN`

Vanilla RNN core with an external memory bank. This layer is not GRU or LSTM: the recurrent cell remains a simple RNN, then the cell reads from and writes to memory slots with a learned vector gate.

At each time step it builds a query from `[x_t; h_{t-1}]`, retrieves a memory read vector using softmax over memory slots, feeds `concat(x_t, memory_read_t)` into the RNN core, then updates the selected memory slot with gated retention:

```ts
k_i <- (1 - g_t) * k_i + g_t * q_t
v_i <- (1 - g_t) * v_i + g_t * c_t
```

The retrieval is attention-like, but it attends over the layer's memory slots rather than over all input tokens.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `units` | `number` | — | Input features per time step |
| `hiddenUnits` | `number` | — | Vanilla RNN hidden state size |
| `activation` | `"tanh" \| "relu"` | `"tanh"` | RNN core activation |
| `memorySlots` | `number` | `32` | Number of external memory slots |
| `memoryDim` | `number` | `hiddenUnits` | Key/value dimension for each memory slot |
| `returnSequences` | `boolean` | `false` | Return all time steps or only the last hidden state |
| `stateful` | `boolean` | `false` | Carry hidden state and memory bank across calls |
| `optimizer` | `Optimzier` | `"adam"` | Optimizer for trainable recurrent/query/write projections |
| `clipGradient` | `number \| boolean` | `5.0` | Gradient clipping limit |

```ts
import { AdaptiveMemoryRNN, Dense, Embedding, Sequential } from "@akhyar11/ml-v1"

const model = new Sequential({
  layers: [
    new Embedding({ vocabSize: 5000, embeddingDim: 128, padTokenId: 0 }),
    new AdaptiveMemoryRNN({
      units: 128,
      hiddenUnits: 256,
      memorySlots: 32,
      memoryDim: 256,
      returnSequences: false,
    }),
    new Dense({
      units: 256,
      outputUnits: 10,
      activation: "softmax",
      status: "output",
      loss: "crossEntropy",
    }),
  ],
});
```

Gradient support:
- Trainable via optimizer updates: `Wxh`, `Whh`, `bh`, `Wq`, `Wm`, `Wg`, `bg`.
- `memoryKeys`, `memoryValues`, and `memoryUsage` are treated as dynamic recurrent state, not optimizer-trained global parameters.
- Slot selection (`selectWriteSlot`) is discrete. Gradients flow through the selected slot's read/write computations, but not through the slot-choice decision itself.
- The JavaScript path remains the correctness reference, and the native backward path is implemented to match it using the same forward caches.

---

## Notes

- For sequence modeling in `Sequential`, recurrent layers are typically followed by a `Dense` output layer.
- `RNN.getState()` returns a `Matrix`, `LSTM.getState()` returns `{ h, c }`, `GRU.getState()` returns `{ forward, backward? }`.
- `AdaptiveMemoryRNN.getState()` returns `{ h, memoryKeys, memoryValues, memoryUsage }`.
- `save()` / `load()` preserve recurrent weights and stateful hidden states.

---

### `MemoryBank`

Generic, reusable runtime memory layer suitable for insertion after any layer. `MemoryBank` provides a short-term, session-persistent key/value memory that is updated at runtime during `forward()` and can be saved/loaded separately from model weights.

Key properties:
- Auto-infers `units` from the first `forward()` if omitted in constructor.
- Memory state (`memoryKeys`, `memoryValues`, `memoryFilled`, `memoryUsage`, `memoryAge`) is runtime state — it is NOT trained by the model optimizer and is saved via `saveMemory()` / `loadMemory()`.
- Trainable read/output weights (`queryKernel`, `needKernel`, `outputKernel`, `outputBias`) are stored in model `save()` / `load()` and are trained via `backward()`.
- Optional write-side weights (`writeValueKernel`, `writeGateKernel`, `writeKeyKernel`) are only present in the matching modes and should be considered auxiliary write-policy parameters, not generic end-to-end memory state.
- Memory persists across `forward()` calls until `resetMemory()` is called.

How `MemoryBank` works conceptually:
- Input `x` first becomes a query vector through `queryKernel`.
- That query is compared against stored `memoryKeys`.
- The best matching slots are read and combined into one `read` vector.
- `needKernel` decides how much of that read vector should matter.
- The layer then combines `x` and memory context according to `mode`.
- If writes are enabled, the layer may also write a new key/value pair into one slot.

Constructor config:
- `units?: number` — input rows; if omitted, inferred on first forward.
- `memorySlots: number` — number of memory slots (required).
- `memoryDim?: number` — key/value dim; defaults to `units` when omitted.
- `outputUnits?: number` — output projection size for `mode='project'` or `mode='read-project'`.
- `mode?: "project" | "concat" | "add" | "read-project"` — how memory read result combines with input.
- `similarity?: "cosine" | "dot"` — how query vectors compare against memory keys.
- `readTopK?: number` — top-K memory reads (default `min(4,memorySlots)`).
- `updateMode?: "replace" | "merge" | "gated-merge"` — how an occupied slot is updated.
- `writePolicy?: "empty-first" | "least-used" | "oldest" | "least-relevant"` — how a write target slot is chosen when memory is full.
- `writeThreshold?: number` — gate threshold to trigger writes (default `0.5`).
- `persistence?: "session" | "manual"` — persistence strategy marker for the layer config.
- `writeEnabled?: boolean` — allow writes during forward (default `true`).
- `forceNeedGate?: number` — override learned read-need gate with a fixed value in `[0,1]`.
- `valueMode?: "identity" | "project"` — how new memory values are generated during writes.
- `writeKeyMode?: "shared-query" | "separate-project"` — how new memory keys are generated during writes.
- `writeGateMode?: "always" | "threshold" | "learned"` — how the write decision is made.
- `trainablePolicy?: boolean` — whether policy projection weights are trainable (default `true`).

API (important methods):
- `forward(x: Matrix): Matrix` — read & optionally write memory per column; shapes:
  - `project` => `[outputUnits, cols]`
  - `read-project` => `[outputUnits, cols]`
  - `concat`  => `[units + memoryDim, cols]`
  - `add`     => `[units, cols]` (requires `memoryDim===units`)
- `backward(y: Matrix, err: Matrix): Matrix` — returns `dx` with gradients to inputs and updates differentiable read/output weights.
- `resetMemory()` / `clearMemory()` — clear runtime memory to empty state.
- `hasMemory()` — returns `boolean` whether any slot is filled.
- `getMemoryState()` / `setMemoryState(state)` — get/set runtime memory snapshot (useful for saving/restoring session memory programmatically).
- `saveMemory(path)` / `loadMemory(path)` — persist runtime memory to a file (JSON).
- `freezeWrites()` / `unfreezeWrites()` / `setWriteFrozen(boolean)` — disable/enable runtime writes while still allowing reads.
- `trainLastWriteKey(targetKey)` — auxiliary supervised update for `writeKeyKernel` in `writeKeyMode='separate-project'`.
- `trainLastWriteValue(targetValue)` — auxiliary supervised update for `writeValueKernel` in `valueMode='project'`.
- `trainLastWriteGate(targetGate)` — auxiliary supervised update for `writeGateKernel` in `writeGateMode='learned'`.

Recommended correctness-first config:

```ts
new MemoryBank({
  memorySlots: 64,
  memoryDim: 32,
  mode: "read-project",
  similarity: "cosine",
  writeKeyMode: "shared-query",
  valueMode: "identity", // only valid if memoryDim === units
  writeGateMode: "always",
  updateMode: "replace",
  writePolicy: "empty-first",
  forceNeedGate: 1,
});
```

This config is the easiest to reason about:
- read path is isolated
- write always happens when enabled
- keys use the same projection space for write and read
- values are stored directly from input
- no hidden learned write gate is blocking memory writes

#### Meaning of Each Enum

##### `MemoryBankMode`

`mode` controls how the final output is produced after memory has been read.

`"project"`
- Output formula: `output = outputKernel * [x; context] + outputBias`
- Meaning: input `x` and memory context are concatenated, then projected to a new output space.
- Use when:
  you want the layer to learn a mixed representation from both input and memory.
- Good for:
  general integration in a trainable model.
- Tradeoff:
  harder to inspect, because output is a learned mixture of input and memory.

`"concat"`
- Output formula: `output = [x; context]`
- Meaning: output is just raw concatenation of original input and memory context.
- Use when:
  the next layer should decide how to mix input and memory.
- Good for:
  debugging or when you want a downstream dense layer to handle fusion.
- Tradeoff:
  output size becomes `units + memoryDim`.

`"add"`
- Output formula: `output = x + context`
- Meaning: memory context is added directly to input.
- Use when:
  memory context and input already live in the same vector space.
- Good for:
  residual-style memory injection.
- Requirements:
  `memoryDim === units`.
- Tradeoff:
  less flexible than `project`, because no learned fusion matrix is used.

`"read-project"`
- Output formula: `output = outputKernel * read + outputBias`
- Meaning: output is driven directly by the memory read vector, not by concatenation with input.
- Use when:
  you want to test or prove that predictions really come from memory.
- Good for:
  diagnostics, episodic memory tests, causal memory checks.
- Tradeoff:
  the direct input path is removed from output, so if memory is empty the output may be weak or uninformative.

Practical intuition:
- If you are confused, start with `read-project` for correctness tests.
- Use `project` later for more flexible real models.

##### `MemorySimilarity`

`similarity` controls how the query compares against stored keys.

`"cosine"`
- Compares direction, not raw magnitude.
- Query and keys are treated as normalized vectors.
- Best when:
  you care about semantic alignment more than vector scale.
- Usually the safest default for memory retrieval.

`"dot"`
- Compares raw dot product.
- Vector magnitude affects the score.
- Best when:
  the magnitude itself carries meaning and you intentionally want strong vectors to dominate.
- Tradeoff:
  more sensitive to scale drift.

Practical intuition:
- Use `"cosine"` unless you have a clear reason to preserve magnitude effects.

##### `MemoryUpdateMode`

`updateMode` controls what happens when the chosen slot is already occupied.

`"replace"`
- Old key/value are overwritten completely.
- Best when:
  each write should fully replace the previous memory content.
- Good for:
  deterministic episodic memory and correctness testing.

`"merge"`
- New state becomes an even blend of old and new.
- Roughly: `0.5 * old + 0.5 * new`
- Best when:
  you want memory to evolve gradually instead of being overwritten immediately.

`"gated-merge"`
- New state becomes a gate-controlled interpolation between old and new.
- Roughly: `(1-gate) * old + gate * new`
- Best when:
  you want partial updates to occupied slots.
- Important detail:
  empty slots are still written as full replace, not partial merge.

Practical intuition:
- Start with `"replace"` if you need correctness and predictability.
- Use `"merge"` or `"gated-merge"` only if you explicitly want soft updates.

##### `MemoryWritePolicy`

`writePolicy` decides which slot should receive a write when no empty slot remains.

`"empty-first"`
- First use any empty slot.
- If all slots are full, fall back to the least-used slot.
- Best when:
  you want simple and stable behavior.
- Recommended default.

`"least-used"`
- Replace the slot with the smallest usage counter.
- Best when:
  rarely-read or rarely-written slots should be recycled first.

`"oldest"`
- Replace the slot with the oldest timestamp.
- Best when:
  you want FIFO-like turnover where older memories expire first.

`"least-relevant"`
- Replace the slot whose key is least relevant to the current query.
- Best when:
  you want memory replacement to depend on current retrieval context.
- Tradeoff:
  replacement becomes input-dependent and more dynamic.

Practical intuition:
- Use `"empty-first"` first.
- Try `"oldest"` if memory should behave like a recency buffer.

##### `MemoryPersistence`

`persistence` is a config label describing how memory is expected to live.

`"session"`
- Memory is expected to persist across forward calls during the current runtime session.
- This is the normal in-memory behavior.

`"manual"`
- Memory lifecycle is expected to be controlled manually via `getMemoryState()`, `setMemoryState()`, `saveMemory()`, and `loadMemory()`.

Important:
- In practice both modes use the same runtime memory object.
- This enum is mainly semantic/config metadata, not a separate storage backend.

##### `MemoryValueMode`

`valueMode` controls how the value written into memory is created.

`"identity"`
- New memory value is the input itself.
- Formula: `newValue = x`
- Requirement:
  `memoryDim === units`
- Best when:
  the input vector already is the thing you want to store.
- Good for:
  correctness tests and deterministic write-read setups.

`"project"`
- New memory value is a learned projection of the input.
- Formula: `newValue = writeValueKernel * x`
- Best when:
  the stored value should be a transformed representation rather than the raw input.
- Important:
  this uses an auxiliary write-side parameter and should only be treated as trainable when you explicitly supervise it or otherwise manage its training intentionally.

Practical intuition:
- Use `"identity"` first if possible.
- Use `"project"` only when `memoryDim !== units` or when you deliberately need a transformed value space.

##### `MemoryWriteKeyMode`

`writeKeyMode` controls how the key for a new write is generated.

`"shared-query"`
- New write key uses the same projection as the read query path.
- Formula: `newKey = queryKernel * x`
- Meaning:
  the key space used for writing and the key space used for reading are shared.
- Best when:
  you want writes and reads to align naturally.
- Recommended default.

`"separate-project"`
- New write key uses a separate learned projection.
- Formula: `newKey = writeKeyKernel * x`
- Meaning:
  write-time key generation can differ from read-time query generation.
- Best when:
  you explicitly need a separate write key policy.
- Important:
  this mode is harder to keep aligned. If write keys and query keys drift apart, retrieval quality drops.
- Recommended usage:
  only use this if you also have a clear training procedure such as `trainLastWriteKey(...)`.

Practical intuition:
- If you do not know why you need separate write keys, you do not need them.

##### `MemoryWriteGateMode`

`writeGateMode` controls whether a write happens.

`"always"`
- If writes are enabled and not frozen, always write.
- Gate value is effectively `1`.
- Best when:
  you want guaranteed memory writes.
- Good for:
  correctness tests and causal memory verification.

`"threshold"`
- Write decision uses the current `need` score directly.
- If `need >= writeThreshold`, the write happens.
- Meaning:
  the more the model "needs" memory for this input, the more likely it is to store it.
- Best when:
  you want a simple deterministic gate linked to read-need behavior.

`"learned"`
- Write decision is produced by `writeGateKernel`.
- Formula: `gate = sigmoid(writeGateKernel * [x; read])`
- If `gate >= writeThreshold`, the write happens.
- Best when:
  you intentionally want a learned write policy.
- Important:
  this is more flexible but also more complex to train and debug.

Practical intuition:
- Start with `"always"`.
- Move to `"threshold"` if you want simple selective writing.
- Use `"learned"` only when you are ready to reason about gate training.

#### Which Parts Are Trainable vs Runtime State?

Trainable by `backward()`:
- `queryKernel`
- `needKernel`
- `outputKernel`
- `outputBias`

Conditionally trainable through explicit auxiliary APIs:
- `writeValueKernel` when `valueMode="project"`
- `writeGateKernel` when `writeGateMode="learned"`
- `writeKeyKernel` when `writeKeyMode="separate-project"`

Runtime state only, not optimizer weights:
- `memoryKeys`
- `memoryValues`
- `memoryFilled`
- `memoryUsage`
- `memoryAge`
- `memoryStep`

This distinction is important:
- memory content changes because the layer writes to memory during `forward()`
- memory content does not change because Adam/SGD optimized it

#### Save/Load Behavior

- `save()` stores:
  config, dimensions, trainable parameters, optimizer settings, and current memory state.
- `load()` restores all of the above.
- `saveMemory()` / `loadMemory()` only store and restore runtime memory state.

#### Simple Usage Example

For a first experiment, use this:

```ts
import { Sequential, Dense, MemoryBank, mj } from "@akhyar11/ml-v1";

const model = new Sequential({
  layers: [
    new Dense({ units: 10, outputUnits: 32, activation: "relu", status: "input" }),
    new MemoryBank({
      memorySlots: 64,
      memoryDim: 32,
      mode: "read-project",
      similarity: "cosine",
      writeKeyMode: "shared-query",
      valueMode: "identity",
      writeGateMode: "always",
      updateMode: "replace",
      writePolicy: "empty-first",
      forceNeedGate: 1,
    }),
    new Dense({ units: 32, outputUnits: 3, activation: "linear", status: "output", loss: "softmaxCrossEntropy" }),
  ],
});
```

If `memoryDim !== units`, then `valueMode: "identity"` is invalid, so change it to:

```ts
valueMode: "project"
```

Usage example:

```ts
import { Sequential, Dense, MemoryBank, mj } from "@akhyar11/ml-v1";

const model = new Sequential({
  layers: [
    new Dense({ units: 10, outputUnits: 32, activation: "relu", status: "input" }),
    new MemoryBank({ memorySlots: 64, memoryDim: 32, mode: "project", readTopK: 4, writeThreshold: 0.5 }),
    new Dense({ units: 32, outputUnits: 3, activation: "linear", status: "output", loss: "softmaxCrossEntropy" }),
  ],
});

const x = mj.matrix(new Array(10).fill(0).map(() => new Array(2).fill(Math.random())));
model.predict(x);
model.saveMemory("session-memory.json");
model.resetMemory();
```

Notes & design constraints:
- Memory content is runtime state, not optimizer state.
- `MemoryBank` is intentionally generic — it does not assume sequences, batches, or tokens. Inputs are handled as `[features, columns]`.
- Slot selection and replacement are non-differentiable runtime operations. The backward pass computes exact gradients through the read/output path using the memory snapshot captured during forward.
- The current implementation is correctness-first:
  read/output path is differentiable through `backward()`, while optional write-side learned policies should be treated as explicit auxiliary training paths rather than automatic full-history BPTT.
