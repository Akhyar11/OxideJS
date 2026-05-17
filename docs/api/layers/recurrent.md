# 🔁 Recurrent Layers (RNN, LSTM, GRU) API Reference

Recurrent layers in **@oxide-js/layers** enable sequence modeling, capturing temporal dependencies across sequential steps (e.g. text processing, time series forecasting, and speech signals). They implement high-performance **Backpropagation Through Time (BPTT)** with native Rust acceleration.

---

## 🏗️ 1. `SimpleRNN` Layer

The **SimpleRNN** layer is a fully-connected RNN where the output at step $t$ is fed back into the next step inputs.

### 📐 Mathematical Formulation
For each step $t \in [1, T]$:
$$\mathbf{h}_t = \tanh(\mathbf{x}_t \mathbf{W} + \mathbf{h}_{t-1} \mathbf{U} + \mathbf{b})$$
where:
* $\mathbf{W}$ is the input kernel matrix.
* $\mathbf{U}$ is the recurrent kernel matrix.
* $\mathbf{b}$ is the bias vector.
* $\mathbf{h}_t$ is the hidden state vector.

### 📌 Configuration Parameters (`SimpleRNNConfig`)
- `units: number` — Dimension of the hidden state space.
- `useBias?: boolean` — Default is `true`.
- `returnSequences?: boolean` — If `true`, returns the full hidden sequence of shape `[batch * seqLen, units]`. If `false`, returns only the last hidden vector of shape `[batch, units]`.
- `sequenceLength?: number` — Sequence steps. Required if input shape is 2D.
- `inputDim?: number` — Features count per sequence step.

---

## 🔋 2. `LSTM` Layer

The **[LSTM](file:///home/akhyar/Dokumen/Code/NODE_JS/Oxide-JS/packages/layers/src/layers/LSTM.ts)** (Long Short-Term Memory) layer mitigates vanishing gradients by using an internal cell state vector managed by three gating mechanisms: input, forget, and output gates.

### 📐 Gated Cell Transitions
For each sequence step $t$:
1. **Forget Gate**: Controls how much cell state to forget.
   $$\mathbf{f}_t = \sigma(\mathbf{x}_t \mathbf{W}_f + \mathbf{h}_{t-1} \mathbf{U}_f + \mathbf{b}_f)$$
2. **Input Gate & Cell Candidate**: Controls what new info to store.
   $$\mathbf{i}_t = \sigma(\mathbf{x}_t \mathbf{W}_i + \mathbf{h}_{t-1} \mathbf{U}_i + \mathbf{b}_i)$$
   $$\tilde{\mathbf{c}}_t = \tanh(\mathbf{x}_t \mathbf{W}_c + \mathbf{h}_{t-1} \mathbf{U}_c + \mathbf{b}_c)$$
3. **Cell State Update**: Update cell memory state.
   $$\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t$$
4. **Output Gate & Hidden State Update**: Compute step hidden state output.
   $$\mathbf{o}_t = \sigma(\mathbf{x}_t \mathbf{W}_o + \mathbf{h}_{t-1} \mathbf{U}_o + \mathbf{b}_o)$$
   $$\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{c}_t)$$

All weights are packed into monolithic parameters of shape `[inputDim, 4 * units]` (`kernel`) and `[units, 4 * units]` (`recurrentKernel`) to achieve maximum processing throughput.

### 📌 Configuration Parameters (`LSTMConfig`)
- `units: number` — Dimension of the cell and hidden states.
- `useBias?: boolean` — Default is `true`.
- `returnSequences?: boolean` — Default is `false`.
- `sequenceLength?: number` — Step sequence length. Required if input shape is 2D.
- `inputDim?: number` — Channels per step.

---

## ⚡ 3. `GRU` Layer

The **GRU** (Gated Recurrent Unit) layer simplifies LSTMs by merging the cell state and hidden state, controlling state transitions through reset and update gates.

### 📐 Gated State Transitions
For each step $t$:
1. **Update Gate**: Controls what memory is carried forward.
   $$\mathbf{z}_t = \sigma(\mathbf{x}_t \mathbf{W}_z + \mathbf{h}_{t-1} \mathbf{U}_z + \mathbf{b}_z)$$
2. **Reset Gate**: Controls how much past memory to forget.
   $$\mathbf{r}_t = \sigma(\mathbf{x}_t \mathbf{W}_r + \mathbf{h}_{t-1} \mathbf{U}_r + \mathbf{b}_r)$$
3. **Candidate State**: New memory candidate.
   $$\tilde{\mathbf{h}}_t = \tanh(\mathbf{x}_t \mathbf{W}_h + (\mathbf{r}_t \odot \mathbf{h}_{t-1}) \mathbf{U}_h + \mathbf{b}_h)$$
4. **Hidden State Update**: Output step state.
   $$\mathbf{h}_t = (1 - \mathbf{z}_t) \odot \mathbf{h}_{t-1} + \mathbf{z}_t \odot \tilde{\mathbf{h}}_t$$

---

## 🛠️ Usage Examples

### 📌 Example (LSTM - Sentence Encoding)
This example shows how to configure an LSTM layer to process sequential token representations and return the final sentence summary encoding vector.

```ts
import { LSTM } from "@oxide-js/layers";
import { Matrix } from "@oxide-js/core";

// 1. Instantiate LSTM layer (sequenceLength = 3, inputDim = 4, hiddenUnits = 8)
const lstm = new LSTM({
  name: "sentence_encoder",
  units: 8,
  returnSequences: false, // returns only the final hidden state vector
  sequenceLength: 3,
  inputDim: 4
});

// 2. Feed sequence inputs [batch * sequenceLength, inputDim] -> [2 * 3, 4] = [6, 4]
const inputs = Matrix.fromFlat(new Float32Array(6 * 4).fill(0.5), [6, 4]);

const outputs = lstm.forward(inputs);
console.log("Outputs Shape (batch, units):", outputs._shape); // [2, 8]
outputs.print();
```

---

### 📌 Example (LSTM - Sequence Tagging / returnSequences)
This example shows how to enable `returnSequences: true` to get predictions for *every* step in the sequence (useful for POS tagging, named entity recognition, or language modeling).

```ts
import { LSTM } from "@oxide-js/layers";
import { Matrix } from "@oxide-js/core";

// 1. Instantiate LSTM with returnSequences enabled
const taggerLstm = new LSTM({
  name: "token_tagger_lstm",
  units: 5,
  returnSequences: true, // returns hidden states for ALL steps
  sequenceLength: 4,
  inputDim: 3
});

// 2. Feed inputs [batch * seqLen, inputDim] -> [2 * 4, 3] = [8, 3]
const inputs = Matrix.fromFlat(new Float32Array(8 * 3).fill(1.0), [8, 3]);

const outputs = taggerLstm.forward(inputs);
console.log("Outputs Shape (batch * seqLen, units):", outputs._shape); // [8, 5]
outputs.print();
```

---

### 📌 Example (SimpleRNN - Time Series Processing)
```ts
import { SimpleRNN } from "@oxide-js/layers";
import { Matrix } from "@oxide-js/core";

// 1. Instantiate SimpleRNN
const rnn = new SimpleRNN({
  name: "timeseries_rnn",
  units: 4,
  returnSequences: false,
  sequenceLength: 5,
  inputDim: 1
});

// 2. Feed 1D sequence features (batch=2, seq=5, feature=1) -> shape [10, 1]
const inputs = Matrix.fromFlat(new Float32Array(10).map((_, i) => i * 0.1), [10, 1]);

const outputs = rnn.forward(inputs);
console.log("RNN Output Shape:", outputs._shape); // [2, 4]
outputs.print();
```

---

### 📌 Example (GRU - Recurrent Gated Modeling)
```ts
import { GRU } from "@oxide-js/layers";
import { Matrix } from "@oxide-js/core";

// 1. Instantiate GRU
const gru = new GRU({
  name: "gru_layer",
  units: 6,
  returnSequences: true,
  sequenceLength: 3,
  inputDim: 5
});

// 2. Feed inputs [batch=2 * seq=3, inputDim=5] -> [6, 5]
const inputs = Matrix.fromFlat(new Float32Array(6 * 5).fill(0.2), [6, 5]);

const outputs = gru.forward(inputs);
console.log("GRU output shape:", outputs._shape); // [6, 6]
outputs.print();
```
