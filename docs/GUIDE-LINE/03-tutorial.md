# Quick Tutorial: Getting Started with ML-V1

This guide will walk you through the basics of using ML-V1, from simple matrix operations to training a small transformer model.

## 1. Basic Matrix Operations

All data in ML-V1 is represented as `Matrix` objects. Use the `math` module (often aliased as `mj`) to perform operations.

```ts
import mj from "./src/math";

// Create a 2x2 matrix
const a = mj.matrix([[1, 2], [3, 4]]);
const b = mj.matrix([[5, 6], [7, 8]]);

// Dot Product Multiplication
const c = mj.dotProduct(a, b);

// Element-wise Addition
const d = mj.add(c, 10);

c.print(); // Print matrix contents to console
console.log("Shape:", d._shape);
```

---

## 2. Building a Simple Model

You can use the `Sequential` class to stack various neural network layers.

```ts
import mj from "./src/math";
import { Sequential } from "./src/models";
import { Dense } from "./src/layers";

const model = new Sequential({
  layers: [
    new Dense({ units: 2, outputUnits: 4, activation: "relu", status: "input" }),
    new Dense({ units: 4, outputUnits: 1, activation: "sigmoid", status: "output", loss: "mse" }),
  ],
});

// Compile model with optimizer and learning rate
model.compile({ alpha: 0.01, optimizer: "adam", error: "mse" });
```

---

## 3. Training the Model (Fit)

Use the `.fit()` method to train the model on a dataset.

```ts
// Simple XOR Data
const X = [
  mj.matrix([[0], [0]]), 
  mj.matrix([[0], [1]]), 
  mj.matrix([[1], [0]]), 
  mj.matrix([[1], [1]])
];
const Y = [
  mj.matrix([[0]]), 
  mj.matrix([[1]]), 
  mj.matrix([[1]]), 
  mj.matrix([[0]])
];

// Train for 500 epochs
model.fit(X, Y, 500, (loss) => {
  console.log(`Current Loss: ${loss.toFixed(6)}`);
});

// Prediction
const pred = model.predict(mj.matrix([[1], [0]]));
console.log("Prediction Result for [1, 0]:");
pred.print();
```

---

## 4. Using the BPE Tokenizer

For NLP tasks, you need to convert text into a sequence of numbers (token IDs).

```ts
import { BPETokenizer } from "./src/tokenizer";

const tokenizer = new BPETokenizer({ vocabSize: 100, minFrequency: 1 });

// Train tokenizer with text data
const corpus = ["i am learning ai", "ai is cool", "learning coding"];
tokenizer.train(corpus);

// Encode text to token IDs
const ids = tokenizer.encodeWithSpecial("i am learning coding");
console.log("Token IDs:", ids);

// Decode back to text
const text = tokenizer.decode(ids);
console.log("Decoded Text:", text);

// Save tokenizer for later use
tokenizer.save("./my-tokenizer.json");
```

---

## 5. Sequence Modeling with GRU

Use recurrent layers when the input is a sequence (common shape: `[features, seqLen]`).

```ts
import mj from "./src/math";
import { Sequential } from "./src/models";
import { GRU, Dense } from "./src/layers";

const model = new Sequential({
  layers: [
    new GRU({ units: 8, hiddenUnits: 16, returnSequences: false, status: "input" }),
    new Dense({ units: 16, outputUnits: 4, activation: "linear", status: "output", loss: "softmaxCrossEntropy" }),
  ],
});

model.compile({ alpha: 0.001, optimizer: "adam", error: "softmaxCrossEntropy" });
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
const y = mj.matrix([[2]]);
model.forward(x);
model.backward(y);
```

---

## 6. Full-Sequence Causal LM with Transformers

For `Transformers`, the training and inference paths are separated, but inference can now be unified via `predictMode`:
- training: logits for all valid token positions
- inference: default last-token logits, or full-sequence if requested

```ts
import mj from "./src/math";
import { Transformers } from "./src/models";

const padTokenId = 0;
const model = new Transformers({
  units: 32,
  seqLen: 6,
  vocabSize: 100,
  heads: 4,
  alpha: 0.001,
  padTokenId,
  predictMode: "next-token",
});

const x = mj.matrix([
  [0],
  [11],
  [12],
  [13],
  [14],
  [15],
]);

const y = mj.matrix([
  [0],
  [12],
  [13],
  [14],
  [15],
  [0],
]);

model.train();
const trainLogits = model.forward(x); // [vocabSize, seqLen * batch]
model.backward(y);

model.eval();
const nextTokenLogits = model.predict(x); // [vocabSize, batch]

model.setPredictMode("full-sequence");
const allTokenLogits = model.predict(x); // [vocabSize, seqLen * batch]
```

---

## Development Tips

- **Training vs Eval Mode**: Use `model.train()` during training and `model.eval()` during inference (especially if using `Dropout` layers).
- **Transformer Predict Mode**: Use `predictMode: "next-token"` for the generation loop, or `predictMode: "full-sequence"` if you want to inspect logits for all positions using the same `predict()` method.
- **Matrix Dimensions**: Always check your matrix shapes. Most layers expect input in the form of `[features, batch_size]` or `[sequence_length, batch_size]`.

---

**Next Steps:**
Explore all available functions in the [API & Function Reference](04-api-functions.md).
