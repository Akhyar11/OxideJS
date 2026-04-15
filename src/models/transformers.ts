import mj from "../math";
import Matrix from "../matrix";
import { Matrix as MatrixType } from "../@types/type";
import Sequential from "./sequential";
import { SelfAttention, Dense, PositionalEncoding, LayerNormalization, Embedding } from "../layers";

interface TransformersConfig {
  units: number;          // d_model (embedding size)
  seqLen: number;         // sequence length
  vocabSize: number;      // vocabulary size
  alpha?: number;
  padTokenId?: number;
}

/**
 * Improved Transformer Model
 * 
 * Arsitektur:
 * Input (Indices) -> Embedding -> PositionalEncoding -> LayerNorm -> SelfAttention + Residual -> LayerNorm -> Flatten -> Dense -> Output
 */
export default class Transformers extends Sequential {
  private embedding: Embedding;
  private pe: PositionalEncoding;
  private ln1: LayerNormalization;
  private attention: SelfAttention;
  private ln2: LayerNormalization;
  private dense: Dense;

  private xInput: Matrix = mj.matrix([]);
  private xEmb: Matrix = mj.matrix([]);
  private xPe: Matrix = mj.matrix([]);
  private xRes: Matrix;
  private errPeRes: Matrix; // Buffer untuk gradient residual path

  constructor({ units, seqLen, vocabSize, alpha = 0.01, padTokenId }: TransformersConfig) {
    const embedding = new Embedding({ vocabSize, embeddingDim: units, alpha, padTokenId });
    const pe = new PositionalEncoding({ dModel: units, maxSeqLen: seqLen });
    const ln1 = new LayerNormalization({ units });
    const attention = new SelfAttention({ units, seqLen, alpha });
    const ln2 = new LayerNormalization({ units });
    const dense = new Dense({
      units: units * seqLen, 
      outputUnits: vocabSize, 
      activation: "linear",
      alpha,
      status: "output",
    });

    // Registrasi layer ke super (Sequential) untuk keperluan summary/save
    super({ layers: [embedding, pe, ln1, attention, ln2, dense] });
    
    this.embedding = embedding;
    this.pe = pe;
    this.ln1 = ln1;
    this.attention = attention;
    this.ln2 = ln2;
    this.dense = dense;

    // Pre-allocate buffers
    this.xRes = mj.zeros([units, seqLen]);
    this.errPeRes = mj.zeros([units, seqLen]);
  }

  forward(x: Matrix): Matrix {
    this.xInput = x;

    // 1. Embedding
    this.xEmb = this.embedding.forward(x);

    // 2. Positional Encoding
    this.xPe = this.pe.forward(this.xEmb);
    
    // 3. LayerNorm 1
    const xLn1 = this.ln1.forward(this.xPe);
    
    // 4. Self-Attention
    const xAttn = this.attention.forward(xLn1);
    
    // 5. Residual Connection (Add)
    this.xRes.copyFrom(this.xPe); 
    this.xRes.addInPlace(xAttn);
    
    // 6. LayerNorm 2
    const xLn2 = this.ln2.forward(this.xRes);

    // 7. Flatten
    const n = xLn2._data.length;
    const flat = mj.reshape(xLn2, [n, 1]);

    // 8. Dense Output
    return this.dense.forward(flat);
  }

  backward(y: Matrix) {
    // 1. Backward Dense
    const errDense = this.dense.backward(y, mj.matrix([[]]));
    this.loss = this.dense.loss;

    // 2. Un-flatten
    const errLn2 = mj.reshape(errDense, this.xRes._shape);

    // 3. Backward LayerNorm 2
    const errRes = this.ln2.backward(y, errLn2);

    // 4. Backward Residual (Gradient split)
    const errAttn = errRes;
    this.errPeRes.copyFrom(errRes); 
    const errPe_ResidualPath = this.errPeRes;

    // 5. Backward Attention
    const errLn1 = this.attention.backward(y, errAttn);

    // 6. Backward LayerNorm 1
    const errPe_AttentionPath = this.ln1.backward(y, errLn1);

    // 7. Gabungkan gradien (In-Place)
    const totalErrPe = errPe_ResidualPath;
    totalErrPe.addInPlace(errPe_AttentionPath);

    // 8. Backward Positional Encoding
    const errEmb = this.pe.backward(y, totalErrPe);

    // 9. Backward Embedding
    this.embedding.backward(y, errEmb);
  }

  load(path: string) {
    super.load(path);
    // Setelah super.load() selesai, this.layers berisi layer-layer baru hasil loading.
    // Kita harus memetakan kembali layer tersebut ke member private Transformers.
    for (const layer of this.layers) {
      if (layer instanceof Embedding) this.embedding = layer;
      else if (layer instanceof PositionalEncoding) this.pe = layer;
      else if (layer instanceof SelfAttention) this.attention = layer;
      else if (layer instanceof Dense) this.dense = layer;
      else if (layer instanceof LayerNormalization) {
        // Tentukan mana LN1 dan LN2 berdasarkan urutan (Embedding -> PE -> LN1 -> Attention -> LN2)
        const idx = this.layers.indexOf(layer);
        if (idx === 2) this.ln1 = layer;
        else if (idx === 4) this.ln2 = layer;
      }
    }
  }

  /**
   * Resize model to accommodate more tokens
   */
  resizeVocab(newVocabSize: number): void {
      this.embedding.resize(newVocabSize);
      this.dense.resize(newVocabSize);
  }

  fit(
    X: Matrix[],
    y: Matrix[],
    epochs: number,
    cb: (loss: number) => any = (_) => { }
  ) {
    for (let i = 0; i < epochs; i++) {
      this.dense.resetLoss();
      let epochLoss = 0;

      for (let j = 0; j < X.length; j++) {
        this.forward(X[j]);
        this.backward(y[j]);
        epochLoss = this.dense.loss;
      }

      this.loss = epochLoss;
      cb(this.loss);

      if (this.loss < 0.01) return 0;
    }
  }
}
