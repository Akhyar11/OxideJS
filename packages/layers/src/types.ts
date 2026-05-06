import Activation from "./activation.js";
import Convolution from "./convolution.js";
import Dense from "./dense.js";
import SelfAttention from "./selfAttention.js";
import Embedding from "./embedding.js";
import Flatten from "./flatten.js";
import PositionalEncoding from "./positionalEncoding.js";
import LayerNormalization from "./layerNormalization.js";
import MultiHeadAttention from "./multiHeadAttention.js";
import RNN from "./rnn.js";
import LSTM from "./lstm.js";
import GRU from "./gru.js";
import AdaptiveMemoryRNN from "./adaptiveMemoryRNN.js";
import MemoryBank from "./memoryBank.js";
import Dropout from "./dropout.js";
import AttentionPooling from "./attentionPooling.js";

export type Layers =
  | Dense
  | Activation
  | Convolution
  | SelfAttention
  | Embedding
  | Flatten
  | PositionalEncoding
  | LayerNormalization
  | MultiHeadAttention
  | Dropout
  | RNN
  | LSTM
  | GRU
  | AdaptiveMemoryRNN
  | MemoryBank
  | AttentionPooling;
