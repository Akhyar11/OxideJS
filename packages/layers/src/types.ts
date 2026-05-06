import Activation from "./activation";
import Convolution from "./convolution";
import Dense from "./dense";
import SelfAttention from "./selfAttention";
import Embedding from "./embedding";
import Flatten from "./flatten";
import PositionalEncoding from "./positionalEncoding";
import LayerNormalization from "./layerNormalization";
import MultiHeadAttention from "./multiHeadAttention";
import RNN from "./rnn";
import LSTM from "./lstm";
import GRU from "./gru";
import AdaptiveMemoryRNN from "./adaptiveMemoryRNN";
import MemoryBank from "./memoryBank";
import Dropout from "./dropout";
import AttentionPooling from "./attentionPooling";

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
