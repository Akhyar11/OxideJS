import linear, { lRelu, relu, sigmoid, softmax, tanh, elu, gelu, hardSigmoid, hardSwish, mish, selu, softplus, softsign, swish } from "../activation/index.js";
import { ActivationType } from "../@types/type.js";

interface Activation {
  activation: ActivationType
  row?: boolean
  alpha?: number
}

export default function setActivation({ activation, alpha = 0.001, row = false }: Activation) {
  switch (activation) {
    case "sigmoid":
      return sigmoid;
    case "tanh":
      return tanh;
    case "relu":
      return relu;
    case "lRelu":
      return lRelu;
    case "linear":
      return linear;
    case "softmax":
      return (a: any) => softmax(a, row);
    case "elu":
      return (a: any) => elu(a, alpha);
    case "gelu":
      return gelu;
    case "hardsigmoid":
      return hardSigmoid;
    case "hardswish":
      return hardSwish;
    case "mish":
      return mish;
    case "selu":
      return selu;
    case "softplus":
      return softplus;
    case "softsign":
      return softsign;
    case "swish":
      return swish;
    default:
      throw new Error(`Activation '${activation}' tidak dikenal. Pilih: sigmoid, tanh, relu, lRelu, linear, softmax, elu, gelu, hardsigmoid, hardswish, mish, selu, softplus, softsign, swish`);
  }
}
