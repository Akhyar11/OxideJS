import linear, { lRelu, relu, sigmoid, softmax, tanh } from "../activation/index.js";
import { ActivationType } from "../@types/type.js";

export default function setActivation(activation: ActivationType) {
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
      return (a: any) => softmax(a, false);
    default:
      throw new Error(`Activation '${activation}' tidak dikenal. Pilih: sigmoid, tanh, relu, lRelu, linear, softmax`);
  }
}

