import { Cost } from "../@types/type.js";
import MeanSquerError from "../cost/mse.js";
import CategoricalCrossEntropy, { BinaryCrossEntropy } from "../cost/crossEntropy.js";
import SoftmaxCrossEntropy from "../cost/softmaxCrossEntropy.js";

export default function setLoss(cost: Cost) {
  switch (cost) {
    case "mse":
      return MeanSquerError;
    case "crossEntropy":
      return CategoricalCrossEntropy;
    case "binaryCrossEntropy":
      return BinaryCrossEntropy;
    case "softmaxCrossEntropy":
      return SoftmaxCrossEntropy;
    default:
      throw new Error(`Loss function '${cost}' tidak dikenal. Pilih: mse, crossEntropy, binaryCrossEntropy, softmaxCrossEntropy`);
  }
}

