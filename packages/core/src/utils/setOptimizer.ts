import { MatrixShape, Optimzier } from "../@types/type.js";
import AdaGrad from "../optimizer/adaGrad.js";
import Adam from "../optimizer/adam.js";
import Momentum from "../optimizer/momentum.js";
import NAG from "../optimizer/nag.js";
import SGD from "../optimizer/sgd.js";

export default function setOptimizer(
  optimzier: Optimzier,
  shape: MatrixShape,
  alpha: number
) {
  switch (optimzier) {
    case "adaGrad":
      return new AdaGrad(shape, alpha);
    case "sgd":
      return new SGD();
    case "momentum":
      return new Momentum(shape);
    case "nag":
      return new NAG(shape);
    case "adam":
      return new Adam(shape);
    default:
      throw new Error(`Optimizer '${optimzier}' tidak dikenal. Pilih: sgd, momentum, nag, adaGrad, adam`);
  }
}
