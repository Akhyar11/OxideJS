import { Cost } from "../@types/type.js";
import MeanSquerError from "../cost/mse.js";
import CategoricalCrossEntropy, { BinaryCrossEntropy } from "../cost/crossEntropy.js";
import SoftmaxCrossEntropy from "../cost/softmaxCrossEntropy.js";

import MeanAbsoluteError from "../cost/mae.js";
import HuberLoss from "../cost/huber.js";
import LogCoshLoss from "../cost/logCosh.js";
import HingeLoss from "../cost/hinge.js";
import SquaredHingeLoss from "../cost/squaredHinge.js";
import KLDivergence from "../cost/klDivergence.js";
import PoissonLoss from "../cost/poisson.js";

export default function setLoss(cost: Cost) {
  switch (cost) {
    case "mse":
      return MeanSquerError;
    case "mae":
      return MeanAbsoluteError;
    case "huber":
      return HuberLoss;
    case "logCosh":
      return LogCoshLoss;
    case "hinge":
      return HingeLoss;
    case "squaredHinge":
      return SquaredHingeLoss;
    case "klDivergence":
      return KLDivergence;
    case "poisson":
      return PoissonLoss;
    case "crossEntropy":
      return CategoricalCrossEntropy;
    case "binaryCrossEntropy":
      return BinaryCrossEntropy;
    case "softmaxCrossEntropy":
      return SoftmaxCrossEntropy;

    default:
      throw new Error(`Loss function '${cost}' tidak dikenal.`);
  }
}

