import Matrix from "../matrix";
import AdaGrad from "../optimizer/adaGrad";
import Adam from "../optimizer/adam";
import Momentum from "../optimizer/momentum";
import NAG from "../optimizer/nag";
import SGD from "../optimizer/sgd";
export type { FitConfig, FitResult } from "./fitConfig";

export type vector = number[];
export type matrix2d = number[][];
export type matrix3d = number[][][];
export type MatrixCollection = Matrix | number;
export type MatrixShape = [number, number];
export type MatrixFlatData = Float32Array | Float64Array;
export { Matrix };
export type ActivationType = "sigmoid" | "tanh" | "relu" | "lRelu" | "linear" | "softmax";
export type StatusLayer =
  | "input"
  | "output"
  | "norm"
  | "outputReduction"
  | "convOutput"
  | "train"
  | "test";
export type Optimzier = "sgd" | "adaGrad" | "momentum" | "nag" | "adam";
export type OptimzierType = SGD | AdaGrad | NAG | Momentum | Adam;
export type Cost = "mse" | "crossEntropy" | "binaryCrossEntropy" | "softmaxCrossEntropy";
// Layers type is now in @oxidejs/layers


export type WorkerData = {
  value: number;
  i: number;
  k: number;
};
