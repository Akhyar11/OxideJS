import { FitConfig, FitResult } from "../@types/fitConfig";
import Matrix from "../matrix";

export interface TrainableModel {
  forward(x: Matrix, batchSize?: number): Matrix;
  backward(y: Matrix, batchSize?: number): void;
  fit(X: Matrix[], y: Matrix[], epochs: number, config?: FitConfig): FitResult;
  predict(x: Matrix): Matrix;
  train(): this;
  eval(): this;
}
