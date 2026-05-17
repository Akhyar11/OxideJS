import { Matrix } from "@oxide-js/core";
import type { MetricLike } from "./types.js";

/**
 * Accuracy metric for multi-class classification.
 * Compares argmax of predictions with argmax of targets.
 */
export function accuracy(yPred: Matrix, yTrue: Matrix): number {
  const predData = yPred._data;
  const trueData = yTrue._data;
  const [rows, cols] = yPred._shape;

  let correct = 0;

  for (let i = 0; i < rows; i++) {
    let maxPredIdx = 0;
    let maxPredVal = predData[i * cols];

    let maxTrueIdx = 0;
    let maxTrueVal = trueData[i * cols];

    for (let j = 0; j < cols; j++) {
      const predVal = predData[i * cols + j];
      const trueVal = trueData[i * cols + j];

      if (predVal > maxPredVal) {
        maxPredVal = predVal;
        maxPredIdx = j;
      }

      if (trueVal > maxTrueVal) {
        maxTrueVal = trueVal;
        maxTrueIdx = j;
      }
    }

    if (maxPredIdx === maxTrueIdx) {
      correct++;
    }
  }

  return correct / rows;
}

/**
 * Categorical accuracy - alias for accuracy.
 */
export function categoricalAccuracy(yPred: Matrix, yTrue: Matrix): number {
  return accuracy(yPred, yTrue);
}

/**
 * Binary accuracy for binary classification (threshold at 0.5).
 */
export function binaryAccuracy(yPred: Matrix, yTrue: Matrix): number {
  const predData = yPred._data;
  const trueData = yTrue._data;
  const totalSamples = predData.length;

  let correct = 0;

  for (let i = 0; i < totalSamples; i++) {
    const pred = predData[i] > 0.5 ? 1 : 0;
    const true_ = trueData[i] > 0.5 ? 1 : 0;

    if (pred === true_) {
      correct++;
    }
  }

  return correct / totalSamples;
}

/**
 * Mean Absolute Error (MAE) metric.
 */
export function mae(yPred: Matrix, yTrue: Matrix): number {
  const predData = yPred._data;
  const trueData = yTrue._data;
  const totalSamples = predData.length;

  let sumAbsErr = 0;

  for (let i = 0; i < totalSamples; i++) {
    const err = Math.abs(predData[i] - trueData[i]);
    sumAbsErr += err;
  }

  return sumAbsErr / totalSamples;
}

/**
 * Mean Squared Error (MSE) metric.
 */
export function mse(yPred: Matrix, yTrue: Matrix): number {
  const predData = yPred._data;
  const trueData = yTrue._data;
  const totalSamples = predData.length;

  let sumSqErr = 0;

  for (let i = 0; i < totalSamples; i++) {
    const err = predData[i] - trueData[i];
    sumSqErr += err * err;
  }

  return sumSqErr / totalSamples;
}

/**
 * Get the name of a metric for display/logging.
 */
export function getMetricName(metric: MetricLike): string {
  if (typeof metric === "string") {
    return metric;
  }

  if (typeof metric === "function") {
    return metric.name ?? "metric";
  }

  if (typeof metric === "object" && metric.name) {
    return metric.name;
  }

  return "metric";
}

/**
 * Compute a metric value given predictions and targets.
 * Resolves string metric names to built-in implementations.
 */
export function computeMetric(metric: MetricLike, yPred: Matrix, yTrue: Matrix): number {
  if (typeof metric === "function") {
    return metric(yPred, yTrue);
  }

  if (typeof metric === "object" && metric.compute) {
    return metric.compute(yPred, yTrue);
  }

  if (typeof metric === "string") {
    const name = metric.toLowerCase();

    switch (name) {
      case "accuracy":
      case "categoricalaccuracy":
        return accuracy(yPred, yTrue);
      case "binaryaccuracy":
        return binaryAccuracy(yPred, yTrue);
      case "mae":
      case "meanabsoluteerror":
        return mae(yPred, yTrue);
      case "mse":
      case "meansquarederror":
        return mse(yPred, yTrue);
      default:
        throw new Error(`Unknown metric: '${metric}'`);
    }
  }

  throw new Error(`Invalid metric type: ${typeof metric}`);
}
