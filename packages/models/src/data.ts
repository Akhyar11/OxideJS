import { Matrix } from "@oxide-js/core";
import type { Batch } from "./types.js";

/**
 * Split data into training and validation sets.
 * First dimension (rows) is treated as batch/sample dimension.
 */
export function trainValidationSplit(
  x: Matrix,
  y: Matrix,
  validationSplit: number,
  shuffle: boolean = false
): {
  xTrain: Matrix;
  yTrain: Matrix;
  xVal: Matrix;
  yVal: Matrix;
} {
  if (validationSplit < 0 || validationSplit > 1) {
    throw new Error(`validationSplit must be between 0 and 1, got ${validationSplit}`);
  }

  const [xRows] = x._shape;
  const [yRows] = y._shape;

  if (xRows !== yRows) {
    throw new Error(
      `x and y must have matching row counts (batch size). x: ${xRows}, y: ${yRows}`
    );
  }

  let indices = Array.from({ length: xRows }, (_, i) => i);

  if (shuffle) {
    // Fisher-Yates shuffle
    for (let i = indices.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [indices[i], indices[j]] = [indices[j], indices[i]];
    }
  }

  const splitIdx = Math.floor(xRows * (1 - validationSplit));

  const trainIndices = indices.slice(0, splitIdx);
  const valIndices = indices.slice(splitIdx);

  return {
    xTrain: sliceMatrixRows(x, trainIndices),
    yTrain: sliceMatrixRows(y, trainIndices),
    xVal: sliceMatrixRows(x, valIndices),
    yVal: sliceMatrixRows(y, valIndices)
  };
}

/**
 * Create mini-batches from data.
 * First dimension (rows) is treated as batch/sample dimension.
 */
export function createBatches(
  x: Matrix,
  y: Matrix,
  batchSize: number,
  shuffle: boolean = false
): Batch[] {
  if (batchSize <= 0) {
    throw new Error(`batchSize must be positive, got ${batchSize}`);
  }

  const [xRows] = x._shape;
  const [yRows] = y._shape;

  if (xRows !== yRows) {
    throw new Error(
      `x and y must have matching row counts (batch size). x: ${xRows}, y: ${yRows}`
    );
  }

  let indices = Array.from({ length: xRows }, (_, i) => i);

  if (shuffle) {
    // Fisher-Yates shuffle
    for (let i = indices.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [indices[i], indices[j]] = [indices[j], indices[i]];
    }
  }

  const batches: Batch[] = [];

  for (let i = 0; i < xRows; i += batchSize) {
    const end = Math.min(i + batchSize, xRows);
    const batchIndices = indices.slice(i, end);

    batches.push({
      x: sliceMatrixRows(x, batchIndices),
      y: sliceMatrixRows(y, batchIndices)
    });
  }

  return batches;
}

/**
 * Helper: slice matrix rows by indices.
 * Assumes 2D matrix [rows, cols].
 */
function sliceMatrixRows(matrix: Matrix, rowIndices: number[]): Matrix {
  const [, cols] = matrix._shape;
  const matrixData = matrix._data;

  // Create new flat data
  const newSize = rowIndices.length * cols;
  const newData = new Float32Array(newSize);

  for (let i = 0; i < rowIndices.length; i++) {
    const srcRow = rowIndices[i];
    const dstRow = i;

    for (let col = 0; col < cols; col++) {
      newData[dstRow * cols + col] = matrixData[srcRow * cols + col];
    }
  }

  return Matrix.fromFlat(newData, [rowIndices.length, cols]);
}
