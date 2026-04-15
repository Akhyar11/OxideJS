import Matrix from "../matrix";

/**
 * Transposisi matrix [i, j] => [j, i] — DIOPTIMASI
 */
export default function transpose(a: Matrix): Matrix {
  const [rows, cols] = a._shape;
  const result = new Float64Array(rows * cols);
  const aData = a._data;

  for (let i = 0; i < rows; i++) {
    const iOffset = i * cols;
    for (let j = 0; j < cols; j++) {
      result[j * rows + i] = aData[iOffset + j];
    }
  }

  return Matrix.fromFlat(result, [cols, rows]);
}
