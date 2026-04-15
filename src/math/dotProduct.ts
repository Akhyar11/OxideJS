import Matrix from "../matrix";

/**
 * Perkalian product matrix a dan b — DIOPTIMASI dengan Float64Array
 * @param a Matrix
 * @param b Matrix
 * @returns Matrix
 */
export default function dotProduct(a: Matrix, b: Matrix): Matrix {
  const aRows = a._shape[0], aCols = a._shape[1];
  const bCols = b._shape[1];

  if (aCols !== b._shape[0]) {
    throw new Error(`row dan col dari matrix harus sama ${aCols}!=${b._shape[0]}`);
  }

  const result = new Float64Array(aRows * bCols);
  const aData = a._data;
  const bData = b._data;

  // Loop order i-k-j untuk cache-friendly access pada b
  for (let i = 0; i < aRows; i++) {
    const iOffset = i * aCols;
    const rOffset = i * bCols;
    for (let k = 0; k < aCols; k++) {
      const aik = aData[iOffset + k];
      const kOffset = k * bCols;
      for (let j = 0; j < bCols; j++) {
        result[rOffset + j] += aik * bData[kOffset + j];
      }
    }
  }

  return Matrix.fromFlat(result, [aRows, bCols]);
}
