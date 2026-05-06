import Matrix from "../matrix/index.js";

/**
 * Pembagian matrix a dengan dirinya sendiri mengebalikan number — DIOPTIMASI
 * @param a Matrix
 * @returns Number
 */
export default function dotDiv(a: Matrix): number {
  let value: number = 1;
  const data = a._data;
  const n = data.length;
  for (let i = 0; i < n; i++) {
    value /= data[i];
  }
  return value;
}
