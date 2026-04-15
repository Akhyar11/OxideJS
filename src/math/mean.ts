import Matrix from "../matrix";

/**
 * Merata rata nilai matrix — DIOPTIMASI
 * @param a Matrix
 * @returns Number
 */
export default function mean(a: Matrix): number {
  let value: number = 0;
  const data = a._data;
  const n = data.length;
  
  for (let i = 0; i < n; i++) {
    value += data[i];
  }

  return value / n;
}
