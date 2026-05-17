import Matrix from "../matrix/index.js";
import { isNativeAvailable, absmNative } from "./rust_backend.js";

/**
 * Memberikan nilai absolute pada matrix — DIOPTIMASI
 * @param a Matrix
 * @returns Matrix
 */
export default function absm(a: Matrix): Matrix {
  const result = new Float32Array(a._data.length);
  const data = a._data;

  if (isNativeAvailable()) {
    absmNative(data, result);
  } else {
    for (let i = 0; i < data.length; i++) {
      result[i] = Math.abs(data[i]);
    }
  }

  return Matrix.fromFlat(result, [a._shape[0], a._shape[1]]);
}
