import Matrix from "../matrix/index.js";
import { isNativeAvailable, clipGradientsNative } from "./rust_backend.js";

/**
 * Membatasi nilai matrix (gradient clipping) secara in-place
 * @param a Matrix
 * @param limit Batas maksimum/minimum (abs)
 */
export default function clipGradients(a: Matrix, limit: number): void {
  if (!Number.isFinite(limit) || limit < 0) {
    throw new Error(`clipGradients: limit must be a finite non-negative number, got ${limit}`);
  }
  if (isNativeAvailable()) {
    clipGradientsNative(a._data, limit);
  } else {
    const data = a._data;
    for (let i = 0; i < data.length; i++) {
        if (data[i] > limit) data[i] = limit;
        else if (data[i] < -limit) data[i] = -limit;
    }
  }
}
