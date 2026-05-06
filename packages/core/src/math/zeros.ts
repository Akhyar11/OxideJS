import { MatrixShape } from "../@types/type.js";
import Matrix from "../matrix/index.js";

/**
 * Memberikan nilai matrix 0 — DIOPTIMASI dengan Float32Array
 */
export default function zeros(shape: MatrixShape): Matrix {
  // Float32Array sudah diisi 0 secara default — tidak perlu loop fill!
  return Matrix.fromFlat(new Float32Array(shape[0] * shape[1]), shape);
}
