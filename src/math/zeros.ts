import { MatrixShape } from "../@types/type";
import Matrix from "../matrix";

/**
 * Memberikan nilai matrix 0 — DIOPTIMASI dengan Float64Array
 */
export default function zeros(shape: MatrixShape): Matrix {
  // Float64Array sudah diisi 0 secara default — tidak perlu loop fill!
  return Matrix.fromFlat(new Float64Array(shape[0] * shape[1]), shape);
}
