import { matrix2d } from "../@types/type.js";
import Matrix from "../matrix/index.js";

/**
 * Mebuat matrix
 * @param value number[][]
 * @returns Matrix
 */
export default function matrix(value: matrix2d): Matrix {
  return new Matrix({ array: value });
}
