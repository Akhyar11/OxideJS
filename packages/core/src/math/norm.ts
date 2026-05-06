import Matrix from "../matrix/index.js";

export default function norm(a: Matrix): number {
  let sum = 0;
  const data = a._data;
  for (let i = 0; i < data.length; i++) {
    sum += data[i] ** 2;
  }
  return Math.sqrt(sum);
}
