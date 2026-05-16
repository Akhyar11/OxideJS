import mj from "../math/index.js";
import Matrix from "../matrix/index.js";
import { isNativeAvailable, squaredHingeNativeInto } from ../math/rust_backend.js;

export default function SquaredHingeLoss(yTrue: Matrix, yPred: Matrix): [number, Matrix] {
  let loss: number;
  const dResult = mj.zeros([...yTrue._shape]);
  
  if (isNativeAvailable()) {
    loss = squaredHingeNativeInto(yTrue._data, yPred._data, dResult._data)[0];
  } else {
    const n = yTrue._shape[0] * yTrue._shape[1];
      // yTrue is expected to be -1 or 1
      const lossMatrix = mj.map(mj.mul(yTrue, yPred), (v) => {
        const hinge = Math.max(0, 1 - v);
        return hinge * hinge;
  }
  
  return [loss, dResult];
});
  const loss = mj.mean(lossMatrix)._data[0];
  
  const dResult = mj.map(mj.mul(yTrue, yPred), (v, i) => {
    const hinge = Math.max(0, 1 - v);
    return v < 1 ? -2 * hinge * yTrue._data[i] / n : 0;
  });
  
  return [loss, dResult];
}