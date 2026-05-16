import mj from "../math/index.js";
import Matrix from "../matrix/index.js";
import { sigmoidNative, isNativeAvailable } from "../math/rust_backend.js";
import { engine } from "../autodiff/engine.js";

export default function sigmoid(a: Matrix): [Matrix, Matrix] {
  let result: Matrix;
  let dResult: Matrix;

  if (isNativeAvailable()) {
    const res = new Float32Array(a._data.length);
    const grad = new Float32Array(a._data.length);
    sigmoidNative(a._data, res, grad);
    result = Matrix.fromFlat(res, a._shape);
    dResult = Matrix.fromFlat(grad, a._shape);
  } else {
    result = mj.map(a, (val) => 1 / (1 + Math.exp(-val)));
    dResult = mj.map(result, (val) => val * (1 - val));
  }

  const tape = engine.tape;
  if (tape) {
    tape.record([a], [result], (grad: Matrix) => {
      const gradA = mj.mul(grad, dResult);
      if (a.grad) a.grad.addInPlace(gradA);
      else a.grad = gradA;
    }, { saveInput: false, saveOutput: true });
  }

  return [result, dResult];
}
