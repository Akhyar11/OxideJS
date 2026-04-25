import { MatrixShape } from "../@types/type";
import mj from "../math";
import Matrix from "../matrix";

export default class AdaGrad {
  shape: MatrixShape;
  sumGradien: Matrix;
  epsilon: number = 0.1;
  private updateBuffer: Matrix;
  constructor(shape: MatrixShape, epsilon: number) {
    this.shape = shape;
    this.sumGradien = mj.zeros(this.shape);
    this.updateBuffer = mj.zeros(this.shape);
    this.epsilon = epsilon;
  }

  calculate(a: Matrix, alpha: number) {
    const gradData = a._data;
    const sumData = this.sumGradien._data;
    const updateData = this.updateBuffer._data;

    for (let i = 0; i < gradData.length; i++) {
      const grad = gradData[i];
      const accumulated = sumData[i] + grad * grad;
      sumData[i] = accumulated;
      updateData[i] = alpha * grad / Math.sqrt(accumulated + this.epsilon);
    }

    return this.updateBuffer;
  }
}
