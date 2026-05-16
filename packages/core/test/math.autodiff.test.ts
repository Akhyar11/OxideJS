import { beforeEach, describe, expect, it } from "vitest";

import Matrix from "../src/matrix/index.js";
import mj from "../src/math/index.js";
import { engine } from "../src/autodiff/engine.js";
import { setForceDisableNative } from "../src/math/rust_backend.js";

type LossBuilder = (...inputs: Matrix[]) => Matrix;

function cloneInputs(inputs: Matrix[]): Matrix[] {
  return inputs.map((m) => m.clone());
}

function scalarValue(m: Matrix): number {
  return m._data[0];
}

function expectGradientClose(actual: number, expected: number, tolerance = 1e-2) {
  expect(Math.abs(actual - expected)).toBeLessThanOrEqual(tolerance);
}

function runGradientCheck(
  inputs: Matrix[],
  buildLoss: LossBuilder,
  options: { epsilon?: number; tolerance?: number } = {}
) {
  const epsilon = options.epsilon ?? 1e-3;
  const tolerance = options.tolerance ?? 2e-2;

  const autodiffInputs = cloneInputs(inputs);
  const tape = engine.grad(() => buildLoss(...autodiffInputs));
  tape.backward(tape.result);

  autodiffInputs.forEach((input, inputIdx) => {
    for (let dataIdx = 0; dataIdx < input._data.length; dataIdx++) {
      const plusInputs = cloneInputs(inputs);
      plusInputs[inputIdx]._data[dataIdx] += epsilon;
      const plusLoss = scalarValue(buildLoss(...plusInputs));

      const minusInputs = cloneInputs(inputs);
      minusInputs[inputIdx]._data[dataIdx] -= epsilon;
      const minusLoss = scalarValue(buildLoss(...minusInputs));

      const numericalGrad = (plusLoss - minusLoss) / (2 * epsilon);
      const autodiffGrad = input.grad!._data[dataIdx];
      expectGradientClose(autodiffGrad, numericalGrad, tolerance);
    }
  });
}

beforeEach(() => {
  setForceDisableNative(true);
});

describe("math autodiff numerical gradient checks", () => {
  it("checks add", () => {
    runGradientCheck(
      [mj.matrix([[1, 2], [3, 4]]), mj.matrix([[5, 6], [7, 8]])],
      (a, b) => mj.mean(mj.add(a, b))
    );
  });

  it("checks sub", () => {
    runGradientCheck(
      [mj.matrix([[5, 6], [7, 8]]), mj.matrix([[1, 2], [3, 4]])],
      (a, b) => mj.mean(mj.sub(a, b))
    );
  });

  it("checks mul", () => {
    runGradientCheck(
      [mj.matrix([[1, 2], [3, 4]]), mj.matrix([[2, 3], [4, 5]])],
      (a, b) => mj.mean(mj.mul(a, b))
    );
  });

  it("checks div", () => {
    runGradientCheck(
      [mj.matrix([[2, 4], [6, 8]]), mj.matrix([[1, 2], [3, 4]])],
      (a, b) => mj.mean(mj.div(a, b))
    );
  });

  it("checks dotProduct", () => {
    runGradientCheck(
      [mj.matrix([[1, 2], [3, 4]]), mj.matrix([[5, 6], [7, 8]])],
      (a, b) => mj.mean(mj.dotProduct(a, b)),
      { tolerance: 3e-2 }
    );
  });

  it("checks concat", () => {
    runGradientCheck(
      [mj.matrix([[1, 2]]), mj.matrix([[3, 4]])],
      (a, b) => mj.mean(mj.concat(a, b))
    );
  });

  it("checks expm", () => {
    runGradientCheck(
      [mj.matrix([[0.2, 0.4], [0.1, 0.3]])],
      (a) => mj.mean(mj.expm(a))
    );
  });

  it("checks flatten", () => {
    runGradientCheck(
      [mj.matrix([[1, 2], [3, 4]])],
      (a) => mj.mean(mj.flatten(a))
    );
  });

  it("checks logm", () => {
    runGradientCheck(
      [mj.matrix([[1.5, 2.5], [3.5, 4.5]])],
      (a) => mj.mean(mj.logm(a))
    );
  });

  it("checks mean", () => {
    runGradientCheck(
      [mj.matrix([[1, 2], [3, 4]])],
      (a) => mj.mean(a)
    );
  });

  it("checks pow", () => {
    runGradientCheck(
      [mj.matrix([[1.5, 2.0], [2.5, 3.0]])],
      (a) => mj.mean(mj.pow(a, 3)),
      { tolerance: 3e-2 }
    );
  });

  it("checks reshape", () => {
    runGradientCheck(
      [mj.matrix([[1, 2], [3, 4]])],
      (a) => mj.mean(mj.reshape(a, [1, 4]))
    );
  });

  it("checks sumAxis", () => {
    runGradientCheck(
      [mj.matrix([[1, 2], [3, 4]])],
      (a) => mj.mean(mj.sumAxis(a, 1))
    );
  });

  it("checks transpose", () => {
    runGradientCheck(
      [mj.matrix([[1, 2], [3, 4]])],
      (a) => mj.mean(mj.transpose(a))
    );
  });

  it("checks addBias", () => {
    runGradientCheck(
      [mj.matrix([[1, 2], [3, 4]]), mj.matrix([[0.5], [1.5]])],
      (a, bias) => {
        mj.addBias(a, bias);
        return mj.mean(a);
      }
    );
  });

  it("checks softmax", () => {
    runGradientCheck(
      [mj.matrix([[1.2], [0.7]])],
      (a) => mj.mean(mj.softmax(a)),
      { tolerance: 3e-2 }
    );
  });

  it("checks convolution", () => {
    runGradientCheck(
      [mj.matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), mj.matrix([[1, 0], [0, -1]])],
      (a, kernel) => mj.mean(mj.convolution(a, kernel)),
      { tolerance: 5e-2 }
    );
  });
});
