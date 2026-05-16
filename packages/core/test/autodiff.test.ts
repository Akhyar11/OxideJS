import { describe, expect, it } from "vitest";

import { engine } from "../src/autodiff/engine.js";
import mj from "../src/math/index.js";

describe("autodiff", () => {
  it("backpropagates through repeated use of the same tensor", () => {
    const x = mj.matrix([[2]]);
    const tape = engine.grad(() => {
      const y = mj.mul(x, x);
      return mj.add(y, y);
    });

    tape.backward(tape.result);

    expect(tape.result._value).toEqual([[8]]);
    expect(x.grad?._value).toEqual([[8]]);
  });

  it("backpropagates matrix multiplication gradients", () => {
    const a = mj.matrix([[1], [2]]);
    const b = mj.matrix([[3, 4]]);
    const tape = engine.grad(() => mj.dotProduct(a, b));

    tape.backward(tape.result);

    expect(a.grad?._value).toEqual([[7], [7]]);
    expect(b.grad?._value).toEqual([[3, 3]]);
  });
});
