import mj from "../src/math";
import LayerNormalization from "../src/layers/layerNormalization";
import { setForceDisableNative } from "../src/math/rust_backend";

function assert(condition: boolean, message: string) {
  if (!condition) throw new Error(message);
}

function assertCloseArray(actual: ArrayLike<number>, expected: number[], tol: number, label: string) {
  assert(actual.length === expected.length, `${label}: length mismatch ${actual.length} != ${expected.length}`);
  for (let i = 0; i < actual.length; i++) {
    const diff = Math.abs(actual[i] - expected[i]);
    if (diff > tol) {
      throw new Error(`${label}: mismatch at ${i} got=${actual[i]} expected=${expected[i]} diff=${diff}`);
    }
  }
}

const layer = new LayerNormalization({
  units: 3,
  status: "train",
  alpha: 0.01,
  optimizer: "sgd",
});
layer.compile({ alpha: 0.01, optimizer: "sgd" });
layer.load(
  [[1.2], [0.8], [1.1]],
  [[0.1], [-0.05], [0.03]]
);

const x = mj.matrix([
  [0.3, -0.4, 0.7, 0.2],
  [0.1, 0.5, -0.2, -0.3],
  [0.8, -0.1, 0.0, 0.4],
]);
const err = mj.matrix([
  [0.2, -0.1, 0.05, 0.1],
  [-0.3, 0.2, -0.2, 0.4],
  [0.1, 0.3, -0.1, -0.2],
]);

const expectedForward = [
  -0.31414997577667236,
  -1.0793516635894775,
  1.3399887084960938,
  0.7199521064758301,
  -0.7453189492225647,
  0.9962344169616699,
  -0.995194137096405,
  -1.2399413585662842,
  1.1751042604446411,
  0.08752919733524323,
  -0.32187244296073914,
  0.5199893116950989,
];

const expectedDx = [
  0.279770165681839,
  -0.6637882590293884,
  0.03527827933430672,
  -0.6612088680267334,
  -0.3036218583583832,
  0.12231669574975967,
  -0.10060200095176697,
  0.7746714353561401,
  0.0238516665995121,
  0.5414716005325317,
  0.06532374024391174,
  0.11973734200000763,
];

const expectedDGamma = [
  0.28377288579940796,
  0.07758095860481262,
  -0.7088972926139832,
];

const expectedDBeta = [
  0.25,
  0.10000000894069672,
  0.09999999403953552,
];

setForceDisableNative(false);
const forward = layer.forward(x);
const backward = layer.backward(mj.matrix([[]]), err);

assertCloseArray(forward._data, expectedForward, 1e-5, "forward regression");
assertCloseArray(backward._data, expectedDx, 1e-5, "dx regression");
assertCloseArray((layer as any).dGammaBuffer._data, expectedDGamma, 1e-5, "dGamma regression");
assertCloseArray((layer as any).dBetaBuffer._data, expectedDBeta, 1e-5, "dBeta regression");

setForceDisableNative(false);
console.log("layernorm_rust_regression passed");
