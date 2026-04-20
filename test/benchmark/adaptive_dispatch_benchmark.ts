import { performance } from "perf_hooks";
import mj from "../../src/math";
import Matrix from "../../src/matrix";
import Adam from "../../src/optimizer/adam";
import Dense from "../../src/layers/dense";
import MultiHeadAttention from "../../src/layers/multiHeadAttention";
import { Transformers } from "../../src/models";
import { isNativeAvailable, setForceDisableNative } from "../../src/math/rust_backend";

type BenchResult = {
  group: string;
  mode: "js" | "native";
  size: "small" | "medium" | "large";
  scenario: string;
  avgMs: number;
};

function makeMatrix(rows: number, cols: number, seed = 1): Matrix {
  const data = new Float32Array(rows * cols);
  let s = seed >>> 0;
  for (let i = 0; i < data.length; i++) {
    s ^= s << 13;
    s ^= s >>> 17;
    s ^= s << 5;
    data[i] = (((s >>> 0) % 10_000) / 5000) - 1;
  }
  return Matrix.fromFlat(data, [rows, cols]);
}

function run(name: string, warmup: number, iterations: number, fn: () => void): number {
  for (let i = 0; i < warmup; i++) fn();
  const t0 = performance.now();
  for (let i = 0; i < iterations; i++) fn();
  const t1 = performance.now();
  return (t1 - t0) / iterations;
}

function withMode<T>(mode: "js" | "native", fn: () => T): T {
  const prev = !isNativeAvailable();
  setForceDisableNative(mode === "js");
  try {
    return fn();
  } finally {
    setForceDisableNative(prev);
  }
}

function benchmarkDot(results: BenchResult[]) {
  const configs = [
    { size: "small" as const, m: 16, k: 16, n: 16, warmup: 40, iter: 300 },
    { size: "medium" as const, m: 64, k: 64, n: 64, warmup: 20, iter: 120 },
    { size: "large" as const, m: 256, k: 256, n: 256, warmup: 4, iter: 20 },
  ];
  for (const cfg of configs) {
    const a = makeMatrix(cfg.m, cfg.k, 11 + cfg.m);
    const b = makeMatrix(cfg.k, cfg.n, 19 + cfg.n);
    const out = mj.zeros([cfg.m, cfg.n]);
    for (const mode of (["js", "native"] as const)) {
      if (mode === "native" && !isNativeAvailable()) continue;
      const avgMs = withMode(mode, () => run("dot", cfg.warmup, cfg.iter, () => {
        mj.dotProduct(a, b, out);
      }));
      results.push({ group: "dotProduct", mode, size: cfg.size, scenario: `${cfg.m}x${cfg.k} * ${cfg.k}x${cfg.n}`, avgMs });
    }
  }
}

function benchmarkSumAxis(results: BenchResult[]) {
  const configs = [
    { size: "small" as const, rows: 32, cols: 32, warmup: 40, iter: 300 },
    { size: "medium" as const, rows: 128, cols: 256, warmup: 20, iter: 140 },
    { size: "large" as const, rows: 256, cols: 1024, warmup: 8, iter: 40 },
  ];
  for (const cfg of configs) {
    const x = makeMatrix(cfg.rows, cfg.cols, 31 + cfg.rows);
    const out = mj.zeros([cfg.rows, 1]);
    for (const mode of (["js", "native"] as const)) {
      if (mode === "native" && !isNativeAvailable()) continue;
      const avgMs = withMode(mode, () => run("sumAxis", cfg.warmup, cfg.iter, () => {
        mj.sumAxis(x, 1, out);
      }));
      results.push({ group: "sumAxis", mode, size: cfg.size, scenario: `${cfg.rows}x${cfg.cols} axis=1`, avgMs });
    }
  }
}

function benchmarkClip(results: BenchResult[]) {
  const configs = [
    { size: "small" as const, rows: 32, cols: 32, warmup: 40, iter: 300 },
    { size: "medium" as const, rows: 128, cols: 256, warmup: 20, iter: 140 },
    { size: "large" as const, rows: 256, cols: 1024, warmup: 8, iter: 40 },
  ];
  for (const cfg of configs) {
    const x = makeMatrix(cfg.rows, cfg.cols, 71 + cfg.rows);
    for (const mode of (["js", "native"] as const)) {
      if (mode === "native" && !isNativeAvailable()) continue;
      const avgMs = withMode(mode, () => run("clip", cfg.warmup, cfg.iter, () => {
        mj.clipGradients(x, 1.0);
      }));
      results.push({ group: "clipGradients", mode, size: cfg.size, scenario: `${cfg.rows}x${cfg.cols}`, avgMs });
    }
  }
}

function benchmarkAdam(results: BenchResult[]) {
  const configs = [
    { size: "small" as const, rows: 1, cols: 512, warmup: 60, iter: 600 },
    { size: "medium" as const, rows: 1, cols: 8192, warmup: 20, iter: 200 },
    { size: "large" as const, rows: 1, cols: 65536, warmup: 6, iter: 50 },
  ];
  for (const cfg of configs) {
    for (const mode of (["js", "native"] as const)) {
      if (mode === "native" && !isNativeAvailable()) continue;
      const avgMs = withMode(mode, () => {
        const optimizer = new Adam([cfg.rows, cfg.cols]);
        const grad = makeMatrix(cfg.rows, cfg.cols, 101 + cfg.cols);
        return run("adam", cfg.warmup, cfg.iter, () => {
          optimizer.calculate(grad, 1e-4);
        });
      });
      results.push({ group: "adamUpdate", mode, size: cfg.size, scenario: `${cfg.rows}x${cfg.cols}`, avgMs });
    }
  }
}

function benchmarkDense(results: BenchResult[]) {
  const configs = [
    { size: "small" as const, units: 32, out: 64, seq: 8, warmup: 20, iter: 120 },
    { size: "medium" as const, units: 128, out: 256, seq: 32, warmup: 8, iter: 60 },
    { size: "large" as const, units: 256, out: 512, seq: 64, warmup: 3, iter: 18 },
  ];
  for (const cfg of configs) {
    for (const mode of (["js", "native"] as const)) {
      if (mode === "native" && !isNativeAvailable()) continue;
      const avgMs = withMode(mode, () => {
        const layer = new Dense({ units: cfg.units, outputUnits: cfg.out, activation: "relu", optimizer: "adam", status: "input" });
        const x = makeMatrix(cfg.units, cfg.seq, 401 + cfg.seq);
        const err = makeMatrix(cfg.out, cfg.seq, 509 + cfg.seq);
        return run("dense", cfg.warmup, cfg.iter, () => {
          layer.forward(x);
          layer.backward(mj.matrix([[]]), err);
        });
      });
      results.push({ group: "dense forward+backward", mode, size: cfg.size, scenario: `${cfg.units}->${cfg.out}, seq=${cfg.seq}`, avgMs });
    }
  }
}

function benchmarkMha(results: BenchResult[]) {
  const configs = [
    { size: "small" as const, units: 64, heads: 8, seq: 16, batch: 2, warmup: 10, iter: 50 },
    { size: "medium" as const, units: 128, heads: 8, seq: 64, batch: 4, warmup: 4, iter: 20 },
    { size: "large" as const, units: 128, heads: 8, seq: 128, batch: 8, warmup: 2, iter: 8 },
  ];
  for (const cfg of configs) {
    for (const mode of (["js", "native"] as const)) {
      if (mode === "native" && !isNativeAvailable()) continue;
      const avgMs = withMode(mode, () => {
        const layer = new MultiHeadAttention({ units: cfg.units, heads: cfg.heads, seqLen: cfg.seq, alpha: 1e-4, status: "input" });
        layer.compile({ alpha: 1e-4, optimizer: "adam" });
        const cols = cfg.seq * cfg.batch;
        const x = makeMatrix(cfg.units, cols, 701 + cols);
        const err = makeMatrix(cfg.units, cols, 709 + cols);
        return run("mha", cfg.warmup, cfg.iter, () => {
          layer.forward(x);
          layer.backward(mj.matrix([[]]), err);
        });
      });
      results.push({ group: "mha forward+backward", mode, size: cfg.size, scenario: `u=${cfg.units},h=${cfg.heads},seq=${cfg.seq},b=${cfg.batch}`, avgMs });
    }
  }
}

function benchmarkTrainingStep(results: BenchResult[]) {
  const configs = [
    { size: "small" as const, seq: 16, batch: 4, units: 64, heads: 8, vocab: 1024, warmup: 1, iter: 8 },
    { size: "medium" as const, seq: 32, batch: 8, units: 64, heads: 8, vocab: 2048, warmup: 1, iter: 5 },
    { size: "large" as const, seq: 64, batch: 8, units: 128, heads: 8, vocab: 4096, warmup: 1, iter: 3 },
  ];
  for (const cfg of configs) {
    for (const mode of (["js", "native"] as const)) {
      if (mode === "native" && !isNativeAvailable()) continue;
      const avgMs = withMode(mode, () => {
        const model = new Transformers({
          units: cfg.units,
          seqLen: cfg.seq,
          vocabSize: cfg.vocab,
          heads: cfg.heads,
          alpha: 1e-4,
          dropoutRate: 0,
          padTokenId: 0,
        });
        model.compile({ alpha: 1e-4, optimizer: "adam", error: "softmaxCrossEntropy" });
        const x = mj.zeros([cfg.seq, cfg.batch]);
        const y = mj.zeros([1, cfg.batch]);
        for (let i = 0; i < cfg.batch; i++) y._data[i] = (i * 7) % cfg.vocab;
        return run("train-step", cfg.warmup, cfg.iter, () => {
          model.forward(x);
          model.backward(y);
        });
      });
      results.push({ group: "training step", mode, size: cfg.size, scenario: `seq=${cfg.seq},batch=${cfg.batch},u=${cfg.units}`, avgMs });
    }
  }
}

function print(results: BenchResult[]) {
  console.log(`# Adaptive Dispatch Benchmark`);
  console.log(`nativeAvailable=${isNativeAvailable()}`);
  console.log(`| Group | Mode | Size | Scenario | Avg ms |`);
  console.log(`|---|---|---|---|---:|`);
  for (const r of results) {
    console.log(`| ${r.group} | ${r.mode} | ${r.size} | ${r.scenario} | ${r.avgMs.toFixed(4)} |`);
  }
}

function main() {
  const results: BenchResult[] = [];
  benchmarkDot(results);
  benchmarkSumAxis(results);
  benchmarkClip(results);
  benchmarkAdam(results);
  benchmarkDense(results);
  benchmarkMha(results);
  benchmarkTrainingStep(results);
  print(results);
}

main();
