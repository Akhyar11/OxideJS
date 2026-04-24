/**
 * Benchmark: trimPadding effect on model.fit() throughput
 *
 * Mengukur perbedaan throughput saat training dengan dan tanpa trimPadding
 * menggunakan data yang memiliki variasi panjang sequence (banyak PAD).
 */
import { performance } from "perf_hooks";
import Matrix from "../../src/matrix";
import mj from "../../src/math";
import { Transformers } from "../../src/models";

// ─── Config ────────────────────────────────────────────────────────

const SEQ_LEN = 256;
const UNITS = 64;
const HEADS = 8;
const NUM_BLOCKS = 1;
const VOCAB_SIZE = 2000;
const PAD_ID = 0;
const ALPHA = 1e-4;
const BATCH_SIZE = 16;
const EPOCHS = 3;
const NUM_SAMPLES = 128;

// ─── Data Generator ────────────────────────────────────────────────

/**
 * Generate samples with varying lengths to simulate real padding scenarios.
 * Short samples = banyak PAD, sehingga trimPadding bisa trim banyak.
 */
function generateVariableLengthData(
    numSamples: number,
    seqLen: number,
    paddingSide: "left" | "right",
): { X: Matrix[]; Y: Matrix[] } {
    const X: Matrix[] = [];
    const Y: Matrix[] = [];

    for (let i = 0; i < numSamples; i++) {
        // Variasi panjang: 20% - 80% dari seqLen
        const actualLen = Math.floor(seqLen * (0.2 + 0.6 * Math.random()));

        const xData = new Float32Array(seqLen);
        const yData = new Float32Array(seqLen);
        xData.fill(PAD_ID);
        yData.fill(PAD_ID);

        if (paddingSide === "right") {
            // Token di awal, PAD di akhir
            for (let j = 0; j < actualLen; j++) {
                xData[j] = 1 + (j % (VOCAB_SIZE - 1));
                yData[j] = j < actualLen - 1 ? 1 + ((j + 1) % (VOCAB_SIZE - 1)) : PAD_ID;
            }
        } else {
            // PAD di awal, token di akhir
            const offset = seqLen - actualLen;
            for (let j = 0; j < actualLen; j++) {
                xData[offset + j] = 1 + (j % (VOCAB_SIZE - 1));
                yData[offset + j] = j < actualLen - 1 ? 1 + ((j + 1) % (VOCAB_SIZE - 1)) : PAD_ID;
            }
        }

        X.push(Matrix.fromFlat(xData, [seqLen, 1]));
        Y.push(Matrix.fromFlat(yData, [seqLen, 1]));
    }

    return { X, Y };
}

// ─── Benchmark Runner ──────────────────────────────────────────────

interface BenchmarkResult {
    label: string;
    paddingSide: "left" | "right";
    trimPadding: boolean;
    epochs: number;
    samples: number;
    batchSize: number;
    seqLen: number;
    totalMs: number;
    msPerEpoch: number;
    msPerSample: number;
    samplesPerSec: number;
}

function runFitBenchmark(
    label: string,
    paddingSide: "left" | "right",
    trimPadding: boolean,
): BenchmarkResult {
    const { X, Y } = generateVariableLengthData(NUM_SAMPLES, SEQ_LEN, paddingSide);

    const model = new Transformers({
        units: UNITS,
        seqLen: SEQ_LEN,
        vocabSize: VOCAB_SIZE,
        heads: HEADS,
        numBlocks: NUM_BLOCKS,
        alpha: ALPHA,
        padTokenId: PAD_ID,
        dropoutRate: 0.0,
    });
    model.compile({ alpha: ALPHA, optimizer: "adam", error: "softmaxCrossEntropy" });

    // Warmup: 1 epoch
    model.fit(X, Y, 1, {
        batchSize: BATCH_SIZE,
        trimPadding,
        paddingSide,
        shuffle: false,
        verbose: false,
    });

    // Measure
    const start = performance.now();
    model.fit(X, Y, EPOCHS, {
        batchSize: BATCH_SIZE,
        trimPadding,
        paddingSide,
        shuffle: false,
        verbose: false,
    });
    const elapsed = performance.now() - start;

    const totalSamples = NUM_SAMPLES * EPOCHS;
    return {
        label,
        paddingSide,
        trimPadding,
        epochs: EPOCHS,
        samples: NUM_SAMPLES,
        batchSize: BATCH_SIZE,
        seqLen: SEQ_LEN,
        totalMs: elapsed,
        msPerEpoch: elapsed / EPOCHS,
        msPerSample: elapsed / totalSamples,
        samplesPerSec: (totalSamples / elapsed) * 1000,
    };
}

// ─── Main ──────────────────────────────────────────────────────────

function main() {
    console.log("╔══════════════════════════════════════════════════════╗");
    console.log("║   Benchmark: trimPadding Effect on model.fit()      ║");
    console.log("╚══════════════════════════════════════════════════════╝\n");
    console.log(`  seqLen=${SEQ_LEN}, units=${UNITS}, heads=${HEADS}, numBlocks=${NUM_BLOCKS}`);
    console.log(`  vocabSize=${VOCAB_SIZE}, batchSize=${BATCH_SIZE}, epochs=${EPOCHS}`);
    console.log(`  samples=${NUM_SAMPLES} (variabel 20%-80% seqLen)\n`);

    const results: BenchmarkResult[] = [];

    // Right padding
    console.log("Running: right-pad WITHOUT trimPadding...");
    results.push(runFitBenchmark("right_no_trim", "right", false));
    console.log(`  → ${results[results.length - 1].samplesPerSec.toFixed(2)} samples/s\n`);

    console.log("Running: right-pad WITH trimPadding...");
    results.push(runFitBenchmark("right_trim", "right", true));
    console.log(`  → ${results[results.length - 1].samplesPerSec.toFixed(2)} samples/s\n`);

    // Left padding
    console.log("Running: left-pad WITHOUT trimPadding...");
    results.push(runFitBenchmark("left_no_trim", "left", false));
    console.log(`  → ${results[results.length - 1].samplesPerSec.toFixed(2)} samples/s\n`);

    console.log("Running: left-pad WITH trimPadding...");
    results.push(runFitBenchmark("left_trim", "left", true));
    console.log(`  → ${results[results.length - 1].samplesPerSec.toFixed(2)} samples/s\n`);

    // Summary table
    console.log("═══════════════════════════════════════════════════════\n");
    console.log("Results:\n");
    console.table(
        results.map((r) => ({
            Label: r.label,
            PaddingSide: r.paddingSide,
            TrimPadding: r.trimPadding ? "YES" : "NO",
            "ms/epoch": r.msPerEpoch.toFixed(2),
            "ms/sample": r.msPerSample.toFixed(2),
            "samples/s": r.samplesPerSec.toFixed(2),
        }))
    );

    // Speedup
    const rightNoTrim = results.find((r) => r.label === "right_no_trim")!;
    const rightTrim = results.find((r) => r.label === "right_trim")!;
    const leftNoTrim = results.find((r) => r.label === "left_no_trim")!;
    const leftTrim = results.find((r) => r.label === "left_trim")!;

    console.log("\nSpeedup from trimPadding:");
    console.log(`  Right-pad: ${(rightTrim.samplesPerSec / rightNoTrim.samplesPerSec).toFixed(2)}x`);
    console.log(`  Left-pad:  ${(leftTrim.samplesPerSec / leftNoTrim.samplesPerSec).toFixed(2)}x`);

    console.log(JSON.stringify({ benchmark: "trimPadding_effect", results }, null, 2));
}

main();
