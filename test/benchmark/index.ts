import { runAllSyntheticBaselineBenchmarks } from "./synthetic_baseline_benchmark";

export async function runBenchmarkSuite() {
  await runAllSyntheticBaselineBenchmarks();
}
