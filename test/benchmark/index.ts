import { runFamilyRnnTrainingBenchmark } from "./testFamilyRnn.test.ts";
import { runTransformerModeBenchmark } from "./testFamilyTransformers.test.ts";
import { fileURLToPath } from "url";

export async function runBenchmarkSuite(): Promise<void> {
  console.log("\n📊 Running Benchmark Suite...");
  runFamilyRnnTrainingBenchmark();
  runTransformerModeBenchmark();
}

const isMain = process.argv[1] && fileURLToPath(import.meta.url) === (process.argv[1]);

if (isMain) {
  runBenchmarkSuite().catch((error) => {
    console.error(error);
    process.exit(1);
  });
}
