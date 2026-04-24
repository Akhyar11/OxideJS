import { runFamilyRnnTrainingBenchmark } from "./testFamilyRnn.test";
import { runTransformerModeBenchmark } from "./testFamilyTransformers.test";

export async function runBenchmarkSuite(): Promise<void> {
  runFamilyRnnTrainingBenchmark();
  runTransformerModeBenchmark();
}

if (require.main === module) {
  runBenchmarkSuite().catch((error) => {
    console.error(error);
    process.exitCode = 1;
  });
}
