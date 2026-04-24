import { runBenchmarkSuite } from "./benchmark";
import { runCorrectnessSuite } from "./correctness";

async function main() {
  runCorrectnessSuite();
  await runBenchmarkSuite();
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
