import "./correctness";
import { runBenchmarkSuite } from "./benchmark";

async function main() {
  await runBenchmarkSuite();
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
