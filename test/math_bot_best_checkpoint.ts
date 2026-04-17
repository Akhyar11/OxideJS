import { shouldSaveBestCheckpoint } from "../project/math-bot/main";

let passed = 0;
let failed = 0;

function assert(condition: boolean, name: string) {
  if (condition) {
    console.log(`  PASS: ${name}`);
    passed++;
    return;
  }

  console.error(`  FAIL: ${name}`);
  failed++;
}

console.log("\n=== Math Bot Best Checkpoint ===");

assert(shouldSaveBestCheckpoint(10, Infinity), "saves first checkpoint");
assert(shouldSaveBestCheckpoint(9.5, 10), "saves when loss improves");
assert(!shouldSaveBestCheckpoint(10, 10), "does not save when loss is equal");
assert(!shouldSaveBestCheckpoint(10.5, 10), "does not save when loss gets worse");

console.log(`\nPASSED: ${passed}  FAILED: ${failed}`);

if (failed > 0) {
  process.exit(1);
}
