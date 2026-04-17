import * as fs from "fs";
import * as os from "os";
import * as path from "path";
import { loadOrCreateMathTokenizer } from "../project/math-bot/main";

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

console.log("\n=== Math Bot Resume Training ===");

const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), "math-bot-resume-"));
const vocabPath = path.join(tempDir, "math_vocab.json");

const initialTokenizer = loadOrCreateMathTokenizer(
  ["instruksi: hitung\ninput: 2 + 2\njawaban: 4"],
  vocabPath,
);
const initialVocabSize = initialTokenizer.getVocabSize();
const initialTokenId = initialTokenizer.getTokenId("4");

const updatedTokenizer = loadOrCreateMathTokenizer(
  [
    "instruksi: hitung\ninput: 2 + 2\njawaban: 4",
    "instruksi: hitung\ninput: 9 + 9\njawaban: 18",
  ],
  vocabPath,
);

assert(fs.existsSync(vocabPath), "keeps saved vocab file");
assert(updatedTokenizer.getVocabSize() >= initialVocabSize, "reuses and updates existing vocab");
assert(updatedTokenizer.getTokenId("4") === initialTokenId, "preserves existing token ids");
assert(updatedTokenizer.encode("jawaban: 18").length > 0, "encodes newly added corpus after update");

console.log(`\nPASSED: ${passed}  FAILED: ${failed}`);

fs.rmSync(tempDir, { recursive: true, force: true });

if (failed > 0) {
  process.exit(1);
}
