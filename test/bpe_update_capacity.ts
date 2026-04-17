import { BPETokenizer } from "../src/tokenizer";
import * as fs from "fs";
import * as os from "os";
import * as path from "path";

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

console.log("\n=== BPE Update Capacity ===");

const tokenizer = new BPETokenizer({
  vocabSize: 12,
  minFrequency: 10,
});

tokenizer.train(["aa aa aa aa"]);
const beforeSize = tokenizer.getVocabSize();
const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), "bpe-capacity-"));
const vocabPath = path.join(tempDir, "vocab.json");

tokenizer.update(["zz zz zz zz"], 12);

tokenizer.save(vocabPath);
const saved = JSON.parse(fs.readFileSync(vocabPath, "utf-8")) as {
  vocab: Record<string, number>;
};
const maxId = Math.max(...Object.values(saved.vocab));
const encoded = tokenizer.encode("zz");

assert(beforeSize === 12, "fills tokenizer to configured capacity");
assert(tokenizer.getVocabSize() === 12, "does not grow past configured capacity on update");
assert(maxId < 12, "does not assign token ids beyond configured capacity");
assert(encoded.length > 0, "still encodes newly added text");
assert(tokenizer.getTokenId("z") !== undefined, "learns new character by reusing spare capacity");

const tokenizerWithReserved = new BPETokenizer({
  vocabSize: 8,
  minFrequency: 10,
  specialTokens: ["<RESERVED_0>", "<RESERVED_1>"],
});

tokenizerWithReserved.train(["aa aa aa aa"]);
const reservedBeforeSize = tokenizerWithReserved.getVocabSize();
const reservedZeroIdBefore = tokenizerWithReserved.getTokenId("<RESERVED_0>");
const reservedOneIdBefore = tokenizerWithReserved.getTokenId("<RESERVED_1>");

tokenizerWithReserved.update(["zz zz zz zz"], 8);

const reservedZeroIdAfter = tokenizerWithReserved.getTokenId("<RESERVED_0>");
const reservedOneIdAfter = tokenizerWithReserved.getTokenId("<RESERVED_1>");
const reservedEncoded = tokenizerWithReserved.encode("zz");

assert(reservedBeforeSize === 8, "fills tokenizer with reserved capacity included");
assert(tokenizerWithReserved.getVocabSize() === 8, "keeps vocab size fixed when only reserved slots remain");
assert(
  tokenizerWithReserved.getTokenId("z") !== undefined,
  "reuses reserved placeholder for new token during update",
);
assert(
  reservedZeroIdAfter === undefined || reservedOneIdAfter === undefined,
  "consumes at least one reserved placeholder slot",
);
assert(
  reservedEncoded.some((id) => id === reservedZeroIdBefore || id === reservedOneIdBefore),
  "new token inherits a former reserved token id",
);

console.log(`\nPASSED: ${passed}  FAILED: ${failed}`);

fs.rmSync(tempDir, { recursive: true, force: true });

if (failed > 0) {
  process.exit(1);
}
