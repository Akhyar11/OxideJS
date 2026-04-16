import * as fs from "fs";
import * as path from "path";
import { BPETokenizer } from "../../src/tokenizer";

/**
 * SCRIPT KHUSUS TRAINING TOKENIZER
 * Digunakan untuk membuat vocabulary (BPE) dari dataset cerita rakyat.
 */

async function main() {
    console.log("=== 1. Loading Text Dataset ===\n");
    const dataPath = path.join(__dirname, "..", "..", "dataset", "kumpulan-kata-indonesia.txt");

    if (!fs.existsSync(dataPath)) {
        console.error(`Error: File dataset tidak ditemukan di ${dataPath}`);
        return;
    }

    const corpus = fs.readFileSync(dataPath, "utf-8").toLowerCase();
    const lines = corpus.split("\n").filter(l => l.trim().length > 0);
    console.log(`Loaded ${lines.length} lines.`);

    // Konfigurasi
    const VOCAB_SIZE = 20000;
    const RESERVED_COUNT = 5000; // Menambah 100 slot kosong untuk token baru di masa depan
    const botDatasetDir = path.join(__dirname, "dataset");
    const generativeVocabPath = path.join(botDatasetDir, "generative_vocab.json");

    if (!fs.existsSync(botDatasetDir)) {
        fs.mkdirSync(botDatasetDir, { recursive: true });
    }

    console.log(`\n=== 2. Training BPE Tokenizer ===`);
    console.log(`Target Vocab    : ${VOCAB_SIZE}`);
    console.log(`Reserved Slots  : ${RESERVED_COUNT} (<RESERVED_0> s/d <RESERVED_99>)`);

    // Membuat daftar reserved tokens sebagai wadah masa depan
    const reservedTokens = Array.from({ length: RESERVED_COUNT }, (_, i) => `<RESERVED_${i}>`);

    const tokenizer = new BPETokenizer({
        vocabSize: VOCAB_SIZE,
        minFrequency: 2,
        specialTokens: ["<SEP>", ...reservedTokens]
    });

    const startTime = Date.now();
    console.log("\nProses training dimulai... (Mungkin memakan waktu beberapa menit untuk 127rb kata)");
    tokenizer.train(lines);
    const duration = (Date.now() - startTime) / 1000;

    console.log(`\nTraining selesai dalam ${duration.toFixed(2)} detik.`);

    // Simpan hasil
    tokenizer.save(generativeVocabPath);
    console.log(`\n[SUCCESS] Tokenizer disimpan ke: ${generativeVocabPath}`);

    // Tampilkan ringkasan
    tokenizer.summary();
}

main().catch(error => {
    console.error("Training Tokenizer failed:", error);
    process.exit(1);
});
