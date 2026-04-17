import * as fs from "fs";

/**
 * BPE (Byte Pair Encoding) Tokenizer
 * 
 * Algoritma:
 * 1. Memulai dari level karakter individual
 * 2. Menghitung pasangan karakter yang paling sering muncul
 * 3. Menggabungkan pasangan tersebut menjadi token baru
 * 4. Mengulangi proses sampai vocabulary mencapai ukuran target
 * 
 * Keunggulan:
 * - Bisa menangani kata-kata baru (OOV / Out-Of-Vocabulary)
 * - Kata yang sering muncul jadi 1 token, kata langka dipecah jadi subword
 * - Efisien untuk vocabulary berukuran terbatas
 */

// Token khusus
const PAD_TOKEN = "<PAD>";
const UNK_TOKEN = "<UNK>";
const BOS_TOKEN = "<BOS>";  // Beginning of Sequence
const EOS_TOKEN = "<EOS>";  // End of Sequence
const WORD_BOUNDARY = "▁";  // Penanda awal kata (seperti SentencePiece)

export interface BPEConfig {
  vocabSize: number;        // Ukuran vocabulary target
  minFrequency?: number;    // Frekuensi minimum untuk merge (default: 2)
  specialTokens?: string[]; // Token khusus tambahan
}

export interface BPEVocabData {
  vocab: Record<string, number>;       // token → id
  merges: [string, string][];          // Daftar merge rules, urut dari pertama dipelajari
  config: BPEConfig;
}

export default class BPETokenizer {
  private vocab: Map<string, number> = new Map();
  private reverseVocab: Map<number, string> = new Map();
  private merges: [string, string][] = [];
  private vocabSize: number;
  private minFrequency: number;
  private specialTokens: string[];

  constructor(config: BPEConfig) {
    this.vocabSize = config.vocabSize;
    this.minFrequency = config.minFrequency ?? 2;
    this.specialTokens = [
      PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN,
      ...(config.specialTokens ?? [])
    ];
  }

  /**
   * Melatih tokenizer dari corpus teks
   * @param texts - Array of strings sebagai training data
   */
  train(texts: string[]): void {
    // === STEP 1: Inisialisasi vocabulary dengan special tokens ===
    this.vocab.clear();
    this.merges = [];
    let nextId = 0;

    for (const token of this.specialTokens) {
      this.vocab.set(token, nextId++);
    }

    // === STEP 2: Tokenisasi awal — pecah setiap kata jadi karakter ===
    // Pre-tokenize: split berdasarkan spasi, tambahkan word boundary marker
    // "saya makan" → ["▁s", "a", "y", "a", " ", "▁m", "a", "k", "a", "n"]
    
    // Hitung frekuensi setiap kata
    const wordFreq: Map<string, number> = new Map();
    for (const text of texts) {
      const words = text.trim().split(/\s+/);
      for (const word of words) {
        if (word.length === 0) continue;
        const key = WORD_BOUNDARY + word; // Tambah boundary marker
        wordFreq.set(key, (wordFreq.get(key) ?? 0) + 1);
      }
    }

    // Pecah setiap kata menjadi karakter individual
    // "▁makan" → ["▁m", "a", "k", "a", "n"] → Tapi kita simpan sebagai ["▁", "m", "a", "k", "a", "n"]
    // Kita gunakan representasi: list of symbols per word
    type WordSymbols = { symbols: string[]; freq: number };
    const corpus: WordSymbols[] = [];

    for (const [word, freq] of wordFreq) {
      const chars = [...word]; // Split Unicode-safe
      corpus.push({ symbols: chars, freq });

      // Tambahkan setiap karakter ke vocab jika belum ada
      for (const char of chars) {
        if (!this.vocab.has(char)) {
          this.vocab.set(char, nextId++);
        }
      }
    }

    this.runBPE(corpus, nextId);
  }

  /**
   * Update tokenizer dengan data baru (Incremental Training)
   * Menambahkan karakter baru atau merge baru tanpa merubah ID lama.
   */
  update(texts: string[], newVocabSize?: number): void {
    if (newVocabSize && newVocabSize > this.vocabSize) {
      this.vocabSize = newVocabSize;
    }

    let nextId = 0;
    for (const id of this.vocab.values()) {
      if (id >= nextId) nextId = id + 1;
    }

    const wordFreq: Map<string, number> = new Map();
    for (const text of texts) {
      const words = text.trim().split(/\s+/);
      for (const word of words) {
        if (word.length === 0) continue;
        const key = WORD_BOUNDARY + word;
        wordFreq.set(key, (wordFreq.get(key) ?? 0) + 1);
      }
    }

    type WordSymbols = { symbols: string[]; freq: number };
    const corpus: WordSymbols[] = [];

    for (const [word, freq] of wordFreq) {
      let symbols = [...word];
      
      // 1. Pastikan karakter dasar ada di vocab
      for (const char of symbols) {
        if (!this.vocab.has(char)) {
          const allocatedId = this.allocateTokenId(char, nextId);
          if (allocatedId !== null && allocatedId >= nextId) {
            nextId = allocatedId + 1;
          }
        }
      }

      // 2. Terapkan merge rules yang sudah ada untuk melihat seberapa "pecah" kata ini
      for (const [left, right] of this.merges) {
        const merged = left + right;
        symbols = this.applyMerge(symbols, left, right, merged);
      }
      
      // Hanya masukkan ke korpus training jika kata tersebut masih "kompleks" (> 3 token)
      // Ini agar BPE fokus memperpendek kata-kata yang masih terlalu panjang.
      if (symbols.length > 3) {
        corpus.push({ symbols, freq });
      }
    }

    if (corpus.length === 0) {
      console.log("[BPE] Semua kata sudah sangat ringkas (<= 3 token). Tidak ada merge baru yang perlu dipelajari.");
      this.buildReverseVocab();
      return;
    }

    console.log(`[BPE] Melanjutkan training dengan ${corpus.length} kata kompleks. Vocab saat ini: ${this.vocab.size}, Target: ${this.vocabSize}`);
    this.runBPE(corpus, nextId);
  }

  private runBPE(corpus: { symbols: string[]; freq: number }[], nextId: number): void {
    // === STEP 3: Iterasi BPE — gabungkan pasangan paling sering ===
    while (this.vocab.size < this.vocabSize) {
      // 3a. Hitung frekuensi semua pasangan yang bersebelahan
      const pairFreq: Map<string, number> = new Map();

      for (const { symbols, freq } of corpus) {
        for (let i = 0; i < symbols.length - 1; i++) {
          // Gunakan separator NULL (\0) yang hampir mustahil ada di text dataset
          const pair = symbols[i] + "\0" + symbols[i + 1];
          pairFreq.set(pair, (pairFreq.get(pair) ?? 0) + freq);
        }
      }

      if (pairFreq.size === 0) break; // Tidak ada lagi yang bisa di-merge

      // 3b. Cari pasangan dengan frekuensi tertinggi
      let bestPair = "";
      let bestFreq = 0;
      for (const [pair, freq] of pairFreq) {
        if (freq > bestFreq) {
          bestFreq = freq;
          bestPair = pair;
        }
      }

      // Hentikan jika frekuensi pasangan terbaik di bawah minimum
      if (bestFreq < this.minFrequency) {
        console.log(`[BPE] Berhenti: frekuensi tertinggi (${bestFreq}) < minimum (${this.minFrequency})`);
        break;
      }

      // 3c. Gabungkan pasangan terbaik
      const separatorIndex = bestPair.indexOf("\0");
      const left = bestPair.substring(0, separatorIndex);
      const right = bestPair.substring(separatorIndex + 1);
      const merged = left + right;

      // HANYA tambah ke merges jika belum ada
      const alreadyMerged = this.merges.some(([l, r]) => l === left && r === right);
      
      if (!alreadyMerged) {
        this.merges.push([left, right]);
        if (!this.vocab.has(merged)) {
          const allocatedId = this.allocateTokenId(merged, nextId);
          if (allocatedId === null) {
            this.merges.pop();
            break;
          }
          if (allocatedId >= nextId) {
            nextId = allocatedId + 1;
          }
        }
        
        // 3d. Terapkan merge ke seluruh corpus
        for (const entry of corpus) {
          entry.symbols = this.applyMerge(entry.symbols, left, right, merged);
        }
      } else {
        // Jika sudah ada tapi kita sampai sini, berarti pasangan ini tidak bisa di-merge lagi
        // (biasanya karena bug split atau data corrupt). Kita hentikan untuk mencegah infinite loop.
        console.warn(`[BPE] Warning: Pasangan "${left}" + "${right}" sudah di-merge tapi terpilih kembali. Menghentikan.`);
        break;
      }

      if (this.merges.length % 100 === 0) {
        console.log(`[BPE] Merge #${this.merges.length}: "${left}" + "${right}" → "${merged}" (freq: ${bestFreq}), vocab: ${this.vocab.size}`);
      }
    }

    // === STEP 4: Isi sisa kapasitas dengan token placeholder jika belum mencapai target ===
    let unusedIdx = 0;
    while (this.vocab.size < this.vocabSize) {
      const placeholder = `<UNUSED_${unusedIdx++}>`;
      if (!this.vocab.has(placeholder)) {
        this.vocab.set(placeholder, nextId++);
      }
    }

    // Build reverse vocab
    this.buildReverseVocab();

    console.log(`[BPE] Selesai! Vocabulary size: ${this.vocab.size}, Merges: ${this.merges.length}`);
  }

  /**
   * Terapkan satu merge rule ke array symbols
   */
  private applyMerge(symbols: string[], left: string, right: string, merged: string): string[] {
    const result: string[] = [];
    let i = 0;
    while (i < symbols.length) {
      if (i < symbols.length - 1 && symbols[i] === left && symbols[i + 1] === right) {
        result.push(merged);
        i += 2; // Skip kedua simbol yang di-merge
      } else {
        result.push(symbols[i]);
        i++;
      }
    }
    return result;
  }

  private allocateTokenId(token: string, nextId: number): number | null {
    const existingId = this.vocab.get(token);
    if (existingId !== undefined) {
      return existingId;
    }

    const reusablePlaceholder = this.findReusablePlaceholder();
    if (reusablePlaceholder) {
      this.vocab.delete(reusablePlaceholder.token);
      this.vocab.set(token, reusablePlaceholder.id);
      return reusablePlaceholder.id;
    }

    if (this.vocab.size >= this.vocabSize) {
      return null;
    }

    this.vocab.set(token, nextId);
    return nextId;
  }

  private findReusablePlaceholder(): { token: string; id: number } | null {
    // Cari token <UNUSED_*> dulu (prioritas pertama)
    for (const [token, id] of this.vocab) {
      if (token.startsWith("<UNUSED_") && token.endsWith(">")) {
        return { token, id };
      }
    }

    // Jika tidak ada <UNUSED_*>, cari <RESERVED_*> (wadah cadangan)
    for (const [token, id] of this.vocab) {
      if (token.startsWith("<RESERVED_") && token.endsWith(">")) {
        return { token, id };
      }
    }

    return null;
  }

  /**
   * Encode teks menjadi array token ID
   */
  encode(text: string): number[] {
    const words = text.trim().split(/\s+/);
    const tokenIds: number[] = [];

    for (const word of words) {
      if (word.length === 0) continue;

      // Pecah kata jadi karakter dengan word boundary
      let symbols = [...(WORD_BOUNDARY + word)];

      // Terapkan semua merge rules secara berurutan
      for (const [left, right] of this.merges) {
        const merged = left + right;
        symbols = this.applyMerge(symbols, left, right, merged);
      }

      // Convert symbols ke ID
      for (const sym of symbols) {
        const id = this.vocab.get(sym);
        if (id !== undefined) {
          tokenIds.push(id);
        } else {
          // Token tidak dikenal → UNK
          tokenIds.push(this.vocab.get(UNK_TOKEN)!);
        }
      }
    }

    return tokenIds;
  }

  /**
   * Encode teks dan bungkus dengan BOS/EOS
   */
  encodeWithSpecial(text: string): number[] {
    const bos = this.vocab.get(BOS_TOKEN)!;
    const eos = this.vocab.get(EOS_TOKEN)!;
    return [bos, ...this.encode(text), eos];
  }

  /**
   * Decode array token ID kembali menjadi teks
   */
  decode(ids: number[]): string {
    const tokens: string[] = [];
    for (const id of ids) {
      const token = this.reverseVocab.get(id);
      if (token && token !== BOS_TOKEN && token !== EOS_TOKEN && token !== PAD_TOKEN) {
        tokens.push(token);
      }
    }
    // Gabungkan lalu ganti word boundary dengan spasi
    return tokens.join("").replace(new RegExp(WORD_BOUNDARY, "g"), " ").trim();
  }

  /**
   * Dapatkan ukuran vocabulary saat ini
   */
  getVocabSize(): number {
    return this.vocab.size;
  }

  /**
   * Dapatkan ID dari sebuah token
   */
  getTokenId(token: string): number | undefined {
    return this.vocab.get(token);
  }

  /**
   * Dapatkan token dari sebuah ID
   */
  getToken(id: number): string | undefined {
    return this.reverseVocab.get(id);
  }

  /**
   * Dapatkan ID untuk PAD token (berguna untuk padding sequences)
   */
  getPadId(): number {
    return this.vocab.get(PAD_TOKEN)!;
  }

  /**
   * Pad sequences agar panjangnya sama
   */
  padSequence(ids: number[], maxLength: number): number[] {
    const padId = this.getPadId();
    if (ids.length >= maxLength) {
      return ids.slice(0, maxLength);
    }
    return [...ids, ...Array(maxLength - ids.length).fill(padId)];
  }

  // === SAVE / LOAD ===

  /**
   * Simpan vocabulary dan merge rules ke file JSON
   */
  save(filepath: string): void {
    const data: BPEVocabData = {
      vocab: Object.fromEntries(this.vocab),
      merges: this.merges,
      config: {
        vocabSize: this.vocabSize,
        minFrequency: this.minFrequency,
        specialTokens: this.specialTokens,
      }
    };
    fs.writeFileSync(filepath, JSON.stringify(data, null, 2), "utf-8");
    console.log(`[BPE] Vocabulary disimpan ke: ${filepath}`);
  }

  /**
   * Muat vocabulary dan merge rules dari file JSON
   */
  static load(filepath: string): BPETokenizer {
    const raw = fs.readFileSync(filepath, "utf-8");
    const data: BPEVocabData = JSON.parse(raw);

    const tokenizer = new BPETokenizer(data.config);
    tokenizer.vocab = new Map(Object.entries(data.vocab).map(([k, v]) => [k, v as number]));
    tokenizer.merges = data.merges;
    tokenizer.buildReverseVocab();

    console.log(`[BPE] Vocabulary dimuat dari: ${filepath} (${tokenizer.vocab.size} tokens, ${tokenizer.merges.length} merges)`);
    return tokenizer;
  }

  /**
   * Build reverse vocab map (id → token)
   */
  private buildReverseVocab(): void {
    this.reverseVocab.clear();
    for (const [token, id] of this.vocab) {
      this.reverseVocab.set(id, token);
    }
  }

  /**
   * Print vocabulary summary
   */
  summary(): void {
    console.log("=== BPE Tokenizer Summary ===");
    console.log(`Vocabulary size : ${this.vocab.size}`);
    console.log(`Merge rules     : ${this.merges.length}`);
    console.log(`Special tokens  : ${this.specialTokens.join(", ")}`);
    console.log(`\nSample vocabulary (first 20):`);
    let count = 0;
    for (const [token, id] of this.vocab) {
      if (count >= 20) break;
      const display = token.replace(WORD_BOUNDARY, "▁");
      console.log(`  [${id}] "${display}"`);
      count++;
    }
    console.log("=============================");
  }
}
