import * as fs from "fs";

export interface MathTrainingRecord {
  instruction?: string;
  input?: string;
  output?: string;
  prompt?: string;
  response?: string;
}

function cleanText(value: unknown): string {
  return typeof value === "string" ? value.trim().toLowerCase() : "";
}

export function normalizeMathRecord(record: MathTrainingRecord): string | null {
  const prompt = cleanText(record.prompt);
  const response = cleanText(record.response);
  const instruction = cleanText(record.instruction);
  const input = cleanText(record.input);
  const output = cleanText(record.output);

  if (prompt || response) {
    if (!response) {
      return null;
    }

    const parts: string[] = [];
    parts.push("instruksi: jawab pertanyaan matematika berikut.");

    if (prompt) {
      parts.push(`input: ${prompt}`);
    }

    parts.push(`jawaban: ${response}`);
    return parts.join("\n");
  }

  if (!output) {
    return null;
  }

  const parts: string[] = [];

  if (instruction) {
    parts.push(`instruksi: ${instruction}`);
  }

  if (input) {
    parts.push(`input: ${input}`);
  }

  parts.push(`jawaban: ${output}`);

  return parts.join("\n");
}

export function recordsToCorpus(records: MathTrainingRecord[]): string[] {
  const corpus: string[] = [];

  for (const record of records) {
    const normalized = normalizeMathRecord(record);
    if (normalized) {
      corpus.push(normalized);
    }
  }

  return corpus;
}

export function buildChatPrompt(question: string): string {
  const normalizedQuestion = cleanText(question);
  return `instruksi: jawab pertanyaan matematika berikut.\ninput: ${normalizedQuestion}\njawaban:`;
}

export function loadMathTrainingCorpus(dataPath: string): string[] {
  if (!fs.existsSync(dataPath)) {
    throw new Error(`Dataset matematika tidak ditemukan: ${dataPath}`);
  }

  const raw = JSON.parse(fs.readFileSync(dataPath, "utf-8")) as unknown;
  if (!Array.isArray(raw)) {
    throw new Error("Format dataset matematika harus berupa array JSON.");
  }

  const corpus = recordsToCorpus(raw as MathTrainingRecord[]);
  if (corpus.length === 0) {
    throw new Error("Dataset matematika tidak memiliki record valid untuk training.");
  }

  return corpus;
}
