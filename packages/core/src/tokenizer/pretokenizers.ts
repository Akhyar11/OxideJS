export type PreTokenizer = (text: string) => string[];

export type BuiltInPreTokenizer =
  | "char"
  | "unicode-grapheme"
  | "unicode-word"
  | "whitespace"
  | "script-aware";

type SegmenterGranularity = "grapheme" | "word";
type SegmentData = { segment: string; isWordLike?: boolean };
type SegmenterLike = {
  segment(input: string): Iterable<SegmentData>;
};
type SegmenterConstructor = new (
  locale?: string | string[],
  options?: { granularity?: SegmenterGranularity }
) => SegmenterLike;

const intlWithSegmenter = Intl as typeof Intl & { Segmenter?: SegmenterConstructor };
const hasUnicodePropertyEscapes = supportsUnicodePropertyEscapes();

export function charPreTokenizer(text: string): string[] {
  return Array.from(text);
}

export function unicodeGraphemePreTokenizer(text: string): string[] {
  const segmenter = createSegmenter(undefined, "grapheme");
  if (!segmenter) return charPreTokenizer(text);
  return Array.from(segmenter.segment(text), (part) => part.segment);
}

export function unicodeWordPreTokenizer(text: string): string[] {
  const segmenter = createSegmenter(undefined, "word");
  if (segmenter) {
    const segments: string[] = [];
    for (const part of segmenter.segment(text)) {
      if (part.isWordLike) segments.push(part.segment);
    }
    return segments;
  }

  const whitespaceParts = whitespacePreTokenizer(text);
  if (whitespaceParts.length > 1) return whitespaceParts;
  return whitespaceParts.length === 1 ? charPreTokenizer(whitespaceParts[0]) : [];
}

export function whitespacePreTokenizer(text: string): string[] {
  const trimmed = text.trim();
  return trimmed.length === 0 ? [] : trimmed.split(/\s+/u);
}

export function scriptAwarePreTokenizer(text: string): string[] {
  if (text.trim().length === 0) return [];

  const thaiWordSegments = tryThaiWordSegmentation(text);
  if (thaiWordSegments) return thaiWordSegments;

  const clusters = unicodeGraphemePreTokenizer(text);
  const tokens: string[] = [];
  let buffer = "";
  let bufferKind: ScriptGroup | null = null;

  const flush = (): void => {
    if (buffer.length > 0) {
      tokens.push(buffer);
      buffer = "";
      bufferKind = null;
    }
  };

  for (const cluster of clusters) {
    if (isWhitespace(cluster)) {
      flush();
      continue;
    }

    if (isEmojiLike(cluster) || isMathSymbol(cluster) || isPunctuation(cluster)) {
      flush();
      tokens.push(cluster);
      continue;
    }

    const kind = getScriptGroup(cluster);

    if (kind === "cjk" || kind === "javanese") {
      flush();
      tokens.push(cluster);
      continue;
    }

    if (kind === "other") {
      flush();
      tokens.push(cluster);
      continue;
    }

    if (bufferKind === kind) {
      buffer += cluster;
      continue;
    }

    flush();
    buffer = cluster;
    bufferKind = kind;
  }

  flush();
  return tokens;
}

export function resolvePreTokenizer(preTokenizer: BuiltInPreTokenizer | PreTokenizer): PreTokenizer {
  if (typeof preTokenizer === "function") return preTokenizer;

  switch (preTokenizer) {
    case "char":
      return charPreTokenizer;
    case "unicode-grapheme":
      return unicodeGraphemePreTokenizer;
    case "unicode-word":
      return unicodeWordPreTokenizer;
    case "whitespace":
      return whitespacePreTokenizer;
    case "script-aware":
      return scriptAwarePreTokenizer;
    default:
      return charPreTokenizer;
  }
}

export function isLatin(ch: string): boolean {
  return hasUnicodePropertyEscapes
    ? /\p{Script=Latin}/u.test(ch)
    : codePointInRanges(ch, [[0x0041, 0x007a], [0x00c0, 0x024f], [0x1e00, 0x1eff]]);
}

export function isArabic(ch: string): boolean {
  return hasUnicodePropertyEscapes
    ? /\p{Script=Arabic}/u.test(ch)
    : codePointInRanges(ch, [[0x0600, 0x06ff], [0x0750, 0x077f], [0x08a0, 0x08ff], [0xfb50, 0xfdff], [0xfe70, 0xfeff]]);
}

export function isCJK(ch: string): boolean {
  return codePointInRanges(ch, [[0x3400, 0x4dbf], [0x4e00, 0x9fff], [0xf900, 0xfaff], [0x20000, 0x2a6df], [0x2a700, 0x2b73f], [0x2b740, 0x2b81f], [0x2b820, 0x2ceaf]]);
}

export function isHiragana(ch: string): boolean {
  return codePointInRanges(ch, [[0x3040, 0x309f]]);
}

export function isKatakana(ch: string): boolean {
  return codePointInRanges(ch, [[0x30a0, 0x30ff], [0x31f0, 0x31ff]]);
}

export function isHangul(ch: string): boolean {
  return codePointInRanges(ch, [[0x1100, 0x11ff], [0x3130, 0x318f], [0xac00, 0xd7af], [0xa960, 0xa97f], [0xd7b0, 0xd7ff]]);
}

export function isThai(ch: string): boolean {
  return codePointInRanges(ch, [[0x0e00, 0x0e7f]]);
}

export function isJavaneseBase(ch: string): boolean {
  const cp = firstCodePoint(ch);
  return cp >= 0xa984 && cp <= 0xa9b2;
}

export function isJavaneseMark(ch: string): boolean {
  return codePointInRanges(ch, [[0xa980, 0xa983], [0xa9b3, 0xa9c0]]);
}

export function isMathSymbol(ch: string): boolean {
  return hasUnicodePropertyEscapes
    ? /[\p{Math}\p{Sm}\u2070-\u209f]/u.test(ch)
    : codePointInRanges(ch, [[0x2070, 0x209f], [0x2190, 0x21ff], [0x2200, 0x22ff], [0x27c0, 0x27ef], [0x2980, 0x29ff], [0x2a00, 0x2aff]]);
}

export function isPunctuation(ch: string): boolean {
  return hasUnicodePropertyEscapes
    ? /\p{P}/u.test(ch)
    : /[!"#$%&'()*,\-./:;<=>?@[\\\]^_`{|}~]/.test(ch);
}

export function isEmojiLike(ch: string): boolean {
  return hasUnicodePropertyEscapes
    ? /\p{Extended_Pictographic}/u.test(ch)
    : codePointInRanges(ch, [[0x1f000, 0x1faff], [0x2600, 0x27bf]]);
}

function createSegmenter(locale: string | string[] | undefined, granularity: SegmenterGranularity): SegmenterLike | null {
  const Segmenter = intlWithSegmenter.Segmenter;
  if (!Segmenter) return null;
  try {
    return new Segmenter(locale, { granularity });
  } catch {
    return null;
  }
}

function tryThaiWordSegmentation(text: string): string[] | null {
  const clusters = unicodeGraphemePreTokenizer(text).filter((cluster) => !isWhitespace(cluster));
  if (clusters.length === 0 || !clusters.every(isThai)) return null;

  const segmenter = createSegmenter("th", "word");
  if (!segmenter) return clusters;

  const segments: string[] = [];
  for (const part of segmenter.segment(text)) {
    if (part.isWordLike) segments.push(part.segment);
  }
  return segments.length > 0 ? segments : clusters;
}

type ScriptGroup = "latin" | "arabic" | "hangul" | "kana" | "thai" | "cjk" | "javanese" | "other";

function getScriptGroup(cluster: string): ScriptGroup {
  if (isLatin(cluster) || isDecimalNumber(cluster)) return "latin";
  if (isArabic(cluster)) return "arabic";
  if (isHangul(cluster)) return "hangul";
  if (isHiragana(cluster) || isKatakana(cluster)) return "kana";
  if (isThai(cluster)) return "thai";
  if (isCJK(cluster)) return "cjk";
  if (isJavaneseBase(cluster) || isJavaneseMark(cluster)) return "javanese";
  return "other";
}

function isWhitespace(ch: string): boolean {
  return /\s/u.test(ch);
}

function isDecimalNumber(ch: string): boolean {
  return hasUnicodePropertyEscapes ? /\p{Nd}/u.test(ch) : /[0-9]/.test(ch);
}

function codePointInRanges(ch: string, ranges: Array<[number, number]>): boolean {
  const cp = firstCodePoint(ch);
  return ranges.some(([start, end]) => cp >= start && cp <= end);
}

function firstCodePoint(ch: string): number {
  return ch.codePointAt(0) ?? -1;
}

function supportsUnicodePropertyEscapes(): boolean {
  try {
    return new RegExp("\\p{Script=Latin}", "u").test("a");
  } catch {
    return false;
  }
}
