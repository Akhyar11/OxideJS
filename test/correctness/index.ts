import { runRecurrentLearningCorrectnessSuite } from "./rnn.learning.test";
import { runTokenizerMultilingualCorrectnessSuite } from "./tokenizer.multilingual.test";
import { runTransformerApiCorrectnessSuite } from "./transformers.api.test";
import { runTransformerLearningCorrectnessSuite } from "./transformers.learning.test";

export function runCorrectnessSuite(): void {
  runTokenizerMultilingualCorrectnessSuite();
  runRecurrentLearningCorrectnessSuite();
  runTransformerApiCorrectnessSuite();
  runTransformerLearningCorrectnessSuite();
}

if (require.main === module) {
  runCorrectnessSuite();
}
