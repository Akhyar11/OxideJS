import { runRecurrentLearningCorrectnessSuite } from "./rnn.learning.test";
import { runTransformerApiCorrectnessSuite } from "./transformers.api.test";
import { runTransformerLearningCorrectnessSuite } from "./transformers.learning.test";

export function runCorrectnessSuite(): void {
  runRecurrentLearningCorrectnessSuite();
  runTransformerApiCorrectnessSuite();
  runTransformerLearningCorrectnessSuite();
}

if (require.main === module) {
  runCorrectnessSuite();
}
