import { runRecurrentLearningCorrectnessSuite } from "./rnn.learning.test";
import { runTransformerLearningCorrectnessSuite } from "./transformers.learning.test";

export function runCorrectnessSuite(): void {
  runRecurrentLearningCorrectnessSuite();
  runTransformerLearningCorrectnessSuite();
}

if (require.main === module) {
  runCorrectnessSuite();
}
