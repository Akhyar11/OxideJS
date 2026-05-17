import { setLoss, setOptimizer, Matrix } from "@oxide-js/core";
import type { LossLike, OptimizerLike, MetricLike, CompileConfig } from "./types.js";
import { accuracy, categoricalAccuracy, binaryAccuracy, mae, mse } from "./metrics.js";

export class PerParameterOptimizerWrapper {
  private optimizersMap = new Map<Matrix, any>();
  private name: string;
  private learningRate: number;

  constructor(name: string, learningRate: number) {
    this.name = name;
    this.learningRate = learningRate;
  }

  public step(params: Matrix[]): void {
    for (const param of params) {
      if (!param.grad) continue;
      let opt = this.optimizersMap.get(param);
      if (!opt) {
        // Resolve optimizer for this specific parameter's shape!
        opt = setOptimizer(this.name as any, param._shape, this.learningRate);
        this.optimizersMap.set(param, opt);
      }
      opt.apply(param, this.learningRate);
    }
  }

  public update(params: Matrix[]): void {
    this.step(params);
  }
}

/**
 * Resolve a loss string or object into a usable loss function.
 * If already a function or object with compute/forward, pass through.
 */
export function resolveLoss(loss: LossLike): any {
  if (typeof loss === "function") {
    return loss;
  }

  if (typeof loss === "object" && (loss.forward || loss.compute)) {
    return loss;
  }

  if (typeof loss !== "string") {
    throw new Error(`Invalid loss type: ${typeof loss}`);
  }

  const name = loss.toLowerCase();
  let normalizedLoss: string;

  if (name === "mse" || name === "meansquarederror") {
    normalizedLoss = "mse";
  } else if (name === "categoricalcrossentropy" || name === "categoricalcrossentropy" || name === "cce" || name === "crossentropy") {
    normalizedLoss = "crossEntropy";
  } else if (name === "binarycrossentropy" || name === "binarycrossentropy" || name === "bce") {
    normalizedLoss = "binaryCrossEntropy";
  } else if (name === "softmaxcrossentropy" || name === "softmaxce") {
    normalizedLoss = "softmaxCrossEntropy";
  } else if (name === "mae" || name === "meanabsoluteerror") {
    normalizedLoss = "mae";
  } else if (name === "huber") {
    normalizedLoss = "huber";
  } else if (name === "logcosh") {
    normalizedLoss = "logCosh";
  } else if (name === "hinge") {
    normalizedLoss = "hinge";
  } else if (name === "squaredhinge") {
    normalizedLoss = "squaredHinge";
  } else if (name === "kldivergence") {
    normalizedLoss = "klDivergence";
  } else if (name === "poisson") {
    normalizedLoss = "poisson";
  } else {
    normalizedLoss = loss; // fallback
  }

  try {
    return setLoss(normalizedLoss as any);
  } catch (e) {
    throw new Error(`Unknown loss: '${loss}'. Supported: mse, mae, crossEntropy, binaryCrossEntropy, softmaxCrossEntropy, etc.`);
  }
}

/**
 * Resolve an optimizer string or object into an optimizer instance.
 * If already an object with step/update, pass through.
 */
export function resolveOptimizer(
  optimizer: OptimizerLike,
  options?: { learningRate?: number }
): any {
  if (typeof optimizer === "object" && (optimizer.step || optimizer.update)) {
    return optimizer;
  }

  if (typeof optimizer !== "string") {
    throw new Error(`Invalid optimizer type: ${typeof optimizer}`);
  }

  const learningRate = options?.learningRate ?? 0.001;
  const name = optimizer.toLowerCase();
  let normalizedName: string;

  if (name === "sgd") {
    normalizedName = "sgd";
  } else if (name === "adam") {
    normalizedName = "adam";
  } else if (name === "momentum") {
    normalizedName = "momentum";
  } else if (name === "nag") {
    normalizedName = "nag";
  } else if (name === "adagrad") {
    normalizedName = "adaGrad";
  } else {
    throw new Error(`Unknown optimizer: '${optimizer}'. Supported: sgd, adam, momentum, nag, adagrad.`);
  }

  return new PerParameterOptimizerWrapper(normalizedName, learningRate);
}

/**
 * Resolve a metric string or function into a normalized metric.
 */
export function resolveMetric(metric: MetricLike): { name: string; compute: (yPred: Matrix, yTrue: Matrix) => number } {
  if (typeof metric === "function") {
    return {
      name: metric.name ?? "metric",
      compute: metric
    };
  }

  if (typeof metric === "object" && metric.compute) {
    return {
      name: metric.name ?? "metric",
      compute: metric.compute
    };
  }

  if (typeof metric !== "string") {
    throw new Error(`Invalid metric type: ${typeof metric}`);
  }

  const name = metric.toLowerCase();
  let computeFn: (yPred: Matrix, yTrue: Matrix) => number;

  switch (name) {
    case "accuracy":
      computeFn = accuracy;
      break;
    case "categoricalaccuracy":
      computeFn = categoricalAccuracy;
      break;
    case "binaryaccuracy":
      computeFn = binaryAccuracy;
      break;
    case "mae":
    case "meanabsoluteerror":
      computeFn = mae;
      break;
    case "mse":
    case "meansquarederror":
      computeFn = mse;
      break;
    default:
      throw new Error(`Unknown metric: '${metric}'`);
  }

  return {
    name: metric,
    compute: computeFn
  };
}

/**
 * Compile configuration with resolved loss and optimizer.
 * This is called internally by BaseModel.compile().
 */
export interface CompiledConfig {
  optimizer: any;
  loss: any;
  metrics: Array<{ name: string; compute: (yPred: Matrix, yTrue: Matrix) => number }>;
  learningRate: number;
}

/**
 * Resolve full compile config.
 * Handles string resolution for loss, optimizer, and metrics.
 */
export function resolveCompileConfig(config: CompileConfig): CompiledConfig {
  const learningRate = config.learningRate ?? 0.001;

  // Resolve loss
  const loss = resolveLoss(config.loss);

  // Resolve optimizer
  const optimizer = resolveOptimizer(config.optimizer, { learningRate });

  // Resolve metrics
  const metrics = (config.metrics ?? []).map(m => resolveMetric(m));

  return {
    optimizer,
    loss,
    metrics,
    learningRate
  };
}
