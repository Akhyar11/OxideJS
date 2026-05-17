import type { Callback, CallbackLogs, HistoryRecord } from "./types.js";

/**
 * Callback that accumulates training history.
 */
export class HistoryCallback implements Callback {
  public history: HistoryRecord[] = [];

  onEpochEnd(epoch: number, logs?: CallbackLogs): void {
    const record: HistoryRecord = {
      epoch,
      loss: logs?.loss,
      val_loss: logs?.val_loss,
      metrics: logs?.metrics,
      val_metrics: logs?.val_metrics
    };

    this.history.push(record);
  }
}

/**
 * Early stopping callback.
 * Monitors a metric and stops training if it doesn't improve.
 */
export class EarlyStopping implements Callback {
  public shouldStop: boolean = false;
  public stoppedEpoch: number = 0;

  private monitor: string;
  private patience: number;
  private minDelta: number;
  private restoreBestWeights: boolean;

  private bestValue: number | null = null;
  private patienceCounter: number = 0;

  constructor(config?: {
    monitor?: string;
    patience?: number;
    minDelta?: number;
    restoreBestWeights?: boolean;
  }) {
    this.monitor = config?.monitor ?? "val_loss";
    this.patience = config?.patience ?? 3;
    this.minDelta = config?.minDelta ?? 0;
    this.restoreBestWeights = config?.restoreBestWeights ?? false;
  }

  onEpochEnd(epoch: number, logs?: CallbackLogs): void {
    const monitoredValue = this.getMonitoredValue(logs);

    if (monitoredValue === undefined) {
      console.warn(
        `[EarlyStopping] Monitored metric '${this.monitor}' not found in logs.`
      );
      return;
    }

    if (
      this.bestValue === null ||
      monitoredValue < this.bestValue - this.minDelta
    ) {
      this.bestValue = monitoredValue;
      this.patienceCounter = 0;
    } else {
      this.patienceCounter++;

      if (this.patienceCounter >= this.patience) {
        this.shouldStop = true;
        this.stoppedEpoch = epoch;
      }
    }
  }

  private getMonitoredValue(logs?: CallbackLogs): number | undefined {
    if (!logs) return undefined;

    // Check direct metrics first
    if (this.monitor === "loss") {
      return logs.loss;
    }

    if (this.monitor === "val_loss") {
      return logs.val_loss;
    }

    // Check metrics object
    if (this.monitor.startsWith("val_")) {
      const metricName = this.monitor.slice(4);
      return logs.val_metrics?.[metricName];
    }

    return logs.metrics?.[this.monitor];
  }
}

/**
 * Simple progress logger callback.
 * Logs progress at each batch/epoch in verbose mode.
 */
export class ProgressLogger implements Callback {
  private verbose: number;

  constructor(verbose: number = 1) {
    this.verbose = verbose;
  }

  onEpochBegin(epoch: number): void {
    if (this.verbose >= 1) {
      process.stdout.write(`Epoch ${epoch} `);
    }
  }

  onBatchEnd(batch: number, logs?: CallbackLogs): void {
    if (this.verbose >= 2) {
      const loss = logs?.loss !== undefined ? logs.loss.toFixed(4) : "?";
      process.stdout.write(`[batch ${batch}: loss=${loss}] `);
    }
  }

  onEpochEnd(epoch: number, logs?: CallbackLogs): void {
    if (this.verbose >= 1) {
      const loss = logs?.loss !== undefined ? logs.loss.toFixed(4) : "?";
      const valLoss =
        logs?.val_loss !== undefined ? logs.val_loss.toFixed(4) : "?";

      let line = `- loss: ${loss}`;

      if (logs?.val_loss !== undefined) {
        line += ` - val_loss: ${valLoss}`;
      }

      if (logs?.metrics) {
        for (const [key, value] of Object.entries(logs.metrics)) {
          line += ` - ${key}: ${(value as number).toFixed(4)}`;
        }
      }

      if (logs?.val_metrics) {
        for (const [key, value] of Object.entries(logs.val_metrics)) {
          line += ` - val_${key}: ${(value as number).toFixed(4)}`;
        }
      }

      console.log(line);
    }
  }
}
