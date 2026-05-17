import { Matrix } from "@oxide-js/core";

export type ModelMode = "training" | "inference";

export interface ModelConfig {
  name?: string;
  trainable?: boolean;
}

export interface CompileConfig {
  optimizer?: any;
  loss?: any;
  metrics?: Array<string | any>;
}

export interface FitConfig {
  epochs?: number;
  batchSize?: number;
  shuffle?: boolean;
  verbose?: 0 | 1;
  validationData?: [Matrix, Matrix];
}

export interface HistoryRecord {
  epoch: number;
  loss?: number;
  [key: string]: any;
}

export interface WeightData {
  name: string;
  shape: number[];
  physicalShape?: number[];
  dtype: "float32" | "float64";
  data: Float32Array;
}

export interface ModelSummaryRow {
  name: string;
  type: string;
  outputShape: string;
  paramCount: number;
  trainable: boolean;
}

export interface SerializedModel {
  class_name: string;
  name: string;
  trainable: boolean;
  config: Record<string, any>;
  layers: any[];
  weights: WeightData[];
}
