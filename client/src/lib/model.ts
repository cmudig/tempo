import type { SliceMetric } from './slices/utils/slice.type';

export type ModelDataSummaryItem = {
  name: string;
  type: string;
  children?: ModelDataSummaryItem[];
  summary?: SliceMetric;
};

export type ModelMetrics = {
  threshold?: number;
  performance: {
    [key: string]: number;
  };
  confusion_matrix?: number[][];
  labels: SliceMetric;
  predictions: SliceMetric;
  n_train: { instances: number; trajectories: number };
  n_val: { instances: number; trajectories: number };
  n_test: { instances: number; trajectories: number };
  trivial_solution_warning?: {
    variables: string[];
    metric: string;
    metric_value: number;
    metric_threshold: number;
    metric_fraction: number;
  };
  class_not_predicted_warnings?: {
    class: number;
    true_positive_fraction: number;
    true_positive_threshold: number;
  }[];
  roc?: {
    thresholds: number[];
    tpr: number[];
    fpr: number[];
    performance: { [key: string]: number }[];
  };
  perclass?: {
    label: string;
    performance: { [key: string]: number };
  }[];
  data_summary?: {
    fields: ModelDataSummaryItem[];
  };

  hist: {
    values: number[][];
    bins: number[];
  };
};

export function metricsHaveWarnings(metrics: ModelMetrics): boolean {
  return (
    !!metrics.trivial_solution_warning ||
    Object.keys(metrics.class_not_predicted_warnings ?? {}).length > 0
  );
}

export const decimalMetrics = ['R^2', 'MSE', 'Macro F1', 'Micro F1', 'F1'];

export enum ModelType {
  BinaryClassification = 'binary_classification',
  MulticlassClassification = 'multiclass_classification',
  Regression = 'regression',
}

export const ModelTypeStrings: { [key in ModelType]: string } = {
  [ModelType.BinaryClassification]: 'Binary Classification',
  [ModelType.MulticlassClassification]: 'Multiclass Classification',
  [ModelType.Regression]: 'Regression',
};

export type ModelSummary = {
  outcome?: string;
  model_type?: ModelType;
  likely_trivial?: boolean;
  cohort: string;
  variables: { [key: string]: VariableDefinition };
  training?: boolean;
  timestep_definition: string;
  description?: string;
  status?: { state: string; message: string };
  output_values?: string[];
  error?: string;

  draft?: {
    outcome?: string;
    cohort: string;
    description?: string;
    variables: { [key: string]: VariableDefinition };
    timestep_definition: string;
  };
};

export type VariableDefinition = {
  category: string;
  query: string;
  enabled?: boolean;
};

export type QueryResult = {
  name?: string;
  values?: SliceMetric & { missingness?: number };
  occurrences?: SliceMetric & { missingness?: number };
  durations?: SliceMetric & { missingness?: number };
};

export type VariableEvaluationSummary = {
  query: string;
  n_values: number;
  n_trajectories: number;
  result: QueryResult;
};

export type SliceFilter = { [key: string]: any } & { type: string };

export type SliceSpec = {
  variables: { [key: string]: VariableDefinition };
  slice_filter: SliceFilter;
};
