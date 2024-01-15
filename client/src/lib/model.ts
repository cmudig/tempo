export type ModelMetrics = {
  threshold?: number;
  performance: {
    [key: string]: number;
  };
  confusion_matrix?: number[][];
  positive_rate?: number;
  n_train: { instances: number; trajectories: number };
  n_val: { instances: number; trajectories: number };
  n_slice_eval: { instances: number; trajectories: number };
  trivial_solution_warning?: {
    variables: string[];
    auc: number;
    auc_threshold: number;
    auc_fraction: number;
  };
  class_not_predicted_warnings?: {
    class: number;
    true_positive_fraction: number;
    true_positive_threshold: number;
  }[];
  roc: {
    thresholds: number[];
    tpr: number[];
    fpr: number[];
    performance: { [key: string]: number }[];
  };
};

export type ModelSummary = {
  outcome?: string;
  regression?: boolean;
  likely_trivial?: boolean;
  metrics?: ModelMetrics;
  training?: boolean;
  timestep_definition: string;
  status?: { state: string; message: string };
};

export enum VariableCategory {
  Demographics = 'Demographics',
  Vitals = 'Vitals',
  Labs = 'Labs',
  Assessments = 'Assessments',
  Procedures = 'Procedures',
  Fluids = 'Fluids',
  Vasopressors = 'Vasopressors',
  Prescriptions = 'Prescriptions',
  Cultures = 'Cultures',
  Other = 'Other',
}

export const AllCategories = [
  VariableCategory.Demographics,
  VariableCategory.Vitals,
  VariableCategory.Labs,
  VariableCategory.Assessments,
  VariableCategory.Procedures,
  VariableCategory.Fluids,
  VariableCategory.Vasopressors,
  VariableCategory.Prescriptions,
  VariableCategory.Cultures,
  VariableCategory.Other,
];

export type VariableDefinition = {
  category: VariableCategory;
  query: string;
  enabled: boolean;
};

export type VariableEvaluationSummary = {
  query: string;
  n_values: number;
  n_trajectories: number;
  type: 'binary' | 'continuous' | 'categorical';
  rate?: number;
  counts?: { [key: string]: number };
  mean?: number;
  std?: number;
  hist?: { counts: number[]; bins: number[] };
};

export type SliceFilter = { [key: string]: any } & { type: string };

export type SliceSpec = {
  variables: { [key: string]: VariableDefinition };
  slice_filter: SliceFilter;
};
