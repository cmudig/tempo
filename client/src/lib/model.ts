export type ModelMetrics = {
  threshold?: number;
  performance: {
    [key: string]: number;
  };
  confusion_matrix?: number[][];
  positive_rate?: number;
  n_train: { instances: number; trajectories: number };
  n_val: { instances: number; trajectories: number };
};

export type ModelSummary = {
  outcome?: string;
  regression?: boolean;
  likely_trivial?: boolean;
  metrics?: ModelMetrics;
  training?: boolean;
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
