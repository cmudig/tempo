export type ModelMetrics = {
  threshold?: number;
  acc?: number;
  roc_auc?: number;
  confusion_matrix?: number[][];
  sensitivity?: number;
  specificity?: number;
  r2_score?: number;
  n_train: number;
  n_val: number;
};

export type ModelSummary = {
  target_name: string;
  type: 'classification' | 'regression';
  likely_trivial?: boolean;
  metrics: ModelMetrics;
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
