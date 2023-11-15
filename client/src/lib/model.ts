export type ModelSummary = {
  target_name: string;
  n_patients: number;
  n_timesteps: number;
  type: 'classification' | 'regression';
  likely_trivial?: boolean;
  metrics: {
    roc_auc?: number;
    sensitivity?: number;
    specificity?: number;
    r2_score?: number;
  };
};
