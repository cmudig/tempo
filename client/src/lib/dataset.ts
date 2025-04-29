export type DataSource = {
  type: string;
  path?: string;
  time_field?: string;
  id_field?: string;
  type_field?: string;
  value_field?: string;
  start_time_field?: string;
  end_time_field?: string;
};
export type DataSplit = { train?: number; val?: number; test?: number };
export type SamplerSettings = {
  min_items_fraction?: number;
  n_samples?: number;
  max_features?: number;
  scoring_fraction?: number;
  num_candidates?: number;
  similarity_threshold?: number;
  n_slices?: number;
};

export type Dataset = {
  description?: string;
  error?: string;
  data: {
    sources?: DataSource[];
    split?: DataSplit;
  };
  slices?: {
    sampler?: SamplerSettings;
  };
};
