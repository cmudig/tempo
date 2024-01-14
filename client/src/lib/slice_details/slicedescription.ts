export type SliceVariableDescription = {
  variable: string;
  enrichments: {
    value: string;
    ratio: number;
  }[];
};

export type SliceVariableValueComparison = {
  base: number[];
  slice: number[];
  values: string[];
};

export type SliceDescription = {
  all_variables: {
    [key: string]: SliceVariableValueComparison;
  };
  top_variables: SliceVariableDescription[];
};

export type SliceChangeDescription = {
  variable: string;
  enrichments: {
    source_value: string;
    destination_value: string;
    base_prob: number;
    slice_prob: number;
    ratio: number;
  }[];
};
