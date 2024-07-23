export type Dataset = {
  sources: {
    type: string;
    path?: string;
    query?: string;
  }[];
  split: {
    train?: number;
    val?: number;
    test?: number;
  };
};
