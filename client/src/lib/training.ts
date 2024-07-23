export type TrainingStatus = {
  id: string;
  info: any;
  status: string;
  status_info?: null | {
    message: string;
    progress?: number;
  };
};

export async function checkTrainingStatus(
  datasetName: string,
  modelNames: string[]
): Promise<TrainingStatus[] | null> {
  let taskStatuses: TrainingStatus[] = await (
    await fetch(import.meta.env.BASE_URL + `/tasks?cmd=train_model&dataset_name=${datasetName}`)
  ).json();
  return taskStatuses.filter((task) =>
    modelNames.includes(task.info.model_name)
  );
}

export type SliceFindingStatus = {
  searching: boolean;
  errors?: { [key: string]: string };
  models: string[];
  n_results: number; // number of slices discovered
  n_runs: number; // number of rows sampled
  last_updated: { [key: string]: string };
  status?: {
    state: string;
    message: string;
    progress?: number;
  };
};
