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
    await fetch(
      import.meta.env.BASE_URL +
        `/tasks?cmd=train_model&dataset_name=${datasetName}`
    )
  ).json();
  return taskStatuses.filter((task) =>
    modelNames.includes(task.info.model_name)
  );
}

export async function checkDatasetBuildStatus(
  datasetName: string,
  taskID: string | null = null
): Promise<TrainingStatus | null> {
  if (!!taskID) {
    let status = await (
      await fetch(import.meta.env.BASE_URL + `/tasks/${taskID}`)
    ).json();
    if (status['status'] == 'complete' || status['status'] == 'error')
      return null;
    return status;
  }
  let taskStatuses: TrainingStatus[] = await (
    await fetch(
      import.meta.env.BASE_URL +
        `/tasks?cmd=build_dataset&dataset_name=${datasetName}`
    )
  ).json();
  return taskStatuses.length > 0 ? taskStatuses[0] : null;
}

export async function taskSuccessful(taskID: string): Promise<boolean | null> {
  let status = await (
    await fetch(import.meta.env.BASE_URL + `/tasks/${taskID}`)
  ).json();
  if (status['status'] == 'complete' || status['status'] == 'error')
    return status['status'] == 'complete';
  return null;
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
