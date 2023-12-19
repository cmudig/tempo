import type { Slice } from './slices/utils/slice.type';

export type TrainingStatus = {
  state: string;
  message: string;
};

export async function checkTrainingStatus(
  modelName: string
): Promise<TrainingStatus | null> {
  let trainingStatus = await (
    await fetch(`/models/status/${modelName}`)
  ).json();
  if (trainingStatus!.state == 'none' || trainingStatus!.state == 'complete')
    trainingStatus = null;
  return trainingStatus;
}

export type SliceFindingStatus = {
  searching: boolean;
  errors?: { [key: string]: string };
  status?: {
    state: string;
    message: string;
    progress?: number;
  };
};

export async function checkSlicingStatus(): Promise<SliceFindingStatus | null> {
  let trainingStatus = await (await fetch(`/slices/status`)).json();
  return trainingStatus;
}
