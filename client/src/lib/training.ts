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
