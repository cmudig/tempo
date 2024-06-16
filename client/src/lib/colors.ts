import { interpolateHsl } from 'd3-interpolate';
export const MetricColors: { [key: string]: string } = {
  Timesteps: '#0284c7',
  Trajectories: '#0284c7',
  Accuracy: '#059669',
  Labels: '#d97706',
  Predictions: '#d97706',
};

export function makeCategoricalColorScale(
  baseColor: string
): (v: number) => string {
  let scale = interpolateHsl(baseColor, '#ffffff');
  // shift away from white a little bit
  return (v: number) => {
    return scale(v * 0.9);
  };
}
