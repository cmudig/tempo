import { areSetsEqual } from './utils/utils';

export type ScoreFunction = {
  type: 'constant' | 'relation' | 'logical' | 'model_property';
  relation?: string;
  model_name?: string;
  property?: string;
  value?: number | number[];
  lhs?: ScoreFunction;
  rhs?: ScoreFunction;
};

const RelationStrings: { [key: string]: string } = {
  '=': 'equals',
  '!=': 'does not equal',
  '<': 'less than',
  '>': 'greater than',
  '<=': 'less than or equal to',
  '>=': 'greater than or equal to',
  in: 'is one of',
  'not-in': 'is not one of',
  and: 'and',
  or: 'or',
};
const ModelPropertyStrings: { [key: string]: string } = {
  label: 'true label',
  prediction: 'prediction',
  prediction_probability: 'prediction probability',
  correctness: 'correctness',
  deviation: 'deviation',
  abs_deviation: 'absolute deviation',
};

// support describing score functions with a simple shorthand when they are simple conceptually
function matchTemplates(scoreFn: ScoreFunction): string | undefined {
  if (scoreFn.type == 'constant' || scoreFn.type == 'model_property') return;

  if (
    scoreFn.type == 'relation' &&
    scoreFn.relation == '=' &&
    scoreFn.lhs?.type == 'model_property' &&
    scoreFn.rhs?.type == 'constant'
  ) {
    let constantVal = scoreFn.rhs?.value;
    let modelName = scoreFn.lhs?.model_name;
    let property = scoreFn.lhs?.property;
    if (property == 'correctness')
      return constantVal == 1 ? `${modelName} correct` : `${modelName} error`;
    if (constantVal == 1) {
      return `${modelName} positive ${property}`;
    } else if (constantVal == 0) {
      return `${modelName} negative ${property}`;
    }
    return;
  }

  if (scoreFn.type == 'logical') {
    if (scoreFn.lhs!.type != 'relation' || scoreFn.rhs!.type != 'relation')
      return;
    let lhs = scoreFn.lhs!;
    let rhs = scoreFn.rhs!;
    if (
      !(lhs.lhs?.type == 'model_property' && lhs.rhs?.type == 'constant') ||
      !(rhs.lhs?.type == 'model_property' && rhs.rhs?.type == 'constant')
    )
      return;
    if (lhs.lhs.model_name != rhs.lhs.model_name) return;
    let modelName = lhs.lhs.model_name!;
    if (
      (lhs.lhs.property == 'label' &&
        lhs.rhs.value == 1 &&
        rhs.lhs.property == 'prediction' &&
        rhs.rhs.value == 0) ||
      (lhs.lhs.property == 'prediction' &&
        lhs.rhs.value == 0 &&
        rhs.lhs.property == 'label' &&
        rhs.rhs.value == 1)
    )
      return `${modelName} false negative`;
    if (
      (lhs.lhs.property == 'label' &&
        lhs.rhs.value == 0 &&
        rhs.lhs.property == 'prediction' &&
        rhs.rhs.value == 1) ||
      (lhs.lhs.property == 'prediction' &&
        lhs.rhs.value == 1 &&
        rhs.lhs.property == 'label' &&
        rhs.rhs.value == 0)
    )
      return `${modelName} false positive`;
  }
}

export function scoreFunctionToString(
  scoreFn: ScoreFunction,
  nested: boolean = false
): string {
  if (!nested) {
    let templated = matchTemplates(scoreFn);
    if (!!templated) return templated;
  }
  if (scoreFn.type == 'constant') return `${scoreFn.value}`;
  else if (scoreFn.type == 'model_property')
    return `${scoreFn.model_name} ${ModelPropertyStrings[scoreFn.property!]}`;
  else if (scoreFn.type == 'relation' || scoreFn.type == 'logical') {
    let base = `${scoreFunctionToString(scoreFn.lhs!, true)} ${
      RelationStrings[scoreFn.relation!]
    } ${scoreFunctionToString(scoreFn.rhs!, true)}`;
    if (nested) return `(${base})`;
    return base;
  }
  return '';
}
