import type {
  Slice,
  SliceFeature,
  SliceFeatureAnd,
  SliceFeatureBase,
  SliceFeatureNegation,
  SliceFeatureOr,
} from './slice.type';

export function sliceToString(slice: Slice): string {
  let featureKeys = Object.keys(slice.featureValues);
  featureKeys.sort();
  return featureKeys.map((k) => `${k}: ${slice.featureValues[k]}`).join(',');
}

export let areSetsEqual = <T>(a: Set<T>, b: Set<T>): boolean =>
  a.size === b.size && [...a].every((value) => b.has(value));

export function areObjectsEqual(x, y) {
  if (x === y) return true;
  // if both x and y are null or undefined and exactly the same

  if (!(x instanceof Object) || !(y instanceof Object)) return false;
  // if they are not strictly equal, they both need to be Objects

  if (x.constructor !== y.constructor) return false;
  // they must have the exact same prototype chain, the closest we can do is
  // test there constructor.

  for (var p in x) {
    if (!x.hasOwnProperty(p)) continue;
    // other properties were tested using x.constructor === y.constructor

    if (!y.hasOwnProperty(p)) return false;
    // allows to compare x[ p ] and y[ p ] when set to undefined

    if (x[p] === y[p]) continue;
    // if they have the same strict value or identity then they are equal

    if (typeof x[p] !== 'object') return false;
    // Numbers, Strings, Functions, Booleans must be strictly equal

    if (!areObjectsEqual(x[p], y[p])) return false;
    // Objects and Arrays must be tested recursively
  }

  for (p in y) if (y.hasOwnProperty(p) && !x.hasOwnProperty(p)) return false;
  // allows x[ p ] to be set to undefined

  return true;
}

export function deepCopy<T>(obj: T): T {
  var copy;

  // Handle the 3 simple types, and null or undefined
  if (null == obj || 'object' != typeof obj) return obj;

  // Handle Date
  if (obj instanceof Date) {
    copy = new Date();
    copy.setTime(obj.getTime());
    return copy as T;
  }

  // Handle Array
  if (obj instanceof Array) {
    copy = [];
    for (var i = 0, len = obj.length; i < len; i++) {
      copy[i] = deepCopy(obj[i]);
    }
    return copy as T;
  }

  // Handle Object
  if (obj instanceof Object) {
    copy = Object.fromEntries(
      Object.entries(obj).map(([attr, val]) => [attr, deepCopy(val)])
    );
    return copy as T;
  }

  throw new Error("Unable to copy obj! Its type isn't supported.");
}

export function cumulativeSum(arr: Array<number>): Array<number> {
  return arr.map(
    (
      (sum) => (value) =>
        (sum += value)
    )(0)
  );
}

// return a copy of featureToCopy that has an identical tree to referenceFeature,
// where instances of searchFeature will be replaced with replaceFeature
export function withToggledFeature(
  featureToCopy: SliceFeatureBase,
  referenceFeature: SliceFeatureBase,
  searchFeature: SliceFeatureBase
): SliceFeatureBase {
  if (areObjectsEqual(searchFeature, referenceFeature)) {
    if (areObjectsEqual(searchFeature, featureToCopy)) return { type: 'base' };
    return Object.assign({}, referenceFeature);
  }
  let copied = Object.assign({}, featureToCopy);
  if (referenceFeature.type == 'negation') {
    (copied as SliceFeatureNegation).feature = withToggledFeature(
      (copied as SliceFeatureNegation).feature,
      (referenceFeature as SliceFeatureNegation).feature,
      searchFeature
    );
  } else if (referenceFeature.type == 'and') {
    (copied as SliceFeatureAnd).lhs = withToggledFeature(
      (copied as SliceFeatureAnd).lhs,
      (referenceFeature as SliceFeatureAnd).lhs,
      searchFeature
    );
    (copied as SliceFeatureAnd).rhs = withToggledFeature(
      (copied as SliceFeatureAnd).rhs,
      (referenceFeature as SliceFeatureAnd).rhs,
      searchFeature
    );
  } else if (referenceFeature.type == 'or') {
    (copied as SliceFeatureOr).lhs = withToggledFeature(
      (copied as SliceFeatureOr).lhs,
      (referenceFeature as SliceFeatureOr).lhs,
      searchFeature
    );
    (copied as SliceFeatureOr).rhs = withToggledFeature(
      (copied as SliceFeatureOr).rhs,
      (referenceFeature as SliceFeatureOr).rhs,
      searchFeature
    );
  }
  return copied;
}

export function featureNeedsParentheses(
  feature: SliceFeatureBase,
  parent: SliceFeatureBase = null
): boolean {
  if (feature.type == 'and' || feature.type == 'or') {
    if (parent.type == 'and' || parent.type == 'or') {
      return feature.type != parent.type;
    }
    if (parent.type == 'negation') return true;
  }
  return false;
}

// returns true if the two features have the same structure except for some
// features being substituted by an empty SliceFeatureBase
export function featuresHaveSameTree(
  feature: SliceFeatureBase,
  otherFeature: SliceFeatureBase
): boolean {
  if (feature.type != otherFeature.type) {
    return feature.type == 'base' || otherFeature.type == 'base';
  }
  if (feature.type == 'feature') {
    return areObjectsEqual(feature, otherFeature);
  }
  if (feature.type == 'negation') {
    return featuresHaveSameTree(
      (feature as SliceFeatureNegation).feature,
      (otherFeature as SliceFeatureNegation).feature
    );
  }
  if (feature.type == 'and' || feature.type == 'or') {
    return (
      featuresHaveSameTree(
        (feature as SliceFeatureAnd).lhs,
        (otherFeature as SliceFeatureAnd).lhs
      ) &&
      featuresHaveSameTree(
        (feature as SliceFeatureAnd).rhs,
        (otherFeature as SliceFeatureAnd).rhs
      )
    );
  }
  return true;
}

const prefixMetrics = ['count', 'timesteps', 'trajectories'];

export function sortMetrics(a: string, b: string): number {
  if (prefixMetrics.includes(a.toLocaleLowerCase())) {
    if (prefixMetrics.includes(b.toLocaleLowerCase()))
      return a.localeCompare(b);
    else return -1;
  } else if (prefixMetrics.includes(b.toLocaleLowerCase())) return 1;
  return a.localeCompare(b);
}

/**
 * Uses canvas.measureText to compute and return the width of the given text of given font in pixels.
 *
 * @param {String} text The text to be rendered.
 * @param {String} font The css font descriptor that text is to be rendered with (e.g. "bold 14px verdana").
 *
 * @see https://stackoverflow.com/questions/118241/calculate-text-width-with-javascript/21015393#21015393
 */
export function getTextWidth(text: string, font: number): number {
  // re-use canvas object for better performance
  var canvas =
    getTextWidth.canvas ||
    (getTextWidth.canvas = document.createElement('canvas'));
  var context = canvas.getContext('2d');
  context.font = font;
  var metrics = context.measureText(text);
  return metrics.width;
}
