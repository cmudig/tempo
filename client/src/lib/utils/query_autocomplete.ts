const constantExpressions = [
  'now',
  'value',
  'mintime',
  'maxtime',
  'indexvalue',
];

export function getAutocompleteOptions(
  dataFields: string[],
  searchQuery: string,
  fullPrefix: string
): { value: string; type: string }[] {
  if (dataFields.length == 0) return [];
  if (searchQuery.length == 0) return [];

  let result = fullPrefix.match(/\{[^}]*?(?:,\s*([^},]*))?$/i);
  if (!!result) {
    if (!!result[1]) searchQuery = result[1];
    if (searchQuery.length == 0) return [];
    return [
      ...dataFields
        .filter((v) =>
          v.toLocaleLowerCase().startsWith(searchQuery.toLocaleLowerCase())
        )
        .sort((a, b) => a.length - b.length),
      ...dataFields
        .filter(
          (v) =>
            v.toLocaleLowerCase().includes(searchQuery.toLocaleLowerCase()) &&
            !v.toLocaleLowerCase().startsWith(searchQuery.toLocaleLowerCase())
        )
        .sort((a, b) => a.length - b.length),
    ].map((v) => ({ value: v, type: 'data_item' }));
  }

  // constant expressions
  result = fullPrefix.match(/#[^\s]*/);
  if (!!result) {
    return [
      ...constantExpressions
        .filter((v) =>
          v.toLocaleLowerCase().startsWith(searchQuery.toLocaleLowerCase())
        )
        .sort((a, b) => a.length - b.length),
      ...constantExpressions
        .filter(
          (v) =>
            v.toLocaleLowerCase().includes(searchQuery.toLocaleLowerCase()) &&
            !v.toLocaleLowerCase().startsWith(searchQuery.toLocaleLowerCase())
        )
        .sort((a, b) => a.length - b.length),
    ].map((v) => ({ value: v, type: 'constant' }));
  }
  return [];
}

export function performAutocomplete(
  item: { value: string; type: string },
  trigger: string,
  fullPrefix: string,
  fullSuffix: string,
  replaceRegion: string
): string {
  if (item.type == 'data_item') {
    let result = replaceRegion.match(/\{([^}]*?,\s*)([^},]*)$/i);
    if (!!result) return `{${result[1]}${item.value}`;
    return `{${item.value}`;
  }
  let closingSpace = fullSuffix.match(/^\s/) ? '' : ' ';
  return `#${item.value}${closingSpace}`;
}
