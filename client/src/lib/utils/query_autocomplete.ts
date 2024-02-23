const constantExpressions = ['now', 'value', 'mintime', 'maxtime'];

export function getAutocompleteOptions(
  dataFields: string[],
  searchQuery: string,
  fullPrefix: string
): { value: string; type: string }[] {
  if (dataFields.length == 0) return [];
  let result = fullPrefix.match(/\{[^}]*$/i);
  if (!!result) {
    return dataFields
      .filter((v) =>
        v.toLocaleLowerCase().includes(searchQuery.toLocaleLowerCase())
      )
      .map((v) => ({ value: v, type: 'data_item' }));
  }

  // constant expressions
  result = fullPrefix.match(/#[^\s]*/);
  if (!!result) {
    return constantExpressions
      .filter((v) =>
        v.toLocaleLowerCase().includes(searchQuery.toLocaleLowerCase())
      )
      .map((v) => ({ value: v, type: 'constant' }));
  }
  return [];
}

export function performAutocomplete(
  item: { value: string; type: string },
  trigger: string,
  fullPrefix: string,
  fullSuffix: string
): string {
  if (item.type == 'data_item') {
    let closingBrace = fullSuffix.match(/^\}/)
      ? ''
      : fullSuffix.match(/^\s/)
      ? '}'
      : '} ';
    return `{${item.value}${closingBrace}`;
  }
  let closingSpace = fullSuffix.match(/^\s/) ? '' : ' ';
  return `#${item.value}${closingSpace}`;
}
