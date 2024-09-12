export const highlightPatterns = [
  {
    name: 'tempo-placeholder-token',
    match: /^####TOKEN####([A-z]+)####ENDTOKEN####/,
  },
  {
    name: 'tempo-string',
    match: /^([\'\"]([^\'\"\#]*(?!####TOKEN####)#?)*[\'\"]?)/,
  },
  {
    name: 'tempo-constant',
    match: /^(#(?:now|mintime|maxtime|indexvalue|value))/i,
  },
  {
    name: 'tempo-data-field',
    match: /^(\{[^\}\#]+\}?)/,
  },
  {
    name: 'tempo-function',
    match: [/^([A-z_]+)\(/, '', '('],
  },
  {
    name: 'tempo-keyword',
    match:
      /^\b(and|or|not|case|when|where|end|else|in|then|every|at|as|from|to|(starts|ends)?with|contains|as|value|rate|duration|exists|mean|median|sum|min|max|first|last|any|all|nonnull|distinct|count|integral|impute|carry|cut|quantiles|bins)\b/i,
  },
];
