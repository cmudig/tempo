export const QueryTemplatesNoTimestepDefs = [
  {
    title: 'Basics',
    children: [
      {
        name: 'Data Field',
        query: '{<fieldname>}',
      },
      {
        name: 'Duration in Hours',
        query: '<number> hours',
      },
      {
        name: 'Duration in Days',
        query: '<number> days',
      },
      {
        name: 'Convert Numerical Result in Seconds to Hours',
        query: '(<expression>) as hours',
      },
      {
        name: 'Convert Numerical Result in Seconds to Days',
        query: '(<expression>) as days',
      },
    ],
  },
  {
    title: 'Extracting Info from Events/Intervals',
    children: [
      { name: 'Time of Event', query: 'time({<field>})' },
      { name: 'Start Time of Interval', query: 'starttime({<field>})' },
      { name: 'End Time of Interval', query: 'endtime({<field>})' },
      { name: 'Events for Each Interval Start', query: 'start({<field>})' },
      { name: 'Events for Each Interval End', query: 'end({<field>})' },
    ],
  },
  {
    title: 'Testing Values',
    children: [
      {
        name: 'Equality',
        query: '<left> = <right>',
      },
      {
        name: 'Between Two Values',
        query: '<expression> between <lower> and <upper>',
      },
      {
        name: 'Contained in List',
        query: '<expression> in [<value>, <value>]',
      },
    ],
  },
  {
    title: 'Logical Operations',
    children: [
      {
        name: 'Logical And',
        query: '<condition> and <condition>',
      },
      {
        name: 'Logical Or',
        query: '<condition> or <condition>',
      },
      {
        name: 'Logical Not',
        query: 'not <condition>',
      },
      {
        name: 'Case Expression',
        query: 'case when <predicate> then <value> else <fallback> end',
      },
    ],
  },
  {
    title: 'String Operations',
    children: [
      {
        name: 'Contains String',
        query: '<field> contains "<string>"',
      },
      {
        name: 'Contains Regex Pattern',
        query: '<field> contains /<pattern>/',
      },
      {
        name: 'Starts With String',
        query: '<field> startswith "<string>"',
      },
      {
        name: 'Ends With String',
        query: '<field> endswith "<string>"',
      },
      {
        name: 'Extract Regex Match',
        query: 'extract(<expression>, /<pattern>/)',
      },
      {
        name: 'Extract Regex Match at Index',
        query: 'extract(<expression>, /<pattern>/, <index>)',
      },
    ],
  },
  {
    title: 'General Event/Interval Aggregations',
    children: [
      {
        name: 'Count of Events',
        query: 'count <expression> from #now - <duration> to #now ',
      },
      {
        name: 'Count of Events with Non-Null Values',
        query: 'count nonnull <expression> from #now - <duration> to #now ',
      },
      {
        name: 'Count of Distinct-Valued Events',
        query: 'count distinct <expression> from #now - <duration> to #now ',
      },
      {
        name: 'Event Exists',
        query: 'exists <expression> from #now - <duration> to #now ',
      },
      {
        name: 'Value of First Event Within Duration',
        query: 'first <expression> from #now - <duration> to #now ',
      },
      {
        name: 'Value of Last Event Within Duration',
        query: 'last <expression> from #now - <duration> to #now ',
      },
      {
        name: 'Any Event Matches Condition',
        query: 'any <condition> from #now - <duration> to #now ',
      },
      {
        name: 'All Events Match Condition',
        query: 'all <condition> from #now - <duration> to #now ',
      },
    ],
  },
  {
    title: 'Aggregations of Numerical Events/Intervals',
    children: [
      {
        name: 'Mean of Values',
        query: 'mean <expression> from #now - <duration> to #now ',
      },
      {
        name: 'Sum of Values',
        query: 'sum <expression> from #now - <duration> to #now ',
      },
      {
        name: 'Min of Values',
        query: 'min <expression> from #now - <duration> to #now ',
      },
      {
        name: 'Max of Values',
        query: 'max <expression> from #now - <duration> to #now ',
      },
      {
        name: 'Mean Duration of Intervals',
        query: 'mean duration <expression> from #now - <duration> to #now ',
      },
      {
        name: 'Integral of Rate Values',
        query: 'integral rate <expression> from #now - <duration> to #now ',
      },
    ],
  },
  {
    title: 'Transformations',
    children: [
      {
        name: 'Filter Values by Predicate',
        query: '<expression> where <predicate>',
      },
      {
        name: 'Carry Values by Time',
        query: '<expression> carry <number> <unit>',
      },
      {
        name: 'Carry Values by Steps',
        query: '<expression> carry <numer> steps',
      },
      {
        name: 'Impute Missing Values with Mean',
        query: '<expression> impute mean',
      },
      {
        name: 'Impute Missing Values with Constant',
        query: '<expression> impute <value>',
      },
    ],
  },
  {
    title: 'Discretizations',
    children: [
      {
        name: 'Discretize into Equally-Spaced Bins',
        query: '<expression> cut <number> bins',
      },
      {
        name: 'Discretize into Equal Quantile Bins',
        query: '<expression> cut <number> quantiles',
      },
      {
        name: 'Discretize into Custom Bins',
        query: '<expression> cut bins [-inf, <value>, <value>, inf]',
      },
      {
        name: 'Discretize into Custom Quantiles',
        query: '<expression> cut quantiles [0, <value>, <value>, 1]',
      },
    ],
  },
  {
    title: 'Inline Variables',
    children: [
      {
        name: 'Define Inline Variable',
        query: '<expression> with <varname> as (<expression>)',
      },
    ],
  },
];

export const QueryTemplatesTimestepDefsOnly = [
  {
    title: 'Timestep Definitions',
    children: [
      {
        name: 'At Regular Intervals While Events Exist',
        query: 'every <duration>',
      },
      {
        name: 'At Regular Intervals Bounded by Attributes',
        query: 'every <duration> from <start> to <end>',
      },
      { name: 'At Each Occurrence of an Event', query: 'at every {<field>}' },
      { name: 'At Specific Times', query: 'at [<value>, <value>]' },
    ],
  },
];
