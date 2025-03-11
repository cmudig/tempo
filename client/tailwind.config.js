import typography from '@tailwindcss/typography';

/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,svelte}'],
  darkMode: 'class',
  theme: {
    extend: {
      typography: {
        DEFAULT: {
          css: {
            'max-width': '85ch',
            'p, h1, h2, h3, li': {
              'margin-top': 0,
              'margin-bottom': '0.5em',
              'line-height': '1.2em',
            },
          },
        },
      },
    },
  },
  plugins: [typography],
  safelist: [
    'bg-blue-200',
    'bg-violet-200',
    'bg-fuchsia-200',
    'bg-emerald-200',
    'bg-cyan-200',
    'bg-emerald-200',
    'bg-lime-200',
    'bg-amber-200',
    'bg-red-200',
    'bg-blue-400',
    'bg-violet-400',
    'bg-fuchsia-400',
    'bg-emerald-400',
    'bg-cyan-400',
    'bg-emerald-400',
    'bg-lime-400',
    'bg-amber-400',
    'bg-red-400',
    'bg-blue-600',
    'bg-violet-600',
    'bg-fuchsia-600',
    'bg-emerald-600',
    'bg-cyan-600',
    'bg-emerald-600',
    'bg-lime-600',
    'bg-amber-600',
    'bg-red-600',
    'bg-blue-800',
    'bg-violet-800',
    'bg-fuchsia-800',
    'bg-emerald-800',
    'bg-cyan-800',
    'bg-emerald-800',
    'bg-lime-800',
    'bg-amber-800',
    'bg-red-800',
  ],
};
