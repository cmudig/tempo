import { writable } from 'svelte/store';
import './app.css';
import App from './App.svelte';

let csrf = document.getElementsByName('csrf-token')[0].content;

const app = new App({
  target: document.getElementById('app'),
  props: {
    csrf: writable(csrf),
  },
});

export default app;
