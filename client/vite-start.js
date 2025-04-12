#!/usr/bin/env node

// Override console methods to ensure all logs are visible
const originalConsole = {
  log: console.log,
  info: console.info,
  warn: console.warn,
  error: console.error,
  debug: console.debug
};

// Prefix all console output
function prefixLog(prefix, originalFn) {
  return function(...args) {
    if (typeof args[0] === 'string') {
      originalFn(prefix + args[0], ...args.slice(1));
    } else {
      originalFn(prefix, ...args);
    }
  };
}

// Apply overrides
console.log = prefixLog('[VITE] ', originalConsole.log);
console.info = prefixLog('[VITE] ', originalConsole.info);
console.warn = prefixLog('[VITE] ', originalConsole.warn);
console.error = prefixLog('[VITE] ', originalConsole.error);
console.debug = prefixLog('[VITE] ', originalConsole.debug);

// Run Vite
require('vite/cli');
