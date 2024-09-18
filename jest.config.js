module.exports = {
  testEnvironment: 'jsdom',
  moduleFileExtensions: ['js', 'json', 'mjs'],
  transform: {
    '^.+\\.(js|mjs)$': 'babel-jest',
  },
  moduleNameMapper: {
    '^@/(.*)$': '<rootDir>/public/js/$1',
    '^three$': '<rootDir>/node_modules/three/build/three.js',
    '^three/examples/jsm/(.*)$': '<rootDir>/node_modules/three/examples/jsm/$1',
    '^3d-force-graph$': '<rootDir>/node_modules/3d-force-graph/dist/3d-force-graph.js',
  },
  testMatch: ['<rootDir>/tests/client/**/*.test.js'],
  setupFilesAfterEnv: ['<rootDir>/jest.setup.js'],
  transformIgnorePatterns: [
    '/node_modules/(?!(three|3d-force-graph)/)'
  ],
  globals: {
    'ts-jest': {
      useESM: true,
    },
  },
};