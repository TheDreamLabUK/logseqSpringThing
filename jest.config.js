module.exports = {
  testEnvironment: 'jsdom',
  moduleFileExtensions: ['js', 'json', 'mjs'],
  transform: {
    '^.+\\.mjs$': 'babel-jest', // Transforms .mjs files with Babel
    '^.+\\.[jt]sx?$': 'babel-jest', // Transforms JS and JSX
  },
  moduleNameMapper: {
    '^@/(.*)$': '<rootDir>/public/js/$1',
  },
  testMatch: ['<rootDir>/tests/client/**/*.test.js'],
  setupFilesAfterEnv: ['<rootDir>/jest.setup.js'],
  transformIgnorePatterns: [
    '/node_modules/(?!three|3d-force-graph|other-es-modules)' // Ensures ES modules are not ignored
  ],
  globals: {
    'babel-jest': {
      useESModules: true, // Ensures ES module support in Babel
    },
  },
};