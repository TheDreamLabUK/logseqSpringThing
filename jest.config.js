module.exports = {
  transform: {
    '^.+\\.jsx?$': 'babel-jest', // Transform JavaScript files using babel-jest
  },
  transformIgnorePatterns: [
    '/node_modules/(?!three/examples|three/examples/jsm|@babel)'
  ],
  moduleNameMapper: {
    '^three$': '<rootDir>/node_modules/three', // Ensure Jest resolves the three module correctly
  },
  setupFilesAfterEnv: ['<rootDir>/jest.setup.js'], // Ensure the setup file is included
  testEnvironment: 'jsdom', // Set the test environment to jsdom for browser-like environment
};