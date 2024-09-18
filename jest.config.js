module.exports = {
  testEnvironment: 'jsdom',
  moduleFileExtensions: ['js', 'json', 'mjs'],
  transform: {
    '^.+\\.mjs$': 'babel-jest',
    '^.+\\.[jt]sx?$': 'babel-jest',
  },
  moduleNameMapper: {
    '^@/(.*)$': '<rootDir>/public/js/$1',
    '^three$': '<rootDir>/node_modules/three',
  },
  testMatch: ['<rootDir>/tests/client/**/*.test.js'],
  setupFilesAfterEnv: ['<rootDir>/jest.setup.js', '<rootDir>/jest.setup.three.js'],
  globals: {
    'babel-jest': {
      useESModules: true,
>>>>>>> 5e408e194b8851565f2b1e0bb7217b0722c0bb15
    },
  },
};