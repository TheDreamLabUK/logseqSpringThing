import { runCLI } from 'jest';
import { fileURLToPath } from 'url';
import { dirname } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const config = {
  verbose: true,
  testEnvironment: 'jsdom',
  transform: {
    '^.+\\.js$': 'babel-jest',
  },
};

runCLI(config, [__dirname])
  .then(({ results }) => {
    console.log('Test Results:');
    console.log(`Total Tests: ${results.numTotalTests}`);
    console.log(`Passed Tests: ${results.numPassedTests}`);
    console.log(`Failed Tests: ${results.numFailedTests}`);
    
    results.testResults.forEach((testFile) => {
      console.log(`\nFile: ${testFile.testFilePath}`);
      testFile.testResults.forEach((test) => {
        console.log(`  ${test.status}: ${test.title}`);
        if (test.status === 'failed') {
          console.log(`    Error: ${test.failureMessages.join('\n')}`);
        }
      });
    });
    
    process.exit(results.success ? 0 : 1);
  })
  .catch((error) => {
    console.error('Error running tests:', error);
    process.exit(1);
  });