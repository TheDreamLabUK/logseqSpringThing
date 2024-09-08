import { initInterface } from '../../public/js/components/interface';

// Declare jest directly
const jest = require('jest-mock');

describe('Interface', () => {
  test('initInterface should initialize interface correctly', () => {
    console.log = jest.fn(); // Mock console.log
    initInterface();
    expect(console.log).toHaveBeenCalledWith('Initializing interface...');
  });
});