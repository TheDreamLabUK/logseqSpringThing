import { initializeXRInteractions } from '../../public/js/xr/xrInteraction';

// Declare jest directly
const jest = require('jest-mock');

describe('XRInteraction', () => {
  test('initializeXRInteractions should initialize XR interactions correctly', () => {
    console.log = jest.fn(); // Mock console.log
    initializeXRInteractions();
    expect(console.log).toHaveBeenCalledWith('Initializing XR interactions...');
  });
});