import { setupXRSession } from '../../public/js/xr/xrSetup';

// Declare jest directly
const jest = require('jest-mock');

describe('XRSetup', () => {
  test('setupXRSession should set up XR session correctly', () => {
    console.log = jest.fn(); // Mock console.log
    setupXRSession();
    expect(console.log).toHaveBeenCalledWith('Setting up XR session...');
  });
});