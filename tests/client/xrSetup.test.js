import { setupXRSession } from '../../public/js/xr/xrSetup';

describe('XRSetup', () => {
  test('setupXRSession should set up XR session correctly', () => {
    const consoleSpy = jest.spyOn(console, 'log');
    setupXRSession();
    expect(consoleSpy).toHaveBeenCalledWith('Setting up XR session...');
    consoleSpy.mockRestore();
  });
});