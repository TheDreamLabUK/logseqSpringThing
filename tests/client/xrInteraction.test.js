import { initializeXRInteractions } from '../../public/js/xr/xrInteraction';

describe('XRInteraction', () => {
  test('initializeXRInteractions should initialize XR interactions correctly', () => {
    const consoleSpy = jest.spyOn(console, 'log');
    initializeXRInteractions();
    expect(consoleSpy).toHaveBeenCalledWith('Initializing XR interactions...');
    consoleSpy.mockRestore();
  });
});