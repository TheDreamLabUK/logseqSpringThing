import { initInterface } from '../../public/js/components/interface';

describe('Interface', () => {
  test('initInterface should initialize interface correctly', () => {
    const consoleSpy = jest.spyOn(console, 'log');
    initInterface();
    expect(consoleSpy).toHaveBeenCalledWith('Initializing interface...');
    consoleSpy.mockRestore();
  });
});