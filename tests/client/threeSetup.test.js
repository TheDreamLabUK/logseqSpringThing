import { setupThreeScene } from '../../public/js/threeJS/threeSetup';

describe('ThreeSetup', () => {
  test('setupThreeScene should set up Three.js scene correctly', () => {
    const consoleSpy = jest.spyOn(console, 'log');
    setupThreeScene();
    expect(consoleSpy).toHaveBeenCalledWith('Setting up Three.js scene...');
    consoleSpy.mockRestore();
  });
});