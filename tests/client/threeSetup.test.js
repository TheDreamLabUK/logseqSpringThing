import { setupThreeScene } from '../../public/js/threeJS/threeSetup';

// Declare jest directly
const jest = require('jest-mock');

describe('ThreeSetup', () => {
  test('setupThreeScene should set up Three.js scene correctly', () => {
    console.log = jest.fn(); // Mock console.log
    setupThreeScene();
    expect(console.log).toHaveBeenCalledWith('Setting up Three.js scene...');
  });
});