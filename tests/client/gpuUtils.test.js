import { initGPU, computeOnGPU, isGPUAvailable } from '../../public/js/gpuUtils';

describe('GPU Utils', () => {
  test('initGPU function exists', () => {
    expect(typeof initGPU).toBe('function');
  });

  test('initGPU returns an object with compute method', () => {
    const gpu = initGPU();
    expect(typeof gpu.compute).toBe('function');
  });

  test('computeOnGPU function exists', () => {
    expect(typeof computeOnGPU).toBe('function');
  });

  test('computeOnGPU returns true', () => {
    const mockGPU = { compute: jest.fn() };
    const mockData = [1, 2, 3];
    expect(computeOnGPU(mockGPU, mockData)).toBe(true);
    expect(mockGPU.compute).toHaveBeenCalledWith(mockData);
  });

  test('isGPUAvailable function exists', () => {
    expect(typeof isGPUAvailable).toBe('function');
  });

  test('isGPUAvailable returns true', () => {
    expect(isGPUAvailable()).toBe(true);
  });
});