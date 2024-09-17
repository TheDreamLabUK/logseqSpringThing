import { initInterface, handleUserInput } from '../../public/js/components/interface';

describe('User Interface', () => {
  test('initInterface function exists', () => {
    expect(typeof initInterface).toBe('function');
  });

  test('initInterface function can be called without errors', () => {
    expect(() => initInterface()).not.toThrow();
  });

  test('handleUserInput function exists', () => {
    expect(typeof handleUserInput).toBe('function');
  });

  test('handleUserInput function returns the input', () => {
    const input = 'test input';
    expect(handleUserInput(input)).toBe(input);
  });
});
