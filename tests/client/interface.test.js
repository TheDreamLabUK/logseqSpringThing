// interface.test.js

const Interface = require('../../public/js/components/interface');

describe('Interface', () => {
  let interfaceInstance;

  beforeEach(() => {
    document.body.innerHTML = `<canvas id="canvas"></canvas>`;
    interfaceInstance = new Interface();
  });

  test('should initialize properly', () => {
    expect(interfaceInstance).toBeDefined();
  });

  test('should handle user input events', () => {
    const onInputSpy = jest.fn();
    interfaceInstance.onUserInput = onInputSpy;

    const event = new Event('keydown');
    document.dispatchEvent(event);

    expect(onInputSpy).toHaveBeenCalled();
  });
});
