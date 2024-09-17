// xrInteraction.test.js

const XRInteraction = require('../../public/js/xr/xrInteraction');
const THREE = require('three');

describe('XRInteraction', () => {
  let xrInteraction;

  beforeEach(() => {
    xrInteraction = new XRInteraction();
  });

  test('should initialize properly', () => {
    expect(xrInteraction).toBeDefined();
    expect(xrInteraction.controller).toBeDefined();
  });

  test('should handle controller input', () => {
    const onSelectStartSpy = jest.spyOn(xrInteraction, 'onSelectStart');
    const onSelectEndSpy = jest.spyOn(xrInteraction, 'onSelectEnd');

    xrInteraction.controller.dispatchEvent({ type: 'selectstart' });
    xrInteraction.controller.dispatchEvent({ type: 'selectend' });

    expect(onSelectStartSpy).toHaveBeenCalled();
    expect(onSelectEndSpy).toHaveBeenCalled();
  });
});
