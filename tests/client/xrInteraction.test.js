import { initXRInteraction, updateXRInteraction, handleXRControllerEvent } from '../../public/js/xr/xrInteraction';

describe('XR Interaction', () => {
  let mockScene, mockCamera, xrInteraction;

  beforeEach(() => {
    mockScene = {};
    mockCamera = {};
    xrInteraction = initXRInteraction(mockScene, mockCamera);
  });

  test('initXRInteraction function exists', () => {
    expect(typeof initXRInteraction).toBe('function');
  });

  test('initXRInteraction returns an object with update and handleControllerEvent methods', () => {
    expect(typeof xrInteraction.update).toBe('function');
    expect(typeof xrInteraction.handleControllerEvent).toBe('function');
  });

  test('updateXRInteraction function exists', () => {
    expect(typeof updateXRInteraction).toBe('function');
  });

  test('updateXRInteraction returns true', () => {
    expect(updateXRInteraction(xrInteraction)).toBe(true);
  });

  test('handleXRControllerEvent function exists', () => {
    expect(typeof handleXRControllerEvent).toBe('function');
  });

  test('handleXRControllerEvent returns true', () => {
    const mockEvent = { type: 'select' };
    expect(handleXRControllerEvent(xrInteraction, mockEvent)).toBe(true);
  });
});
