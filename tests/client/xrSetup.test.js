// xrSetup.test.js

const XRSetup = require('../../public/js/xr/xrSetup');

describe('XRSetup', () => {
  let xrSetup;

  beforeEach(() => {
    xrSetup = new XRSetup();
  });

  test('should initialize properly', () => {
    expect(xrSetup).toBeDefined();
    expect(xrSetup.renderer).toBeDefined();
  });

  test('should check for XR support', async () => {
    global.navigator.xr = {
      isSessionSupported: jest.fn().mockResolvedValue(true),
    };

    const result = await xrSetup.checkXRSupport();

    expect(global.navigator.xr.isSessionSupported).toHaveBeenCalledWith(
      'immersive-vr'
    );
    expect(result).toBe(true);
  });
});
