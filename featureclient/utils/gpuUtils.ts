// This file is kept as a placeholder for future GPU-specific utilities.
// Position updates are now handled directly in the binaryUpdate store.

export function isGPUAvailable(): boolean {
    return 'gpu' in navigator;
}

export function getGPUTier(): number {
    // Simple GPU tier detection
    // Returns:
    // 0 = no GPU/unknown
    // 1 = integrated GPU
    // 2 = discrete GPU
    if (!isGPUAvailable()) return 0;

    const canvas = document.createElement('canvas');
    const gl = canvas.getContext('webgl2');
    if (!gl) return 0;

    const debugInfo = gl.getExtension('WEBGL_debug_renderer_info');
    if (!debugInfo) return 1;

    const renderer = gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL).toLowerCase();
    
    // Check for common discrete GPU identifiers
    const discreteGPUIdentifiers = [
        'nvidia',
        'radeon',
        'geforce',
        'rx',
        'rtx',
        'gtx'
    ];

    return discreteGPUIdentifiers.some(id => renderer.includes(id)) ? 2 : 1;
}
