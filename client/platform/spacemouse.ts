// SpaceMouse HID interface types
interface SpaceMouseEvent {
    x: number;
    y: number;
    z: number;
}

interface HIDDevice {
    open(): Promise<void>;
    productName: string;
    addEventListener(event: string, handler: (event: any) => void): void;
}

const SPACEMOUSE_VENDOR_ID = 0x256F; // 3Dconnexion vendor ID
const SPACEMOUSE_PRODUCT_ID = 0xC635; // SpaceMouse Compact product ID

let spacemouseDevice: HIDDevice | null = null;

async function requestHIDAccess(): Promise<void> {
    try {
        // @ts-ignore - HID API types not available
        const devices = await navigator.hid.requestDevice({
            filters: [{ vendorId: SPACEMOUSE_VENDOR_ID, productId: SPACEMOUSE_PRODUCT_ID }]
        });
        
        if (devices.length > 0) {
            const device = devices[0];
            if (!device) {
                throw new Error('No device found');
            }
            
            await device.open();
            console.log('HID device opened:', device.productName);
            device.addEventListener('inputreport', handleHIDInput);
            spacemouseDevice = device;
        }
    } catch (error) {
        console.error('HID access denied:', error);
    }
}

function handleHIDInput(event: { data: DataView }): void {
    const { data } = event;
    
    // Parse the input data
    const x = data.getInt16(1, true);
    const y = data.getInt16(3, true);
    const z = data.getInt16(5, true);

    // Normalize values (adjust as needed based on your Spacemouse model)
    const normalizedX = x / 350;
    const normalizedY = y / 350;
    const normalizedZ = z / 350;

    // Create a typed event
    const detail: SpaceMouseEvent = {
        x: normalizedX,
        y: normalizedY,
        z: normalizedZ
    };

    // Dispatch the event
    window.dispatchEvent(new CustomEvent('spacemouse-move', { detail }));
}

// Function to check if WebHID is supported
function isHIDSupported(): boolean {
    // @ts-ignore - HID API types not available
    return 'hid' in navigator;
}

// Function to check if device is connected
function isConnected(): boolean {
    return spacemouseDevice !== null;
}

// Function to be called when the "Enable Spacemouse" button is clicked
function enableSpacemouse(): void {
    if (isHIDSupported()) {
        requestHIDAccess();
    } else {
        console.error('WebHID is not supported in this browser');
        alert('WebHID is not supported in this browser. Please use a compatible browser like Chrome or Edge.');
    }
}

export { 
    enableSpacemouse,
    isHIDSupported,
    isConnected,
    type SpaceMouseEvent 
};
