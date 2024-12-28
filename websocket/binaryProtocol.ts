import { BINARY_PROTOCOL, HEADER_SIZE } from './constants';

export enum MessageType {
    POSITION_UPDATE = 1,
    FORCE_UPDATE = 2,
    STATE_UPDATE = 3
}

export class BinaryProtocol {
    static validateHeader(data: ArrayBuffer): boolean {
        const view = new DataView(data);
        const version = view.getUint32(0);
        return version === BINARY_PROTOCOL.VERSION;
    }

    static getMessageType(data: ArrayBuffer): MessageType {
        const view = new DataView(data);
        return view.getUint32(4);
    }

    static createPositionUpdate(positions: Float32Array): ArrayBuffer {
        const buffer = new ArrayBuffer(HEADER_SIZE + positions.byteLength);
        const view = new DataView(buffer);
        
        // Write header
        view.setUint32(0, BINARY_PROTOCOL.VERSION);
        view.setUint32(4, MessageType.POSITION_UPDATE);
        
        // Write positions
        new Float32Array(buffer, HEADER_SIZE).set(positions);
        
        return buffer;
    }

    static parsePositionUpdate(data: ArrayBuffer): Float32Array {
        if (!this.validateHeader(data)) {
            throw new Error('Invalid protocol version');
        }
        
        const messageType = this.getMessageType(data);
        if (messageType !== MessageType.POSITION_UPDATE) {
            throw new Error('Invalid message type');
        }
        
        return new Float32Array(data, HEADER_SIZE);
    }
} 