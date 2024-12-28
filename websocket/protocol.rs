use bytes::{BytesMut, BufMut};
use std::io::{self, Cursor};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use super::constants::{BINARY_PROTOCOL_VERSION, NODE_POSITION_SIZE, MessageType};
use log;
use flate2::{write::GzEncoder, read::GzDecoder, Compression};

pub struct BinaryProtocol;

impl BinaryProtocol {
    pub fn create_header(msg_type: MessageType) -> BytesMut {
        let mut header = BytesMut::with_capacity(8);
        header.put_u32_le(BINARY_PROTOCOL_VERSION);
        header.put_u32_le(msg_type as u32);
        header
    }

    pub fn validate_header(data: &[u8]) -> io::Result<(u32, MessageType)> {
        let mut cursor = Cursor::new(data);
        let version = cursor.read_u32::<LittleEndian>()?;
        
        if version != BINARY_PROTOCOL_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Invalid protocol version: {}", version)
            ));
        }

        let msg_type = cursor.read_u32::<LittleEndian>()?;
        Ok((version, MessageType::from(msg_type)))
    }

    pub fn create_position_update(positions: &[[f32; 3]]) -> BytesMut {
        let mut buffer = Self::create_header(MessageType::PositionUpdate);
        buffer.reserve(positions.len() * NODE_POSITION_SIZE);

        for pos in positions {
            buffer.put_f32_le(pos[0]);
            buffer.put_f32_le(pos[1]);
            buffer.put_f32_le(pos[2]);
        }

        buffer
    }

    pub fn parse_position_update(data: &[u8]) -> io::Result<Vec<[f32; 3]>> {
        let (_, msg_type) = Self::validate_header(data)?;
        
        if !matches!(msg_type, MessageType::PositionUpdate) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid message type for position update"
            ));
        }

        let mut cursor = Cursor::new(&data[8..]);
        let mut positions = Vec::new();

        while cursor.position() < (data.len() - 8) as u64 {
            let x = cursor.read_f32::<LittleEndian>()?;
            let y = cursor.read_f32::<LittleEndian>()?;
            let z = cursor.read_f32::<LittleEndian>()?;
            positions.push([x, y, z]);
        }

        Ok(positions)
    }

    pub fn parse_gpu_compute_status(data: &[u8]) -> io::Result<bool> {
        let (_, msg_type) = Self::validate_header(data)?;
        
        if !matches!(msg_type, MessageType::GpuComputeStatus) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid message type for GPU compute status"
            ));
        }

        let mut cursor = Cursor::new(&data[8..]);
        let enabled = cursor.read_u8()? != 0;
        Ok(enabled)
    }

    pub fn create_gpu_compute_status(enabled: bool) -> BytesMut {
        let mut buffer = Self::create_header(MessageType::GpuComputeStatus);
        buffer.put_u8(if enabled { 1 } else { 0 });
        buffer
    }

    pub fn create_compressed_update(data: &[u8]) -> io::Result<BytesMut> {
        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(data)?;
        let compressed = encoder.finish()?;
        
        let mut buffer = Self::create_header(MessageType::CompressedData);
        buffer.extend_from_slice(&compressed);
        Ok(buffer)
    }

    pub fn parse_compressed_update(data: &[u8]) -> io::Result<Vec<u8>> {
        let (_, msg_type) = Self::validate_header(data)?;
        
        if !matches!(msg_type, MessageType::CompressedData) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid message type for compressed data"
            ));
        }

        let mut decoder = GzDecoder::new(&data[8..]);
        let mut decompressed = Vec::new();
        decoder.read_to_end(&mut decompressed)?;
        Ok(decompressed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_protocol_roundtrip() {
        let positions = vec![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let buffer = BinaryProtocol::create_position_update(&positions);
        let parsed = BinaryProtocol::parse_position_update(&buffer).unwrap();
        assert_eq!(positions, parsed);
    }

    #[test]
    fn test_invalid_version() {
        let mut buffer = BytesMut::with_capacity(8);
        buffer.put_u32_le(999); // Invalid version
        buffer.put_u32_le(MessageType::PositionUpdate as u32);
        
        let result = BinaryProtocol::validate_header(&buffer);
        assert!(result.is_err());
    }

    #[test]
    fn test_gpu_compute_status() {
        let enabled = true;
        let buffer = BinaryProtocol::create_gpu_compute_status(enabled);
        let parsed = BinaryProtocol::parse_gpu_compute_status(&buffer).unwrap();
        assert_eq!(enabled, parsed);
    }
} 