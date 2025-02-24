use bytemuck::{Pod, Zeroable};
use cudarc::driver::{DeviceRepr, ValidAsZeroBits};

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct EdgeData {
    pub source_idx: i32,
    pub target_idx: i32,
    pub weight: f32,
}

// Implement DeviceRepr for EdgeData
unsafe impl DeviceRepr for EdgeData {}

// Implement ValidAsZeroBits for EdgeData
unsafe impl ValidAsZeroBits for EdgeData {}