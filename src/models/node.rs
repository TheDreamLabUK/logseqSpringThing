use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use bytemuck::{Pod, Zeroable};

/// Core node structure optimized for GPU computation and network transfer
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct NodeData {
    pub position: [f32; 3],  // 12 bytes - matches THREE.Vector3
    pub velocity: [f32; 3],  // 12 bytes - matches THREE.Vector3
    pub mass: u8,            // 1 byte - quantized mass
    pub flags: u8,           // 1 byte - node state flags
    pub padding: [u8; 2],    // 2 bytes - alignment padding
}

/// Node metadata and rendering properties
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct Node {
    // Core data
    pub id: String,
    pub label: String,
    pub data: NodeData,

    // Metadata
    #[serde(skip_serializing_if = "HashMap::is_empty")]
    pub metadata: HashMap<String, String>,
    #[serde(skip)]
    pub file_size: u64,

    // Rendering properties
    #[serde(rename = "type")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub node_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub color: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub weight: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub group: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_data: Option<HashMap<String, String>>,
}

impl Node {
    pub fn new(id: String) -> Self {
        Self {
            id: id.clone(),
            label: id,
            data: NodeData {
                position: [0.0; 3],
                velocity: [0.0; 3],
                mass: 127, // Default mass
                flags: 0,
                padding: [0; 2],
            },
            metadata: HashMap::new(),
            file_size: 0,
            node_type: None,
            size: None,
            color: None,
            weight: None,
            group: None,
            user_data: None,
        }
    }

    /// Update mass based on file size
    pub fn update_mass(&mut self) {
        if self.file_size == 0 {
            self.data.mass = 127; // Default mass
            return;
        }
        
        // Scale file size logarithmically to 0-255 range
        let log_size = (self.file_size as f64).log2();
        let max_log = (1024.0 * 1024.0 * 1024.0_f64).log2(); // 1GB
        let normalized = (log_size / max_log).min(1.0);
        self.data.mass = (normalized * 255.0) as u8;
    }
}

impl Default for Node {
    fn default() -> Self {
        Self::new(String::new())
    }
}

// WGSL struct reference:
/// ```wgsl
/// struct Node {
///     position: vec3<f32>,  // 12 bytes
///     velocity: vec3<f32>,  // 12 bytes
///     mass: u8,            // 1 byte
///     flags: u8,           // 1 byte
///     padding: vec2<u8>,   // 2 bytes padding
/// }
