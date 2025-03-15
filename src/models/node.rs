use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering, AtomicBool};
use crate::utils::socket_flow_messages::BinaryNodeData;
use crate::types::vec3::Vec3Data;
use std::path::Path;
use std::fs;
use log::{info, warn};

// Static counter for generating unique numeric IDs
static NEXT_NODE_ID: AtomicU32 = AtomicU32::new(1);  // Start from 1 (0 could be reserved)
static ID_INITIALIZED: AtomicBool = AtomicBool::new(false);

// Constants for ID management
const MAX_NODE_ID_FILE: &str = "data/metadata/max_node_id.txt";
const MAX_U16_VALUE: u32 = 65535; // Maximum value for u16 to ensure compatibility with binary protocol

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct Node {
    // Core data
    pub id: String,
    pub metadata_id: String,  // Store the original filename for lookup
    pub label: String,
    pub data: BinaryNodeData,

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
    /// Initialize the NEXT_NODE_ID counter from stored max ID
    pub fn initialize_id_counter() {
        // Only initialize once to prevent race conditions
        if ID_INITIALIZED.swap(true, Ordering::SeqCst) {
            return;
        }

        let max_id = Node::load_max_id_from_storage();
        
        // Update the counter if we found a valid ID
        if max_id > 0 {
            // Add 1 to ensure next ID is unique
            let next_id = max_id + 1;
            
            // Ensure we don't exceed u16 limit for binary protocol compatibility
            if next_id > MAX_U16_VALUE {
                warn!("Loaded max node ID {} exceeds u16 limit. Resetting to 1.", max_id);
                NEXT_NODE_ID.store(1, Ordering::SeqCst);
            } else {
                info!("Initialized node ID counter with value {} (loaded max ID: {})", next_id, max_id);
                NEXT_NODE_ID.store(next_id, Ordering::SeqCst);
            }
        } else {
            info!("No valid max node ID found, starting from 1");
            NEXT_NODE_ID.store(1, Ordering::SeqCst);
        }
    }
    
    /// Load the maximum node ID from storage
    fn load_max_id_from_storage() -> u32 {
        // Check if the file exists
        if !Path::new(MAX_NODE_ID_FILE).exists() {
            return 0;
        }
        
        // Try to read the file
        match fs::read_to_string(MAX_NODE_ID_FILE) {
            Ok(content) => {
                // Try to parse the content as u32
                match content.trim().parse::<u32>() {
                    Ok(id) => {
                        if id > 0 && id <= MAX_U16_VALUE {
                            return id;
                        } else {
                            warn!("Invalid node ID in storage: {}, must be between 1 and {}", id, MAX_U16_VALUE);
                        }
                    },
                    Err(e) => warn!("Failed to parse node ID from storage: {}", e)
                }
            },
            Err(e) => warn!("Failed to read max node ID file: {}", e)
        }
        
        0 // Return 0 if loading failed
    }
    
    /// Save the current maximum node ID to storage
    fn save_max_id_to_storage(id: u32) {
        // Create parent directory if it doesn't exist
        if let Some(parent) = Path::new(MAX_NODE_ID_FILE).parent() {
            if !parent.exists() {
                if let Err(e) = fs::create_dir_all(parent) {
                    warn!("Failed to create directory for max node ID: {}", e);
                    return;
                }
            }
        }
        
        // Write the ID to the file
        if let Err(e) = fs::write(MAX_NODE_ID_FILE, id.to_string()) {
            warn!("Failed to save max node ID to storage: {}", e);
        }
    }
    
    pub fn new(metadata_id: String) -> Self {
        // Ensure ID counter is initialized
        if !ID_INITIALIZED.load(Ordering::SeqCst) {
            Node::initialize_id_counter();
        }
        
        Self::new_with_id(metadata_id, None)
    }

    pub fn new_with_id(metadata_id: String, provided_id: Option<String>) -> Self {
        // Ensure ID counter is initialized
        if !ID_INITIALIZED.load(Ordering::SeqCst) {
            Node::initialize_id_counter();
        }
        
        // Always generate a new ID on the server side
        // Use provided ID only if it's a valid numeric string (from a previous session)
        let id = match provided_id {
            Some(id) if !id.is_empty() && id != "0" && id.parse::<u32>().is_ok() => {
                let parsed_id = id.parse::<u32>().unwrap();
                
                // Ensure we update our counter if this ID is higher
                if parsed_id > NEXT_NODE_ID.load(Ordering::SeqCst) {
                    NEXT_NODE_ID.store(parsed_id + 1, Ordering::SeqCst);
                    // Save the new maximum ID
                    Node::save_max_id_to_storage(parsed_id);
                }
                
                // Return the provided ID
                id.clone()
            },
            _ => {
                // Generate a new unique ID and save it
                let new_id = NEXT_NODE_ID.fetch_add(1, Ordering::SeqCst);
                
                // Save the new maximum ID periodically (every 10 IDs)
                if new_id % 10 == 0 {
                    Node::save_max_id_to_storage(new_id);
                }
                
                // Return the new ID as string
                new_id.to_string()
            }
        };
        
        Self {
            id,
            metadata_id: metadata_id.clone(),
            label: String::new(), // Initialize as empty string, will be set from metadata later
            data: BinaryNodeData {
                position: Vec3Data::zero(),
                velocity: Vec3Data::zero(),
                mass: 0,
                flags: 1, // Active by default
                padding: [0, 0],
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

    pub fn set_file_size(&mut self, size: u64) {
        self.file_size = size;
        // Calculate mass using log scale to prevent extremely large masses
        let base_mass = ((size + 1) as f32).log10() / 4.0;
        // Scale to 0-255 range for u8
        self.data.mass = ((base_mass.max(0.1).min(10.0) * 25.5) as u8).max(1);
    }

    pub fn with_position(mut self, x: f32, y: f32, z: f32) -> Self {
        self.data.position = Vec3Data::new(x, y, z);
        self
    }

    pub fn with_velocity(mut self, vx: f32, vy: f32, vz: f32) -> Self {
        self.data.velocity = Vec3Data::new(vx, vy, vz);
        self
    }

    pub fn with_label(mut self, label: String) -> Self {
        self.label = label;
        self
    }

    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }

    pub fn with_type(mut self, node_type: String) -> Self {
        self.node_type = Some(node_type);
        self
    }

    pub fn with_size(mut self, size: f32) -> Self {
        self.size = Some(size);
        self
    }

    pub fn with_color(mut self, color: String) -> Self {
        self.color = Some(color);
        self
    }

    pub fn with_weight(mut self, weight: f32) -> Self {
        self.weight = Some(weight);
        self
    }

    pub fn with_group(mut self, group: String) -> Self {
        self.group = Some(group);
        self
    }

    // Convenience getters/setters for position and velocity
    pub fn x(&self) -> f32 { self.data.position.x }
    pub fn y(&self) -> f32 { self.data.position.y }
    pub fn z(&self) -> f32 { self.data.position.z }
    pub fn vx(&self) -> f32 { self.data.velocity.x }
    pub fn vy(&self) -> f32 { self.data.velocity.y }
    pub fn vz(&self) -> f32 { self.data.velocity.z }
    
    pub fn set_x(&mut self, val: f32) { self.data.position.x = val; }
    pub fn set_y(&mut self, val: f32) { self.data.position.y = val; }
    pub fn set_z(&mut self, val: f32) { self.data.position.z = val; }
    pub fn set_vx(&mut self, val: f32) { self.data.velocity.x = val; }
    pub fn set_vy(&mut self, val: f32) { self.data.velocity.y = val; }
    pub fn set_vz(&mut self, val: f32) { self.data.velocity.z = val; }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::Ordering;
    use super::*;

    #[test]
    fn test_numeric_id_generation() {
        // Read the current value of the counter (it might have been incremented elsewhere)
        let start_value = NEXT_NODE_ID.load(Ordering::SeqCst);
        
        // Create two nodes with different metadata IDs
        let node1 = Node::new("test-file-1.md".to_string());
        let node2 = Node::new("test-file-2.md".to_string());
        
        // Verify each node has a unique numeric ID
        assert_ne!(node1.id, node2.id);
        
        // Verify metadata_id is stored correctly
        assert_eq!(node1.metadata_id, "test-file-1.md");
        assert_eq!(node2.metadata_id, "test-file-2.md");
        
        // Verify IDs are consecutive numbers (as strings)
        let id1: u32 = node1.id.parse().unwrap();
        let id2: u32 = node2.id.parse().unwrap();
        assert_eq!(id1 + 1, id2);
        
        // Verify final counter value
        let end_value = NEXT_NODE_ID.load(Ordering::SeqCst);
        assert_eq!(end_value, start_value + 2);
    }

    #[test]
    fn test_node_creation() {
        let node = Node::new("test".to_string())
            .with_label("Test Node".to_string())
            .with_position(1.0, 2.0, 3.0)
            .with_velocity(0.1, 0.2, 0.3)
            .with_type("test_type".to_string())
            .with_size(1.5)
            .with_color("#FF0000".to_string())
            .with_weight(2.0)
            .with_group("group1".to_string());

        // ID should be a numeric string now, not "test"
        assert!(node.id.parse::<u32>().is_ok(), "ID should be numeric, got: {}", node.id);
        assert_eq!(node.metadata_id, "test");
        assert_eq!(node.label, "Test Node");
        assert_eq!(node.data.position.x, 1.0);
        assert_eq!(node.data.position.y, 2.0);
        assert_eq!(node.data.position.z, 3.0);
        assert_eq!(node.data.velocity.x, 0.1);
        assert_eq!(node.data.velocity.y, 0.2);
        assert_eq!(node.data.velocity.z, 0.3);
        assert_eq!(node.node_type, Some("test_type".to_string()));
        assert_eq!(node.size, Some(1.5));
        assert_eq!(node.color, Some("#FF0000".to_string()));
        assert_eq!(node.weight, Some(2.0));
        assert_eq!(node.group, Some("group1".to_string()));
    }

    #[test]
    fn test_position_velocity_getters_setters() {
        let mut node = Node::new("test".to_string());
        
        node.set_x(1.0);
        node.set_y(2.0);
        node.set_z(3.0);
        node.set_vx(0.1);
        node.set_vy(0.2);
        node.set_vz(0.3);

        assert_eq!(node.x(), 1.0);
        assert_eq!(node.y(), 2.0);
        assert_eq!(node.z(), 3.0);
        assert_eq!(node.vx(), 0.1);
        assert_eq!(node.vy(), 0.2);
        assert_eq!(node.vz(), 0.3);
    }

    #[test]
    fn test_mass_calculation() {
        let mut node = Node::new("test".to_string());
        
        // Test small file
        node.set_file_size(100);  // 100 bytes
        assert!(node.data.mass > 0 && node.data.mass < 128);

        // Test large file
        node.set_file_size(1_000_000);  // 1MB
        assert!(node.data.mass > 128 && node.data.mass < 255);
    }
}
