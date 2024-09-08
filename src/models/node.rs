use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Default, Serialize, Deserialize)]
pub struct Node {
    pub id: String,
    pub label: String,
    pub metadata: HashMap<String, String>,
}
