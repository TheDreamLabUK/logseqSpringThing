// src/models/edge.rs
use serde::{Serialize, Deserialize};

#[derive(Clone, Serialize, Deserialize)]
pub struct Edge {
    pub source: u32,
    pub target: u32,
    pub weight: f32,
}

impl Edge {
    pub fn new(source: u32, target: u32, weight: f32) -> Self {
        Edge { source, target, weight }
    }
}
