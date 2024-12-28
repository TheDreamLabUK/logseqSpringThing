use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use serde_json;
use tokio::fs::File;
use tokio::io::AsyncWriteExt;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    pub id: String,
    pub position: [f32; 3],
    pub velocity: [f32; 3],
    pub mass: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Edge {
    pub source: String,
    pub target: String,
    pub strength: f32,
}

pub struct Graph {
    nodes: HashMap<String, Node>,
    edges: Vec<Edge>,
}

impl Graph {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            edges: Vec::new(),
        }
    }

    pub fn get_nodes(&self) -> Vec<Node> {
        self.nodes.values().cloned().collect()
    }

    pub fn get_edges(&self) -> Vec<Edge> {
        self.edges.clone()
    }

    pub fn get_nodes_paginated(&self, start: usize, limit: usize) -> Vec<Node> {
        self.nodes.values()
            .skip(start)
            .take(limit)
            .cloned()
            .collect()
    }

    pub fn get_edges_paginated(&self, start: usize, limit: usize) -> Vec<Edge> {
        self.edges.iter()
            .skip(start)
            .take(limit)
            .cloned()
            .collect()
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    pub async fn update_graph(&mut self, nodes: Vec<Node>, edges: Vec<Edge>) -> Result<(), String> {
        // Validate the graph update
        for edge in &edges {
            if !self.nodes.contains_key(&edge.source) || !self.nodes.contains_key(&edge.target) {
                return Err("Edge references non-existent node".to_string());
            }
        }

        // Update nodes
        for node in nodes {
            self.nodes.insert(node.id.clone(), node);
        }

        // Update edges
        self.edges = edges;

        Ok(())
    }

    pub async fn save_to_file<P: AsRef<Path>>(&self, path: P) -> io::Result<()> {
        let data = serde_json::to_string_pretty(&GraphData {
            nodes: self.get_nodes(),
            edges: self.get_edges(),
        })?;

        let mut file = File::create(path).await?;
        file.write_all(data.as_bytes()).await?;
        file.sync_all().await?;

        Ok(())
    }

    pub async fn load_from_file<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let data = fs::read_to_string(path)?;
        let graph_data: GraphData = serde_json::from_str(&data)?;

        let mut graph = Graph::new();
        for node in graph_data.nodes {
            graph.nodes.insert(node.id.clone(), node);
        }
        graph.edges = graph_data.edges;

        Ok(graph)
    }
}

#[derive(Serialize, Deserialize)]
struct GraphData {
    nodes: Vec<Node>,
    edges: Vec<Edge>,
} 