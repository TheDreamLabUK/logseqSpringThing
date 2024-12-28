use std::sync::Arc;
use tempfile::TempDir;
use crate::state::AppState;
use crate::models::graph::{Node, Edge, Graph};

pub struct TestContext {
    pub state: Arc<AppState>,
    pub temp_dir: TempDir,
}

impl TestContext {
    pub async fn new() -> Self {
        let temp_dir = tempfile::tempdir().unwrap();
        let state = Arc::new(AppState::new(temp_dir.path().to_path_buf()).unwrap());
        
        Self {
            state,
            temp_dir,
        }
    }

    pub async fn create_test_graph(&self) -> Graph {
        let mut graph = Graph::new();
        
        // Add test nodes
        let nodes = vec![
            Node {
                id: "1".to_string(),
                position: [0.0, 0.0, 0.0],
                velocity: [0.0, 0.0, 0.0],
                mass: 1.0,
            },
            Node {
                id: "2".to_string(),
                position: [1.0, 1.0, 1.0],
                velocity: [0.0, 0.0, 0.0],
                mass: 1.0,
            },
        ];

        // Add test edges
        let edges = vec![
            Edge {
                source: "1".to_string(),
                target: "2".to_string(),
                strength: 1.0,
            },
        ];

        graph.update_graph(nodes, edges).await.unwrap();
        graph
    }
}

pub async fn setup_test_context() -> TestContext {
    TestContext::new().await
}

pub fn assert_graph_equality(g1: &Graph, g2: &Graph) {
    assert_eq!(g1.node_count(), g2.node_count());
    assert_eq!(g1.edge_count(), g2.edge_count());

    let nodes1 = g1.get_nodes();
    let nodes2 = g2.get_nodes();
    
    for (n1, n2) in nodes1.iter().zip(nodes2.iter()) {
        assert_eq!(n1.id, n2.id);
        assert_eq!(n1.position, n2.position);
        assert_eq!(n1.velocity, n2.velocity);
        assert_eq!(n1.mass, n2.mass);
    }

    let edges1 = g1.get_edges();
    let edges2 = g2.get_edges();
    
    for (e1, e2) in edges1.iter().zip(edges2.iter()) {
        assert_eq!(e1.source, e2.source);
        assert_eq!(e1.target, e2.target);
        assert_eq!(e1.strength, e2.strength);
    }
} 