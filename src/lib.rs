#![recursion_limit = "256"]

extern crate log;

// Declare modules
pub mod utils;
pub mod app_state;
pub mod config;
pub mod handlers;
pub mod models;
pub mod services;
pub mod types;
pub mod state;

// Re-export standard logging if needed
pub use log::{debug, error, info, warn};

// Re-export GPU compute
pub use crate::utils::gpu_compute::GPUCompute;

// Re-export socket flow handler
pub use crate::handlers::socket_flow_handler::{SocketFlowServer, socket_flow_handler};

// Public re-exports
pub use app_state::AppState;
pub use config::Settings;
pub use models::position_update::PositionUpdate;
pub use models::metadata::MetadataStore;
pub use models::simulation_params::SimulationParams;
pub use models::graph::GraphData;
pub use services::graph_service::GraphService;
pub use services::file_service::FileService;
pub use services::perplexity_service::PerplexityService;
pub use services::ragflow_service::{RAGFlowService, RAGFlowError};
pub use services::github::{GitHubClient, ContentAPI};

// Re-export handlers
pub use handlers::api_handler::files as file_handler;
pub use handlers::api_handler::graph as graph_handler;
pub use handlers::health_handler;
pub use handlers::pages_handler;
pub use handlers::perplexity_handler;
pub use handlers::ragflow_handler;
pub use handlers::api_handler::visualization as visualization_handler;
pub use handlers::settings_handler;
