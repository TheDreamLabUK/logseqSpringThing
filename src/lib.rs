#[macro_use]
extern crate log;

// Re-export macros at crate root
#[macro_use]
pub mod utils;

// Re-export debug settings
pub use utils::debug_logging::init_debug_settings;

// Module declarations
pub mod app_state;
pub mod config;
pub mod handlers;
pub mod models;
pub mod services;

// Public re-exports
pub use app_state::AppState;
pub use config::Settings;
pub use models::position_update::PositionUpdate;
pub use models::metadata::MetadataStore;
pub use models::simulation_params::SimulationParams;
pub use models::graph::GraphData;
pub use services::graph_service::GraphService;
pub use services::file_service::{RealGitHubService, FileService};
pub use services::perplexity_service::PerplexityService;
pub use services::ragflow_service::{RAGFlowService, RAGFlowError};
pub use services::github_service::RealGitHubPRService;
pub use utils::gpu_compute::GPUCompute;
pub use utils::socket_flow_handler::{SocketFlowServer, ws_handler};

// Re-export handlers
pub use handlers::file_handler;
pub use handlers::graph_handler;
pub use handlers::perplexity_handler;
pub use handlers::ragflow_handler;
pub use handlers::visualization_handler;
