pub mod app_state;
pub mod config;
pub mod handlers;
pub mod models;
pub mod services;
pub mod utils;

// Re-export commonly used types
pub use app_state::AppState;
pub use models::graph::{GraphData, Edge};
pub use models::node::Node;
pub use models::metadata::Metadata;
pub use services::file_service::{FileService, GitHubService, GithubFile, ProcessedFile};
pub use services::perplexity_service::{
    ApiClient,
    PerplexityRequest,
    PerplexityError,
    call_perplexity_api,
    process_markdown,
    RealApiClient,
    PerplexityService,
    clean_logseq_links,
    process_markdown_block,
    select_context_blocks,
    PerplexityResponse,
    Message,
    Choice,
    Delta,
    Usage,
};

// Re-export config
pub use config::Settings;