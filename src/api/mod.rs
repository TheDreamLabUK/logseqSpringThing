//! # WebXR Graph API Documentation
//! 
//! This module provides the API endpoints for the WebXR graph visualization system.
//! 
//! ## WebXR Session Management
//! 
//! ```rust
//! /// Initializes a WebXR session with the specified features
//! #[post("/api/xr/session")]
//! async fn init_xr_session(
//!     features: web::Json<XRFeatureRequest>,
//!     state: web::Data<AppState>,
//! ) -> Result<HttpResponse, Error>;
//! 
//! /// Updates XR controller positions and interactions
//! #[post("/api/xr/input")]
//! async fn update_xr_input(
//!     input: web::Json<XRInputState>,
//!     state: web::Data<AppState>,
//! ) -> Result<HttpResponse, Error>;
//! ```
//! 
//! ## Logseq Integration
//! 
//! ```rust
//! /// Fetches and processes Logseq graph data
//! #[get("/api/logseq/graph")]
//! async fn get_logseq_graph(
//!     state: web::Data<AppState>,
//! ) -> Result<HttpResponse, Error>;
//! 
//! /// Updates Logseq content through RAGFlow
//! #[post("/api/logseq/update")]
//! async fn update_logseq_content(
//!     content: web::Json<LogseqUpdate>,
//!     state: web::Data<AppState>,
//! ) -> Result<HttpResponse, Error>;
//! ```
//! 
//! ## RAGFlow Integration
//! 
//! ```rust
//! /// Processes a question using RAGFlow
//! #[post("/api/ragflow/query")]
//! async fn process_ragflow_query(
//!     query: web::Json<RAGFlowQuery>,
//!     state: web::Data<AppState>,
//! ) -> Result<HttpResponse, Error>;
//! ``` 