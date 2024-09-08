use crate::handlers::file_handler::fetch_and_process_files;
use crate::app_state::AppState;
use std::sync::{Arc, RwLock};

#[tokio::test]
async fn test_fetch_and_process_files() {
    let app_state = AppState {
        graph_data: Arc::new(RwLock::new(Default::default())),
        file_cache: Arc::new(RwLock::new(Default::default())),
    };

    let result = fetch_and_process_files(&app_state).await;
    assert!(result.is_ok());
    
    let processed_files = result.unwrap();

    // Assuming we expect two processed files
    assert_eq!(processed_files.len(), 2); 
    // Here you can check actual file contents 
    assert!(processed_files.contains(&"expected_file_name_1".to_string()));
    assert!(processed_files.contains(&"expected_file_name_2".to_string()));
}