use crate::*;
use services::openwebui_service::OpenWebUiService;

#[tokio::test]
async fn test_process_file() {
    let file = "Test file content".to_string();
    let result = OpenWebUiService::process_file(file).await;
    assert!(result.is_ok());
}