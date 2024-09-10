use mockall::predicate::*;
use mockall::automock;
use async_trait::async_trait;

#[automock]
#[async_trait]
trait OpenWebUiServiceTrait {
    async fn process_file(&self, file_content: String) -> Result<String, Box<dyn std::error::Error>>;
}

#[tokio::test]
async fn test_process_file_success() {
    let mut mock = MockOpenWebUiServiceTrait::new();
    mock.expect_process_file()
        .with(eq("Test file content".to_string()))
        .times(1)
        .returning(|_| Ok("Processed content".to_string()));

    let result = mock.process_file("Test file content".to_string()).await;
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), "Processed content");
}

#[tokio::test]
async fn test_process_file_error() {
    let mut mock = MockOpenWebUiServiceTrait::new();
    mock.expect_process_file()
        .with(eq("Test file content".to_string()))
        .times(1)
        .returning(|_| Err(Box::new(std::io::Error::new(std::io::ErrorKind::Other, "API Error"))));

    let result = mock.process_file("Test file content".to_string()).await;
    assert!(result.is_err());
    assert_eq!(result.unwrap_err().to_string(), "API Error");
}

#[tokio::test]
async fn test_process_file_empty_input() {
    let mut mock = MockOpenWebUiServiceTrait::new();
    mock.expect_process_file()
        .with(eq("".to_string()))
        .times(1)
        .returning(|_| Ok("Empty input processed".to_string()));

    let result = mock.process_file("".to_string()).await;
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), "Empty input processed");
}

#[tokio::test]
async fn test_process_file_large_input() {
    let large_input = "A".repeat(1000000); // 1 MB of data
    let mut mock = MockOpenWebUiServiceTrait::new();
    mock.expect_process_file()
        .with(eq(large_input.clone()))
        .times(1)
        .returning(|_| Ok("Large input processed".to_string()));

    let result = mock.process_file(large_input).await;
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), "Large input processed");
}