use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct ProcessedFile {
    pub content: String,
}

pub struct OpenWebUiService;

impl OpenWebUiService {
    pub async fn process_file(file: String) -> Result<ProcessedFile, reqwest::Error> {
        // Placeholder implementation
        Ok(ProcessedFile { content: format!("Processed: {}", file) })
    }
}
