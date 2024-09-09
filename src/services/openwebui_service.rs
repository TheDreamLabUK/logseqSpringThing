use crate::services::file_service::ProcessedFile;

pub struct OpenWebUiService;

impl OpenWebUiService {
    pub async fn process_file(file: String) -> Result<ProcessedFile, reqwest::Error> {
        // Implementation goes here
        Ok(ProcessedFile::default())
    }
}
