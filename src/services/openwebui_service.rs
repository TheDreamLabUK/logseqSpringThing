pub struct OpenWebUiService;

impl OpenWebUiService {
    pub async fn process_file(file: String) -> Result<ProcessedFile, reqwest::Error> {
        // Logic to process file using OpenWebUI
        Ok(ProcessedFile::default())
    }
}
