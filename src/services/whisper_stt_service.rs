use reqwest::{Client, multipart};
use serde::{Deserialize};
use std::sync::Arc;
use log::{info, error, debug, warn};
use tokio::time::{sleep, Duration};
use url::Url;

use crate::config::AppFullSettings;

const POLLING_INTERVAL_SECONDS: u64 = 2;
const MAX_POLLING_ATTEMPTS: u32 = 30; // Results in a total of POLLING_INTERVAL_SECONDS * MAX_POLLING_ATTEMPTS seconds timeout

#[derive(Deserialize, Debug, Clone, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum TaskStatus {
    Pending,
    InProgress,
    Completed,
    Failed,
    Cancelled,
    Queued,
    Paused,
    Retrying,
}

#[derive(Deserialize, Debug)]
pub struct QueueResponse {
    pub identifier: String,
    pub status: TaskStatus,
    pub message: String,
}

#[derive(Deserialize, Debug)]
pub struct TaskResult {
    // Assuming the transcription text is nested under a "text" or "transcription" field.
    // This needs to be confirmed by inspecting an actual successful response.
    // For now, let's try to deserialize common patterns.
    pub text: Option<String>,
    pub transcription: Option<String>,
    // Add other fields from the 'result' object if known, e.g., segments, language
}

#[derive(Deserialize, Debug)]
pub struct TaskStatusResponse {
    pub identifier: String,
    pub status: TaskStatus,
    pub task_type: Option<String>,
    pub result_type: Option<String>,
    pub result: Option<serde_json::Value>, // Keep as Value to inspect, then try to parse into TaskResult
    pub task_params: Option<serde_json::Value>,
    pub error: Option<String>,
    pub duration: Option<f64>,
    pub progress: Option<f64>,
}


#[derive(Debug)]
pub enum SttError {
    RequestError(reqwest::Error),
    ApiError(String),
    PollingTimeout(String),
    TaskFailed(String),
    NoTranscription,
    ConfigError(String),
    UrlParseError(url::ParseError),
}

impl std::fmt::Display for SttError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SttError::RequestError(e) => write!(f, "STT HTTP request error: {}", e),
            SttError::ApiError(msg) => write!(f, "STT API error: {}", msg),
            SttError::PollingTimeout(id) => write!(f, "STT task polling timed out for identifier: {}", id),
            SttError::TaskFailed(msg) => write!(f, "STT task failed: {}", msg),
            SttError::NoTranscription => write!(f, "STT service did not return a transcription."),
            SttError::ConfigError(msg) => write!(f, "STT configuration error: {}", msg),
            SttError::UrlParseError(e) => write!(f, "STT URL parse error: {}", e),
        }
    }
}

impl std::error::Error for SttError {}

impl From<reqwest::Error> for SttError {
    fn from(err: reqwest::Error) -> Self {
        SttError::RequestError(err)
    }
}
impl From<url::ParseError> for SttError {
    fn from(err: url::ParseError) -> Self {
        SttError::UrlParseError(err)
    }
}


pub struct WhisperSttService {
    http_client: Arc<Client>,
    settings: Arc<tokio::sync::RwLock<AppFullSettings>>,
    base_api_url: String,
}

impl WhisperSttService {
    pub fn new(settings: Arc<tokio::sync::RwLock<AppFullSettings>>, http_client: Arc<Client>) -> Result<Self, SttError> {
        let base_api_url = {
            let s = settings.blocking_read();
            s.whisper.as_ref()
                .and_then(|w_settings| w_settings.api_url.clone())
                .ok_or_else(|| SttError::ConfigError("Whisper API base URL not configured".to_string()))?
        };
        info!("WhisperSttService initialized with base API URL: {}", base_api_url);
        Ok(Self {
            http_client,
            settings,
            base_api_url,
        })
    }

    async fn get_transcription_url(&self) -> Result<Url, SttError> {
        Url::parse(&self.base_api_url)?
            .join("/transcription/")
            .map_err(SttError::UrlParseError)
    }

    async fn get_task_status_url(&self, task_id: &str) -> Result<Url, SttError> {
        Url::parse(&self.base_api_url)?
            .join(&format!("/task/{}/", task_id)) // Ensure trailing slash if API expects it
            .map_err(SttError::UrlParseError)
    }


    pub async fn transcribe(&self, audio_data: Vec<u8>) -> Result<String, SttError> {
        if audio_data.is_empty() {
            return Err(SttError::ApiError("Audio data is empty".to_string()));
        }

        let s = self.settings.read().await;
        let whisper_settings = s.whisper.as_ref();

        let model_size = whisper_settings.and_then(|ws| ws.model.clone()).unwrap_or_else(|| "large-v2".to_string());
        let lang = whisper_settings.and_then(|ws| ws.language.clone());
        // Add other parameters as needed, e.g., word_timestamps
        // let word_timestamps = whisper_settings.and_then(|ws| ws.word_timestamps).unwrap_or(false);


        // 1. Submit audio for transcription
        let transcription_url = self.get_transcription_url().await?;
        debug!("Submitting {} bytes of audio data to Whisper STT: {}", audio_data.len(), transcription_url);

        let audio_part = multipart::Part::bytes(audio_data)
            .file_name("audio.wav") // Assuming WAV, client needs to send compatible format
            .mime_str("audio/wav") // Common MIME type for WAV
            .map_err(|e| SttError::ApiError(format!("Failed to create audio part: {}", e)))?;
        
        let form = multipart::Form::new().part("file", audio_part);

        let mut query_params = vec![("model_size", model_size)];
        if let Some(l) = lang {
            query_params.push(("lang", l));
        }
        // query_params.push(("word_timestamps", word_timestamps.to_string()));
        // ... add other query params from settings ...

        let response = self.http_client.post(transcription_url)
            .query(&query_params)
            .multipart(form)
            .send()
            .await?;

        if response.status().as_u16() != 201 { // API returns 201 Created
            let status = response.status();
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown API error".to_string());
            error!("Whisper API submission error {}: {}", status, error_text);
            return Err(SttError::ApiError(format!("Whisper API request failed with status {}: {}", status, error_text)));
        }

        let queue_response = response.json::<QueueResponse>().await?;
        debug!("Whisper API submission response: {:?}", queue_response);
        let task_id = queue_response.identifier;

        // 2. Poll for task status
        let mut attempts = 0;
        loop {
            if attempts >= MAX_POLLING_ATTEMPTS {
                return Err(SttError::PollingTimeout(task_id.clone()));
            }
            attempts += 1;

            sleep(Duration::from_secs(POLLING_INTERVAL_SECONDS)).await;
            
            let task_status_url = self.get_task_status_url(&task_id).await?;
            debug!("Polling task status for {}: {}", task_id, task_status_url);
            
            let status_response = self.http_client.get(task_status_url).send().await?;

            if !status_response.status().is_success() {
                let status = status_response.status();
                let error_text = status_response.text().await.unwrap_or_else(|_| "Unknown API error during polling".to_string());
                warn!("Whisper API polling error for task {}: {} - {}", task_id, status, error_text);
                // Decide if this is a retryable error or a permanent one for polling
                continue; // Simple retry for now
            }

            let task_status_data = status_response.json::<TaskStatusResponse>().await?;
            debug!("Whisper API task status for {}: {:?}", task_id, task_status_data.status);

            match task_status_data.status {
                TaskStatus::Completed => {
                    if let Some(result_value) = task_status_data.result {
                        // Try to parse result_value into TaskResult
                        match serde_json::from_value::<TaskResult>(result_value.clone()) {
                            Ok(task_result) => {
                                if let Some(text) = task_result.text.or(task_result.transcription) {
                                    info!("Transcription completed for task {}: {}", task_id, text);
                                    return Ok(text);
                                } else {
                                    error!("Task {} completed but no transcription text found in result: {:?}", task_id, result_value);
                                    return Err(SttError::NoTranscription);
                                }
                            }
                            Err(e) => {
                                error!("Failed to parse TaskResult from JSON value for task {}: {:?}. Error: {}", task_id, result_value, e);
                                return Err(SttError::ApiError(format!("Could not parse transcription result for task {}", task_id)));
                            }
                        }
                    } else {
                        error!("Task {} completed but result field is missing.", task_id);
                        return Err(SttError::NoTranscription);
                    }
                }
                TaskStatus::Failed => {
                    let error_msg = task_status_data.error.unwrap_or_else(|| "Unknown task failure".to_string());
                    error!("Whisper task {} failed: {}", task_id, error_msg);
                    return Err(SttError::TaskFailed(error_msg));
                }
                TaskStatus::Pending | TaskStatus::InProgress | TaskStatus::Queued | TaskStatus::Retrying => {
                    // Continue polling
                }
                TaskStatus::Cancelled | TaskStatus::Paused => {
                     error!("Whisper task {} is {} - stopping polling.", task_id, format!("{:?}", task_status_data.status).to_lowercase());
                    return Err(SttError::TaskFailed(format!("Task status is {:?}", task_status_data.status)));
                }
            }
        }
    }
}