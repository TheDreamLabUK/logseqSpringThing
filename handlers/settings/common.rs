use bincode::{serialize, deserialize};
use serde::{Serialize, Deserialize};

#[derive(Debug, thiserror::Error)]
pub enum SettingsError {
    #[error("Category not found: {0}")]
    CategoryNotFound(String),
    
    #[error("Setting not found: {0}")]
    SettingNotFound(String),
    
    #[error("Invalid setting value: {0}")]
    InvalidValue(String),
    
    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),
    
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

pub type SettingResult<T> = Result<T, SettingsError>;

pub struct SettingResponse {
    pub value: Option<serde_json::Value>,
    pub error: Option<SettingsError>,
}

pub struct CategorySettingsResponse {
    pub values: Option<serde_json::Map<String, serde_json::Value>>,
    pub error: Option<SettingsError>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Settings {
    pub categories: HashMap<String, CategorySettings>,
}

impl Settings {
    pub fn to_bytes(&self) -> SettingResult<Vec<u8>> {
        serialize(self).map_err(|e| SettingsError::SerializationError(e.into()))
    }

    pub fn from_bytes(bytes: &[u8]) -> SettingResult<Self> {
        deserialize(bytes).map_err(|e| SettingsError::SerializationError(e.into()))
    }

    pub async fn save_to_file(&self, path: &Path) -> SettingResult<()> {
        let bytes = self.to_bytes()?;
        tokio::fs::write(path, bytes).await
            .map_err(SettingsError::IoError)
    }

    pub async fn load_from_file(path: &Path) -> SettingResult<Self> {
        let bytes = tokio::fs::read(path).await
            .map_err(SettingsError::IoError)?;
        Self::from_bytes(&bytes)
    }
} 