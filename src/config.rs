use config::{ConfigBuilder, ConfigError, Environment, File};
use log::{debug, info};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::PathBuf;

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "snake_case")]
pub struct Settings {
    // Core visualization settings (shared with client)
    #[serde(default)]
    pub visualization: VisualizationSettings,

    // System settings
    #[serde(default)]
    pub system: SystemSettings,

    // XR settings
    #[serde(default)]
    pub xr: XRSettings,

    // Server-only settings
    #[serde(default)]
    pub github: GitHubSettings,
    #[serde(default)]
    pub ragflow: RagFlowSettings,
    #[serde(default)]
    pub perplexity: PerplexitySettings,
    #[serde(default)]
    pub openai: OpenAISettings,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "snake_case")]
pub struct VisualizationSettings {
    #[serde(default)]
    pub nodes: NodeSettings,
    #[serde(default)]
    pub edges: EdgeSettings,
    #[serde(default)]
    pub physics: PhysicsSettings,
    #[serde(default)]
    pub rendering: RenderingSettings,
    #[serde(default)]
    pub animations: AnimationSettings,
    #[serde(default)]
    pub labels: LabelSettings,
    #[serde(default)]
    pub bloom: BloomSettings,
    #[serde(default)]
    pub hologram: HologramSettings,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "snake_case")]
pub struct SystemSettings {
    #[serde(default)]
    pub network: NetworkSettings,
    #[serde(default)]
    pub websocket: WebSocketSettings,
    #[serde(default)]
    pub security: SecuritySettings,
    #[serde(default)]
    pub debug: DebugSettings,
}

// Rest of the settings structs remain unchanged...

impl Settings {
    pub fn new() -> Result<Self, ConfigError> {
        debug!("Initializing settings");

        // Load .env file first
        dotenvy::dotenv().ok();

        let settings_path = std::env::var("SETTINGS_FILE_PATH")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("/app/settings.yaml"));

        debug!("Loading settings from: {:?}", settings_path);

        // Read and parse YAML file
        let yaml_content = std::fs::read_to_string(&settings_path)
            .map_err(|e| ConfigError::NotFound(format!("Failed to read settings file: {}", e)))?;
            
        debug!("Deserializing settings from YAML");
        let mut settings: Settings = serde_yaml::from_str(&yaml_content)
            .map_err(|e| ConfigError::Message(format!("Failed to parse YAML: {}", e)))?;
            
        // Apply environment variables on top of YAML settings
        if let Ok(env_settings) = Settings::from_env() {
            settings.merge_env(env_settings);
        }

        // Load required GitHub settings from environment variables
        settings.github.token = std::env::var("GITHUB_TOKEN")
            .map_err(|_| ConfigError::NotFound("GITHUB_TOKEN".into()))?;
        settings.github.owner = std::env::var("GITHUB_OWNER")
            .map_err(|_| ConfigError::NotFound("GITHUB_OWNER".into()))?;
        settings.github.repo = std::env::var("GITHUB_REPO")
            .map_err(|_| ConfigError::NotFound("GITHUB_REPO".into()))?;
        settings.github.base_path = std::env::var("GITHUB_BASE_PATH")
            .map_err(|_| ConfigError::NotFound("GITHUB_BASE_PATH".into()))?;

        // Validate GitHub settings are not empty
        if settings.github.token.is_empty()
            || settings.github.owner.is_empty()
            || settings.github.repo.is_empty()
            || settings.github.base_path.is_empty()
        {
            return Err(ConfigError::Message(
                "Required GitHub settings cannot be empty".into(),
            ));
        }

        info!(
            "GitHub settings loaded: owner={}, repo={}, base_path={}",
            settings.github.owner, settings.github.repo, settings.github.base_path
        );

        Ok(settings)
    }

    pub fn merge_env(&mut self, env_settings: Settings) {
        // Merge environment settings, only overwriting if they are non-default values
        if !env_settings.github.token.is_empty() {
            self.github.token = env_settings.github.token;
        }
        if !env_settings.github.owner.is_empty() {
            self.github.owner = env_settings.github.owner;
        }
        if !env_settings.github.repo.is_empty() {
            self.github.repo = env_settings.github.repo;
        }
        if !env_settings.github.base_path.is_empty() {
            self.github.base_path = env_settings.github.base_path;
        }
        // Add other environment-specific settings as needed
    }

    pub fn merge(&mut self, value: Value) -> Result<(), String> {
        // Convert incoming JSON value to snake_case
        let snake_case_value = self.to_snake_case_value(value);
        
        // Deserialize the value into a temporary Settings
        let new_settings: Settings = serde_json::from_value(snake_case_value)
            .map_err(|e| format!("Failed to deserialize settings: {}", e))?;
        
        // Update only the fields that were present in the input
        // This preserves existing values for fields that weren't included in the update
        if let Ok(visualization) = serde_json::to_value(&new_settings.visualization) {
            if !visualization.is_null() {
                self.visualization = new_settings.visualization;
            }
        }
        if let Ok(system) = serde_json::to_value(&new_settings.system) {
            if !system.is_null() {
                self.system = new_settings.system;
            }
        }
        if let Ok(xr) = serde_json::to_value(&new_settings.xr) {
            if !xr.is_null() {
                self.xr = new_settings.xr;
            }
        }
        
        Ok(())
    }

    pub fn save(&self) -> Result<(), String> {
        let settings_path = std::env::var("SETTINGS_FILE_PATH")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("/app/settings.yaml"));
            
        // Convert to YAML
        let yaml = serde_yaml::to_string(&self)
            .map_err(|e| format!("Failed to serialize settings to YAML: {}", e))?;
            
        // Write to file
        std::fs::write(&settings_path, yaml)
            .map_err(|e| format!("Failed to write settings file: {}", e))?;
            
        Ok(())
    }

    fn to_snake_case_value(&self, value: Value) -> Value {
        match value {
            Value::Object(map) => {
                let converted: serde_json::Map<String, Value> = map
                    .into_iter()
                    .map(|(k, v)| {
                        let snake_case_key = crate::utils::case_conversion::to_snake_case(&k);
                        (snake_case_key, self.to_snake_case_value(v))
                    })
                    .collect();
                Value::Object(converted)
            }
            Value::Array(arr) => Value::Array(
                arr.into_iter()
                    .map(|v| self.to_snake_case_value(v))
                    .collect(),
            ),
            _ => value,
        }
    }

    pub fn from_env() -> Result<Self, ConfigError> {
        let builder = ConfigBuilder::<config::builder::DefaultState>::default();
        let config = builder
            .add_source(Environment::default().separator("_").try_parsing(true))
            .build()?;

        config.try_deserialize()
    }
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            visualization: VisualizationSettings::default(),
            system: SystemSettings::default(),
            xr: XRSettings::default(),
            github: GitHubSettings::default(),
            ragflow: RagFlowSettings::default(),
            perplexity: PerplexitySettings::default(),
            openai: OpenAISettings::default(),
        }
    }
}

impl Default for VisualizationSettings {
    fn default() -> Self {
        Self {
            nodes: NodeSettings::default(),
            edges: EdgeSettings::default(),
            physics: PhysicsSettings::default(),
            rendering: RenderingSettings::default(),
            animations: AnimationSettings::default(),
            labels: LabelSettings::default(),
            bloom: BloomSettings::default(),
            hologram: HologramSettings::default(),
        }
    }
}

impl Default for SystemSettings {
    fn default() -> Self {
        Self {
            network: NetworkSettings::default(),
            websocket: WebSocketSettings::default(),
            security: SecuritySettings::default(),
            debug: DebugSettings::default(),
        }
    }
}