use serde::{Deserialize, Serialize};
use config::{ConfigBuilder, ConfigError, Environment};
use log::debug;
use crate::models::simulation_params::SimulationParams;

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
#[serde(rename_all = "snake_case")]
pub struct Settings {
    // Core visualization settings
    #[serde(default)]
    pub visualization: VisualizationSettings,

    // XR-specific settings
    #[serde(default)]
    pub xr: XRSettings,

    // System settings
    #[serde(default)]
    pub system: SystemSettings,

    // Graph settings
    #[serde(default)]
    pub graph: GraphSettings,

    // Service settings from .env (server-side only)
    #[serde(default)]
    pub github: GitHubSettings,
    #[serde(default)]
    pub ragflow: RagFlowSettings,
    #[serde(default)]
    pub perplexity: PerplexitySettings,
    #[serde(default)]
    pub openai: OpenAISettings,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
#[serde(rename_all = "snake_case")]
pub struct GraphSettings {
    pub simulation_params: SimulationParams,
    pub layout_params: LayoutParams,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
#[serde(rename_all = "snake_case")]
pub struct LayoutParams {
    pub use_force_layout: bool,
    pub force_iterations: u32,
    pub link_distance: f32,
    pub link_strength: f32,
    pub charge_strength: f32,
    pub center_strength: f32,
    pub collision_radius: f32,
}

// Placeholder structs for other settings
// These should be moved to their own modules
#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct VisualizationSettings {
    pub animations: AnimationSettings,
    pub bloom: BloomSettings,
    pub edges: EdgeSettings,
    pub hologram: HologramSettings,
    pub labels: LabelSettings,
    pub nodes: NodeSettings,
    pub physics: PhysicsSettings,
    pub rendering: RenderingSettings,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct AnimationSettings {
    pub enable_node_animations: bool,
    pub enable_motion_blur: bool,
    pub motion_blur_strength: f32,
    pub selection_wave_enabled: bool,
    pub pulse_enabled: bool,
    pub pulse_speed: f32,
    pub pulse_strength: f32,
    pub wave_speed: f32,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct BloomSettings {
    pub enabled: bool,
    pub strength: f32,
    pub radius: f32,
    pub edge_bloom_strength: f32,
    pub node_bloom_strength: f32,
    pub environment_bloom_strength: f32,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct EdgeSettings {
    pub color: String,
    pub opacity: f32,
    pub arrow_size: f32,
    pub base_width: f32,
    pub enable_arrows: bool,
    pub width_range: (f32, f32),
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct HologramSettings {
    pub ring_count: u32,
    pub ring_sizes: Vec<f32>,
    pub ring_rotation_speed: f32,
    pub global_rotation_speed: f32,
    pub ring_color: String,
    pub ring_opacity: f32,
    pub enable_buckminster: bool,
    pub buckminster_scale: f32,
    pub buckminster_opacity: f32,
    pub enable_geodesic: bool,
    pub geodesic_scale: f32,
    pub geodesic_opacity: f32,
    pub enable_triangle_sphere: bool,
    pub triangle_sphere_scale: f32,
    pub triangle_sphere_opacity: f32,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct LabelSettings {
    pub enable_labels: bool,
    pub desktop_font_size: u32,
    pub text_color: String,
    pub text_outline_color: String,
    pub text_outline_width: f32,
    pub text_resolution: u32,
    pub text_padding: u32,
    pub billboard_mode: bool,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct NodeSettings {
    pub quality: String,
    pub enable_instancing: bool,
    pub enable_hologram: bool,
    pub enable_metadata_shape: bool,
    pub enable_metadata_visualization: bool,
    pub base_size: f32,
    pub size_range: (f32, f32),
    pub base_color: String,
    pub opacity: f32,
    pub color_range_age: (String, String),
    pub color_range_links: (String, String),
    pub metalness: f32,
    pub roughness: f32,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct PhysicsSettings {
    pub enabled: bool,
    pub attraction_strength: f32,
    pub repulsion_strength: f32,
    pub spring_strength: f32,
    pub damping: f32,
    pub iterations: u32,
    pub max_velocity: f32,
    pub collision_radius: f32,
    pub enable_bounds: bool,
    pub bounds_size: f32,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct RenderingSettings {
    pub ambient_light_intensity: f32,
    pub directional_light_intensity: f32,
    pub environment_intensity: f32,
    pub background_color: String,
    pub enable_ambient_occlusion: bool,
    pub enable_antialiasing: bool,
    pub enable_shadows: bool,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct XRSettings {}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct SystemSettings {
    pub websocket: WebSocketSettings,
    pub debug: DebugSettings,
    pub paths: PathSettings,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct PathSettings {
    pub data_root: String,
    pub markdown_dir: String,
    pub metadata_dir: String,
    pub data_dir: String,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct WebSocketSettings {
    pub update_rate: u32,
    pub max_message_size: u32,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct DebugSettings {
    pub enabled: bool,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct GitHubSettings {
    pub api_key: String,
    pub api_url: String,
    pub token: String,
    pub owner: String,
    pub repo: String,
    pub base_path: String,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct RagFlowSettings {
    pub api_key: String,
    pub api_base_url: String,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct PerplexitySettings {
    pub api_key: String,
    pub api_url: String,
    pub model: String,
    pub max_tokens: u32,
    pub temperature: f32,
    pub top_p: f32,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
    pub timeout: u64,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct OpenAISettings {
    pub api_key: String,
}

impl Settings {
    pub fn new() -> Result<Self, ConfigError> {
        debug!("Initializing settings with client-side defaults");
        Ok(Settings::default())
    }

    pub fn from_env() -> Result<Self, ConfigError> {
        let builder = ConfigBuilder::<config::builder::DefaultState>::default();
        let config = builder
            .add_source(
                Environment::default()
                    .separator("_")
                    .try_parsing(true)
            )
            .build()?;

        let mut settings = Settings::default();

        // Load path settings from environment
        if let Ok(data_root) = std::env::var("DATA_ROOT") {
            settings.system.paths.data_root = data_root;
        }
        if let Ok(markdown_dir) = std::env::var("MARKDOWN_DIR") {
            settings.system.paths.markdown_dir = markdown_dir;
        }
        if let Ok(metadata_dir) = std::env::var("METADATA_DIR") {
            settings.system.paths.metadata_dir = metadata_dir;
        }
        if let Ok(data_dir) = std::env::var("DATA_DIR") {
            settings.system.paths.data_dir = data_dir;
        }
        
        // Load GitHub settings from environment
        if let Ok(token) = std::env::var("GITHUB_TOKEN") {
            settings.github.token = token;
        }
        if let Ok(owner) = std::env::var("GITHUB_OWNER") {
            settings.github.owner = owner;
        }
        if let Ok(repo) = std::env::var("GITHUB_REPO") {
            settings.github.repo = repo;
        }
        if let Ok(base_path) = std::env::var("GITHUB_BASE_PATH") {
            settings.github.base_path = base_path;
        }

        // Load other settings from config
        if let Ok(other_settings) = config.try_deserialize::<Settings>() {
            settings.visualization = other_settings.visualization;
            settings.xr = other_settings.xr;
            settings.system = other_settings.system;
            settings.graph = other_settings.graph;
            settings.ragflow = other_settings.ragflow;
            settings.perplexity = other_settings.perplexity;
            settings.openai = other_settings.openai;
        }

        Ok(settings)
    }
}
