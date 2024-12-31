use serde::{Deserialize, Serialize};
use config::{ConfigBuilder, ConfigError, Environment};
use log::debug;
use crate::models::simulation_params::SimulationParams;

#[derive(Debug, Serialize, Deserialize, Clone)]
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

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "snake_case")]
pub struct GraphSettings {
    pub simulation_params: SimulationParams,
    pub layout_params: LayoutParams,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
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

impl Default for GraphSettings {
    fn default() -> Self {
        Self {
            simulation_params: SimulationParams::new(),
            layout_params: LayoutParams {
                use_force_layout: true,
                force_iterations: 300,
                link_distance: 30.0,
                link_strength: 1.0,
                charge_strength: -30.0,
                center_strength: 0.1,
                collision_radius: 5.0,
            },
        }
    }
}

// Placeholder structs for other settings
// These should be moved to their own modules
#[derive(Debug, Serialize, Deserialize, Clone)]
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

#[derive(Debug, Serialize, Deserialize, Clone)]
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

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct BloomSettings {
    pub enabled: bool,
    pub strength: f32,
    pub radius: f32,
    pub edge_bloom_strength: f32,
    pub node_bloom_strength: f32,
    pub environment_bloom_strength: f32,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct EdgeSettings {
    pub color: String,
    pub opacity: f32,
    pub arrow_size: f32,
    pub base_width: f32,
    pub enable_arrows: bool,
    pub width_range: (f32, f32),
}

#[derive(Debug, Serialize, Deserialize, Clone)]
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

#[derive(Debug, Serialize, Deserialize, Clone)]
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

#[derive(Debug, Serialize, Deserialize, Clone)]
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

#[derive(Debug, Serialize, Deserialize, Clone)]
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

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct RenderingSettings {
    pub ambient_light_intensity: f32,
    pub directional_light_intensity: f32,
    pub environment_intensity: f32,
    pub background_color: String,
    pub enable_ambient_occlusion: bool,
    pub enable_antialiasing: bool,
    pub enable_shadows: bool,
}

impl Default for VisualizationSettings {
    fn default() -> Self {
        Self {
            animations: AnimationSettings::default(),
            bloom: BloomSettings::default(),
            edges: EdgeSettings::default(),
            hologram: HologramSettings::default(),
            labels: LabelSettings::default(),
            nodes: NodeSettings::default(),
            physics: PhysicsSettings::default(),
            rendering: RenderingSettings::default(),
        }
    }
}

impl Default for AnimationSettings {
    fn default() -> Self {
        Self {
            enable_node_animations: true,
            enable_motion_blur: false,
            motion_blur_strength: 0.5,
            selection_wave_enabled: false,
            pulse_enabled: false,
            pulse_speed: 1.0,
            pulse_strength: 0.5,
            wave_speed: 1.0,
        }
    }
}

impl Default for BloomSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            strength: 0.5,
            radius: 1.0,
            edge_bloom_strength: 0.5,
            node_bloom_strength: 0.5,
            environment_bloom_strength: 0.5,
        }
    }
}

impl Default for EdgeSettings {
    fn default() -> Self {
        Self {
            color: String::from("#ffffff"),
            opacity: 0.8,
            arrow_size: 3.0,
            base_width: 0.1,
            enable_arrows: true,
            width_range: (1.0, 5.0),
        }
    }
}

impl Default for HologramSettings {
    fn default() -> Self {
        Self {
            ring_count: 3,
            ring_sizes: vec![1.0, 1.5, 2.0],
            ring_rotation_speed: 0.1,
            global_rotation_speed: 0.05,
            ring_color: String::from("#00FFFF"),
            ring_opacity: 0.5,
            enable_buckminster: true,
            buckminster_scale: 1.0,
            buckminster_opacity: 0.3,
            enable_geodesic: true,
            geodesic_scale: 1.2,
            geodesic_opacity: 0.4,
            enable_triangle_sphere: true,
            triangle_sphere_scale: 1.1,
            triangle_sphere_opacity: 0.35,
        }
    }
}

impl Default for LabelSettings {
    fn default() -> Self {
        Self {
            enable_labels: true,
            desktop_font_size: 48,
            text_color: String::from("#ffffff"),
            text_outline_color: String::from("#000000"),
            text_outline_width: 0.1,
            text_resolution: 512,
            text_padding: 16,
            billboard_mode: true,
        }
    }
}

impl Default for NodeSettings {
    fn default() -> Self {
        Self {
            quality: String::from("medium"),
            enable_instancing: true,
            enable_hologram: true,
            enable_metadata_shape: true,
            enable_metadata_visualization: true,
            base_size: 1.0,
            size_range: (0.5, 2.0),
            base_color: String::from("#ffffff"),
            opacity: 0.8,
            color_range_age: (String::from("#ff0000"), String::from("#00ff00")),
            color_range_links: (String::from("#0000ff"), String::from("#ff00ff")),
            metalness: 0.5,
            roughness: 0.2,
        }
    }
}

impl Default for PhysicsSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            attraction_strength: 0.1,
            repulsion_strength: 0.1,
            spring_strength: 0.1,
            damping: 0.5,
            iterations: 1,
            max_velocity: 10.0,
            collision_radius: 1.0,
            enable_bounds: true,
            bounds_size: 100.0,
        }
    }
}

impl Default for RenderingSettings {
    fn default() -> Self {
        Self {
            ambient_light_intensity: 0.5,
            directional_light_intensity: 0.8,
            environment_intensity: 1.0,
            background_color: String::from("#000000"),
            enable_ambient_occlusion: true,
            enable_antialiasing: true,
            enable_shadows: true,
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct XRSettings {}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SystemSettings {
    pub websocket: WebSocketSettings,
    pub debug: DebugSettings,
    pub paths: PathSettings,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct PathSettings {
    pub data_root: String,
    pub markdown_dir: String,
    pub metadata_dir: String,
    pub data_dir: String,
}

impl Default for PathSettings {
    fn default() -> Self {
        Self {
            data_root: String::from("/app"),
            markdown_dir: String::from("/app/data/markdown"),
            metadata_dir: String::from("/app/data/metadata"),
            data_dir: String::from("/app/data"),
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct WebSocketSettings {
    pub update_rate: u32,
    pub max_message_size: u32,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct DebugSettings {
    pub enabled: bool,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct GitHubSettings {
    pub api_key: String,
    pub api_url: String,
    pub token: String,
    pub owner: String,
    pub repo: String,
    pub base_path: String,
}

impl Default for WebSocketSettings {
    fn default() -> Self {
        Self {
            update_rate: 100,
            max_message_size: 65536, // 64KB default max message size
        }
    }
}

impl Default for GitHubSettings {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            api_url: String::from("https://api.github.com"),
            token: String::new(),
            owner: String::new(),
            repo: String::new(),
            base_path: String::new(),
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct RagFlowSettings {
    pub api_key: String,
    pub api_base_url: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
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

#[derive(Debug, Serialize, Deserialize, Clone)]
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

impl Default for Settings {
    fn default() -> Self {
        Self {
            visualization: VisualizationSettings::default(),
            xr: XRSettings::default(),
            system: SystemSettings::default(),
            graph: GraphSettings::default(),
            github: GitHubSettings::default(),
            ragflow: RagFlowSettings::default(),
            perplexity: PerplexitySettings::default(),
            openai: OpenAISettings::default(),
        }
    }
}

impl Default for SystemSettings {
    fn default() -> Self {
        Self {
            websocket: WebSocketSettings {
                update_rate: 100,
                max_message_size: 65536,
            },
            debug: DebugSettings {
                enabled: false,
            },
            paths: PathSettings::default(),
        }
    }
}

impl Default for PerplexitySettings {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            api_url: String::from("https://api.perplexity.ai"),
            model: String::from("mistral-7b-instruct"),
            max_tokens: 1024,
            temperature: 0.7,
            top_p: 0.9,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            timeout: 30,
        }
    }
}

impl Default for RagFlowSettings {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            api_base_url: String::from("http://localhost:8000"),
        }
    }
}

impl Default for OpenAISettings {
    fn default() -> Self {
        Self {
            api_key: String::new(),
        }
    }
}
