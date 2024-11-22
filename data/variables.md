| Variable | Initialized In | Updated In |
| --- | --- | --- |
| debug_mode | Settings::new() in config.rs | N/A |
| prompt | Settings::new() in config.rs | N/A |
| domain | Settings::new() in config.rs | N/A |
| port | Settings::new() in config.rs | N/A |
| bind_address | Settings::new() in config.rs | N/A |
| enable_tls | Settings::new() in config.rs | N/A |
| min_tls_version | Settings::new() in config.rs | N/A |
| enable_http2 | Settings::new() in config.rs | N/A |
| max_request_size | Settings::new() in config.rs | N/A |
| enable_rate_limiting | Settings::new() in config.rs | N/A |
| rate_limit_requests | Settings::new() in config.rs | N/A |
| rate_limit_window | Settings::new() in config.rs | N/A |
| enable_cors | Settings::new() in config.rs | N/A |
| allowed_origins | Settings::new() in config.rs | N/A |
| enable_csrf | Settings::new() in config.rs | N/A |
| csrf_token_timeout | Settings::new() in config.rs | N/A |
| session_timeout | Settings::new() in config.rs | N/A |
| cookie_secure | Settings::new() in config.rs | N/A |
| cookie_httponly | Settings::new() in config.rs | N/A |
| cookie_samesite | Settings::new() in config.rs | N/A |
| enable_security_headers | Settings::new() in config.rs | N/A |
| enable_request_validation | Settings::new() in config.rs | N/A |
| enable_audit_logging | Settings::new() in config.rs | N/A |
| audit_log_path | Settings::new() in config.rs | N/A |
| github_access_token | Settings::new() in config.rs | N/A |
| github_owner | Settings::new() in config.rs | N/A |
| github_repo | Settings::new() in config.rs | N/A |
| github_directory | Settings::new() in config.rs | N/A |
| github_api_version | Settings::new() in config.rs | N/A |
| github_rate_limit_enabled | Settings::new() in config.rs | N/A |
| ragflow_api_key | Settings::new() in config.rs | N/A |
| ragflow_api_base_url | Settings::new() in config.rs | N/A |
| ragflow_timeout | Settings::new() in config.rs | N/A |
| ragflow_max_retries | Settings::new() in config.rs | N/A |
| perplexity_api_key | Settings::new() in config.rs | N/A |
| perplexity_model | Settings::new() in config.rs | N/A |
| perplexity_api_url | Settings::new() in config.rs | N/A |
| perplexity_max_tokens | Settings::new() in config.rs | N/A |
| perplexity_temperature | Settings::new() in config.rs | N/A |
| perplexity_top_p | Settings::new() in config.rs | N/A |
| perplexity_presence_penalty | Settings::new() in config.rs | N/A |
| perplexity_frequency_penalty | Settings::new() in config.rs | N/A |
| perplexity_timeout | Settings::new() in config.rs | N/A |
| perplexity_rate_limit | Settings::new() in config.rs | N/A |
| openai_api_key | Settings::new() in config.rs | N/A |
| openai_base_url | Settings::new() in config.rs | N/A |
| openai_timeout | Settings::new() in config.rs | N/A |
| openai_rate_limit | Settings::new() in config.rs | N/A |
| max_concurrent_requests | Settings::new() in config.rs | N/A |
| max_retries | Settings::new() in config.rs | N/A |
| retry_delay | Settings::new() in config.rs | N/A |
| api_client_timeout | Settings::new() in config.rs | N/A |
| max_payload_size | Settings::new() in config.rs | N/A |
| enable_request_logging | Settings::new() in config.rs | N/A |
| enable_metrics | Settings::new() in config.rs | N/A |
| metrics_port | Settings::new() in config.rs | N/A |
| log_level | Settings::new() in config.rs | N/A |
| log_format | Settings::new() in config.rs | N/A |
| node_color | Settings::new() in config.rs | updateFeature in NodeManager |
| edge_color | Settings::new() in config.rs | updateFeature in NodeManager |
| hologram_color | Settings::new() in config.rs | N/A |
| node_size_scaling_factor | Settings::new() in config.rs | updateFeature in NodeManager |
| hologram_scale | Settings::new() in config.rs | N/A |
| hologram_opacity | Settings::new() in config.rs | N/A |
| edge_opacity | Settings::new() in config.rs | updateFeature in NodeManager |
| label_font_size | Settings::new() in config.rs | updateFeature in NodeManager |
| fog_density | Settings::new() in config.rs | N/A |
| force_directed_iterations | Settings::new() in config.rs | updateFeature in GraphDataManager |
| force_directed_spring | Settings::new() in config.rs | updateFeature in GraphDataManager |
| force_directed_repulsion | Settings::new() in config.rs | updateFeature in GraphDataManager |
| force_directed_attraction | Settings::new() in config.rs | updateFeature in GraphDataManager |
| force_directed_damping | Settings::new() in config.rs | updateFeature in GraphDataManager |
| node_bloom_strength | Settings::new() in config.rs | N/A |
| node_bloom_radius | Settings::new() in config.rs | N/A |
| node_bloom_threshold | Settings::new() in config.rs | N/A |
| edge_bloom_strength | Settings::new() in config.rs | N/A |
| edge_bloom_radius | Settings::new() in config.rs | N/A |
| edge_bloom_threshold | Settings::new() in config.rs | N/A |
| environment_bloom_strength | Settings::new() in config.rs | N/A |
| environment_bloom_radius | Settings::new() in config.rs | N/A |
| environment_bloom_threshold | Settings::new() in config.rs | N/A |
| fisheye_enabled | Settings::new() in config.rs | updateFisheyeParams in GPUCompute |
| fisheye_strength | Settings::new() in config.rs | updateFisheyeParams in GPUCompute |
| fisheye_radius | Settings::new() in config.rs | updateFisheyeParams in GPUCompute |
| fisheye_focus_x | Settings::new() in config.rs | updateFisheyeParams in GPUCompute |
| fisheye_focus_y | Settings::new() in config.rs | updateFisheyeParams in GPUCompute |
| fisheye_focus_z | Settings::new() in config.rs | updateFisheyeParams in GPUCompute |
The variables in the .env_template file are used to override the default values in the Settings struct, which is initialized in the Settings::new() function in the config.rs file. The variables in the settings.toml file are also used to initialize the Settings struct.
The variables in the JavaScript files (app.js, ControlPanel.vue, chatManager.vue, core.js, etc.) are initialized and updated within the corresponding components and services.
The variables in the Rust files (main.rs, app_state.rs, config.rs, handlers/*.rs, services/*.rs, utils/*.rs) are initialized and updated within the corresponding modules and functions.