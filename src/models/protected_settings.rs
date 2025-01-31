use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProtectedSettings {
    pub network: NetworkSettings,
    pub security: SecuritySettings,
    pub websocket_server: WebSocketServerSettings,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkSettings {
    pub bind_address: String,
    pub domain: String,
    pub port: u16,
    pub enable_http2: bool,
    pub enable_tls: bool,
    pub min_tls_version: String,
    pub max_request_size: usize,
    pub enable_rate_limiting: bool,
    pub rate_limit_requests: u32,
    pub rate_limit_window: u32,
    pub tunnel_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecuritySettings {
    pub allowed_origins: Vec<String>,
    pub audit_log_path: String,
    pub cookie_httponly: bool,
    pub cookie_samesite: String,
    pub cookie_secure: bool,
    pub csrf_token_timeout: u32,
    pub enable_audit_logging: bool,
    pub enable_request_validation: bool,
    pub session_timeout: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebSocketServerSettings {
    pub max_connections: usize,
    pub max_message_size: usize,
    pub url: String,
}

impl Default for ProtectedSettings {
    fn default() -> Self {
        Self {
            network: NetworkSettings {
                bind_address: "127.0.0.1".to_string(),
                domain: "localhost".to_string(),
                port: 3000,
                enable_http2: true,
                enable_tls: false,
                min_tls_version: "TLS1.2".to_string(),
                max_request_size: 10 * 1024 * 1024, // 10MB
                enable_rate_limiting: true,
                rate_limit_requests: 100,
                rate_limit_window: 60,
                tunnel_id: String::new(),
            },
            security: SecuritySettings {
                allowed_origins: vec!["http://localhost:3000".to_string()],
                audit_log_path: "./audit.log".to_string(),
                cookie_httponly: true,
                cookie_samesite: "Lax".to_string(),
                cookie_secure: false,
                csrf_token_timeout: 3600,
                enable_audit_logging: true,
                enable_request_validation: true,
                session_timeout: 86400,
            },
            websocket_server: WebSocketServerSettings {
                max_connections: 100,
                max_message_size: 32 * 1024 * 1024, // 32MB
                url: String::new(),
            },
        }
    }
}

impl ProtectedSettings {
    pub fn merge(&mut self, other: serde_json::Value) -> Result<(), String> {
        if let Some(network) = other.get("network") {
            if let Ok(network_settings) = serde_json::from_value(network.clone()) {
                self.network = network_settings;
            }
        }

        if let Some(security) = other.get("security") {
            if let Ok(security_settings) = serde_json::from_value(security.clone()) {
                self.security = security_settings;
            }
        }

        if let Some(websocket) = other.get("websocketServer") {
            if let Ok(websocket_settings) = serde_json::from_value(websocket.clone()) {
                self.websocket_server = websocket_settings;
            }
        }

        Ok(())
    }

    pub fn load(path: &str) -> Result<Self, String> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| format!("Failed to read protected settings: {}", e))?;
        
        serde_json::from_str(&content)
            .map_err(|e| format!("Failed to parse protected settings: {}", e))
    }

    pub fn save(&self, path: &str) -> Result<(), String> {
        let content = serde_json::to_string_pretty(self)
            .map_err(|e| format!("Failed to serialize protected settings: {}", e))?;
        
        std::fs::write(path, content)
            .map_err(|e| format!("Failed to write protected settings: {}", e))
    }
}