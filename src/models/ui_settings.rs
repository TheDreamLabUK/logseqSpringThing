use serde::{Deserialize, Serialize};
use crate::config::{
    DebugSettings, Settings, VisualizationSettings, XRSettings,
};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct UISettings {
    pub visualization: VisualizationSettings,
    pub system: UISystemSettings,
    pub xr: XRSettings,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct UISystemSettings {
    pub websocket: WebSocketClientSettings,
    pub debug: DebugSettings,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct WebSocketClientSettings {
    pub reconnect_attempts: u32,
    pub reconnect_delay: u64,
    pub binary_chunk_size: usize,
    pub compression_enabled: bool,
    pub compression_threshold: usize,
    pub update_rate: u32,
}

impl From<&Settings> for UISettings {
    fn from(settings: &Settings) -> Self {
        Self {
            visualization: settings.visualization.clone(),
            system: UISystemSettings {
                websocket: WebSocketClientSettings {
                    reconnect_attempts: settings.system.websocket.reconnect_attempts,
                    reconnect_delay: settings.system.websocket.reconnect_delay,
                    binary_chunk_size: settings.system.websocket.binary_chunk_size,
                    compression_enabled: settings.system.websocket.compression_enabled,
                    compression_threshold: settings.system.websocket.compression_threshold,
                    update_rate: settings.system.websocket.update_rate,
                },
                debug: settings.system.debug.clone(),
            },
            xr: settings.xr.clone(),
        }
    }
}

impl UISettings {
    pub fn merge_into_settings(&self, settings: &mut Settings) {
        settings.visualization = self.visualization.clone();
        settings.system.websocket.reconnect_attempts = self.system.websocket.reconnect_attempts;
        settings.system.websocket.reconnect_delay = self.system.websocket.reconnect_delay;
        settings.system.websocket.binary_chunk_size = self.system.websocket.binary_chunk_size;
        settings.system.websocket.compression_enabled = self.system.websocket.compression_enabled;
        settings.system.websocket.compression_threshold = self.system.websocket.compression_threshold;
        settings.system.websocket.update_rate = self.system.websocket.update_rate;
        settings.system.debug = self.system.debug.clone();
        settings.xr = self.xr.clone();
    }
}