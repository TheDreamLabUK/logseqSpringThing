use serde::{Deserialize, Serialize};
// Import the necessary structs from config
use crate::config::{
    AppFullSettings, // Use the full server settings struct
    ClientWebSocketSettings, // The structure expected by the client for websocket settings
    DebugSettings, 
    VisualisationSettings, 
    XRSettings,
    // Settings as ClientFacingSettings, // No longer needed for the From impl
};

// UISettings remains the structure sent to the client
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct UISettings {
    pub visualisation: VisualisationSettings,
    pub system: UISystemSettings,
    pub xr: XRSettings, // Assuming XRSettings structure is compatible enough for UI
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct UISystemSettings {
    // This must use the client-expected structure
    pub websocket: ClientWebSocketSettings, 
    pub debug: DebugSettings,
    // Note: persist_settings from client SystemSettings is not included here,
    // as it's likely not needed for direct UI rendering based on UISettings.
    // Add it if necessary.
}

// WebSocketClientSettings definition remains the same as it defines the client structure
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

// Updated From implementation to convert from AppFullSettings
impl From<&AppFullSettings> for UISettings {
    fn from(settings: &AppFullSettings) -> Self {
        Self {
            visualisation: settings.visualisation.clone(),
            system: UISystemSettings {
                // Map fields from ServerFullWebSocketSettings to ClientWebSocketSettings
                websocket: ClientWebSocketSettings {
                    reconnect_attempts: settings.system.websocket.reconnect_attempts,
                    reconnect_delay: settings.system.websocket.reconnect_delay,
                    binary_chunk_size: settings.system.websocket.binary_chunk_size,
                    compression_enabled: settings.system.websocket.compression_enabled,
                    compression_threshold: settings.system.websocket.compression_threshold,
                    update_rate: settings.system.websocket.update_rate,
                },
                // Debug settings structure is assumed compatible
                debug: settings.system.debug.clone(), 
            },
            // XR settings structure is assumed compatible enough for UI purposes
            xr: settings.xr.clone(), 
        }
    }
}

// Commenting out merge_into_settings as the merge logic is now centralized
// in settings_handler.rs for better control, especially with AppFullSettings.
// impl UISettings {
//     pub fn merge_into_settings(&self, settings: &mut AppFullSettings) {
//         settings.visualisation = self.visualisation.clone();
//         // Careful mapping needed here, especially for websocket
//         let server_ws = &mut settings.system.websocket;
//         let ui_ws = &self.system.websocket;
//         server_ws.reconnect_attempts = ui_ws.reconnect_attempts;
//         server_ws.reconnect_delay = ui_ws.reconnect_delay;
//         server_ws.binary_chunk_size = ui_ws.binary_chunk_size;
//         server_ws.compression_enabled = ui_ws.compression_enabled;
//         server_ws.compression_threshold = ui_ws.compression_threshold;
//         server_ws.update_rate = ui_ws.update_rate;
//         // Other server_ws fields remain untouched by UISettings merge
        
//         settings.system.debug = self.system.debug.clone();
//         settings.xr = self.xr.clone();
//         // persist_settings? auth? AI settings? - Not part of UISettings merge
//     }
// }