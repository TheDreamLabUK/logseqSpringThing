use std::env;
use std::fs;
use std::path::PathBuf;
use log::{info, warn};

/// Represents the access control configuration for various features and user roles
pub struct FeatureAccess {
    // Base access control
    pub approved_pubkeys: Vec<String>,
    
    // Feature-specific access
    pub perplexity_enabled: Vec<String>,
    pub openai_enabled: Vec<String>,
    pub ragflow_enabled: Vec<String>,
    
    // Role-based access control
    pub power_users: Vec<String>,
    pub settings_sync_enabled: Vec<String>,
}

impl FeatureAccess {
    /// Creates a new FeatureAccess instance from environment variables
    pub fn from_env() -> Self {
        Self {
            // Base access
            approved_pubkeys: Self::load_pubkeys_from_env("APPROVED_PUBKEYS"),
            
            // Feature access
            perplexity_enabled: Self::load_pubkeys_from_env("PERPLEXITY_ENABLED_PUBKEYS"),
            openai_enabled: Self::load_pubkeys_from_env("OPENAI_ENABLED_PUBKEYS"),
            ragflow_enabled: Self::load_pubkeys_from_env("RAGFLOW_ENABLED_PUBKEYS"),
            
            // Role-based access
            power_users: Self::load_pubkeys_from_env("POWER_USER_PUBKEYS"),
            settings_sync_enabled: Self::load_pubkeys_from_env("SETTINGS_SYNC_ENABLED_PUBKEYS"),
        }
    }

    /// Helper function to load and parse pubkeys from environment variables
    fn load_pubkeys_from_env(var_name: &str) -> Vec<String> {
        env::var(var_name)
            .unwrap_or_default()
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect()
    }

    /// Registers a new user with basic access and default features
    pub fn register_new_user(&mut self, pubkey: &str) -> bool {
        let pubkey = pubkey.to_string();
        
        // Don't register if already approved
        if self.approved_pubkeys.contains(&pubkey) {
            return false;
        }

        // Add to approved pubkeys
        self.approved_pubkeys.push(pubkey.clone());
        
        // Grant RAGFlow access by default
        self.ragflow_enabled.push(pubkey.clone());
        
        // Grant OpenAI (Kokoros) access by default
        self.openai_enabled.push(pubkey.clone());

        // Update the environment file
        self.save_to_env_file();

        info!("Registered new user: {}", pubkey);
        true
    }

    /// Saves the current access configuration back to the .env file
    fn save_to_env_file(&self) {
        let env_path = PathBuf::from(".env");
        if let Ok(content) = fs::read_to_string(&env_path) {
            let mut lines: Vec<String> = content
                .lines()
                .map(|line| line.to_string())
                .collect();

            // Update the relevant lines
            self.update_env_line(&mut lines, "APPROVED_PUBKEYS", &self.approved_pubkeys);
            self.update_env_line(&mut lines, "RAGFLOW_ENABLED_PUBKEYS", &self.ragflow_enabled);
            self.update_env_line(&mut lines, "OPENAI_ENABLED_PUBKEYS", &self.openai_enabled);

            if let Err(e) = fs::write(&env_path, lines.join("\n")) {
                warn!("Failed to update .env file: {}", e);
            }
        }
    }

    fn update_env_line(&self, lines: &mut Vec<String>, var_name: &str, pubkeys: &[String]) {
        let new_line = format!("{}={}", var_name, pubkeys.join(","));
        if let Some(pos) = lines.iter().position(|line| line.starts_with(var_name)) {
            lines[pos] = new_line;
        } else {
            lines.push(new_line);
        }
    }

    /// Checks if a pubkey has basic access
    pub fn has_access(&self, pubkey: &str) -> bool {
        self.approved_pubkeys.contains(&pubkey.to_string())
    }

    /// Checks if a pubkey has access to Perplexity features
    pub fn has_perplexity_access(&self, pubkey: &str) -> bool {
        self.perplexity_enabled.contains(&pubkey.to_string())
    }

    /// Checks if a pubkey has access to OpenAI features
    pub fn has_openai_access(&self, pubkey: &str) -> bool {
        self.openai_enabled.contains(&pubkey.to_string())
    }

    /// Checks if a pubkey has access to RagFlow features
    pub fn has_ragflow_access(&self, pubkey: &str) -> bool {
        self.ragflow_enabled.contains(&pubkey.to_string())
    }

    /// Checks if a pubkey has power user status
    pub fn is_power_user(&self, pubkey: &str) -> bool {
        self.power_users.contains(&pubkey.to_string())
    }

    /// Checks if a pubkey has settings sync access
    pub fn can_sync_settings(&self, pubkey: &str) -> bool {
        // Power users automatically get settings sync access
        self.is_power_user(pubkey) || self.settings_sync_enabled.contains(&pubkey.to_string())
    }

    /// Checks if a pubkey has access to a specific feature
    pub fn has_feature_access(&self, pubkey: &str, feature: &str) -> bool {
        match feature {
            "perplexity" => self.has_perplexity_access(pubkey),
            "openai" => self.has_openai_access(pubkey),
            "ragflow" => self.has_ragflow_access(pubkey),
            "settings_sync" => self.can_sync_settings(pubkey),
            _ => false,
        }
    }

    /// Gets all features available to a pubkey
    pub fn get_available_features(&self, pubkey: &str) -> Vec<String> {
        let mut features = Vec::new();
        
        if self.has_perplexity_access(pubkey) {
            features.push("perplexity".to_string());
        }
        if self.has_openai_access(pubkey) {
            features.push("openai".to_string());
        }
        if self.has_ragflow_access(pubkey) {
            features.push("ragflow".to_string());
        }
        if self.can_sync_settings(pubkey) {
            features.push("settings_sync".to_string());
        }
        if self.is_power_user(pubkey) {
            features.push("power_user".to_string());
        }
        
        features
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    fn setup_test_env() {
        env::set_var("APPROVED_PUBKEYS", "pub1,pub2");
        env::set_var("POWER_USER_PUBKEYS", "pub1");
        env::set_var("PERPLEXITY_ENABLED_PUBKEYS", "pub1,pub2");
        env::set_var("OPENAI_ENABLED_PUBKEYS", "pub1");
        env::set_var("SETTINGS_SYNC_ENABLED_PUBKEYS", "pub2");
    }

    #[test]
    fn test_basic_access() {
        setup_test_env();
        let access = FeatureAccess::from_env();
        
        assert!(access.has_access("pub1"));
        assert!(access.has_access("pub2"));
        assert!(!access.has_access("pub3"));
    }

    #[test]
    fn test_power_user_status() {
        setup_test_env();
        let access = FeatureAccess::from_env();
        
        assert!(access.is_power_user("pub1"));
        assert!(!access.is_power_user("pub2"));
    }

    #[test]
    fn test_feature_access() {
        setup_test_env();
        let access = FeatureAccess::from_env();
        
        // Test pub1 (power user)
        assert!(access.has_perplexity_access("pub1"));
        assert!(access.has_openai_access("pub1"));
        assert!(access.can_sync_settings("pub1")); // Power users can always sync
        
        // Test pub2 (regular user with some features)
        assert!(access.has_perplexity_access("pub2"));
        assert!(!access.has_openai_access("pub2"));
        assert!(access.can_sync_settings("pub2")); // Explicitly granted
    }

    #[test]
    fn test_available_features() {
        setup_test_env();
        let access = FeatureAccess::from_env();
        
        let pub1_features = access.get_available_features("pub1");
        assert!(pub1_features.contains(&"power_user".to_string()));
        assert!(pub1_features.contains(&"perplexity".to_string()));
        assert!(pub1_features.contains(&"openai".to_string()));
        assert!(pub1_features.contains(&"settings_sync".to_string()));
        
        let pub2_features = access.get_available_features("pub2");
        assert!(!pub2_features.contains(&"power_user".to_string()));
        assert!(pub2_features.contains(&"perplexity".to_string()));
        assert!(pub2_features.contains(&"settings_sync".to_string()));
    }
}