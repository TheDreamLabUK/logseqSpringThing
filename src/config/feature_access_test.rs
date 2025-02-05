use super::FeatureAccess;
use std::env;

fn setup_test_env() {
    // Clear any existing environment variables
    env::remove_var("APPROVED_PUBKEYS");
    env::remove_var("POWER_USER_PUBKEYS");
    env::remove_var("SETTINGS_SYNC_ENABLED_PUBKEYS");
    env::remove_var("PERPLEXITY_ENABLED_PUBKEYS");
    env::remove_var("OPENAI_ENABLED_PUBKEYS");
    env::remove_var("RAGFLOW_ENABLED_PUBKEYS");
}

fn setup_test_pubkeys() {
    env::set_var("APPROVED_PUBKEYS", "pub1,pub2,pub3");
    env::set_var("POWER_USER_PUBKEYS", "pub1");
    env::set_var("SETTINGS_SYNC_ENABLED_PUBKEYS", "pub2");
    env::set_var("PERPLEXITY_ENABLED_PUBKEYS", "pub1,pub2");
    env::set_var("OPENAI_ENABLED_PUBKEYS", "pub1,pub3");
    env::set_var("RAGFLOW_ENABLED_PUBKEYS", "pub1,pub2");
}

#[test]
fn test_environment_loading() {
    setup_test_env();
    setup_test_pubkeys();
    
    let access = FeatureAccess::from_env();
    
    assert_eq!(access.approved_pubkeys.len(), 3);
    assert_eq!(access.power_users.len(), 1);
    assert_eq!(access.settings_sync_enabled.len(), 1);
    assert_eq!(access.perplexity_enabled.len(), 2);
    assert_eq!(access.openai_enabled.len(), 2);
    assert_eq!(access.ragflow_enabled.len(), 2);
}

#[test]
fn test_empty_environment() {
    setup_test_env();
    
    let access = FeatureAccess::from_env();
    
    assert!(access.approved_pubkeys.is_empty());
    assert!(access.power_users.is_empty());
    assert!(access.settings_sync_enabled.is_empty());
    assert!(access.perplexity_enabled.is_empty());
    assert!(access.openai_enabled.is_empty());
    assert!(access.ragflow_enabled.is_empty());
}

#[test]
fn test_power_user_privileges() {
    setup_test_env();
    setup_test_pubkeys();
    
    let access = FeatureAccess::from_env();
    
    // Test power user (pub1)
    assert!(access.is_power_user("pub1"));
    assert!(access.can_sync_settings("pub1")); // Power users can always sync
    
    // Test regular user with sync access (pub2)
    assert!(!access.is_power_user("pub2"));
    assert!(access.can_sync_settings("pub2")); // Explicitly granted
    
    // Test regular user without sync access (pub3)
    assert!(!access.is_power_user("pub3"));
    assert!(!access.can_sync_settings("pub3"));
}

#[test]
fn test_feature_access_combinations() {
    setup_test_env();
    setup_test_pubkeys();
    
    let access = FeatureAccess::from_env();
    
    // Test power user (pub1) - should have access to everything
    assert!(access.has_perplexity_access("pub1"));
    assert!(access.has_openai_access("pub1"));
    assert!(access.has_ragflow_access("pub1"));
    
    // Test user with some features (pub2)
    assert!(access.has_perplexity_access("pub2"));
    assert!(!access.has_openai_access("pub2"));
    assert!(access.has_ragflow_access("pub2"));
    
    // Test user with limited access (pub3)
    assert!(!access.has_perplexity_access("pub3"));
    assert!(access.has_openai_access("pub3"));
    assert!(!access.has_ragflow_access("pub3"));
}

#[test]
fn test_feature_access_helper() {
    setup_test_env();
    setup_test_pubkeys();
    
    let access = FeatureAccess::from_env();
    
    // Test power user (pub1)
    assert!(access.has_feature_access("pub1", "perplexity"));
    assert!(access.has_feature_access("pub1", "openai"));
    assert!(access.has_feature_access("pub1", "ragflow"));
    assert!(access.has_feature_access("pub1", "settings_sync"));
    
    // Test regular users
    assert!(access.has_feature_access("pub2", "perplexity"));
    assert!(!access.has_feature_access("pub2", "openai"));
    assert!(access.has_feature_access("pub3", "openai"));
    assert!(!access.has_feature_access("pub3", "perplexity"));
}

#[test]
fn test_available_features() {
    setup_test_env();
    setup_test_pubkeys();
    
    let access = FeatureAccess::from_env();
    
    // Test power user (pub1)
    let pub1_features = access.get_available_features("pub1");
    assert!(pub1_features.contains(&"power_user".to_string()));
    assert!(pub1_features.contains(&"perplexity".to_string()));
    assert!(pub1_features.contains(&"openai".to_string()));
    assert!(pub1_features.contains(&"ragflow".to_string()));
    assert!(pub1_features.contains(&"settings_sync".to_string()));
    
    // Test user with some features (pub2)
    let pub2_features = access.get_available_features("pub2");
    assert!(!pub2_features.contains(&"power_user".to_string()));
    assert!(pub2_features.contains(&"perplexity".to_string()));
    assert!(!pub2_features.contains(&"openai".to_string()));
    assert!(pub2_features.contains(&"ragflow".to_string()));
    assert!(pub2_features.contains(&"settings_sync".to_string()));
    
    // Test user with limited access (pub3)
    let pub3_features = access.get_available_features("pub3");
    assert!(!pub3_features.contains(&"power_user".to_string()));
    assert!(!pub3_features.contains(&"perplexity".to_string()));
    assert!(pub3_features.contains(&"openai".to_string()));
    assert!(!pub3_features.contains(&"settings_sync".to_string()));
}

#[test]
fn test_invalid_pubkeys() {
    setup_test_env();
    setup_test_pubkeys();
    
    let access = FeatureAccess::from_env();
    let invalid_pubkey = "invalid_pubkey";
    
    assert!(!access.has_access(invalid_pubkey));
    assert!(!access.is_power_user(invalid_pubkey));
    assert!(!access.can_sync_settings(invalid_pubkey));
    assert!(!access.has_perplexity_access(invalid_pubkey));
    assert!(!access.has_openai_access(invalid_pubkey));
    assert!(!access.has_ragflow_access(invalid_pubkey));
    
    let features = access.get_available_features(invalid_pubkey);
    assert!(features.is_empty());
}