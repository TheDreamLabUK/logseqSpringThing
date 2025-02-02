use std::env;
use std::error::Error;
use std::fmt;

#[derive(Debug)]
pub enum GitHubConfigError {
    MissingEnvVar(String),
    ValidationError(String),
}

impl fmt::Display for GitHubConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingEnvVar(var) => write!(f, "Missing environment variable: {}", var),
            Self::ValidationError(msg) => write!(f, "Configuration validation error: {}", msg),
        }
    }
}

impl Error for GitHubConfigError {}

#[derive(Debug, Clone)]
pub struct GitHubConfig {
    pub token: String,
    pub owner: String,
    pub repo: String,
    pub base_path: String,
    pub rate_limit: bool,
    pub version: String,
}

impl GitHubConfig {
    pub fn from_env() -> Result<Self, GitHubConfigError> {
        let token = env::var("GITHUB_TOKEN")
            .map_err(|_| GitHubConfigError::MissingEnvVar("GITHUB_TOKEN".to_string()))?;
            
        let owner = env::var("GITHUB_OWNER")
            .map_err(|_| GitHubConfigError::MissingEnvVar("GITHUB_OWNER".to_string()))?;
            
        let repo = env::var("GITHUB_REPO")
            .map_err(|_| GitHubConfigError::MissingEnvVar("GITHUB_REPO".to_string()))?;
            
        let base_path = env::var("GITHUB_BASE_PATH")
            .map_err(|_| GitHubConfigError::MissingEnvVar("GITHUB_BASE_PATH".to_string()))?;

        // Optional settings with defaults
        let rate_limit = env::var("GITHUB_RATE_LIMIT")
            .map(|v| v.parse::<bool>().unwrap_or(true))
            .unwrap_or(true);

        let version = env::var("GITHUB_API_VERSION")
            .unwrap_or_else(|_| "v3".to_string());

        let config = Self {
            token,
            owner,
            repo,
            base_path,
            rate_limit,
            version,
        };

        config.validate()?;

        Ok(config)
    }

    fn validate(&self) -> Result<(), GitHubConfigError> {
        if self.token.is_empty() {
            return Err(GitHubConfigError::ValidationError(
                "GitHub token cannot be empty".to_string(),
            ));
        }

        if self.owner.is_empty() {
            return Err(GitHubConfigError::ValidationError(
                "GitHub owner cannot be empty".to_string(),
            ));
        }

        if self.repo.is_empty() {
            return Err(GitHubConfigError::ValidationError(
                "GitHub repository cannot be empty".to_string(),
            ));
        }

        if self.base_path.is_empty() {
            return Err(GitHubConfigError::ValidationError(
                "GitHub base path cannot be empty".to_string(),
            ));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    #[test]
    fn test_missing_required_vars() {
        env::remove_var("GITHUB_TOKEN");
        env::remove_var("GITHUB_OWNER");
        env::remove_var("GITHUB_REPO");
        env::remove_var("GITHUB_BASE_PATH");

        match GitHubConfig::from_env() {
            Err(GitHubConfigError::MissingEnvVar(var)) => {
                assert_eq!(var, "GITHUB_TOKEN");
            }
            _ => panic!("Expected MissingEnvVar error"),
        }
    }

    #[test]
    fn test_empty_values() {
        env::set_var("GITHUB_TOKEN", "");
        env::set_var("GITHUB_OWNER", "owner");
        env::set_var("GITHUB_REPO", "repo");
        env::set_var("GITHUB_BASE_PATH", "path");

        match GitHubConfig::from_env() {
            Err(GitHubConfigError::ValidationError(msg)) => {
                assert!(msg.contains("token cannot be empty"));
            }
            _ => panic!("Expected ValidationError"),
        }
    }

    #[test]
    fn test_valid_config() {
        env::set_var("GITHUB_TOKEN", "token");
        env::set_var("GITHUB_OWNER", "owner");
        env::set_var("GITHUB_REPO", "repo");
        env::set_var("GITHUB_BASE_PATH", "path");

        let config = GitHubConfig::from_env().unwrap();
        assert_eq!(config.token, "token");
        assert_eq!(config.owner, "owner");
        assert_eq!(config.repo, "repo");
        assert_eq!(config.base_path, "path");
        assert!(config.rate_limit); // Default value
        assert_eq!(config.version, "v3"); // Default value
    }

    #[test]
    fn test_optional_settings() {
        env::set_var("GITHUB_TOKEN", "token");
        env::set_var("GITHUB_OWNER", "owner");
        env::set_var("GITHUB_REPO", "repo");
        env::set_var("GITHUB_BASE_PATH", "path");
        env::set_var("GITHUB_RATE_LIMIT", "false");
        env::set_var("GITHUB_API_VERSION", "v4");

        let config = GitHubConfig::from_env().unwrap();
        assert!(!config.rate_limit);
        assert_eq!(config.version, "v4");
    }
}