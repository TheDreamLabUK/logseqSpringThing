use actix::prelude::*;
use log::info;
use serde_json::Value;

use crate::models::protected_settings::{ProtectedSettings, NostrUser, ApiKeys};

pub struct ProtectedSettingsActor {
    settings: ProtectedSettings,
}

impl ProtectedSettingsActor {
    pub fn new(settings: ProtectedSettings) -> Self {
        info!("ProtectedSettingsActor created");
        Self { settings }
    }
}

impl Actor for ProtectedSettingsActor {
    type Context = Context<Self>;
    
    fn started(&mut self, _ctx: &mut Self::Context) {
        info!("ProtectedSettingsActor started");
    }
}

// Message to get API keys for a user
#[derive(Message)]
#[rtype(result = "ApiKeys")]
pub struct GetApiKeys {
    pub pubkey: String,
}

impl Handler<GetApiKeys> for ProtectedSettingsActor {
    type Result = MessageResult<GetApiKeys>;
    
    fn handle(&mut self, msg: GetApiKeys, _ctx: &mut Self::Context) -> Self::Result {
        MessageResult(self.settings.get_api_keys(&msg.pubkey))
    }
}

// Message to validate client token
#[derive(Message)]
#[rtype(result = "bool")]
pub struct ValidateClientToken {
    pub pubkey: String,
    pub token: String,
}

impl Handler<ValidateClientToken> for ProtectedSettingsActor {
    type Result = bool;
    
    fn handle(&mut self, msg: ValidateClientToken, _ctx: &mut Self::Context) -> Self::Result {
        self.settings.validate_client_token(&msg.pubkey, &msg.token)
    }
}

// Message to store client token
#[derive(Message)]
#[rtype(result = "()")]
pub struct StoreClientToken {
    pub pubkey: String,
    pub token: String,
}

impl Handler<StoreClientToken> for ProtectedSettingsActor {
    type Result = ();
    
    fn handle(&mut self, msg: StoreClientToken, _ctx: &mut Self::Context) -> Self::Result {
        self.settings.store_client_token(msg.pubkey, msg.token);
    }
}

// Message to update user API keys
#[derive(Message)]
#[rtype(result = "Result<NostrUser, String>")]
pub struct UpdateUserApiKeys {
    pub pubkey: String,
    pub api_keys: ApiKeys,
}

impl Handler<UpdateUserApiKeys> for ProtectedSettingsActor {
    type Result = Result<NostrUser, String>;
    
    fn handle(&mut self, msg: UpdateUserApiKeys, _ctx: &mut Self::Context) -> Self::Result {
        self.settings.update_user_api_keys(&msg.pubkey, msg.api_keys)
    }
}

// Message to cleanup expired tokens
#[derive(Message)]
#[rtype(result = "()")]
pub struct CleanupExpiredTokens {
    pub max_age_hours: i64,
}

impl Handler<CleanupExpiredTokens> for ProtectedSettingsActor {
    type Result = ();
    
    fn handle(&mut self, msg: CleanupExpiredTokens, _ctx: &mut Self::Context) -> Self::Result {
        self.settings.cleanup_expired_tokens(msg.max_age_hours);
    }
}

// Message to merge settings
#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct MergeSettings {
    pub settings: Value,
}

impl Handler<MergeSettings> for ProtectedSettingsActor {
    type Result = Result<(), String>;
    
    fn handle(&mut self, msg: MergeSettings, _ctx: &mut Self::Context) -> Self::Result {
        self.settings.merge(msg.settings)
    }
}

// Message to save settings to file
#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct SaveSettings {
    pub path: String,
}

impl Handler<SaveSettings> for ProtectedSettingsActor {
    type Result = Result<(), String>;
    
    fn handle(&mut self, msg: SaveSettings, _ctx: &mut Self::Context) -> Self::Result {
        self.settings.save(&msg.path)
    }
}

// Message to get user by pubkey
#[derive(Message)]
#[rtype(result = "Option<NostrUser>")]
pub struct GetUser {
    pub pubkey: String,
}

impl Handler<GetUser> for ProtectedSettingsActor {
    type Result = Option<NostrUser>;
    
    fn handle(&mut self, msg: GetUser, _ctx: &mut Self::Context) -> Self::Result {
        self.settings.users.get(&msg.pubkey).cloned()
    }
}