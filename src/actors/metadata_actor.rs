//! Metadata Actor to replace Arc<RwLock<MetadataStore>>

use actix::prelude::*;
use log::{debug, info};

use crate::actors::messages::*;
use crate::models::metadata::MetadataStore;

pub struct MetadataActor {
    metadata: MetadataStore,
}

impl MetadataActor {
    pub fn new(metadata: MetadataStore) -> Self {
        Self { metadata }
    }

    pub fn get_metadata(&self) -> &MetadataStore {
        &self.metadata
    }

    pub fn update_metadata(&mut self, new_metadata: MetadataStore) {
        self.metadata = new_metadata;
        debug!("Metadata updated with {} files", self.metadata.len()); // Changed .files.len() to .len()
    }

    pub fn refresh_metadata(&mut self) -> Result<(), String> {
        // This would typically reload metadata from disk/database
        // For now, we'll just log that a refresh was requested
        info!("Metadata refresh requested");
        
        // TODO: Implement actual metadata refresh logic
        // This might involve:
        // 1. Re-scanning the file system
        // 2. Re-parsing markdown files
        // 3. Updating the metadata store
        
        Ok(())
    }

    pub fn get_file_count(&self) -> usize {
        self.metadata.len() // Changed .files.len() to .len()
    }

    // TODO: Re-implement or remove get_files_by_tag and get_files_by_type
    // The Metadata struct no longer has a `properties` field.
    // Tag and type information needs to be stored differently or derived.
    /*
    pub fn get_files_by_tag(&self, tag: &str) -> Vec<String> {
        self.metadata // Changed .files.iter() to .iter()
            .iter()
            .filter_map(|(filename, file_meta)| {
                // if let Some(ref properties) = file_meta.properties { // .properties does not exist
                //     if let Some(tags) = properties.get("tags") {
                //         if let Some(tag_array) = tags.as_array() {
                //             if tag_array.iter().any(|t| t.as_str() == Some(tag)) {
                //                 return Some(filename.clone());
                //             }
                //         }
                //     }
                // }
                None // Placeholder
            })
            .collect()
    }

    pub fn get_files_by_type(&self, file_type: &str) -> Vec<String> {
        self.metadata // Changed .files.iter() to .iter()
            .iter()
            .filter_map(|(filename, file_meta)| {
                // if let Some(ref properties) = file_meta.properties { // .properties does not exist
                //     if let Some(type_value) = properties.get("type") {
                //         if type_value.as_str() == Some(file_type) {
                //             return Some(filename.clone());
                //         }
                //     }
                // }
                None // Placeholder
            })
            .collect()
    }
    */
}

impl Actor for MetadataActor {
    type Context = Context<Self>;

    fn started(&mut self, _ctx: &mut Self::Context) {
        info!("MetadataActor started with {} files", self.metadata.len()); // Changed .files.len() to .len()
    }

    fn stopped(&mut self, _ctx: &mut Self::Context) {
        info!("MetadataActor stopped");
    }
}

impl Handler<GetMetadata> for MetadataActor {
    type Result = Result<MetadataStore, String>;

    fn handle(&mut self, _msg: GetMetadata, _ctx: &mut Self::Context) -> Self::Result {
        Ok(self.metadata.clone())
    }
}

impl Handler<UpdateMetadata> for MetadataActor {
    type Result = Result<(), String>;

    fn handle(&mut self, msg: UpdateMetadata, _ctx: &mut Self::Context) -> Self::Result {
        self.update_metadata(msg.metadata);
        Ok(())
    }
}

impl Handler<RefreshMetadata> for MetadataActor {
    type Result = Result<(), String>;

    fn handle(&mut self, _msg: RefreshMetadata, _ctx: &mut Self::Context) -> Self::Result {
        self.refresh_metadata()
    }
}