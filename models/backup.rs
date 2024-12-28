use std::path::{Path, PathBuf};
use chrono::{DateTime, Utc};
use tokio::fs;
use serde::{Serialize, Deserialize};
use crate::models::graph::{Graph, Node, Edge};

#[derive(Debug, Serialize, Deserialize)]
struct BackupMetadata {
    timestamp: DateTime<Utc>,
    version: u32,
    node_count: usize,
    edge_count: usize,
}

pub struct BackupManager {
    backup_dir: PathBuf,
    max_backups: usize,
}

impl BackupManager {
    pub fn new<P: AsRef<Path>>(backup_dir: P, max_backups: usize) -> io::Result<Self> {
        let backup_dir = backup_dir.as_ref().to_path_buf();
        fs::create_dir_all(&backup_dir)?;
        
        Ok(Self {
            backup_dir,
            max_backups,
        })
    }

    pub async fn create_backup(&self, graph: &Graph) -> io::Result<PathBuf> {
        let timestamp = Utc::now();
        let filename = format!("graph_backup_{}.json", timestamp.timestamp());
        let backup_path = self.backup_dir.join(&filename);

        let metadata = BackupMetadata {
            timestamp,
            version: 1,
            node_count: graph.node_count(),
            edge_count: graph.edge_count(),
        };

        // Save metadata
        let metadata_path = backup_path.with_extension("meta.json");
        fs::write(&metadata_path, serde_json::to_string_pretty(&metadata)?).await?;

        // Save graph data
        graph.save_to_file(&backup_path).await?;

        // Cleanup old backups
        self.cleanup_old_backups().await?;

        Ok(backup_path)
    }

    pub async fn list_backups(&self) -> io::Result<Vec<(PathBuf, BackupMetadata)>> {
        let mut backups = Vec::new();
        let mut entries = fs::read_dir(&self.backup_dir).await?;

        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();
            if path.extension().and_then(|ext| ext.to_str()) == Some("json") {
                if let Ok(metadata) = self.read_backup_metadata(&path).await {
                    backups.push((path, metadata));
                }
            }
        }

        backups.sort_by(|a, b| b.1.timestamp.cmp(&a.1.timestamp));
        Ok(backups)
    }

    pub async fn restore_from_backup<P: AsRef<Path>>(&self, backup_path: P) -> io::Result<Graph> {
        Graph::load_from_file(backup_path).await
    }

    async fn read_backup_metadata<P: AsRef<Path>>(&self, backup_path: P) -> io::Result<BackupMetadata> {
        let metadata_path = backup_path.as_ref().with_extension("meta.json");
        let metadata_str = fs::read_to_string(metadata_path).await?;
        Ok(serde_json::from_str(&metadata_str)?)
    }

    async fn cleanup_old_backups(&self) -> io::Result<()> {
        let backups = self.list_backups().await?;
        if backups.len() <= self.max_backups {
            return Ok(());
        }

        for (path, _) in backups.into_iter().skip(self.max_backups) {
            fs::remove_file(&path).await?;
            fs::remove_file(path.with_extension("meta.json")).await?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_backup_manager() {
        let temp_dir = tempdir().unwrap();
        let manager = BackupManager::new(temp_dir.path(), 2).unwrap();

        // Create a test graph
        let mut graph = Graph::new();
        // Add some test data...

        // Create backup
        let backup_path = manager.create_backup(&graph).await.unwrap();
        assert!(backup_path.exists());

        // List backups
        let backups = manager.list_backups().await.unwrap();
        assert_eq!(backups.len(), 1);

        // Restore from backup
        let restored_graph = manager.restore_from_backup(&backup_path).await.unwrap();
        assert_eq!(restored_graph.node_count(), graph.node_count());
    }
} 