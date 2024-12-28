use tokio::sync::RwLock;
use crate::models::graph::Graph;
use crate::models::backup::BackupManager;
use crate::utils::metrics::MetricsCollector;
use crate::utils::recovery::RecoverySystem;
use crate::utils::connection_pool::ConnectionPoolManager;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::io;

pub struct AppState {
    pub graph: RwLock<Graph>,
    pub backup_manager: BackupManager,
    pub metrics: MetricsCollector,
    pub gpu_compute_enabled: AtomicBool,
    pub recovery: RecoverySystem,
    pub db_pool: ConnectionPoolManager,
}

impl AppState {
    pub async fn new(backup_dir: PathBuf) -> io::Result<Self> {
        let backup_manager = BackupManager::new(backup_dir, 5)?;
        let metrics = MetricsCollector::new();
        let recovery = RecoverySystem::new(30);
        let db_url = std::env::var("DATABASE_URL")
            .unwrap_or_else(|_| "postgres://localhost/logseq_xr".to_string());
        let db_pool = ConnectionPoolManager::new(&db_url, 16).await
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
        
        Ok(Self {
            graph: RwLock::new(Graph::new()),
            backup_manager,
            metrics,
            gpu_compute_enabled: AtomicBool::new(false),
            recovery,
            db_pool,
        })
    }

    pub async fn create_backup(&self) -> io::Result<PathBuf> {
        let graph = self.graph.read().await;
        self.backup_manager.create_backup(&graph).await
    }
} 