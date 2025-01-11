use log::LevelFilter;
use log::{error, info};
use simplelog::{CombinedLogger, Config, TermLogger, TerminalMode, WriteLogger};
use std::fs::File;
use std::io;

#[derive(Debug)]
pub struct LogConfig {
    file_level: LevelFilter,
    console_level: LevelFilter,
    log_path: String,
}

impl Default for LogConfig {
    fn default() -> Self {
        Self {
            file_level: LevelFilter::Debug,
            console_level: LevelFilter::Info,
            log_path: "/tmp/webxr.log".to_string(),
        }
    }
}

impl LogConfig {
    pub fn new(file_level: &str, console_level: &str, log_path: &str) -> Self {
        Self {
            file_level: match file_level {
                "trace" => LevelFilter::Trace,
                "debug" => LevelFilter::Debug,
                "info" => LevelFilter::Info,
                "warn" => LevelFilter::Warn,
                "error" => LevelFilter::Error,
                _ => LevelFilter::Info,
            },
            console_level: match console_level {
                "trace" => LevelFilter::Trace,
                "debug" => LevelFilter::Debug,
                "info" => LevelFilter::Info,
                "warn" => LevelFilter::Warn,
                "error" => LevelFilter::Error,
                _ => LevelFilter::Info,
            },
            log_path: log_path.to_string(),
        }
    }
}

pub fn init_logging_with_config(config: LogConfig) -> io::Result<()> {
    let log_file = File::create("/tmp/webxr.log")?;
    
    CombinedLogger::init(vec![
        TermLogger::new(
            config.console_level,
            Config::default(),
            TerminalMode::Mixed,
            simplelog::ColorChoice::Auto,
        ),
        WriteLogger::new(
            config.file_level,
            Config::default(),
            log_file,
        ),
    ]).map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;

    info!("Logging initialized with level file:{:?} console:{:?}", 
          config.file_level, config.console_level);
    Ok(())
}

pub fn init_logging() -> io::Result<()> {
    init_logging_with_config(LogConfig::default())
} 