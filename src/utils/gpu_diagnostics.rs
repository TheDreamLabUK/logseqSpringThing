use crate::utils::gpu_compute::GPUCompute;
use log::{info, warn, error};
use std::env;
use std::path::Path;
use std::io::{Error, ErrorKind};

pub fn run_gpu_diagnostics() -> String {
    let mut report = String::new();
    report.push_str("==== GPU DIAGNOSTIC REPORT ====\n");
    
    // Check environment variables
    report.push_str("Environment Variables:\n");
    for var in &["NVIDIA_GPU_UUID", "NVIDIA_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"] {
        match env::var(var) {
            Ok(val) => {
                report.push_str(&format!("  {} = {}\n", var, val));
                info!("GPU Diagnostic: {} = {}", var, val);
            },
            Err(_) => {
                report.push_str(&format!("  {} = <not set>\n", var));
                warn!("GPU Diagnostic: {} not set", var);
            }
        }
    }
    
    // Check for PTX file
    let ptx_paths = ["/app/src/utils/compute_forces.ptx", "./src/utils/compute_forces.ptx"];
    report.push_str("\nPTX File Status:\n");
    let mut ptx_found = false;
    
    for path in &ptx_paths {
        if Path::new(path).exists() {
            ptx_found = true;
            report.push_str(&format!("  ✅ PTX file found at: {}\n", path));
            info!("GPU Diagnostic: PTX file found at {}", path);
            // Try to get file size
            match std::fs::metadata(path) {
                Ok(metadata) => {
                    report.push_str(&format!("     Size: {} bytes\n", metadata.len()));
                    info!("GPU Diagnostic: PTX file size = {} bytes", metadata.len());
                },
                Err(e) => {
                    report.push_str(&format!("     Error getting file info: {}\n", e));
                    warn!("GPU Diagnostic: Error getting PTX file info: {}", e);
                }
            }
        } else {
            report.push_str(&format!("  ❌ PTX file NOT found at: {}\n", path));
            warn!("GPU Diagnostic: PTX file NOT found at {}", path);
        }
    }
    
    if !ptx_found {
        error!("GPU Diagnostic: No PTX file found at any expected location");
        // This is a critical error for GPU computation
        report.push_str("  ⚠️ CRITICAL ERROR: No PTX file found. GPU physics will not work.\n");
    }
    
    // Check GPU device creation
    report.push_str("\nCUDA Device Detection:\n");
    match GPUCompute::test_gpu() {
        Ok(_) => {
            report.push_str("  ✅ CUDA device successfully detected and tested\n");
            info!("GPU Diagnostic: CUDA device detected and tested successfully");
        },
        Err(e) => {
            report.push_str(&format!("  ❌ CUDA device test failed: {}\n", e));
            error!("GPU Diagnostic: CUDA device test failed: {}", e);
            
            // This is likely why GPU physics isn't working
            report.push_str("  ⚠️ GPU PHYSICS WILL NOT WORK: Could not create CUDA device\n");
        }
    }
    
    report.push_str("=============================\n");
    info!("GPU diagnostic report complete");
    report
}

pub fn fix_cuda_environment() -> Result<(), Error> {
    info!("Attempting to fix CUDA environment...");
    
    // Check and set CUDA_VISIBLE_DEVICES if not set
    if env::var("CUDA_VISIBLE_DEVICES").is_err() {
        info!("CUDA_VISIBLE_DEVICES not set, setting to 0");
        env::set_var("CUDA_VISIBLE_DEVICES", "0");
    }
    
    // Check if PTX file exists; if not, try to find it or create a symlink
    let primary_path = "/app/src/utils/compute_forces.ptx";
    let alternative_path = "./src/utils/compute_forces.ptx";
    
    if !Path::new(primary_path).exists() {
        info!("Primary PTX file not found at {}", primary_path);
        
        if Path::new(alternative_path).exists() {
            info!("Alternative PTX file found at {}, attempting to create symlink", alternative_path);
            
            let alt_path_abs = std::fs::canonicalize(alternative_path)
                .map_err(|e| Error::new(ErrorKind::Other, format!("Failed to get canonical path: {}", e)))?;
                
            let dir_path = Path::new(primary_path).parent()
                .ok_or_else(|| Error::new(ErrorKind::Other, "Invalid PTX path"))?;
                
            if !dir_path.exists() {
                std::fs::create_dir_all(dir_path)
                    .map_err(|e| Error::new(ErrorKind::Other, format!("Failed to create PTX directory: {}", e)))?;
            }
            
            #[cfg(unix)]
            std::os::unix::fs::symlink(&alt_path_abs, primary_path)
                .map_err(|e| Error::new(ErrorKind::Other, format!("Failed to create symlink: {}", e)))?;
                
            #[cfg(not(unix))]
            std::fs::copy(&alt_path_abs, primary_path)
                .map_err(|e| Error::new(ErrorKind::Other, format!("Failed to copy PTX file: {}", e)))?;
                
            info!("Successfully created PTX file at {}", primary_path);
        } else {
            return Err(Error::new(ErrorKind::NotFound, "No PTX file found anywhere. GPU physics will not work."));
        }
    }
    
    info!("CUDA environment has been fixed");
    Ok(())
}