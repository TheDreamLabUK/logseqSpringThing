use crate::*;
use utils::gpu_compute::GPUCompute;

#[test]
fn test_initialize_gpu() {
    let gpu_compute = GPUCompute {};
    let result = gpu_compute.initialize_gpu();
    assert!(result.is_ok());
}

#[test]
fn test_compute_forces() {
    let gpu_compute = GPUCompute {};
    let result = gpu_compute.compute_forces();
    assert!(result.is_ok());
}

#[test]
fn test_update_positions() {
    let gpu_compute = GPUCompute {};
    let result = gpu_compute.update_positions();
    assert!(result.is_ok());
}