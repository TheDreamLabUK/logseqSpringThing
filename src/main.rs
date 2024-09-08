use std::env;
use dotenv::dotenv;

#[tokio::main]
async fn main() {
    dotenv().ok();
    env_logger::init();

    println!("WebXR Graph Visualization Server starting...");

    // TODO: Implement server logic here

    println!("Server started successfully!");
}