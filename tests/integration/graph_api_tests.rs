use actix_web::{test, web, App};
use serde_json::json;
use crate::tests::utils::{setup_test_context, TestContext};
use crate::handlers::{graph_handler, settings};

#[actix_web::test]
async fn test_graph_crud_operations() {
    let ctx = setup_test_context().await;
    
    let app = test::init_service(
        App::new()
            .app_data(web::Data::new(ctx.state.clone()))
            .service(
                web::scope("/api/graph")
                    .route("/data", web::get().to(graph_handler::get_graph_data))
                    .route("/data/paginated", web::get().to(graph_handler::get_paginated_data))
                    .route("/update", web::post().to(graph_handler::update_graph))
            )
    ).await;

    // Test GET /api/graph/data
    let req = test::TestRequest::get().uri("/api/graph/data").to_request();
    let resp = test::call_service(&app, req).await;
    assert!(resp.status().is_success());

    // Test POST /api/graph/update
    let test_data = json!({
        "nodes": [
            {
                "id": "1",
                "position": [0.0, 0.0, 0.0],
                "velocity": [0.0, 0.0, 0.0],
                "mass": 1.0
            }
        ],
        "edges": []
    });

    let req = test::TestRequest::post()
        .uri("/api/graph/update")
        .set_json(&test_data)
        .to_request();
    let resp = test::call_service(&app, req).await;
    assert!(resp.status().is_success());

    // Test GET /api/graph/data/paginated
    let req = test::TestRequest::get()
        .uri("/api/graph/data/paginated?page=1&page_size=10")
        .to_request();
    let resp = test::call_service(&app, req).await;
    assert!(resp.status().is_success());
} 