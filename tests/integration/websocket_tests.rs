use actix_web::{test, web, App};
use actix_web_actors::ws;
use futures::StreamExt;
use crate::tests::utils::setup_test_context;
use crate::websocket::{handler::WebSocketConnection, protocol::BinaryProtocol};

#[actix_web::test]
async fn test_websocket_connection() {
    let ctx = setup_test_context().await;
    
    let app = test::init_service(
        App::new()
            .app_data(web::Data::new(ctx.state.clone()))
            .route("/ws", web::get().to(crate::websocket::routes::ws_route))
    ).await;

    // Connect to WebSocket
    let req = test::TestRequest::get().uri("/ws").to_request();
    let mut resp = test::call_service(&app, req).await;
    assert!(resp.status().is_success());

    // Perform WebSocket handshake
    let (response, mut framed) = actix_web_actors::ws::start(WebSocketConnection::new(ctx.state), &req).unwrap();
    assert!(response.status().is_success());

    // Test binary message handling
    let test_positions = vec![[0.0f32, 0.0, 0.0], [1.0, 1.0, 1.0]];
    let binary_msg = BinaryProtocol::create_position_update(&test_positions);
    
    framed.send(ws::Message::Binary(binary_msg.to_vec())).await.unwrap();
    
    if let Some(Ok(ws::Frame::Binary(response))) = framed.next().await {
        let positions = BinaryProtocol::parse_position_update(&response).unwrap();
        assert_eq!(positions.len(), test_positions.len());
    } else {
        panic!("Expected binary response");
    }
} 