use actix_web::{dev::ServiceRequest, Error, HttpMessage};
use actix_web::error::ErrorUnauthorized;
use futures::future::{ok, Ready};

pub async fn check_auth(req: ServiceRequest) -> Result<ServiceRequest, Error> {
    if let Some(auth_header) = req.headers().get("Authorization") {
        if let Ok(auth_str) = auth_header.to_str() {
            if auth_str.starts_with("Bearer ") {
                // Validate token here
                return Ok(req);
            }
        }
    }
    Err(ErrorUnauthorized("Invalid authorization"))
}

pub fn cors_config() -> actix_cors::Cors {
    actix_cors::Cors::default()
        .allowed_origin_fn(|origin, _req_head| {
            origin.as_bytes().starts_with(b"http://localhost:") ||
            origin.as_bytes().starts_with(b"https://localhost:")
        })
        .allowed_methods(vec!["GET", "POST", "PUT", "DELETE"])
        .allowed_headers(vec!["Authorization", "Content-Type"])
        .max_age(3600)
} 