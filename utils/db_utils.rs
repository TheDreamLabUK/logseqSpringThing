use crate::utils::db_error::DatabaseError;
use deadpool_postgres::Client;
use tokio_postgres::Row;
use std::future::Future;
use std::pin::Pin;

pub async fn execute_query<F, T>(
    client: &Client,
    query: &str,
    params: &[&(dyn tokio_postgres::types::ToSql + Sync)],
    row_mapper: F,
) -> Result<Vec<T>, DatabaseError>
where
    F: Fn(Row) -> T,
{
    let rows = client.query(query, params).await?;
    Ok(rows.into_iter().map(row_mapper).collect())
}

pub async fn execute_single_query<F, T>(
    client: &Client,
    query: &str,
    params: &[&(dyn tokio_postgres::types::ToSql + Sync)],
    row_mapper: F,
) -> Result<T, DatabaseError>
where
    F: Fn(Row) -> T,
{
    let row = client.query_one(query, params).await?;
    Ok(row_mapper(row))
} 