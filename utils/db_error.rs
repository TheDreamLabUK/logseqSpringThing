use std::error::Error;
use std::fmt;
use tokio_postgres::Error as PgError;
use deadpool_postgres::PoolError;

#[derive(Debug)]
pub enum DatabaseError {
    Connection(PoolError),
    Query(PgError),
    NoResults,
}

impl fmt::Display for DatabaseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DatabaseError::Connection(e) => write!(f, "Database connection error: {}", e),
            DatabaseError::Query(e) => write!(f, "Database query error: {}", e),
            DatabaseError::NoResults => write!(f, "No results found"),
        }
    }
}

impl Error for DatabaseError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            DatabaseError::Connection(e) => Some(e),
            DatabaseError::Query(e) => Some(e),
            DatabaseError::NoResults => None,
        }
    }
}

impl From<PoolError> for DatabaseError {
    fn from(error: PoolError) -> Self {
        DatabaseError::Connection(error)
    }
}

impl From<PgError> for DatabaseError {
    fn from(error: PgError) -> Self {
        DatabaseError::Query(error)
    }
} 