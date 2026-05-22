use crate::types::{ConnectionConfig, PoolError};
use reqwest::Client;
use reqwest::ClientBuilder;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Semaphore;

pub struct ConnectionPool {
    _client: Client,
    config: ConnectionConfig,
    semaphore: Arc<Semaphore>,
}

impl ConnectionPool {
    pub fn new(config: ConnectionConfig) -> Result<Self, PoolError> {
        let timeout = Duration::from_secs(config.timeout_seconds);

        let client = ClientBuilder::new()
            .timeout(timeout)
            .pool_max_idle_per_host(config.max_connections)
            .pool_idle_timeout(Duration::from_secs(90))
            .build()
            .map_err(|e| PoolError::ConnectionFailed(e.to_string()))?;

        let semaphore = Arc::new(Semaphore::new(config.max_connections));

        Ok(Self {
            _client: client,
            config,
            semaphore,
        })
    }

    #[allow(dead_code)]
    pub async fn acquire(&self) -> Result<AcquiredConnection<'_>, PoolError> {
        let permit = self
            .semaphore
            .acquire()
            .await
            .map_err(|_| PoolError::Exhausted)?;

        Ok(AcquiredConnection {
            client: self._client.clone(),
            _permit: permit,
        })
    }

    #[allow(dead_code)]
    pub async fn get(&self) -> &Client {
        &self._client
    }

    pub fn max_connections(&self) -> usize {
        self.config.max_connections
    }

    pub fn active_connections(&self) -> usize {
        self.config.max_connections - self.semaphore.available_permits()
    }
}

#[allow(dead_code)]
pub struct AcquiredConnection<'a> {
    client: Client,
    _permit: tokio::sync::SemaphorePermit<'a>,
}

impl<'a> AcquiredConnection<'a> {
    #[allow(dead_code)]
    pub fn client(&self) -> &Client {
        &self.client
    }
}
