use crate::types::{ConnectionConfig, PoolError};
use reqwest::Client;
use reqwest::ClientBuilder;
use std::time::Duration;
use tokio::sync::Semaphore;
use std::sync::Arc;

pub struct ConnectionPool {
    client: Client,
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
            client,
            config,
            semaphore,
        })
    }
    
    pub async fn acquire(&self) -> Result<AcquiredConnection, PoolError> {
        let permit = self.semaphore.acquire().await
            .map_err(|_| PoolError::Exhausted)?;
        
        Ok(AcquiredConnection {
            client: self.client.clone(),
            _permit: permit,
        })
    }
    
    pub async fn get(&self) -> &Client {
        &self.client
    }
    
    pub fn max_connections(&self) -> usize {
        self.config.max_connections
    }
    
    pub fn active_connections(&self) -> usize {
        self.config.max_connections - self.semaphore.available_permits()
    }
}

pub struct AcquiredConnection {
    client: Client,
    _permit: tokio::sync::SemaphorePermit<'static>,
}

impl AcquiredConnection {
    pub fn client(&self) -> &Client {
        &self.client
    }
}
