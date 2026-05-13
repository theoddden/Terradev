use crate::types::Event;
use crossbeam::channel::{unbounded, Receiver, Sender};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use std::thread;
use uuid::Uuid;

pub struct EventBus {
    event_tx: Sender<Event>,
    subscribers: Arc<RwLock<HashMap<String, Sender<Event>>>>,
}

impl EventBus {
    pub fn new() -> Self {
        let (event_tx, event_rx): (Sender<Event>, Receiver<Event>) = unbounded();
        let subscribers = Arc::new(RwLock::new(HashMap::new()));
        
        let subscribers_clone = subscribers.clone();
        thread::spawn(move || {
            Self::dispatch_events(event_rx, subscribers_clone);
        });
        
        Self {
            event_tx,
            subscribers,
        }
    }
    
    pub fn publish(&self, event: Event) -> Result<(), crossbeam::channel::SendError<Event>> {
        self.event_tx.send(event)
    }
    
    pub fn subscribe(&self) -> String {
        let id = Uuid::new_v4().to_string();
        let (tx, _rx): (Sender<Event>, Receiver<Event>) = unbounded();
        
        let mut subscribers = self.subscribers.write();
        subscribers.insert(id.clone(), tx);
        
        id
    }
    
    pub fn unsubscribe(&self, id: &str) {
        let mut subscribers = self.subscribers.write();
        subscribers.remove(id);
    }
    
    pub fn subscriber_count(&self) -> usize {
        let subscribers = self.subscribers.read();
        subscribers.len()
    }
    
    fn dispatch_events(event_rx: Receiver<Event>, subscribers: Arc<RwLock<HashMap<String, Sender<Event>>>>) {
        for event in event_rx {
            let subscribers = subscribers.read();
            let failed: Vec<String> = subscribers
                .iter()
                .filter(|(_, tx)| tx.send(event.clone()).is_err())
                .map(|(id, _)| id.clone())
                .collect();
            
            drop(subscribers);
            
            if !failed.is_empty() {
                let mut subscribers = subscribers.write();
                for id in failed {
                    subscribers.remove(&id);
                }
            }
        }
    }
}
