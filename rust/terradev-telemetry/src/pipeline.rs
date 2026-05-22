use crate::types::{HistogramSnapshot, Metric};
use crossbeam::channel::{unbounded, Receiver, Sender};
use hdrhistogram::Histogram;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use std::thread;

pub struct TelemetryPipeline {
    metric_tx: Sender<Metric>,
    histograms: Arc<RwLock<HashMap<String, Histogram<u64>>>>,
}

impl TelemetryPipeline {
    pub fn new() -> Self {
        let (metric_tx, metric_rx): (Sender<Metric>, Receiver<Metric>) = unbounded();
        let histograms = Arc::new(RwLock::new(HashMap::new()));

        let histograms_clone = histograms.clone();
        thread::spawn(move || {
            Self::process_metrics(metric_rx, histograms_clone);
        });

        Self {
            metric_tx,
            histograms,
        }
    }

    pub fn record(&self, metric: Metric) -> Result<(), crossbeam::channel::SendError<Metric>> {
        self.metric_tx.send(metric)
    }

    pub fn record_value(
        &self,
        name: String,
        value: f64,
        tags: Vec<(String, String)>,
    ) -> Result<(), crossbeam::channel::SendError<Metric>> {
        let metric = Metric {
            name,
            value,
            timestamp: chrono::Utc::now(),
            tags,
        };
        self.metric_tx.send(metric)
    }

    pub fn get_histogram(&self, name: &str) -> Option<HistogramSnapshot> {
        let histograms = self.histograms.read();
        histograms.get(name).map(Self::snapshot_histogram)
    }

    pub fn list_histograms(&self) -> Vec<String> {
        let histograms = self.histograms.read();
        histograms.keys().cloned().collect()
    }

    fn process_metrics(
        metric_rx: Receiver<Metric>,
        histograms: Arc<RwLock<HashMap<String, Histogram<u64>>>>,
    ) {
        for metric in metric_rx {
            let mut histograms = histograms.write();
            let hist = histograms
                .entry(metric.name.clone())
                .or_insert_with(|| Histogram::new(3).expect("Failed to create histogram"));

            let value = (metric.value as u64).max(1);
            let _ = hist.record(value);
        }
    }

    fn snapshot_histogram(hist: &Histogram<u64>) -> HistogramSnapshot {
        let count = hist.len();
        let sum = if count > 0 {
            hist.iter_recorded()
                .map(|v| v.value_iterated_to() * v.count_at_value())
                .sum::<u64>() as f64
        } else {
            0.0
        };

        HistogramSnapshot {
            min: hist.min() as f64,
            max: hist.max() as f64,
            mean: hist.mean(),
            p50: hist.value_at_quantile(0.5) as f64,
            p95: hist.value_at_quantile(0.95) as f64,
            p99: hist.value_at_quantile(0.99) as f64,
            count,
            sum,
        }
    }
}
