use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstanceType {
    pub name: String,
    pub provider: String,
    pub region: String,
    pub hourly_cost_usd: Decimal,
    pub spot_discount_percent: Decimal,
    pub gpu_count: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostBreakdown {
    pub hourly_cost_usd: Decimal,
    pub monthly_cost_usd: Decimal,
    pub spot_hourly_cost_usd: Decimal,
    pub spot_monthly_cost_usd: Decimal,
    pub spot_savings_usd: Decimal,
    pub gpu_count: u32,
}
