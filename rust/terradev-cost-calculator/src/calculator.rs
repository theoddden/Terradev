use crate::types::{CostBreakdown, InstanceType};
use rust_decimal::Decimal;
use std::collections::HashMap;

pub struct CostCalculator {
    instance_types: HashMap<String, InstanceType>,
}

impl CostCalculator {
    pub fn new() -> Self {
        Self {
            instance_types: HashMap::new(),
        }
    }
    
    pub fn add_instance_type(&mut self, instance_type: InstanceType) {
        self.instance_types.insert(instance_type.name.clone(), instance_type);
    }
    
    pub fn calculate_cost(
        &self,
        instance_type_name: &str,
        hours: Decimal,
        use_spot: bool,
    ) -> Option<CostBreakdown> {
        let instance_type = self.instance_types.get(instance_type_name)?;
        
        let hourly_cost = if use_spot {
            let discount = instance_type.spot_discount_percent / Decimal::from(100);
            instance_type.hourly_cost_usd * (Decimal::from(1) - discount)
        } else {
            instance_type.hourly_cost_usd
        };
        
        let _total_cost = hourly_cost * hours;
        let monthly_hours = Decimal::from(730); // 24 * 30.4
        let monthly_cost = hourly_cost * monthly_hours;
        
        let spot_hourly_cost = {
            let discount = instance_type.spot_discount_percent / Decimal::from(100);
            instance_type.hourly_cost_usd * (Decimal::from(1) - discount)
        };
        
        let spot_monthly_cost = spot_hourly_cost * monthly_hours;
        let spot_savings = (instance_type.hourly_cost_usd - spot_hourly_cost) * monthly_hours;
        
        Some(CostBreakdown {
            hourly_cost_usd: instance_type.hourly_cost_usd,
            monthly_cost_usd: monthly_cost,
            spot_hourly_cost_usd: spot_hourly_cost,
            spot_monthly_cost_usd: spot_monthly_cost,
            spot_savings_usd: spot_savings,
            gpu_count: instance_type.gpu_count,
        })
    }
    
    pub fn calculate_multi_instance_cost(
        &self,
        instances: &[(String, Decimal, bool)],
    ) -> CostBreakdown {
        let mut total_hourly = Decimal::from(0);
        let mut total_monthly = Decimal::from(0);
        let mut total_spot_monthly = Decimal::from(0);
        let mut total_spot_savings = Decimal::from(0);
        let mut total_gpu_count = 0;
        
        for (instance_type_name, hours, use_spot) in instances {
            if let Some(breakdown) = self.calculate_cost(instance_type_name, *hours, *use_spot) {
                total_hourly += breakdown.hourly_cost_usd;
                total_monthly += breakdown.monthly_cost_usd;
                total_spot_monthly += breakdown.spot_monthly_cost_usd;
                total_spot_savings += breakdown.spot_savings_usd;
                total_gpu_count += breakdown.gpu_count;
            }
        }
        
        CostBreakdown {
            hourly_cost_usd: total_hourly,
            monthly_cost_usd: total_monthly,
            spot_hourly_cost_usd: Decimal::from(0),
            spot_monthly_cost_usd: total_spot_monthly,
            spot_savings_usd: total_spot_savings,
            gpu_count: total_gpu_count,
        }
    }
}
