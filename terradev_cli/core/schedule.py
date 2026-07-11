#!/usr/bin/env python3
"""
Terradev Schedule Command - Spot-Aware Scheduling

Cron-aware scheduling specifically optimized for spot pricing windows.
Not just a generic cron caller - aware of spot market dynamics.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
import json
import re

logger = logging.getLogger(__name__)


class SpotPricingWindow:
    """Represents a spot pricing window for cost-optimized scheduling"""
    
    def __init__(self, start_hour: int, end_hour: int, gpu_type: str, max_price: float):
        self.start_hour = start_hour  # UTC hour (0-23)
        self.end_hour = end_hour      # UTC hour (0-23)
        self.gpu_type = gpu_type
        self.max_price = max_price
        self.current_price = max_price
    
    def is_active(self, current_time: datetime) -> bool:
        """Check if this pricing window is currently active"""
        hour = current_time.hour
        if self.start_hour <= self.end_hour:
            return self.start_hour <= hour < self.end_hour
        else:  # Window crosses midnight
            return hour >= self.start_hour or hour < self.end_hour
    
    def time_until_active(self, current_time: datetime) -> timedelta:
        """Calculate time until this window becomes active"""
        hour = current_time.hour
        if self.is_active(current_time):
            return timedelta(0)
        
        if self.start_hour <= self.end_hour:
            if hour < self.start_hour:
                return timedelta(hours=self.start_hour - hour)
            else:
                return timedelta(hours=(24 - hour) + self.start_hour)
        else:  # Window crosses midnight
            if hour < self.start_hour:
                return timedelta(hours=self.start_hour - hour)
            else:
                return timedelta(0)  # Should be active
    
    def __repr__(self) -> str:
        return f"SpotPricingWindow({self.start_hour:02d}:00-{self.end_hour:02d}:00, {self.gpu_type}, ${self.max_price:.4f}/hr)"


class SpotAwareScheduler:
    """Scheduler optimized for spot pricing windows"""
    
    def __init__(self):
        self.pricing_windows: List[SpotPricingWindow] = []
        self.scheduled_jobs: Dict[str, Dict[str, Any]] = {}
        self._load_default_pricing_windows()
    
    def _load_default_pricing_windows(self):
        """Load default spot pricing windows based on historical data"""
        # These are typical low-cost windows for major GPU types
        # In production, these would be loaded from historical pricing data
        self.pricing_windows = [
            # Overnight windows (UTC) - typically lower demand
            SpotPricingWindow(start_hour=0, end_hour=6, gpu_type="A100", max_price=0.80),
            SpotPricingWindow(start_hour=0, end_hour=8, gpu_type="H100", max_price=2.50),
            SpotPricingWindow(start_hour=1, end_hour=7, gpu_type="V100", max_price=0.40),
            SpotPricingWindow(start_hour=2, end_hour=9, gpu_type="RTX4090", max_price=0.30),
            
            # Weekend-like windows (lower demand periods)
            SpotPricingWindow(start_hour=10, end_hour=14, gpu_type="A100", max_price=0.90),
            SpotPricingWindow(start_hour=11, end_hour=15, gpu_type="H100", max_price=2.80),
        ]
    
    def add_pricing_window(self, window: SpotPricingWindow):
        """Add a custom pricing window"""
        self.pricing_windows.append(window)
    
    def get_active_windows(self, current_time: Optional[datetime] = None) -> List[SpotPricingWindow]:
        """Get currently active pricing windows"""
        if current_time is None:
            current_time = datetime.utcnow()
        
        return [w for w in self.pricing_windows if w.is_active(current_time)]
    
    def get_next_window(self, gpu_type: str, current_time: Optional[datetime] = None) -> Optional[SpotPricingWindow]:
        """Get the next available pricing window for a GPU type"""
        if current_time is None:
            current_time = datetime.utcnow()
        
        gpu_windows = [w for w in self.pricing_windows if w.gpu_type == gpu_type]
        if not gpu_windows:
            return None
        
        # Find the next active window
        active_windows = [w for w in gpu_windows if w.is_active(current_time)]
        if active_windows:
            return active_windows[0]  # Return currently active window
        
        # Find the next upcoming window
        upcoming_windows = []
        for window in gpu_windows:
            time_until = window.time_until_active(current_time)
            upcoming_windows.append((time_until, window))
        
        if upcoming_windows:
            upcoming_windows.sort(key=lambda x: x[0])
            return upcoming_windows[0][1]
        
        return None
    
    def schedule_job(
        self,
        job_id: str,
        gpu_type: str,
        command: str,
        max_wait_hours: int = 24,
        prefer_current: bool = True
    ) -> Dict[str, Any]:
        """Schedule a job for the next optimal spot pricing window"""
        current_time = datetime.utcnow()
        
        # Get next available window
        next_window = self.get_next_window(gpu_type, current_time)
        
        if next_window is None:
            return {
                "status": "failed",
                "reason": f"No pricing windows available for GPU type: {gpu_type}",
                "job_id": job_id
            }
        
        # Check if window is currently active
        is_active = next_window.is_active(current_time)
        time_until = next_window.time_until_active(current_time)
        
        # If prefer_current and window is active, run now
        if prefer_current and is_active:
            schedule_time = current_time
            status = "immediate"
        else:
            # Schedule for window start
            if is_active:
                schedule_time = current_time
            else:
                schedule_time = current_time + time_until
            
            # Check if wait time is acceptable
            if time_until.total_seconds() > max_wait_hours * 3600:
                return {
                    "status": "failed",
                    "reason": f"Next window is {time_until.total_seconds()/3600:.1f}h away, exceeds max wait of {max_wait_hours}h",
                    "job_id": job_id,
                    "next_window": str(next_window)
                }
            
            status = "scheduled"
        
        # Store job
        self.scheduled_jobs[job_id] = {
            "job_id": job_id,
            "gpu_type": gpu_type,
            "command": command,
            "schedule_time": schedule_time.isoformat(),
            "pricing_window": str(next_window),
            "status": status,
            "max_price": next_window.max_price
        }
        
        return {
            "status": "success",
            "job_id": job_id,
            "gpu_type": gpu_type,
            "schedule_time": schedule_time.isoformat(),
            "pricing_window": str(next_window),
            "execution_status": status,
            "time_until": time_until.total_seconds() if not is_active else 0,
            "estimated_cost": next_window.max_price
        }
    
    def list_scheduled_jobs(self) -> List[Dict[str, Any]]:
        """List all scheduled jobs"""
        return list(self.scheduled_jobs.values())
    
    def remove_job(self, job_id: str) -> bool:
        """Remove a scheduled job"""
        if job_id in self.scheduled_jobs:
            del self.scheduled_jobs[job_id]
            return True
        return False


class CronExpression:
    """Parse and validate cron expressions"""
    
    def __init__(self, expression: str):
        self.expression = expression
        self.minute = "*"
        self.hour = "*"
        self.day_of_month = "*"
        self.month = "*"
        self.day_of_week = "*"
        self._parse()
    
    def _parse(self):
        """Parse cron expression into components"""
        parts = self.expression.split()
        if len(parts) < 5:
            raise ValueError(f"Invalid cron expression: {self.expression}")
        
        self.minute = parts[0]
        self.hour = parts[1]
        self.day_of_month = parts[2]
        self.month = parts[3]
        self.day_of_week = parts[4] if len(parts) > 4 else "*"
    
    def matches(self, dt: datetime) -> bool:
        """Check if datetime matches cron expression"""
        return (
            self._matches_field(self.minute, dt.minute, 0, 59) and
            self._matches_field(self.hour, dt.hour, 0, 23) and
            self._matches_field(self.day_of_month, dt.day, 1, 31) and
            self._matches_field(self.month, dt.month, 1, 12) and
            self._matches_field(self.day_of_week, dt.weekday(), 0, 6)
        )
    
    def _matches_field(self, field: str, value: int, min_val: int, max_val: int) -> bool:
        """Check if a field matches the cron specification"""
        if field == "*":
            return True
        
        # Handle lists (e.g., "1,2,3")
        if "," in field:
            return any(self._matches_field(part.strip(), value, min_val, max_val) for part in field.split(","))
        
        # Handle ranges (e.g., "1-5")
        if "-" in field:
            start, end = field.split("-")
            return min_val <= value <= max_val and int(start) <= value <= int(end)
        
        # Handle step values (e.g., "*/5")
        if "/" in field:
            base, step = field.split("/")
            if base == "*":
                return value % int(step) == 0
            return self._matches_field(base, value, min_val, max_val) and value % int(step) == 0
        
        # Simple value match
        try:
            return int(field) == value
        except ValueError:
            return False
    
    def next_run(self, current_time: Optional[datetime] = None) -> datetime:
        """Calculate next run time from cron expression"""
        if current_time is None:
            current_time = datetime.utcnow()
        
        # Simple implementation - check next 24 hours
        # In production, use a proper cron library like croniter
        for i in range(1, 1440):  # Check next 24 hours (minute by minute)
            next_time = current_time + timedelta(minutes=i)
            if self.matches(next_time):
                return next_time
        
        return current_time + timedelta(days=1)  # Fallback


async def schedule_spot_job(
    command: str,
    gpu_type: str,
    cron_expression: Optional[str] = None,
    max_wait_hours: int = 24,
    job_name: Optional[str] = None,
    prefer_current_window: bool = True
) -> Dict[str, Any]:
    """Schedule a job with spot pricing awareness
    
    Args:
        command: Terradev command to execute
        gpu_type: GPU type to use
        cron_expression: Optional cron expression for recurring jobs
        max_wait_hours: Maximum hours to wait for optimal pricing
        job_name: Optional job name
        prefer_current_window: Prefer currently active pricing window
    
    Returns:
        Dictionary with scheduling results
    """
    scheduler = SpotAwareScheduler()
    
    job_id = job_name or f"spot_job_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    
    print(f"📅 Scheduling Spot-Aware Job")
    print(f"   Job ID: {job_id}")
    print(f"   GPU Type: {gpu_type}")
    print(f"   Command: {command}")
    
    # Get current pricing windows
    current_time = datetime.utcnow()
    active_windows = scheduler.get_active_windows(current_time)
    
    print(f"   Current Time (UTC): {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Active Pricing Windows: {len(active_windows)}")
    
    for window in active_windows:
        print(f"     - {window}")
    
    # Schedule the job
    result = scheduler.schedule_job(
        job_id=job_id,
        gpu_type=gpu_type,
        command=command,
        max_wait_hours=max_wait_hours,
        prefer_current=prefer_current_window
    )
    
    if result["status"] == "success":
        print(f"\n✓ Job Scheduled Successfully")
        print(f"   Status: {result['execution_status']}")
        print(f"   Schedule Time: {result['schedule_time']}")
        print(f"   Pricing Window: {result['pricing_window']}")
        print(f"   Estimated Max Price: ${result['estimated_cost']:.4f}/hr")
        
        if result['time_until'] > 0:
            hours = result['time_until'] / 3600
            print(f"   Time Until Execution: {hours:.1f}h")
        
        if cron_expression:
            print(f"   Recurring: {cron_expression}")
            cron = CronExpression(cron_expression)
            next_run = cron.next_run()
            print(f"   Next Cron Run: {next_run.strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        print(f"\n✗ Job Scheduling Failed")
        print(f"   Reason: {result['reason']}")
    
    return result


async def schedule_list() -> Dict[str, Any]:
    """List all scheduled jobs"""
    scheduler = SpotAwareScheduler()
    jobs = scheduler.list_scheduled_jobs()
    
    print(f"📋 Scheduled Jobs: {len(jobs)}")
    
    for job in jobs:
        print(f"\n   Job ID: {job['job_id']}")
        print(f"   GPU Type: {job['gpu_type']}")
        print(f"   Schedule Time: {job['schedule_time']}")
        print(f"   Status: {job['status']}")
        print(f"   Pricing Window: {job['pricing_window']}")
    
    return {"jobs": jobs, "count": len(jobs)}


async def schedule_pricing_windows(gpu_type: Optional[str] = None) -> Dict[str, Any]:
    """Show available spot pricing windows"""
    scheduler = SpotAwareScheduler()
    current_time = datetime.utcnow()
    
    print(f"💰 Spot Pricing Windows")
    print(f"   Current Time (UTC): {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    if gpu_type:
        windows = [w for w in scheduler.pricing_windows if w.gpu_type == gpu_type]
        print(f"   GPU Type: {gpu_type}")
    else:
        windows = scheduler.pricing_windows
    
    print(f"   Available Windows: {len(windows)}\n")
    
    active_windows = []
    upcoming_windows = []
    
    for window in windows:
        is_active = window.is_active(current_time)
        time_until = window.time_until_active(current_time)
        
        window_info = {
            "window": str(window),
            "is_active": is_active,
            "time_until_hours": time_until.total_seconds() / 3600 if not is_active else 0
        }
        
        if is_active:
            active_windows.append(window_info)
        else:
            upcoming_windows.append(window_info)
    
    if active_windows:
        print("   🟢 Active Windows:")
        for w in active_windows:
            print(f"     - {w['window']}")
    
    if upcoming_windows:
        print("\n   ⏳ Upcoming Windows:")
        upcoming_windows.sort(key=lambda x: x['time_until_hours'])
        for w in upcoming_windows:
            print(f"     - {w['window']} (in {w['time_until_hours']:.1f}h)")
    
    return {
        "active_windows": active_windows,
        "upcoming_windows": upcoming_windows,
        "current_time": current_time.isoformat()
    }
