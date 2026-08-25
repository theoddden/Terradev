# Terraform Backend Configuration
# Per-job state management for parallel operations and easy cleanup

terraform {
  backend "local" {
    # Dynamic state file path based on job ID
    # Each job gets its own state file for parallel operations
    path = "~/.terradev/state/${var.job_id}/terraform.tfstate"
    
    # Enable state locking for concurrent operations
    lock_file = "~/.terradev/state/${var.job_id}/terraform.tfstate.lock"
    
    # State file configuration
    workspace_dir = "~/.terradev/state/${var.job_id}/workspaces"
    
    # Enable state backup for disaster recovery
    backup_file = "~/.terradev/state/${var.job_id}/terraform.tfstate.backup"
    
    # Enable state encryption for security
    encrypt = true
    
    # Enable state compression for storage efficiency
    compress = true
  }
}

# Variables for backend configuration
variable "job_id" {
  description = "Unique identifier for the training job"
  type        = string
  
  validation {
    condition     = can(regex("^[a-zA-Z0-9_-]+$", var.job_id))
    error_message = "Job ID must contain only alphanumeric characters, hyphens, and underscores."
  }
  
  validation {
    condition     = length(var.job_id) >= 3 && length(var.job_id) <= 64
    error_message = "Job ID must be between 3 and 64 characters."
  }
}

variable "state_backend" {
  description = "Backend type for state storage (local, s3, gcs, azure)"
  type        = string
  default     = "local"
  
  validation {
    condition     = contains(["local", "s3", "gcs", "azure"], var.state_backend)
    error_message = "State backend must be one of: local, s3, gcs, azure."
  }
}

# Conditional remote backend configuration
terraform {
  # Override based on state_backend variable
  backend "s3" {
    bucket = "terradev-terraform-state"
    key    = "${var.job_id}/terraform.tfstate"
    region = "us-east-1"
    
    # Enable state locking
    dynamodb_table = "terradev-terraform-locks"
    
    # Enable encryption
    encrypt = true
    
    # Enable versioning
    bucket = "terradev-terraform-state"
    
    # State file configuration
    workspace_key_prefix = "${var.job_id}/workspaces"
    
    # Enable state backup
    backup = true
    
    # Enable state compression
    compress = true
  }
}

# Local variables for state management
locals {
  # State file path
  state_file_path = "${var.state_backend == "local" ? "~/.terradev/state/${var.job_id}/terraform.tfstate" : ""}"
  
  # State lock file path
  state_lock_file_path = "${var.state_backend == "local" ? "~/.terradev/state/${var.job_id}/terraform.tfstate.lock" : ""}"
  
  # State backup file path
  state_backup_file_path = "${var.state_backend == "local" ? "~/.terradev/state/${var.job_id}/terraform.tfstate.backup" : ""}"
  
  # Workspace directory path
  workspace_dir_path = "${var.state_backend == "local" ? "~/.terradev/state/${var.job_id}/workspaces" : ""}"
  
  # State management configuration
  state_config = {
    backend_type = var.state_backend
    job_id      = var.job_id
    state_file  = local.state_file_path
    lock_file   = local.state_lock_file_path
    backup_file = local.state_backup_file_path
    workspace_dir = local.workspace_dir_path
  }
}

# Output state configuration
output "state_configuration" {
  description = "Terraform state configuration"
  value = local.state_config
}

# Output state file location
output "state_file_location" {
  description = "Location of the Terraform state file"
  value = local.state_file_path
}

# Output lock file location
output "lock_file_location" {
  description = "Location of the Terraform lock file"
  value = local.state_lock_file_path
}

# Output workspace directory
output "workspace_directory" {
  description = "Location of the Terraform workspace directory"
  value = local.workspace_dir_path
}

# Resource for state management
resource "local_file" "state_info" {
  filename = "${path.module}/state_info.json"
  content = jsonencode({
    job_id = var.job_id
    backend = var.state_backend
    state_file = local.state_file_path
    lock_file = local.state_lock_file_path
    backup_file = local.state_backup_file_path
    workspace_dir = local.workspace_dir_path
    created_at = timestamp()
  })
}

# Resource for state directory creation
resource "local_file" "state_directory" {
  filename = "${path.module}/.state_directory"
  content = "State directory for job ${var.job_id}"
  
  # Create state directory if it doesn't exist
  provisioner "local-exec" {
    command = "mkdir -p ~/.terradev/state/${var.job_id}"
  }
  
  # Create workspace directory if it doesn't exist
  provisioner "local-exec" {
    command = "mkdir -p ~/.terradev/state/${var.job_id}/workspaces"
  }
}

# Resource for state cleanup
resource "null_resource" "state_cleanup" {
  triggers = {
    # This will trigger cleanup when the job is destroyed
    job_id = var.job_id
  }
  
  # Cleanup state files when job is destroyed
  provisioner "local-exec" {
    when    = destroy
    command = "rm -rf ~/.terradev/state/${var.job_id}"
  }
}

# Resource for state backup
resource "local_file" "state_backup" {
  filename = "${path.module}/state_backup.sh"
  content = <<-EOT
#!/bin/bash
# State backup script for job ${var.job_id}

# Create backup directory
mkdir -p ~/.terradev/state/backups

# Backup state file
if [ -f ~/.terradev/state/${var.job_id}/terraform.tfstate ]; then
  cp ~/.terradev/state/${var.job_id}/terraform.tfstate ~/.terradev/state/backups/${var.job_id}-$(date +%Y%m%d-%H%M%S).tfstate
  echo "State backed up for job ${var.job_id}"
else
  echo "No state file found for job ${var.job_id}"
fi

# Backup lock file
if [ -f ~/.terradev/state/${var.job_id}/terraform.tfstate.lock ]; then
  cp ~/.terradev/state/${var.job_id}/terraform.tfstate.lock ~/.terradev/state/backups/${var.job_id}-$(date +%Y%m%d-%H%M%S).lock
  echo "Lock file backed up for job ${var.job_id}"
fi

# List backups
echo "Available backups for job ${var.job_id}:"
ls -la ~/.terradev/state/backups/${var.job_id}-* 2>/dev/null || echo "No backups found"
EOT
  
  # Make backup script executable
  provisioner "local-exec" {
    command = "chmod +x ${path.module}/state_backup.sh"
  }
}

# Resource for state restoration
resource "local_file" "state_restore" {
  filename = "${path.module}/state_restore.sh"
  content = <<-EOT
#!/bin/bash
# State restoration script for job ${var.job_id}

# List available backups
echo "Available backups for job ${var.job_id}:"
ls -la ~/.terradev/state/backups/${var.job_id}-* 2>/dev/null || echo "No backups found"

# Prompt for backup to restore
if [ $# -eq 0 ]; then
  echo "Usage: $0 <backup_file>"
  echo "Available backups:"
  ls -la ~/.terradev/state/backups/${var.job_id}-* 2>/dev/null
  exit 1
fi

backup_file=$1

# Check if backup file exists
if [ ! -f "$backup_file" ]; then
  echo "Backup file not found: $backup_file"
  exit 1
fi

# Restore state file
if [[ $backup_file == *.tfstate ]]; then
  cp "$backup_file" ~/.terradev/state/${var.job_id}/terraform.tfstate
  echo "State file restored from $backup_file"
fi

# Restore lock file
if [[ $backup_file == *.lock ]]; then
  cp "$backup_file" ~/.terradev/state/${var.job_id}/terraform.tfstate.lock
  echo "Lock file restored from $backup_file"
fi

echo "State restoration completed for job ${var.job_id}"
EOT
  
  # Make restore script executable
  provisioner "local-exec" {
    command = "chmod +x ${path.module}/state_restore.sh"
  }
}

# Resource for state validation
resource "local_file" "state_validate" {
  filename = "${path.module}/state_validate.sh"
  content = <<-EOT
#!/bin/bash
# State validation script for job ${var.job_id}

# Check if state file exists
if [ ! -f ~/.terradev/state/${var.job_id}/terraform.tfstate ]; then
  echo "❌ State file not found for job ${var.job_id}"
  exit 1
fi

# Check if state file is valid
if ! terraform validate -state=~/.terradev/state/${var.job_id}/terraform.tfstate; then
  echo "❌ State file is invalid for job ${var.job_id}"
  exit 1
fi

# Check if lock file exists and is stale
if [ -f ~/.terradev/state/${var.job_id}/terraform.tfstate.lock ]; then
  lock_age=$(find ~/.terradev/state/${var.job_id}/terraform.tfstate.lock -mtime +1 -print 2>/dev/null)
  if [ -n "$lock_age" ]; then
    echo "⚠️  Stale lock file found for job ${var.job_id}"
    echo "Removing stale lock file..."
    rm ~/.terradev/state/${var.job_id}/terraform.tfstate.lock
  fi
fi

# Check state file size
state_size=$(du -h ~/.terradev/state/${var.job_id}/terraform.tfstate | cut -f1)
echo "✅ State file size: $state_size"

# Check state file age
state_age=$(find ~/.terradev/state/${var.job_id}/terraform.tfstate -mtime -1 -print 2>/dev/null)
if [ -n "$state_age" ]; then
  echo "⚠️  State file is older than 1 day"
else
  echo "✅ State file is recent"
fi

echo "✅ State validation completed for job ${var.job_id}"
EOT
  
  # Make validation script executable
  provisioner "local-exec" {
    command = "chmod +x ${path.module}/state_validate.sh"
  }
}

# Resource for state monitoring
resource "local_file" "state_monitor" {
  filename = "${path.module}/state_monitor.sh"
  content = <<-EOT
#!/bin/bash
# State monitoring script for job ${var.job_id}

# Monitor state file changes
if [ -f ~/.terradev/state/${var.job_id}/terraform.tfstate ]; then
  echo "📊 State file monitoring for job ${var.job_id}"
  
  # Get state file size
  state_size=$(du -h ~/.terradev/state/${var.job_id}/terraform.tfstate | cut -f1)
  echo "Current size: $state_size"
  
  # Get state file age
  state_age=$(find ~/.terradev/state/${var.job_id}/terraform.tfstate -mtime -1 -print 2>/dev/null)
  if [ -n "$state_age" ]; then
    echo "State file age: $(find ~/.terradev/state/${var.job_id}/terraform.tfstate -mtime +1 -print | cut -d' ' -f1)"
  else
    echo "State file age: less than 1 day"
  fi
  
  # Check for lock file
  if [ -f ~/.terradev/state/${var.job_id}/terraform.tfstate.lock ]; then
    echo "Lock file: present"
  else
    echo "Lock file: not present"
  fi
  
  # Check for backup files
  backup_count=$(ls -1 ~/.terradev/state/backups/${var.job_id}-* 2>/dev/null | wc -l)
  echo "Backup files: $backup_count"
  
  # Check state file integrity
  if terraform validate -state=~/.terradev/state/${var.job_id}/terraform.tfstate >/dev/null 2>&1; then
    echo "State integrity: valid"
  else
    echo "State integrity: invalid"
  fi
else
  echo "❌ No state file found for job ${var.job_id}"
fi
EOT
  
  # Make monitoring script executable
  provisioner "local-exec" {
    command = "chmod +x ${path.module}/state_monitor.sh"
  }
}

# Resource for state migration
resource "local_file" "state_migrate" {
  filename = "${path.module}/state_migrate.sh"
  content = <<-EOT
#!/bin/bash
# State migration script for job ${var.job_id}

# Check if source state file exists
if [ ! -f ~/.terradev/state/${var.job_id}/terraform.tfstate ]; then
  echo "❌ Source state file not found for job ${var.job_id}"
  exit 1
fi

# Check if destination backend is provided
if [ $# -eq 0 ]; then
  echo "Usage: $0 <destination_backend>"
  echo "Available backends: s3, gcs, azure"
  exit 1
fi

destination_backend=$1

# Validate destination backend
if [[ ! "$destination_backend" =~ ^(s3|gcs|azure)$ ]]; then
  echo "❌ Invalid destination backend: $destination_backend"
  echo "Available backends: s3, gcs, azure"
  exit 1
fi

# Create destination directory
mkdir -p ~/.terradev/state/${var.job_id}/migrate

# Copy state file to destination
cp ~/.terradev/state/${var.job_id}/terraform.tfstate ~/.terradev/state/${var.job_id}/migrate/terraform.tfstate

echo "State file copied for migration to $destination_backend"
echo "Next steps:"
echo "1. Update backend configuration to $destination_backend"
echo "2. Run terraform init -migrate-state"
echo "3. Run terraform plan"
echo "4. Run terraform apply"
EOT
  
  # Make migration script executable
  provisioner "local-exec" {
    command = "chmod +x ${path.module}/state_migrate.sh"
  }
}

# Resource for state cleanup
resource "local_file" "state_cleanup" {
  filename = "${path.module}/state_cleanup.sh"
  content = <<-EOT
#!/bin/bash
# State cleanup script for job ${var.job_id}

echo "🧹 Cleaning up state for job ${var.job_id}"

# Remove state file
if [ -f ~/.terradev/state/${var.job_id}/terraform.tfstate ]; then
  rm ~/.terradev/state/${var.job_id}/terraform.tfstate
  echo "✅ State file removed"
fi

# Remove lock file
if [ -f ~/.terradev/state/${var.job_id}/terraform.tfstate.lock ]; then
  rm ~/.terradev/state/${var.job_id}/terraform.tfstate.lock
  echo "✅ Lock file removed"
fi

# Remove workspace directory
if [ -d ~/.terradev/state/${var.job_id}/workspaces ]; then
  rm -rf ~/.terradev/state/${var.job_id}/workspaces
  echo "✅ Workspace directory removed"
fi

# Remove state directory if empty
if [ -d ~/.terradev/state/${var.job_id} ] && [ -z "$(ls -A ~/.terradev/state/${var.job_id})" ]; then
  rmdir ~/.terradev/state/${var.job_id}
  echo "✅ State directory removed"
fi

# Remove old backups (older than 30 days)
find ~/.terradev/state/backups -name "${var.job_id}-*" -mtime +30 -delete 2>/dev/null
echo "✅ Old backups removed"

echo "🎉 State cleanup completed for job ${var.job_id}"
EOT
  
  # Make cleanup script executable
  provisioner "local-exec" {
    command = "chmod +x ${path.module}/state_cleanup.sh"
  }
}

# Resource for state backup automation
resource "local_file" "state_backup_automation" {
  filename = "${path.module}/state_backup_automation.sh"
  content = <<-EOT
#!/bin/bash
# Automated state backup script for job ${var.job_id}

# Create backup directory
mkdir -p ~/.terradev/state/backups

# Create timestamped backup
timestamp=$(date +%Y%m%d-%H%M%S)
backup_name="${var.job_id}-${timestamp}"

# Backup state file
if [ -f ~/.terradev/state/${var.job_id}/terraform.tfstate ]; then
  cp ~/.terradev/state/${var.job_id}/terraform.tfstate ~/.terradev/state/backups/${backup_name}.tfstate
  echo "✅ State file backed up: ${backup_name}.tfstate"
fi

# Backup lock file
if [ -f ~/.terradev/state/${var.job_id}/terraform.tfstate.lock ]; then
  cp ~/.terradev/state/${var.job_id}/terraform.tfstate.lock ~/.terradev/state/backups/${backup_name}.lock
  echo "✅ Lock file backed up: ${backup_name}.lock"
fi

# Clean up old backups (keep last 10)
ls -t ~/.terradev/state/backups/${var.job_id}-*.tfstate 2>/dev/null | tail -n +11 | xargs rm -f
ls -t ~/.terradev/state/backups/${var.job_id}-*.lock 2>/dev/null | tail -n +11 | xargs rm -f

echo "🎯 Automated backup completed for job ${var.job_id}"
echo "📊 Backup retention: last 10 backups"
EOT
  
  # Make automation script executable
  provisioner "local-exec" {
    command = "chmod +x ${path.module}/state_backup_automation.sh"
  }
}

# Resource for state health check
resource "local_file" "state_health_check" {
  filename = "${path.module}/state_health_check.sh"
  content = <<-EOT
#!/bin/bash
# State health check script for job ${var.job_id}

echo "🏥 State health check for job ${var.job_id}"

# Check state file existence
if [ ! -f ~/.terradev/state/${var.job_id}/terraform.tfstate ]; then
  echo "❌ State file not found"
  exit 1
fi

# Check state file size
state_size=$(du -h ~/.terradev/state/${var.job_id}/terraform.tfstate | cut -f1)
echo "📊 State file size: $state_size"

# Check state file age
state_age=$(find ~/.terradev/state/${var.job_id}/terraform.tfstate -mtime +1 -print 2>/dev/null)
if [ -n "$state_age" ]; then
  echo "⚠️  State file is older than 1 day"
  health_score=$((health_score - 10))
else
  echo "✅ State file is recent"
  health_score=$((health_score + 10))
fi

# Check lock file status
if [ -f ~/.terradev/state/${var.job_id}/terraform.tfstate.lock ]; then
  lock_age=$(find ~/.terradev/state/${var.job_id}/terraform.tfstate.lock -mtime +1 -print 2>/dev/null)
  if [ -n "$lock_age" ]; then
    echo "⚠️  Stale lock file found"
    health_score=$((health_score - 20))
  else
    echo "🔒 Active lock file found"
    health_score=$((health_score - 5))
  fi
else
  echo "✅ No lock file"
  health_score=$((health_score + 5))
fi

# Check state file integrity
if terraform validate -state=~/.terradev/state/${var.job_id}/terraform.tfstate >/dev/null 2>&1; then
  echo "✅ State file is valid"
  health_score=$((health_score + 15))
else
  echo "❌ State file is invalid"
  health_score=$((health_score - 30))
fi

# Check backup availability
backup_count=$(ls -1 ~/.terradev/state/backups/${var.job_id}-* 2>/dev/null | wc -l)
if [ $backup_count -gt 0 ]; then
  echo "✅ $backup_count backup(s) available"
  health_score=$((health_score + 10))
else
  echo "⚠️  No backups available"
  health_score=$((health_score - 10))
fi

# Calculate overall health score
health_score=$((health_score + 50))  # Base score of 50
if [ $health_score -ge 80 ]; then
  echo "🟢 Overall health: Excellent ($health_score/100)"
elif [ $health_score -ge 60 ]; then
  echo "🟡 Overall health: Good ($health_score/100)"
elif [ $health_score -ge 40 ]; then
  echo "🟠 Overall health: Fair ($health_score/100)"
else
  echo "🔴 Overall health: Poor ($health_score/100)"
fi

echo "🏥 Health check completed for job ${var.job_id}"
EOT
  
  # Make health check script executable
  provisioner "local-exec" {
    command = "chmod +x ${path.module}/state_health_check.sh"
  }
}

# Output state management scripts
output "state_management_scripts" {
  description = "State management scripts for the job"
  value = {
    backup = "${path.module}/state_backup.sh"
    restore = "${path.module}/state_restore.sh"
    validate = "${path.module}/state_validate.sh"
    monitor = "${path.module}/state_monitor.sh"
    migrate = "${path.module}/state_migrate.sh"
    cleanup = "${path.module}/state_cleanup.sh"
    backup_automation = "${path.module}/state_backup_automation.sh"
    health_check = "${path.module}/state_health_check.sh"
  }
}

# Output state management commands
output "state_management_commands" {
  description = "State management commands for the job"
  value = {
    backup = "bash ${path.module}/state_backup.sh"
    restore = "bash ${path.module}/state_restore.sh <backup_file>"
    validate = "bash ${path.module}/state_validate.sh"
    monitor = "bash ${path.module}/state_monitor.sh"
    migrate = "bash ${path.module}/state_migrate.sh <destination_backend>"
    cleanup = "bash ${path.module}/state_cleanup.sh"
    backup_automation = "bash ${path.module}/state_backup_automation.sh"
    health_check = "bash ${path.module}/state_health_check.sh"
  }
}
