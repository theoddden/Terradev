# Terradev CLI Tool
# Essential command-line tool for DevOps and ML teams

resource "local_file" "cli_tool" {
  filename = "${path.module}/cli/terradev"
  content = <<-EOT
  #!/bin/bash
  # Terradev CLI - GPU Arbitrage Command Line Tool
  
  VERSION="1.0.0"
  API_ENDPOINT="${TERRADEV_API_ENDPOINT:-https://api.terradev.io/v1}"
  API_KEY="${TERRADEV_API_KEY}"
  
  # Colors for output
  RED='\033[0;31m'
  GREEN='\033[0;32m'
  YELLOW='\033[1;33m'
  BLUE='\033[0;34m'
  NC='\033[0m' # No Color
  
  # Help function
  show_help() {
      cat << EOF
  Terradev GPU Arbitrage CLI v$VERSION
  
  USAGE:
      terradev [COMMAND] [OPTIONS]
  
  COMMANDS:
      find-gpu      Find optimal GPU configuration
      deploy        Deploy GPU to optimal provider
      status        Check deployment status
      savings       Show savings summary
      list-gpus     List available GPU types
      list-providers List supported providers
      config        Show configuration
      version       Show version
  
  OPTIONS:
      --gpu-type TEXT        GPU type (a100, h100, a10g, rtx4090, rtx3090)
      --hours INTEGER        Number of hours needed
      --confidence FLOAT     Confidence threshold (0.0-1.0)
      --max-risk FLOAT       Maximum risk score (0.0-1.0)
      --provider TEXT        Specific provider to use
      --region TEXT          Preferred region
      --spot-only            Only consider spot instances
      --format TEXT          Output format (json, table, yaml)
      --quiet                Minimal output
      --verbose              Detailed output
  
  EXAMPLES:
      terradev find-gpu --gpu-type a100 --hours 8
      terradev deploy --gpu-type h100 --hours 4 --confidence 0.8
      terradev status --deployment-id deploy-123456
      terradev savings --this-month
      terradev list-gpus --format table
  
  CONFIGURATION:
      Set environment variables:
      export TERRADEV_API_KEY="your-api-key"
      export TERRADEV_API_ENDPOINT="https://api.terradev.io/v1"
  
  For more information, visit: https://terradev.io/docs/cli
  EOF
  }
  
  # API request function
  api_request() {
      local method="$1"
      local endpoint="$2"
      local data="$3"
      
      if [ -z "$API_KEY" ]; then
          echo -e "${RED}Error: TERRADEV_API_KEY not set${NC}"
          exit 1
      fi
      
      local curl_opts=(
          -s
          -X "$method"
          -H "Authorization: Bearer $API_KEY"
          -H "Content-Type: application/json"
      )
      
      if [ -n "$data" ]; then
          curl_opts+=(-d "$data")
      fi
      
      curl "${curl_opts[@]}" "$API_ENDPOINT$endpoint"
  }
  
  # Format output
  format_output() {
      local data="$1"
      local format="$2"
      
      case "$format" in
          "json")
              echo "$data" | jq '.'
              ;;
          "yaml")
              echo "$data" | yq eval -P -
              ;;
          "table")
              echo "$data" | jq -r '
                  if type == "object" then
                      to_entries[] | 
                      "\(.key): \(.value)"
                  else
                      .
                  end
              '
              ;;
          *)
              echo "$data"
              ;;
      esac
  }
  
  # Find optimal GPU
  find_gpu() {
      local gpu_type=""
      local hours=4
      local confidence=0.7
      local max_risk=0.5
      local format="json"
      local quiet=false
      
      while [[ $# -gt 0 ]]; do
          case $1 in
              --gpu-type)
                  gpu_type="$2"
                  shift 2
                  ;;
              --hours)
                  hours="$2"
                  shift 2
                  ;;
              --confidence)
                  confidence="$2"
                  shift 2
                  ;;
              --max-risk)
                  max_risk="$2"
                  shift 2
                  ;;
              --format)
                  format="$2"
                  shift 2
                  ;;
              --quiet)
                  quiet=true
                  shift
                  ;;
              *)
                  echo -e "${RED}Unknown option: $1${NC}"
                  exit 1
                  ;;
          esac
      done
      
      if [ -z "$gpu_type" ]; then
          echo -e "${RED}Error: --gpu-type is required${NC}"
          exit 1
      fi
      
      local data=$(cat << EOF
  {
      "gpu_type": "$gpu_type",
      "hours": $hours,
      "confidence_threshold": $confidence,
      "max_risk_score": $max_risk
  }
  EOF
      )
      
      local response=$(api_request "POST" "/find-optimal-gpu" "$data")
      
      if [ "$quiet" = false ]; then
          echo -e "${BLUE}🔍 Finding optimal GPU for $gpu_type ($hours hours)${NC}"
          
          local should_deploy=$(echo "$response" | jq -r '.should_deploy // false')
          local provider=$(echo "$response" | jq -r '.provider // "unknown"')
          local price=$(echo "$response" | jq -r '.price // 0')
          local savings=$(echo "$response" | jq -r '.savings_percentage // 0')
          local conf=$(echo "$response" | jq -r '.arbitrage_confidence // 0')
          local risk=$(echo "$response" | jq -r '.risk_score // 0')
          
          if [ "$should_deploy" = "true" ]; then
              echo -e "${GREEN}✅ Optimal GPU found!${NC}"
              echo -e "   Provider: ${YELLOW}$provider${NC}"
              echo -e "   GPU Type: ${YELLOW}$gpu_type${NC}"
              echo -e "   Price: ${YELLOW}\$$price/hour${NC}"
              echo -e "   Savings: ${GREEN}$savings%${NC}"
              echo -e "   Confidence: ${GREEN}$conf${NC}"
              echo -e "   Risk Score: ${YELLOW}$risk${NC}"
              echo ""
              echo -e "${BLUE}Deploy with: terradev deploy --gpu-type $gpu_type --hours $hours${NC}"
          else
              echo -e "${YELLOW}⏸️  Conditions not optimal for deployment${NC}"
              echo -e "   Confidence: $conf (threshold: $confidence)"
              echo -e "   Risk Score: $risk (max: $max_risk)"
              echo -e "   Recommendation: $(echo "$response" | jq -r '.optimal_timing // "unknown"')"
          fi
      fi
      
      format_output "$response" "$format"
  }
  
  # Deploy GPU
  deploy_gpu() {
      local gpu_type=""
      local hours=4
      local confidence=0.7
      local max_risk=0.5
      local provider=""
      local format="json"
      local quiet=false
      
      while [[ $# -gt 0 ]]; do
          case $1 in
              --gpu-type)
                  gpu_type="$2"
                  shift 2
                  ;;
              --hours)
                  hours="$2"
                  shift 2
                  ;;
              --confidence)
                  confidence="$2"
                  shift 2
                  ;;
              --max-risk)
                  max_risk="$2"
                  shift 2
                  ;;
              --provider)
                  provider="$2"
                  shift 2
                  ;;
              --format)
                  format="$2"
                  shift 2
                  ;;
              --quiet)
                  quiet=true
                  shift
                  ;;
              *)
                  echo -e "${RED}Unknown option: $1${NC}"
                  exit 1
                  ;;
          esac
      done
      
      if [ -z "$gpu_type" ]; then
          echo -e "${RED}Error: --gpu-type is required${NC}"
          exit 1
      fi
      
      if [ "$quiet" = false ]; then
          echo -e "${BLUE}🚀 Deploying $gpu_type for $hours hours...${NC}"
      fi
      
      local data=$(cat << EOF
  {
      "gpu_type": "$gpu_type",
      "hours": $hours,
      "confidence_threshold": $confidence,
      "max_risk_score": $max_risk
  }
  EOF
      )
      
      local response=$(api_request "POST" "/deploy" "$data")
      
      local deployment_id=$(echo "$response" | jq -r '.deployment_id // "unknown"')
      local provider_used=$(echo "$response" | jq -r '.provider // "unknown"')
      local status=$(echo "$response" | jq -r '.status // "unknown"')
      local estimated_cost=$(echo "$response" | jq -r '.total_estimated_cost // 0')
      
      if [ "$quiet" = false ]; then
          if [ "$deployment_id" != "unknown" ]; then
              echo -e "${GREEN}✅ GPU deployment started!${NC}"
              echo -e "   Deployment ID: ${YELLOW}$deployment_id${NC}"
              echo -e "   Provider: ${YELLOW}$provider_used${NC}"
              echo -e "   Status: ${GREEN}$status${NC}"
              echo -e "   Estimated Cost: ${YELLOW}\$$estimated_cost${NC}"
              echo ""
              echo -e "${BLUE}Check status with: terradev status --deployment-id $deployment_id${NC}"
          else
              echo -e "${RED}❌ Deployment failed${NC}"
              echo "$response" | jq -r '.error // "Unknown error"'
          fi
      fi
      
      format_output "$response" "$format"
  }
  
  # Check deployment status
  check_status() {
      local deployment_id=""
      local format="json"
      
      while [[ $# -gt 0 ]]; do
          case $1 in
              --deployment-id)
                  deployment_id="$2"
                  shift 2
                  ;;
              --format)
                  format="$2"
                  shift 2
                  ;;
              *)
                  echo -e "${RED}Unknown option: $1${NC}"
                  exit 1
                  ;;
          esac
      done
      
      if [ -z "$deployment_id" ]; then
          echo -e "${RED}Error: --deployment-id is required${NC}"
          exit 1
      fi
      
      local response=$(api_request "GET" "/deployments/$deployment_id")
      
      local status=$(echo "$response" | jq -r '.status // "unknown"')
      local provider=$(echo "$response" | jq -r '.provider // "unknown"')
      local gpu_type=$(echo "$response" | jq -r '.gpu_type // "unknown"')
      local start_time=$(echo "$response" | jq -r '.start_time // "unknown"')
      
      echo -e "${BLUE}📊 Deployment Status${NC}"
      echo -e "   ID: ${YELLOW}$deployment_id${NC}"
      echo -e "   Status: ${GREEN}$status${NC}"
      echo -e "   Provider: ${YELLOW}$provider${NC}"
      echo -e "   GPU Type: ${YELLOW}$gpu_type${NC}"
      echo -e "   Start Time: ${YELLOW}$start_time${NC}"
      
      format_output "$response" "$format"
  }
  
  # Show savings
  show_savings() {
      local this_month=false
      local format="json"
      
      while [[ $# -gt 0 ]]; do
          case $1 in
              --this-month)
                  this_month=true
                  shift
                  ;;
              --format)
                  format="$2"
                  shift 2
                  ;;
              *)
                  echo -e "${RED}Unknown option: $1${NC}"
                  exit 1
                  ;;
          esac
      done
      
      local endpoint="/savings"
      if [ "$this_month" = true ]; then
          endpoint="$endpoint?period=month"
      fi
      
      local response=$(api_request "GET" "$endpoint")
      
      local total_saved=$(echo "$response" | jq -r '.total_saved // 0')
      local monthly_savings=$(echo "$response" | jq -r '.monthly_savings // 0')
      local deployments=$(echo "$response" | jq -r '.deployment_count // 0')
      local avg_savings=$(echo "$response" | jq -r '.average_savings_percentage // 0')
      
      echo -e "${GREEN}💰 Terradev Savings Summary${NC}"
      echo -e "   Total Saved: ${YELLOW}\$$total_saved${NC}"
      echo -e "   This Month: ${YELLOW}\$$monthly_savings${NC}"
      echo -e "   Deployments: ${YELLOW}$deployments${NC}"
      echo -e "   Average Savings: ${GREEN}$avg_savings%${NC}"
      
      format_output "$response" "$format"
  }
  
  # List GPU types
  list_gpus() {
      local format="table"
      
      while [[ $# -gt 0 ]]; do
          case $1 in
              --format)
                  format="$2"
                  shift 2
                  ;;
              *)
                  echo -e "${RED}Unknown option: $1${NC}"
                  exit 1
                  ;;
          esac
      done
      
      local response=$(api_request "GET" "/gpu-types")
      
      if [ "$format" = "table" ]; then
          echo -e "${BLUE}🎮 Available GPU Types${NC}"
          echo ""
          printf "%-15s %-10s %-15s %-10s\n" "GPU Type" "Memory" "Use Case" "Availability"
          echo "--------------------------------------------------------"
          
          echo "$response" | jq -r '.[] | "\(.type)\t\(.memory)\t\(.use_case)\t\(.availability)"' | while IFS=$'\t' read -r type memory use_case availability; do
              local color="$GREEN"
              if [ "$availability" = "limited" ]; then
                  color="$YELLOW"
              elif [ "$availability" = "rare" ]; then
                  color="$RED"
              fi
              printf "%-15s %-10s %-15s ${color}%-10s${NC}\n" "$type" "$memory" "$use_case" "$availability"
          done
      else
          format_output "$response" "$format"
      fi
  }
  
  # List providers
  list_providers() {
      local format="table"
      
      while [[ $# -gt 0 ]]; do
          case $1 in
              --format)
                  format="$2"
                  shift 2
                  ;;
              *)
                  echo -e "${RED}Unknown option: $1${NC}"
                  exit 1
                  ;;
          esac
      done
      
      local response=$(api_request "GET" "/providers")
      
      if [ "$format" = "table" ]; then
          echo -e "${BLUE}☁️  Supported Cloud Providers${NC}"
          echo ""
          printf "%-15s %-10s %-15s %-10s\n" "Provider" "Region" "Reliability" "Avg Savings"
          echo "--------------------------------------------------------"
          
          echo "$response" | jq -r '.[] | "\(.name)\t\(.primary_region)\t\(.reliability)\t\(.average_savings)"' | while IFS=$'\t' read -r name region reliability savings; do
              local rel_color="$GREEN"
              if [ "$reliability" = "medium" ]; then
                  rel_color="$YELLOW"
              elif [ "$reliability" = "low" ]; then
                  rel_color="$RED"
              fi
              printf "%-15s %-10s ${rel_color}%-15s${NC} ${GREEN}%-10s${NC}\n" "$name" "$region" "$reliability" "$savings%"
          done
      else
          format_output "$response" "$format"
      fi
  }
  
  # Show configuration
  show_config() {
      echo -e "${BLUE}⚙️  Terradev Configuration${NC}"
      echo -e "   API Endpoint: ${YELLOW}$API_ENDPOINT${NC}"
      echo -e "   API Key: ${GREEN}[REDACTED]${NC}"
      echo -e "   Version: ${YELLOW}$VERSION${NC}"
      
      # Test connection
      local response=$(api_request "GET" "/health" 2>/dev/null)
      if [ $? -eq 0 ]; then
          echo -e "   Connection: ${GREEN}✅ Connected${NC}"
      else
          echo -e "   Connection: ${RED}❌ Failed${NC}"
      fi
  }
  
  # Show version
  show_version() {
      echo "Terradev CLI v$VERSION"
  }
  
  # Main command router
  case "${1:-}" in
      "find-gpu")
          shift
          find_gpu "$@"
          ;;
      "deploy")
          shift
          deploy_gpu "$@"
          ;;
      "status")
          shift
          check_status "$@"
          ;;
      "savings")
          shift
          show_savings "$@"
          ;;
      "list-gpus")
          shift
          list_gpus "$@"
          ;;
      "list-providers")
          shift
          list_providers "$@"
          ;;
      "config")
          show_config
          ;;
      "version")
          show_version
          ;;
      "help"|"--help"|"-h")
          show_help
          ;;
      "")
          echo -e "${RED}Error: No command specified${NC}"
          echo ""
          show_help
          exit 1
          ;;
      *)
          echo -e "${RED}Unknown command: $1${NC}"
          echo ""
          show_help
          exit 1
          ;;
  esac
  EOT
}

# CLI Installation Script
resource "local_file" "cli_install" {
  filename = "${path.module}/cli/install.sh"
  content = <<-EOT
  #!/bin/bash
  # Terradev CLI Installation Script
  
  set -e
  
  INSTALL_DIR="${HOME}/.local/bin"
  CONFIG_DIR="${HOME}/.terradev"
  
  echo "🚀 Installing Terradev CLI..."
  
  # Create directories
  mkdir -p "$INSTALL_DIR"
  mkdir -p "$CONFIG_DIR"
  
  # Download CLI binary
  if [[ "$OSTYPE" == "linux-gnu"* ]]; then
      PLATFORM="linux"
  elif [[ "$OSTYPE" == "darwin"* ]]; then
      PLATFORM="macos"
  else
      echo "❌ Unsupported platform: $OSTYPE"
      exit 1
  fi
  
  # Download latest version
  LATEST_VERSION=$(curl -s https://api.github.com/repos/terradev/cli/releases/latest | jq -r '.tag_name')
  DOWNLOAD_URL="https://github.com/terradev/cli/releases/download/$LATEST_VERSION/terradev-$PLATFORM-amd64"
  
  echo "Downloading Terradev CLI $LATEST_VERSION for $PLATFORM..."
  curl -L "$DOWNLOAD_URL" -o "$INSTALL_DIR/terradev"
  
  # Make executable
  chmod +x "$INSTALL_DIR/terradev"
  
  # Add to PATH if not already there
  if [[ ":$PATH:" != *":$INSTALL_DIR:"* ]]; then
      echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
      echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.zshrc"
      echo "📝 Added $INSTALL_DIR to PATH"
      echo "⚠️  Please restart your terminal or run: source ~/.bashrc"
  fi
  
  # Create config template
  cat > "$CONFIG_DIR/config.json" << EOF
  {
    "api_endpoint": "https://api.terradev.io/v1",
    "default_gpu_type": "a100",
    "default_hours": 4,
    "confidence_threshold": 0.7,
    "max_risk_score": 0.5,
    "preferred_providers": ["aws", "gcp", "azure"],
    "output_format": "json"
  }
  EOF
  
  echo "✅ Terradev CLI installed successfully!"
  echo ""
  echo "🔧 Next steps:"
  echo "1. Set your API key: export TERRADEV_API_KEY=\"your-api-key\""
  echo "2. Test installation: terradev version"
  echo "3. Find your first GPU: terradev find-gpu --gpu-type a100 --hours 4"
  echo ""
  echo "📚 Documentation: https://terradev.io/docs/cli"
  EOT
}
