#!/bin/bash
# Terradev CLI Installation Script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Detect OS and architecture
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OS="linux"
        if [[ $(uname -m) == "x86_64" ]]; then
            ARCH="amd64"
        elif [[ $(uname -m) == "aarch64" ]]; then
            ARCH="arm64"
        else
            print_error "Unsupported architecture: $(uname -m)"
            exit 1
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="darwin"
        if [[ $(uname -m) == "x86_64" ]]; then
            ARCH="amd64"
        elif [[ $(uname -m) == "arm64" ]]; then
            ARCH="arm64"
        else
            print_error "Unsupported architecture: $(uname -m)"
            exit 1
        fi
    else
        print_error "Unsupported OS: $OSTYPE"
        exit 1
    fi
    
    print_status "Detected OS: $OS, Architecture: $ARCH"
}

# Check dependencies
check_dependencies() {
    print_status "Checking dependencies..."
    
    # Check for curl
    if ! command -v curl &> /dev/null; then
        print_error "curl is required but not installed. Please install curl first."
        exit 1
    fi
    
    # Check for jq (optional but recommended)
    if command -v jq &> /dev/null; then
        print_status "jq is installed"
    else
        print_warning "jq is not installed (optional but recommended)"
    fi
    
    # Check for terraform (optional)
    if command -v terraform &> /dev/null; then
        TF_VERSION=$(terraform version -json | jq -r '.terraform_version' 2>/dev/null || echo "unknown")
        print_status "Terraform is installed (version: $TF_VERSION)"
    else
        print_warning "Terraform is not installed. Please install Terraform to use generated code."
    fi
}

# Download Terradev CLI
download_terradev() {
    print_status "Downloading Terradev CLI..."
    
    # For now, we'll use a local copy since we don't have a real release URL
    # In production, this would download from GitHub releases
    TERRADEV_VERSION="1.0.0"
    DOWNLOAD_URL="https://github.com/terradev/cli/releases/download/v${TERRADEV_VERSION}/terradev-${OS}-${ARCH}"
    
    # Create temporary directory
    TEMP_DIR=$(mktemp -d)
    TERRADEV_BIN="$TEMP_DIR/terradev"
    
    # For demo purposes, we'll copy the local Python script and create a wrapper
    if [[ -f "terradev_cli.py" ]]; then
        print_status "Creating Terradev CLI from local source..."
        
        # Create a wrapper script
        cat > "$TERRADEV_BIN" << 'EOF'
#!/bin/bash
# Terradev CLI Wrapper

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/terradev_cli.py"

if [[ ! -f "$PYTHON_SCRIPT" ]]; then
    echo "Error: terradev_cli.py not found in $SCRIPT_DIR"
    exit 1
fi

# Check if Python 3 is available
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "Error: Python 3 is required but not found"
    exit 1
fi

# Run the Python script
exec "$PYTHON_CMD" "$PYTHON_SCRIPT" "$@"
EOF
        
        chmod +x "$TERRADEV_BIN"
        print_success "Terradev CLI created successfully"
    else
        print_error "terradev_cli.py not found in current directory"
        rm -rf "$TEMP_DIR"
        exit 1
    fi
}

# Install Terradev CLI
install_terradev() {
    print_status "Installing Terradev CLI..."
    
    # Determine installation directory
    if [[ -w "/usr/local/bin" ]]; then
        INSTALL_DIR="/usr/local/bin"
    elif [[ -w "$HOME/.local/bin" ]]; then
        INSTALL_DIR="$HOME/.local/bin"
        mkdir -p "$INSTALL_DIR"
        # Add to PATH if not already there
        if [[ ":$PATH:" != *":$INSTALL_DIR:"* ]]; then
            echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
            export PATH="$HOME/.local/bin:$PATH"
        fi
    else
        INSTALL_DIR="$HOME/bin"
        mkdir -p "$INSTALL_DIR"
        # Add to PATH if not already there
        if [[ ":$PATH:" != *":$INSTALL_DIR:"* ]]; then
            echo 'export PATH="$HOME/bin:$PATH"' >> "$HOME/.bashrc"
            export PATH="$HOME/bin:$PATH"
        fi
    fi
    
    # Copy the binary
    cp "$TERRADEV_BIN" "$INSTALL_DIR/terradev"
    chmod +x "$INSTALL_DIR/terradev"
    
    # Copy the Python script
    cp "terradev_cli.py" "$INSTALL_DIR/terradev_cli.py"
    
    print_success "Terradev CLI installed to $INSTALL_DIR/terradev"
}

# Verify installation
verify_installation() {
    print_status "Verifying installation..."
    
    # Check if terradev is in PATH
    if command -v terradev &> /dev/null; then
        print_success "Terradev CLI is now available in PATH"
        
        # Test basic functionality
        if terradev --help &> /dev/null; then
            print_success "Terradev CLI is working correctly"
        else
            print_warning "Terradev CLI installed but may have issues"
        fi
    else
        print_error "Terradev CLI not found in PATH"
        print_status "Please add $INSTALL_DIR to your PATH or restart your shell"
        print_status "You can run terradev directly with: $INSTALL_DIR/terradev"
    fi
}

# Setup configuration directory
setup_config() {
    print_status "Setting up configuration directory..."
    
    CONFIG_DIR="$HOME/.terradev"
    mkdir -p "$CONFIG_DIR"
    
    # Create example credentials file
    if [[ ! -f "$CONFIG_DIR/credentials.json" ]]; then
        cat > "$CONFIG_DIR/credentials.json" << 'EOF'
{
  "aws": {
    "credentials": {
      "access_key_id": "",
      "secret_access_key": "",
      "region": "us-west-2"
    },
    "enabled": false,
    "last_validated": null
  },
  "vast_ai": {
    "credentials": {
      "api_key": ""
    },
    "enabled": false,
    "last_validated": null
  }
}
EOF
        chmod 600 "$CONFIG_DIR/credentials.json"
        print_success "Created credentials configuration file"
    fi
    
    # Create example provenance file
    if [[ ! -f "$CONFIG_DIR/provenance.json" ]]; then
        echo "[]" > "$CONFIG_DIR/provenance.json"
        print_success "Created provenance database"
    fi
    
    print_success "Configuration directory setup complete"
}

# Show next steps
show_next_steps() {
    print_success "Installation complete!"
    echo
    echo "🚀 Next Steps:"
    echo "1. Configure your cloud providers:"
    echo "   terradev configure aws --interactive"
    echo "   terradev configure vast_ai --interactive"
    echo
    echo "2. List configured providers:"
    echo "   terradev list"
    echo
    echo "3. Optimize your first GPU deployment:"
    echo "   terrradev optimize --gpu-type A100 --duration 24 --user-id your-email@company.com --team-id your-team --project-id your-project"
    echo
    echo "4. View provenance records:"
    echo "   terradev provenance"
    echo
    echo "📚 For more information, see: https://docs.terradev.com"
}

# Cleanup
cleanup() {
    if [[ -n "$TEMP_DIR" && -d "$TEMP_DIR" ]]; then
        rm -rf "$TEMP_DIR"
    fi
}

# Main installation flow
main() {
    echo "🚀 Terradev CLI Installation"
    echo "=================================="
    echo
    
    # Detect OS and architecture
    detect_os
    echo
    
    # Check dependencies
    check_dependencies
    echo
    
    # Download Terradev CLI
    download_terradev
    echo
    
    # Install Terradev CLI
    install_terradev
    echo
    
    # Verify installation
    verify_installation
    echo
    
    # Setup configuration directory
    setup_config
    echo
    
    # Show next steps
    show_next_steps
    
    # Cleanup
    trap cleanup EXIT
}

# Handle script arguments
case "${1:-}" in
    "uninstall")
        echo "🗑️  Uninstalling Terradev CLI..."
        
        # Find and remove terradev binary
        if command -v terradev &> /dev/null; then
            TERRADEV_PATH=$(which terradev)
            rm -f "$TERRADEV_PATH"
            print_success "Removed $TERRADEV_PATH"
        fi
        
        # Remove configuration directory (optional)
        read -p "Remove configuration directory $HOME/.terradev? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf "$HOME/.terradev"
            print_success "Removed configuration directory"
        fi
        
        print_success "Terradev CLI uninstalled"
        ;;
    
    "help"|"-h"|"--help")
        echo "Terradev CLI Installation Script"
        echo
        echo "Usage: $0 [command]"
        echo
        echo "Commands:"
        echo "  install    Install Terradev CLI (default)"
        echo "  uninstall  Remove Terradev CLI"
        echo "  help       Show this help message"
        echo
        echo "Examples:"
        echo "  $0              # Install Terradev CLI"
        echo "  $0 install      # Install Terradev CLI"
        echo "  $0 uninstall    # Remove Terradev CLI"
        echo "  $0 help         # Show this help"
        ;;
    
    *)
        # Default to installation
        main
        ;;
esac
