#!/bin/bash
# Lambda Labs GPU Node Bootstrap Script

set -e

# Variables
CLUSTER_NAME="${cluster_name}"
TAILSCALE_AUTHKEY="${tailscale_authkey}"
K8S_JOIN_SCRIPT="${k8s_join_script}"
GPU_TYPE="${gpu_type}"
PROVIDER="${provider}"

# Labels
LABELS="${labels}"

echo "🚀 Bootstrapping Lambda Labs GPU Node for ${CLUSTER_NAME}"
echo "🎮 GPU Type: ${GPU_TYPE}"
echo "🏷️  Provider: ${PROVIDER}"

# Update system
echo "📦 Updating system packages..."
apt-get update && apt-get upgrade -y

# Install required packages
echo "📦 Installing required packages..."
apt-get install -y \
    curl \
    wget \
    gnupg \
    software-properties-common \
    ca-certificates \
    apt-transport-https \
    htop \
    git \
    vim \
    jq

# Install NVIDIA drivers
echo "🎮 Installing NVIDIA drivers..."
wget -O /tmp/nvidia-driver.run https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i /tmp/nvidia-driver.run
apt-get update
apt-get install -y cuda-toolkit-12-2 cuda-drivers

# Install Docker
echo "🐳 Installing Docker..."
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null
apt-get update
apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# Configure Docker for NVIDIA
echo "⚙️  Configuring Docker for NVIDIA..."
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | tee /etc/apt/sources.list.d/nvidia-docker.list
apt-get update && apt-get install -y nvidia-docker2
systemctl restart docker

# Install Kubernetes components
echo "☸️  Installing Kubernetes components..."
curl -fsSL https://pkgs.k8s.io/core:/stable:/v1.28/deb/Release.key | gpg --dearmor -o /etc/apt/keyrings/kubernetes-apt-keyring.gpg
echo 'deb [signed-by=/etc/apt/keyrings/kubernetes-apt-keyring.gpg] https://pkgs.k8s.io/core:/stable:/v1.28/deb/ /' | tee /etc/apt/sources.list.d/kubernetes.list
apt-get update
apt-get install -y kubelet kubeadm kubectl
apt-mark hold kubelet kubeadm kubectl

# Install Tailscale
echo "🔗 Installing Tailscale..."
curl -fsSL https://tailscale.com/install.sh | sh

# Start and enable Tailscale
echo "🔗 Starting Tailscale..."
tailscale up --authkey="${TAILSCALE_AUTHKEY}" --hostname="${CLUSTER_NAME}-lambda-node" --advertise-routes=10.0.0.0/24 --accept-routes

# Wait for Tailscale to be ready
echo "⏳ Waiting for Tailscale to be ready..."
while ! tailscale status | grep -q "Connected"; do
    echo "Waiting for Tailscale connection..."
    sleep 5
done

# Get Tailscale IP
TAILSCALE_IP=$(tailscale ip -4)
echo "🔗 Tailscale IP: ${TAILSCALE_IP}"

# Configure Kubernetes
echo "⚙️  Configuring Kubernetes..."
cat > /etc/docker/daemon.json <<EOF
{
  "exec-opts": ["native.cgroupdriver=systemd"],
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "100m"
  },
  "storage-driver": "overlay2"
}
EOF

mkdir -p /etc/systemd/system/docker.service.d
cat > /etc/systemd/system/docker.service.d/override.conf <<EOF
[Service]
ExecStart=
ExecStart=/usr/bin/dockerd
EOF

systemctl daemon-reload
systemctl restart docker
systemctl enable docker

# Disable swap
echo "💾 Disabling swap..."
swapoff -a
sed -i '/ swap / s/^\(.*\)$/#\1/g' /etc/fstab

# Enable kernel modules
echo "🔧 Enabling kernel modules..."
cat > /etc/modules-load.d/k8s.conf <<EOF
br_netfilter
overlay
EOF

modprobe br_netfilter
modprobe overlay

# Set sysctl parameters
echo "⚙️  Setting sysctl parameters..."
cat > /etc/sysctl.d/k8s.conf <<EOF
net.bridge.bridge-nf-call-iptables  = 1
net.bridge.bridge-nf-call-ip6tables = 1
net.ipv4.ip_forward                 = 1
EOF

sysctl --system

# Join Kubernetes cluster
if [ -n "$K8S_JOIN_SCRIPT" ] && [ -f "$K8S_JOIN_SCRIPT" ]; then
    echo "☸️  Joining Kubernetes cluster..."
    bash "$K8S_JOIN_SCRIPT"
    
    # Wait for node to be ready
    echo "⏳ Waiting for node to be ready..."
    while ! kubectl get nodes | grep -q "$(hostname)"; do
        echo "Waiting for node registration..."
        sleep 10
    done
    
    # Apply labels
    echo "🏷️  Applying labels..."
    kubectl label node $(hostname) ${LABELS} --overwrite
    
    # Apply GPU taint
    kubectl taint node $(hostname) nvidia.com/gpu=true:NoSchedule --overwrite
else
    echo "⚠️  No Kubernetes join script provided, skipping cluster join"
fi

# Install NVIDIA device plugin
echo "🎮 Installing NVIDIA device plugin..."
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.0/nvidia-device-plugin.yml

# Test GPU
echo "🧪 Testing GPU..."
nvidia-smi

# Create GPU test script
cat > /usr/local/bin/test-gpu.sh <<'EOF'
#!/bin/bash
echo "🎮 GPU Test Results:"
nvidia-smi
echo ""
echo "🐳 Docker GPU Test:"
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi
echo ""
echo "☸️  Kubernetes GPU Test:"
kubectl get nodes -L nvidia.com/gpu
EOF

chmod +x /usr/local/bin/test-gpu.sh

# Create status script
cat > /usr/local/bin/status.sh <<'EOF'
#!/bin/bash
echo "🚀 Terradev GPU Node Status"
echo "=========================="
echo "🔗 Tailscale Status:"
tailscale status
echo ""
echo "🎮 GPU Status:"
nvidia-smi
echo ""
echo "🐳 Docker Status:"
systemctl status docker --no-pager -l
echo ""
echo "☸️  Kubernetes Status:"
if command -v kubectl &> /dev/null; then
    kubectl get nodes -o wide
    kubectl get pods -A
else
    echo "Kubernetes not configured"
fi
EOF

chmod +x /usr/local/bin/status.sh

# Create cleanup script
cat > /usr/local/bin/cleanup.sh <<'EOF'
#!/bin/bash
echo "🧹 Cleaning up GPU node..."
# Leave Kubernetes cluster
if command -v kubectl &> /dev/null; then
    kubectl drain $(hostname) --ignore-daemonsets --delete-emptydir-data --force
    kubectl delete node $(hostname)
fi
# Leave Tailscale
tailscale down
# Stop services
systemctl stop kubelet docker
echo "✅ Cleanup complete"
EOF

chmod +x /usr/local/bin/cleanup.sh

echo "✅ Bootstrap complete!"
echo "🎮 GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits)"
echo "🔗 Tailscale IP: ${TAILSCALE_IP}"
echo "☸️  Kubernetes: $(command -v kubectl &> /dev/null && echo 'Configured' || echo 'Not configured')"
echo ""
echo "📊 Run 'status.sh' for detailed status"
echo "🧪 Run 'test-gpu.sh' to test GPU functionality"
echo "🧹 Run 'cleanup.sh' to clean up node"
