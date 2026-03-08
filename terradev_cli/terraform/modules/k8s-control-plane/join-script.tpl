#!/bin/bash
# Kubernetes Node Join Script for ${cluster_name}

set -e

# Variables
CLUSTER_NAME="${cluster_name}"
ENDPOINT="${endpoint}"
CA_DATA="${ca_data}"
REGION="${region}"

echo "🚀 Joining Kubernetes cluster: ${CLUSTER_NAME}"
echo "🔗 Endpoint: ${ENDPOINT}"
echo "🌍 Region: ${REGION}"

# Create kubeconfig directory
mkdir -p /root/.kube

# Create temporary kubeconfig
cat > /root/.kube/config <<EOF
apiVersion: v1
kind: Config
clusters:
- name: ${CLUSTER_NAME}
  cluster:
    certificate-authority-data: ${CA_DATA}
    server: ${ENDPOINT}
contexts:
- name: ${CLUSTER_NAME}
  context:
    cluster: ${CLUSTER_NAME}
    user: ${CLUSTER_NAME}
    namespace: default
current-context: ${CLUSTER_NAME}
kind: Config
preferences: {}
users:
- name: ${CLUSTER_NAME}
  user:
    exec:
      apiVersion: client.authentication.k8s.io/v1beta1
      command: aws
      args:
      - eks
      - get-token
      - --cluster-name
      - ${CLUSTER_NAME}
      - --region
      - ${REGION}
EOF

# Get join command from control plane
echo "📋 Getting join command..."
JOIN_COMMAND=$(aws eks describe-cluster \
  --name ${CLUSTER_NAME} \
  --region ${REGION} \
  --query "cluster.nodeRoleArn" \
  --output text)

if [ -z "$JOIN_COMMAND" ]; then
    echo "❌ Failed to get cluster information"
    exit 1
fi

# Create join config for kubeadm
cat > /tmp/join-config.yaml <<EOF
apiVersion: kubeadm.k8s.io/v1beta3
kind: JoinConfiguration
discovery:
  bootstrapToken:
    token: $(aws eks create-token --cluster-name ${CLUSTER_NAME} --region ${REGION} | jq -r '.status.token')
    apiServerEndpoint: ${ENDPOINT}
    caCertHashes:
    - $(aws eks describe-cluster --name ${CLUSTER_NAME} --region ${REGION} --query "cluster.certificateAuthority.data" --output text | sha256sum | awk '{print $1}')
nodeRegistration:
  kubeletExtraArgs:
    node-labels: "terradev.io/provider=aws,terradev.io/node-type=gpu"
EOF

# Join the cluster
echo "🔗 Joining cluster..."
kubeadm join --config /tmp/join-config.yaml

# Wait for node to be ready
echo "⏳ Waiting for node to be ready..."
sleep 30

# Verify node status
echo "📊 Checking node status..."
kubectl get nodes -o wide

# Clean up
rm -f /tmp/join-config.yaml

echo "✅ Node joined cluster successfully!"
