apiVersion: v1
kind: Config
clusters:
- name: ${cluster_name}
  cluster:
    certificate-authority-data: ${ca_data}
    server: ${endpoint}
contexts:
- name: ${cluster_name}
  context:
    cluster: ${cluster_name}
    user: ${cluster_name}
    namespace: default
current-context: ${cluster_name}
kind: Config
preferences: {}
users:
- name: ${cluster_name}
  user:
    exec:
      apiVersion: client.authentication.k8s.io/v1beta1
      command: aws
      args:
      - eks
      - get-token
      - --cluster-name
      - ${cluster_name}
      - --region
      - ${region}
