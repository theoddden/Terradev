# Complete Account & API Setup Guide
# Exact list of accounts, APIs, and credentials needed

resource "local_file" "account_setup_guide" {
  filename = "${path.module}/setup/account-setup-guide.md"
  content = <<-EOT
  # Complete Account & API Setup Guide
  
  ## 🎯 Accounts Needed (All Free Tiers)
  
  ### 1. Geolocation & Network APIs
  
  #### IPapi.co
  - **Account**: Free (no signup required)
  - **API Key**: None needed
  - **Rate Limit**: None (unlimited)
  - **Setup**: Direct API access
  - **Credentials**: None
  
  #### ip-api.com  
  - **Account**: Free (no signup required)
  - **API Key**: None needed
  - **Rate Limit**: None (unlimited)
  - **Setup**: Direct API access
  - **Credentials**: None
  
  #### IPinfo.io
  - **Account**: https://ipinfo.io/signup
  - **API Key**: Free token (1000 requests/month)
  - **Rate Limit**: 1000 requests/month
  - **Setup**: Signup → Get free token
  - **Credentials**: `IPINFO_API_KEY`
  
  #### MaxMind GeoLite2
  - **Account**: https://www.maxmind.com/en/geolite2/signup
  - **API Key**: Account ID + License Key
  - **Rate Limit**: None (database download)
  - **Setup**: Signup → Download GeoLite2-City.mmdb
  - **Credentials**: `MAXMIND_ACCOUNT_ID`, `MAXMIND_LICENSE_KEY`
  
  ### 2. Cloud Provider APIs
  
  #### AWS (Amazon Web Services)
  - **Account**: https://aws.amazon.com/free
  - **API Key**: Access Key ID + Secret Access Key
  - **Rate Limit**: Free tier limits
  - **Setup**: Create AWS account → Create IAM user → Get credentials
  - **Credentials**: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`
  - **Services Needed**:
    - CloudWatch (monitoring)
    - Route 53 (health checks)
    - X-Ray (tracing)
    - SageMaker (inference endpoints)
  
  #### Google Cloud Platform (GCP)
  - **Account**: https://cloud.google.com/free
  - **API Key**: Service account JSON key
  - **Rate Limit**: Free tier quotas
  - **Setup**: Create GCP project → Create service account → Download JSON key
  - **Credentials**: `GOOGLE_APPLICATION_CREDENTIALS` (path to JSON file)
  - **Services Needed**:
    - Cloud Monitoring
    - Cloud Trace
    - Vertex AI
    - Cloud Functions
  
  #### Microsoft Azure
  - **Account**: https://azure.microsoft.com/free
  - **API Key**: Service principal credentials
  - **Rate Limit**: Free tier limits
  - **Setup**: Create Azure account → Create app registration → Get credentials
  - **Credentials**: `AZURE_CLIENT_ID`, `AZURE_CLIENT_SECRET`, `AZURE_TENANT_ID`
  - **Services Needed**:
    - Azure Monitor
    - Application Insights
    - Azure ML
    - Front Door
  
  ### 3. Specialized GPU Providers
  
  #### RunPod
  - **Account**: https://runpod.io/signup
  - **API Key**: API token
  - **Rate Limit**: Standard API limits
  - **Setup**: Signup → Settings → API → Generate token
  - **Credentials**: `RUNPOD_API_KEY`
  
  #### Lambda Labs
  - **Account**: https://lambdalabs.com/signup
  - **API Key**: API token
  - **Rate Limit**: Standard API limits
  - **Setup**: Signup → Account → API → Generate token
  - **Credentials**: `LAMBDA_API_KEY`
  
  #### CoreWeave
  - **Account**: https://coreweave.com/signup
  - **API Key**: API token
  - **Rate Limit**: Standard API limits
  - **Setup**: Signup → Dashboard → API → Generate token
  - **Credentials**: `COREWEAVE_API_KEY`
  
  ### 4. Edge Computing & CDN
  
  #### Cloudflare
  - **Account**: https://dash.cloudflare.com/sign-up
  - **API Key**: Global API Key + Token
  - **Rate Limit**: Free tier limits
  - **Setup**: Signup → My Profile → API Tokens → Create token
  - **Credentials**: `CLOUDFLARE_API_KEY`, `CLOUDFLARE_EMAIL`
  - **Services Needed**:
    - Workers AI (inference)
    - Radar (network quality)
    - Edge locations
  
  #### Fastly
  - **Account**: https://www.fastly.com/signup
  - **API Key**: API token
  - **Rate Limit**: Free tier limits
  - **Setup**: Signup → Account → Personal API tokens
  - **Credentials**: `FASTLY_API_KEY`
  
  ### 5. DevOps & Monitoring
  
  #### Grafana
  - **Account**: https://grafana.com/signup
  - **API Key**: API token
  - **Rate Limit**: Free tier limits
  - **Setup**: Signup → Organization → API Keys → Create token
  - **Credentials**: `GRAFANA_API_KEY`, `GRAFANA_URL`
  
  #### Jenkins
  - **Account**: Self-hosted (free)
  - **API Key**: API token
  - **Rate Limit**: Local limits
  - **Setup**: Install Jenkins → Manage Jenkins → Configure Security → API Token
  - **Credentials**: `JENKINS_API_TOKEN`, `JENKINS_URL`
  
  #### Kubernetes
  - **Account**: Various (GKE, EKS, AKS, or self-hosted)
  - **API Key**: Kubeconfig or service account
  - **Rate Limit**: Cluster limits
  - **Setup**: Deploy cluster → Get kubeconfig
  - **Credentials**: `KUBECONFIG` path or in-cluster service account
  
  #### Databricks
  - **Account**: https://databricks.com/signup
  - **API Key**: Personal access token
  - **Rate Limit**: Free tier limits
  - **Setup**: Signup → Workspace → User Settings → Generate token
  - **Credentials**: `DATABRICKS_API_TOKEN`, `DATABRICKS_HOST`
  
  ### 6. Communication & Integration
  
  #### GitHub
  - **Account**: https://github.com/signup
  - **API Key**: Personal access token
  - **Rate Limit**: 5000 requests/hour (authenticated)
  - **Setup**: Signup → Settings → Developer settings → Personal access tokens
  - **Credentials**: `GITHUB_TOKEN`
  
  #### Slack
  - **Account**: https://slack.com/create
  - **API Key**: Bot token + App token
  - **Rate Limit**: Free tier limits
  - **Setup**: Create workspace → Create app → Get bot token
  - **Credentials**: `SLACK_BOT_TOKEN`, `SLACK_APP_TOKEN`
  
  ### 7. Performance Testing
  
  #### WebPageTest
  - **Account**: https://www.webpagetest.org/signup
  - **API Key**: API key
  - **Rate Limit**: 200 tests/day (free)
  - **Setup**: Signup → Account → API Key
  - **Credentials**: `WEBPAGETEST_API_KEY`
  
  #### GTmetrix
  - **Account**: https://gtmetrix.com/signup
  - **API Key**: API key
  - **Rate Limit**: Free tier limits
  - **Setup**: Signup → Account → API
  - **Credentials**: `GTMETRIX_API_KEY`
  
  ## 🔧 Environment Variables Setup
  
  ### Complete .env file template
  ```bash
  # Geolocation APIs
  IPINFO_API_KEY=your_ipinfo_token
  MAXMIND_ACCOUNT_ID=your_maxmind_account_id
  MAXMIND_LICENSE_KEY=your_maxmind_license_key
  
  # Cloud Providers
  AWS_ACCESS_KEY_ID=your_aws_access_key
  AWS_SECRET_ACCESS_KEY=your_aws_secret_key
  AWS_REGION=us-east-1
  
  GOOGLE_APPLICATION_CREDENTIALS=/path/to/gcp-service-account.json
  
  AZURE_CLIENT_ID=your_azure_client_id
  AZURE_CLIENT_SECRET=your_azure_client_secret
  AZURE_TENANT_ID=your_azure_tenant_id
  
  # GPU Providers
  RUNPOD_API_KEY=your_runpod_api_key
  LAMBDA_API_KEY=your_lambda_api_key
  COREWEAVE_API_KEY=your_coreweave_api_key
  
  # Edge Computing
  CLOUDFLARE_API_KEY=your_cloudflare_api_key
  CLOUDFLARE_EMAIL=your_cloudflare_email
  FASTLY_API_KEY=your_fastly_api_key
  
  # DevOps & Monitoring
  GRAFANA_API_KEY=your_grafana_api_key
  GRAFANA_URL=https://your-grafana-instance.com
  
  JENKINS_API_TOKEN=your_jenkins_api_token
  JENKINS_URL=https://your-jenkins-instance.com
  
  DATABRICKS_API_TOKEN=your_databricks_api_token
  DATABRICKS_HOST=https://your-databricks-instance.com
  
  # Communication
  GITHUB_TOKEN=your_github_token
  SLACK_BOT_TOKEN=your_slack_bot_token
  SLACK_APP_TOKEN=your_slack_app_token
  
  # Performance Testing
  WEBPAGETEST_API_KEY=your_webpagetest_api_key
  GTMETRIX_API_KEY=your_gtmetrix_api_key
  
  # Kubernetes
  KUBECONFIG=/path/to/kubeconfig
  ```
  
  ## 📋 Setup Priority & Timeline
  
  ### Week 1: Core APIs (Essential)
  - [ ] IPapi.co (no setup needed)
  - [ ] ip-api.com (no setup needed)
  - [ ] RunPod account + API key
  - [ ] Cloudflare account + API key
  - [ ] GitHub account + token
  
  ### Week 2: Cloud Providers
  - [ ] AWS account + IAM credentials
  - [ ] GCP account + service account
  - [ ] Azure account + service principal
  
  ### Week 3: Additional Providers
  - [ ] Lambda Labs account + API key
  - [ ] CoreWeave account + API key
  - [ ] Fastly account + API key
  
  ### Week 4: DevOps & Monitoring
  - [ ] Grafana account + API key
  - [ ] Jenkins setup + API token
  - [ ] Kubernetes cluster + kubeconfig
  - [ ] Databricks account + token
  
  ### Week 5: Communication & Testing
  - [ ] Slack workspace + bot tokens
  - [ ] WebPageTest account + API key
  - [ ] GTmetrix account + API key
  
  ### Week 6: Optional Enhancements
  - [ ] IPinfo.io account + API key
  - [ ] MaxMind account + database download
  
  ## 🎯 Quick Setup Script
  
  ```bash
  #!/bin/bash
  # Quick setup script for Terradev accounts
  
  echo "🚀 Setting up Terradev accounts..."
  
  # Create .env file
  touch .env
  
  # Core APIs (no setup needed)
  echo "# Geolocation APIs (no setup needed)" >> .env
  echo "# IPapi.co - Direct access" >> .env
  echo "# ip-api.com - Direct access" >> .env
  echo "" >> .env
  
  # Prompt for essential API keys
  echo "🔧 Setting up essential APIs..."
  
  echo "Enter your RunPod API key:"
  read -s RUNPOD_API_KEY
  echo "RUNPOD_API_KEY=$RUNPOD_API_KEY" >> .env
  
  echo "Enter your Cloudflare API key:"
  read -s CLOUDFLARE_API_KEY
  echo "Enter your Cloudflare email:"
  read CLOUDFLARE_EMAIL
  echo "CLOUDFLARE_API_KEY=$CLOUDFLARE_API_KEY" >> .env
  echo "CLOUDFLARE_EMAIL=$CLOUDFLARE_EMAIL" >> .env
  
  echo "Enter your GitHub token:"
  read -s GITHUB_TOKEN
  echo "GITHUB_TOKEN=$GITHUB_TOKEN" >> .env
  
  echo "✅ Essential setup complete!"
  echo "📋 Next steps:"
  echo "1. Setup AWS account (https://aws.amazon.com/free)"
  echo "2. Setup GCP account (https://cloud.google.com/free)"
  echo "3. Setup Azure account (https://azure.microsoft.com/free)"
  echo "4. Run the full setup script"
  ```
  
  ## 🔐 Security Best Practices
  
  ### 1. API Key Management
  - Store API keys in environment variables
  - Use .env files (add to .gitignore)
  - Rotate API keys regularly
  - Use principle of least privilege
  
  ### 2. IAM Policies
  - Create specific IAM users for each service
  - Grant minimum required permissions
  - Use temporary credentials when possible
  - Enable MFA on all accounts
  
  ### 3. Network Security
  - Use HTTPS for all API calls
  - Implement rate limiting
  - Add request signing for cloud providers
  - Monitor API usage and anomalies
  
  ## 📊 Cost Summary
  
  | Service | Cost | Setup Time | Priority |
  |---------|------|------------|----------|
  | IPapi.co | Free | 0 min | 🔴 Critical |
  | ip-api.com | Free | 0 min | 🔴 Critical |
  | RunPod | Free | 5 min | 🔴 Critical |
  | Cloudflare | Free | 10 min | 🔴 Critical |
  | GitHub | Free | 5 min | 🔴 Critical |
  | AWS | Free tier | 15 min | 🟡 High |
  | GCP | Free tier | 15 min | 🟡 High |
  | Azure | Free tier | 20 min | 🟡 High |
  | Lambda Labs | Free | 5 min | 🟡 High |
  | CoreWeave | Free | 5 min | 🟡 High |
  | Fastly | Free | 5 min | 🟢 Medium |
  | Grafana | Free | 10 min | 🟢 Medium |
  | Jenkins | Free | 30 min | 🟢 Medium |
  | Slack | Free | 10 min | 🟢 Medium |
  | WebPageTest | Free | 5 min | 🟢 Medium |
  
  **Total Setup Time**: 2-3 hours
  **Total Cost**: $0/month (all free tiers)
  **Accounts Needed**: ~15 services
  **API Keys Required**: ~12 tokens
  
  ## 🎯 Success Checklist
  
  ### Phase 1: Minimum Viable Product (Week 1)
  - [x] IPapi.co access (no setup)
  - [x] ip-api.com access (no setup)
  - [ ] RunPod API key
  - [ ] Cloudflare API key
  - [ ] GitHub token
  
  ### Phase 2: Cloud Integration (Week 2)
  - [ ] AWS credentials
  - [ ] GCP service account
  - [ ] Azure service principal
  
  ### Phase 3: Full Integration (Week 3-4)
  - [ ] All GPU provider API keys
  - [ ] Edge provider API keys
  - [ ] DevOps API keys
  
  ### Phase 4: Production Ready (Week 5-6)
  - [ ] Monitoring setup
  - [ ] Communication setup
  - [ ] Performance testing setup
  
  ## 🚀 Next Steps
  
  1. **Start with essential APIs** (Week 1)
  2. **Add cloud providers** (Week 2)
  3. **Integrate all providers** (Week 3-4)
  4. **Setup monitoring** (Week 5-6)
  5. **Launch MVP** (Week 7)
  
  This setup gives you everything needed to build a production-ready latency arbitrage system with zero upfront costs!
  EOT
}
