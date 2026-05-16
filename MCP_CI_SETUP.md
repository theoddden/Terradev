# MCP Server CI/CD Setup Instructions

## Overview
This document explains how to configure the MCP server to work with GitHub Actions CI/CD for Terradev.

## GitHub Token Configuration

The MCP server requires a GitHub token for authentication. You need to add the token as a GitHub secret.

### Step 1: Add GitHub Secret

1. Go to your Terradev repository on GitHub
2. Navigate to: Settings → Secrets and variables → Actions
3. Click "New repository secret"
4. Name: `TERRADEV_GITHUB_TOKEN`
5. Value: `<your-github-token-here>`
6. Click "Add secret"

### Step 2: Verify Workflow Configuration

The CI workflow (`.github/workflows/ci.yml`) has been updated to:
- Load the MCP server module during testing
- Use the `TERRADEV_GITHUB_TOKEN` secret for authentication
- Test MCP server functionality as part of the test suite

## MCP Server Test Coverage

The CI workflow now includes:
1. **MCP Server Loading Test**: Verifies the MCP server module can be imported
2. **Tool Count Test**: Verifies the MCP server has the expected number of tools loaded
3. **Existing MCP Handler Tests**: The existing test suite includes comprehensive MCP handler tests in `tests/test_mcp_handlers.py`

## What Gets Tested

The MCP server tests verify:
- Server module can be imported
- Tool definitions are valid
- Helper functions exist and work
- Tool categories are organized correctly
- Command map routing works
- JSON schema validation passes
- Handler patterns are correct
- Structured output format is valid
- Tool batch counts match expectations

## CI Workflow Changes

The following changes were made to `.github/workflows/ci.yml`:

1. Added `GITHUB_TOKEN` and `TERRADEV_GITHUB_TOKEN` environment variables to the test job
2. Added a dedicated "Test MCP server loading" step that:
   - Imports the MCP server module
   - Verifies tools can be listed
   - Uses the configured GitHub tokens for authentication

## Troubleshooting

If the MCP server tests fail in CI:

1. **Secret Not Set**: Ensure `TERRADEV_GITHUB_TOKEN` is set in repository secrets
2. **Import Error**: Check that all MCP dependencies are installed (they should be via the package install step)
3. **Tool Loading Error**: Verify the MCP server code is syntactically correct and all dependencies are available

## Security Notes

- The GitHub token is stored as a repository secret, not in the workflow file
- The built-in `GITHUB_TOKEN` is automatically provided by GitHub Actions
- The custom `TERRADEV_GITHUB_TOKEN` is used for MCP server-specific authentication needs
