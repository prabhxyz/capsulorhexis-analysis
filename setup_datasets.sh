#!/usr/bin/env bash
# 
# setup_datasets.sh
#
# Installs synapseclient, creates two folders (Cataract-1k-Phase and Cataract-1k-Seg),
# and downloads the associated Synapse data.
#
# Usage:
#   1) chmod +x setup_datasets.sh
#   2) ./setup_datasets.sh <auth_token>
#      or
#      export SYNAPSE_AUTH_TOKEN="my_token_value"
#      ./setup_datasets.sh
#
# If you want Synapse to remember your token for future commands, 
# you can add "--rememberMe" to the synapse login command.

# Step 1: Install synapseclient if not installed
echo "Installing synapseclient..."
pip install synapseclient

# Step 2: Retrieve auth token from either:
#   1) Script argument ($1), or
#   2) SYNAPSE_AUTH_TOKEN environment variable, or
#   3) Fallback to empty (will ask for login).
TOKEN="${1:-$SYNAPSE_AUTH_TOKEN}"

if [ -z "$TOKEN" ]; then
  echo "No auth token provided. If you haven't logged in before, Synapse might ask for credentials."
  echo "If you prefer automated login, pass the token as an argument or set SYNAPSE_AUTH_TOKEN."
  # A manual login might prompt for user/password or token if no credentials are found.
  synapse login
else
  echo "Logging into Synapse with provided auth token..."
  synapse login --authToken "$TOKEN" # optionally: --rememberMe
fi

# Step 3: Create and download Cataract-1k-Phase
echo "Creating folder Cataract-1k-Phase and downloading data (syn53395146)..."
mkdir -p Cataract-1k-Phase
cd Cataract-1k-Phase
synapse get -r syn53395146
cd ..

# Step 4: Create and download Cataract-1k-Seg
echo "Creating folder Cataract-1k-Seg and downloading data (syn53395479)..."
mkdir -p Cataract-1k-Seg
cd Cataract-1k-Seg
synapse get -r syn53395479
cd ..

echo "Dataset setup complete."
