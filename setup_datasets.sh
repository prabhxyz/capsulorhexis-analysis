#!/usr/bin/env bash
# 
# setup_datasets.sh
#
# Installs synapseclient, creates two folders (Cataract-1k-Phase and Cataract-1k-Seg),
# and downloads the associated Synapse data.
#
# Usage:
#   1) chmod +x setup_datasets.sh
#   2) ./setup_datasets.sh 

# Step 1: Install synapseclient if not installed
echo "Installing synapseclient..."
pip install synapseclient

# Step 2: Create and download Cataract-1k-Phase
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
