#!/bin/bash

# Define the root directory of the project (assuming it's the current directory)
ROOT_DIR=$(pwd)

# Function to create directories and set up environments
create_structure_and_envs() {
    echo "Creating project structure and setting up environments..."

    # Function to create a directory and add a .keep file to ensure it exists in version control
    create_dir() {
        mkdir -p "$1"
        touch "$1/.keep"
    }

    # Function to initialize npm if package.json does not exist
    init_npm() {
        if [ ! -f "$1/package.json" ]; then
            (cd "$1" && npm init -y)
        else
            echo "npm package already initialized in $1"
        fi
    }

    # Function to initialize Cargo if Cargo.toml does not exist
    init_cargo() {
        if [ ! -f "$1/Cargo.toml" ]; then
            (cd "$1" && cargo init --lib)
        else
            echo "Cargo package already initialized in $1"
        fi
    }

    # Create dml_hardhat directories and set up npm environment
    create_dir "$ROOT_DIR/dml_hardhat/DP/contracts"
    create_dir "$ROOT_DIR/dml_hardhat/DP/models"
    create_dir "$ROOT_DIR/dml_hardhat/DP/scripts"
    init_npm "$ROOT_DIR/dml_hardhat/DP"

    create_dir "$ROOT_DIR/dml_hardhat/MP/contracts"
    create_dir "$ROOT_DIR/dml_hardhat/MP/models"
    create_dir "$ROOT_DIR/dml_hardhat/MP/scripts"
    init_npm "$ROOT_DIR/dml_hardhat/MP"

    create_dir "$ROOT_DIR/dml_hardhat/TP/contracts"
    create_dir "$ROOT_DIR/dml_hardhat/TP/models"
    create_dir "$ROOT_DIR/dml_hardhat/TP/scripts"
    init_npm "$ROOT_DIR/dml_hardhat/TP"

    create_dir "$ROOT_DIR/dml_hardhat/zk-snark"

    # Create dml_substrate directories and set up Rust environment
    create_dir "$ROOT_DIR/dml_substrate/DP/substrate"
    create_dir "$ROOT_DIR/dml_substrate/DP/models"
    create_dir "$ROOT_DIR/dml_substrate/DP/integration"
    init_cargo "$ROOT_DIR/dml_substrate/DP"

    create_dir "$ROOT_DIR/dml_substrate/MP/substrate"
    create_dir "$ROOT_DIR/dml_substrate/MP/models"
    create_dir "$ROOT_DIR/dml_substrate/MP/integration"
    init_cargo "$ROOT_DIR/dml_substrate/MP"

    create_dir "$ROOT_DIR/dml_substrate/TP/substrate"
    create_dir "$ROOT_DIR/dml_substrate/TP/models"
    create_dir "$ROOT_DIR/dml_substrate/TP/integration"
    init_cargo "$ROOT_DIR/dml_substrate/TP"

    # Create testnet connection directories and set up Python virtual environment
    create_dir "$ROOT_DIR/testnet_connection/scripts"
    create_dir "$ROOT_DIR/testnet_connection/configs"
    if [ ! -d "$ROOT_DIR/testnet_connection/venv" ]; then
        (cd "$ROOT_DIR/testnet_connection" && python3 -m venv venv)
    else
        echo "Python virtual environment already exists in $ROOT_DIR/testnet_connection"
    fi

    # Create profiling and comparison directories
    create_dir "$ROOT_DIR/profiling_comparison/dml_hardhat"
    create_dir "$ROOT_DIR/profiling_comparison/dml_substrate"

    # Create evaluation directories
    create_dir "$ROOT_DIR/evaluation/analysis"

    # Create general scripts directory
    create_dir "$ROOT_DIR/scripts"

    echo "Project structure and environments created."
}

# Execute the function
create_structure_and_envs

echo "Project setup complete!"