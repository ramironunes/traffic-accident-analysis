# Traffic Accident Analysis

This repository aims to analyze traffic accident data using Python and various data science tools. The project includes a Docker setup to ensure a consistent development environment with all necessary dependencies.

## Technologies Used

- **Python**: The main programming language used for data analysis and processing.
- **Conda**: Used to manage dependencies and create isolated environments.
- **Orange3**: A comprehensive suite for machine learning and data mining.
- **Docker**: Ensures a consistent and reproducible environment for development and deployment.

## Getting Started

### Prerequisites

Ensure you have Docker and Docker Compose installed on your system.

### Installing Docker and Docker Compose

To install Docker and Docker Compose, follow these steps:

1. **Install Docker:**
   ```bash
   sudo apt-get update
   sudo apt-get install -y docker-ce docker-ce-cli containerd.io
   ```

2. **Start the Docker Container:**
   ```bash
   docker compose -f docker/docker-compose.dev.yml up -d
   ```

3. **Access the Container:**
   To access the container in interactive mode, run:
   ```bash
   docker compose -f docker/docker-compose.dev.yml exec traffic-analysis bash
   ```

4. **Initialize Conda and Activate the Environment:**
   Inside the container, initialize Conda and activate the environment:
   ```bash
   source /opt/conda/etc/profile.d/conda.sh
   conda activate traffic_analysis
   ```

5. **Start Orange:**
   Once the environment is activated, start Orange with:
   ```bash
   orange-canvas
   ```

### Project Dependencies

Dependencies are managed via Conda and are specified in the deploy/requirements.txt file. Key dependencies include:
- **Orange3**: A comprehensive suite for machine learning and data mining.

### Project Scripts

- `setup.py`: Configuration for the Python package, including project metadata and dependencies.

### Docker Configuration

- `docker/Dockerfile.dev`: Defines the Docker image for the development environment, based on Ubuntu 24.04, and sets up Conda and necessary Python packages.
- `docker/docker-compose.dev.yml`: Docker Compose configuration for setting up the development environment.
