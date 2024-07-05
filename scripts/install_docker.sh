#!/bin/bash
# ============================================================================
# @Author: Ramiro Luiz Nunes
# @Date:   2024-07-05 14:25:55
# @Info:   Install docker
# ============================================================================

# Update the package database
sudo apt-get update

# Install prerequisites
sudo apt-get install -y \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg \
    lsb-release

# Add Docker’s official GPG key
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

# Add Docker APT repository
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Update the package database again
sudo apt-get update

# Install Docker Engine and Docker Compose
sudo apt-get install -y docker-ce docker-ce-cli containerd.io

# Install Docker Compose separately
sudo curl -L "https://github.com/docker/compose/releases/download/1.29.2/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Verify the installation
docker --version
docker-compose --version

# Add current user to the docker group to avoid using sudo with Docker commands
sudo usermod -aG docker $USER

echo "Docker and Docker Compose installation completed. Please log out and log back in to apply the user group changes."
