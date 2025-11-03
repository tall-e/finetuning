#!/bin/bash
set -e

echo " Setting up new Lambda instance..."

# System updates
echo "Updating system..."
sudo apt-get update && sudo apt-get upgrade -y
sudo apt-get install -y git vim tmux htop

# Install uv
echo " Installing uv..."
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
    echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.bashrc
fi

# Git config
echo "Configuring git..."
git config --global user.name "tall-e"
git config --global user.email "tollypowell0x@gmail.com"
git config --global credential.helper store
echo "Git configured as: tall-e <tollypowell0x@gmail.com>"

# Clone repo with HTTPS
echo ""
echo "📥 Cloning repository..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "You'll be prompted for GitHub credentials:"
echo "  Username: tall-e"
echo "  Password: your-personal-access-token"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
cd ~
git clone https://github.com/tall-e/finetuning.git
cd finetuning

# Install dependencies
echo ""
echo "Installing dependencies (this takes 5-10 minutes)..."
uv sync

# Activate environment
source .venv/bin/activate

# Authenticate HuggingFace
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "HuggingFace "
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Get your token from: https://huggingface.co/settings/tokens"
huggingface-cli login

# Authenticate Wandb
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Wandb "
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Get your API key from: https://wandb.ai/authorize"
wandb login

# Test setup
echo ""
echo "Testing setup..."
python test_setup.py

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Setup complete!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Instance IP: $(curl -s ifconfig.me)"
echo ""
echo "Next steps on your LOCAL machine:"
echo "  1. Update ~/.ssh/config with IP: $(curl -s ifconfig.me)"
echo "  2. VSCode: Cmd+Shift+P → 'Remote-SSH: Connect to Host' → lambda-cloud"
echo "  3. Open folder: /home/ubuntu/finetuning"
echo "  4. Start coding!"
echo ""