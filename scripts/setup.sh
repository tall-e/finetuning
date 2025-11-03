#!/bin/bash
# Save this in your repo: scripts/setup_new_instance.sh
# Run on fresh Lambda instance

set -e

echo "Setting up new Lambda instance..."

echo "Updating system..."
sudo apt-get update && sudo apt-get upgrade -y
sudo apt-get install -y git vim tmux htop

echo "Installing uv..."
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
    echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.bashrc
fi

echo "🔧 Configuring git..."
git_name="tall-e"                
git_email="powelltolly@gmail.com"             
git config --global user.name "$git_name"
git config --global user.email "$git_email"
git config --global credential.helper store
echo "Git configured as: $git_name <$git_email>"

# Clone repo with HTTPS
echo ""
echo "Cloning repository..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "You'll be prompted for GitHub credentials:"
echo "  Username: your-github-username"
echo "  Password: your-personal-access-token"
echo "           (NOT your GitHub password!)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
cd ~
git clone https://github.com/$git_name/finetuning.git  
cd finetuning

# Install dependencies
echo ""
echo "Installing dependencies..."
uv sync

# Activate environment
source .venv/bin/activate

# Authenticate HuggingFace
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔐 HuggingFace Authentication"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Get your token from: https://huggingface.co/settings/tokens"
huggingface-cli login

# Authenticate Wandb
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔐 Wandb Authentication"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Get your API key from: https://wandb.ai/authorize"
wandb login

# Test setup
echo ""
echo "Testing setup..."
python test_setup.py

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎉 Setup complete!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Instance IP: $(curl -s ifconfig.me)"
echo ""
echo "Next steps on your LOCAL machine:"
echo "  1. Update ~/.ssh/config with IP: $(curl -s ifconfig.me)"
echo "  2. VSCode: Cmd+Shift+P → 'Remote-SSH: Connect to Host' → lambda-gpu"
echo "  3. Open folder: /home/ubuntu/finetuning"
echo "  4. Start coding!"
echo ""