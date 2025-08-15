#!/bin/bash

# Install npm and Claude Code
# This script installs Node.js (which includes npm) and Claude Code

set -e  # Exit on any error

echo "🚀 Starting installation of npm and Claude Code..."

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Install Node.js and npm
echo "📦 Installing Node.js and npm..."

if command_exists node && command_exists npm; then
    echo "✅ Node.js and npm are already installed"
    echo "Node.js version: $(node --version)"
    echo "npm version: $(npm --version)"
else
    # Detect OS and install accordingly
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        if command_exists apt-get; then
            # Ubuntu/Debian
            echo "🐧 Detected Ubuntu/Debian system"
            apt-get update
            apt-get install -y curl
            curl -fsSL https://deb.nodesource.com/setup_lts.x | bash -
            apt-get install -y nodejs
        elif command_exists yum; then
            # CentOS/RHEL/Fedora
            echo "🐧 Detected CentOS/RHEL/Fedora system"
            curl -fsSL https://rpm.nodesource.com/setup_lts.x | bash -
            yum install -y nodejs npm
        elif command_exists pacman; then
            # Arch Linux
            echo "🐧 Detected Arch Linux system"
            pacman -S nodejs npm
        else
            echo "❌ Unsupported Linux distribution"
            exit 1
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        echo "🍎 Detected macOS system"
        if command_exists brew; then
            brew install node
        else
            echo "❌ Homebrew not found. Please install Homebrew first or install Node.js manually"
            echo "Visit: https://nodejs.org/en/download/"
            exit 1
        fi
    else
        echo "❌ Unsupported operating system: $OSTYPE"
        echo "Please install Node.js manually from: https://nodejs.org/en/download/"
        exit 1
    fi
fi

# Verify npm installation
if command_exists npm; then
    echo "✅ npm successfully installed"
    echo "npm version: $(npm --version)"
else
    echo "❌ npm installation failed"
    exit 1
fi

# Install Claude Code
echo "🤖 Installing Claude Code..."

if command_exists claude-code; then
    echo "✅ Claude Code is already installed"
    echo "Claude Code version: $(claude-code --version)"
else
    echo "📥 Installing Claude Code via npm..."
    npm install -g @anthropic-ai/claude-code
    
    # Verify installation
    if command_exists claude-code; then
        echo "✅ Claude Code successfully installed"
        echo "Claude Code version: $(claude-code --version)"
    else
        echo "❌ Claude Code installation failed"
        echo "💡 Try running: npm install -g @anthropic-ai/claude-code"
        exit 1
    fi
fi

echo ""
echo "🎉 Installation complete!"
echo ""
echo "📋 Next steps:"
echo "1. Set up your Anthropic API key:"
echo "   export ANTHROPIC_API_KEY='your-api-key-here'"
echo "   (Add this to your ~/.bashrc or ~/.zshrc for persistence)"
echo ""
echo "2. Start using Claude Code:"
echo "   claude-code --help"
echo ""
echo "📚 For more information about Claude Code, visit Anthropic's blog"