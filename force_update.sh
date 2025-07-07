#!/bin/bash

# Force Update Script for Server
# Discards all local changes and syncs with remote

echo "🚀 Force Server Update"
echo "====================="

# Check if we're in a git repository
if [ ! -d ".git" ]; then
    echo "❌ Not in a git repository!"
    exit 1
fi

# Show current status
echo "📊 Current status:"
git status --short

# Fetch latest changes
echo ""
echo "📥 Fetching latest changes..."
git fetch origin

# Force reset to match remote exactly
echo "🔄 Force resetting to remote..."
git reset --hard origin/main

# Clean any untracked files (optional)
echo "🧹 Cleaning untracked files..."
git clean -fd

echo ""
echo "✅ Force update completed!"
echo "📊 Final status:"
git status --short

echo ""
echo "🎉 Server is now exactly matching GitHub!" 