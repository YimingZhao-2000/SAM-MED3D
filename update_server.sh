#!/bin/bash

# Safe Server Update Script
# Uses fetch + reset instead of pull to avoid conflicts

echo "🔄 Safe Server Update"
echo "===================="

# Check if we're in a git repository
if [ ! -d ".git" ]; then
    echo "❌ Not in a git repository!"
    exit 1
fi

# Check current status
echo "📊 Current status:"
git status --porcelain

# Check for uncommitted changes
if [ -n "$(git status --porcelain)" ]; then
    echo ""
    echo "⚠️  Warning: You have uncommitted changes!"
    echo "Current changes:"
    git status --short
    
    read -p "Do you want to stash changes before updating? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "📦 Stashing changes..."
        git stash push -m "Auto-stash before server update $(date)"
        STASHED=true
    else
        echo "❌ Update cancelled due to uncommitted changes"
        exit 1
    fi
else
    STASHED=false
fi

# Fetch latest changes
echo ""
echo "📥 Fetching latest changes from remote..."
git fetch origin

# Check if we're behind
LOCAL=$(git rev-parse HEAD)
REMOTE=$(git rev-parse origin/main)

if [ "$LOCAL" = "$REMOTE" ]; then
    echo "✅ Already up to date!"
else
    echo "🔄 Updating to latest version..."
    echo "Local:  $LOCAL"
    echo "Remote: $REMOTE"
    
    # Reset to match remote exactly
    git reset --hard origin/main
    
    echo "✅ Successfully updated to latest version!"
fi

# Restore stashed changes if any
if [ "$STASHED" = true ]; then
    echo ""
    echo "📦 Restoring stashed changes..."
    if git stash pop; then
        echo "✅ Changes restored successfully!"
    else
        echo "⚠️  Warning: Could not restore stashed changes"
        echo "Use 'git stash list' and 'git stash pop' to restore manually"
    fi
fi

echo ""
echo "🎉 Server update completed!"
echo "📊 Final status:"
git status --porcelain 