#!/bin/bash

echo "🔍 Diagnosing Frontend Configuration..."
echo "========================================"
echo ""

echo "📁 Checking src/main.jsx:"
echo "------------------------"
docker compose exec frontend sh -c "cat src/main.jsx 2>/dev/null || echo 'File not found'"
echo ""

echo "📁 Checking src/index.css:"
echo "-------------------------"
docker compose exec frontend sh -c "cat src/index.css 2>/dev/null || echo 'File not found'"
echo ""

echo "📁 Checking postcss.config.js:"
echo "------------------------------"
docker compose exec frontend sh -c "cat postcss.config.js 2>/dev/null || echo 'File not found'"
echo ""

echo "📁 Checking vite.config.js:"
echo "---------------------------"
docker compose exec frontend sh -c "cat vite.config.js 2>/dev/null || echo 'File not found'"
echo ""

echo "📁 Directory structure:"
echo "----------------------"
docker compose exec frontend sh -c "ls -la src/"
echo ""

echo "📦 Tailwind version:"
echo "-------------------"
docker compose exec frontend sh -c "npm list tailwindcss"
echo ""

echo "========================================"
