#!/bin/bash
set -e

echo "🔧 Fixing Tailwind CSS v4 Configuration..."
echo "=========================================="
echo ""

# Tailwind v4 uses CSS-based configuration instead of tailwind.config.js
# Let's set it up properly

echo "🎨 Step 1/4: Creating Tailwind CSS v4 configuration..."
docker compose exec frontend sh -c "cat > src/index.css << 'EOFCSS'
@import \"tailwindcss\";

/* Your custom styles */
@layer base {
  ::-webkit-scrollbar {
    width: 8px;
    height: 8px;
  }

  ::-webkit-scrollbar-track {
    @apply bg-gray-100 rounded;
  }

  ::-webkit-scrollbar-thumb {
    @apply bg-gray-400 rounded;
  }

  ::-webkit-scrollbar-thumb:hover {
    @apply bg-gray-600;
  }

  html {
    scroll-behavior: smooth;
  }

  *, *::before, *::after {
    box-sizing: border-box;
  }
}

@layer utilities {
  .text-shadow {
    text-shadow: 0 2px 4px rgba(0,0,0,0.1);
  }
}
EOFCSS"
echo "✅ Tailwind CSS v4 configuration created"
echo ""

echo "🔗 Step 2/4: Ensuring main.jsx imports the CSS..."
docker compose exec frontend sh -c "cat > src/main.jsx << 'EOFMAIN'
import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.jsx'
import './index.css'

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)
EOFMAIN"
echo "✅ main.jsx updated"
echo ""

echo "⚙️  Step 3/4: Verifying PostCSS configuration..."
docker compose exec frontend sh -c "cat > postcss.config.js << 'EOFPOSTCSS'
export default {
  plugins: {
    tailwindcss: {},
    autoprefixer: {},
  },
}
EOFPOSTCSS"
echo "✅ PostCSS configured"
echo ""

echo "🔄 Step 4/4: Restarting frontend container..."
docker compose restart frontend
echo "✅ Frontend restarted"
echo ""

echo "=========================================="
echo "✅ Tailwind CSS v4 fix completed!"
echo ""
echo "📋 Next steps:"
echo "   1. Wait 5-10 seconds for Vite to rebuild"
echo "   2. Open http://localhost:5173"
echo "   3. Hard refresh: Ctrl+Shift+R (Windows/Linux) or Cmd+Shift+R (Mac)"
echo "   4. Clear browser cache if needed"
echo ""
echo "🔍 Verify with:"
echo "   docker compose logs frontend --tail=30"
echo ""
echo "✨ Your UI should now have beautiful Tailwind styling!"
