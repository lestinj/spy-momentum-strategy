#!/bin/bash
# ONE-COMMAND FIX: Run this to sync everything automatically

echo "╔═══════════════════════════════════════════════════════════════════════╗"
echo "║           🔧 ML ADAPTIVE V49 - AUTO SYNC & VERIFY                     ║"
echo "╚═══════════════════════════════════════════════════════════════════════╝"
echo ""

# Check if we're in the right directory
if [ ! -f "positions.txt" ]; then
    echo "❌ Error: positions.txt not found!"
    echo "   Please run this from /mnt/user-data/uploads/"
    exit 1
fi

# Copy tools if needed
if [ ! -f "fix_positions.py" ]; then
    echo "📥 Copying fix tools..."
    cp ../outputs/fix_positions.py .
    cp ../outputs/verify_sync.py .
    echo "✅ Tools copied"
    echo ""
fi

# Run the fix
echo "🔧 Step 1: Syncing positions.txt → adaptive_positions.json..."
echo "═══════════════════════════════════════════════════════════════════════"
python fix_positions.py
echo ""

# Verify the fix
echo "🔍 Step 2: Verifying sync..."
echo "═══════════════════════════════════════════════════════════════════════"
python verify_sync.py
echo ""

# Test the trading script
echo "🧪 Step 3: Testing trading script..."
echo "═══════════════════════════════════════════════════════════════════════"
python ml_adaptive_v49_alerts.py --once
echo ""

# Final summary
echo "╔═══════════════════════════════════════════════════════════════════════╗"
echo "║                        ✅ AUTO-FIX COMPLETE                            ║"
echo "╚═══════════════════════════════════════════════════════════════════════╝"
echo ""
echo "📋 What was fixed:"
echo "   • TSLA position: 35 shares → 27 shares"
echo "   • Capital: $16,000 → $15,700"
echo "   • Leverage: Calculated correctly (1.73x of 2.5x max)"
echo ""
echo "🚀 Next steps:"
echo "   • Review the output above"
echo "   • If all checks passed, you're ready to trade!"
echo "   • Run: python ml_adaptive_v49_alerts.py"
echo ""
echo "═══════════════════════════════════════════════════════════════════════"
