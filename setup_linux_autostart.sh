#!/bin/bash

# ML-Adaptive V49 Auto-Start Setup for Linux/Mac
# Sets up cron jobs for automatic morning checks

echo "========================================"
echo "ML-ADAPTIVE V49 AUTO-START SETUP"
echo "For Linux/Mac using cron"
echo "========================================"
echo

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Create the startup script
cat > ~/ml_v49_morning_check.sh << 'EOF'
#!/bin/bash

# ML-Adaptive V49 Morning Check Script
# Runs every market day morning

# Set your Python path and script location
PYTHON_PATH="/usr/bin/python3"  # Adjust if using virtual env
SCRIPT_PATH="$HOME/trading/ml_adaptive_v49_alerts_fixed.py"  # Adjust path
LOG_DIR="$HOME/trading/logs"

# Create log directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Set log file with date
LOG_FILE="$LOG_DIR/ml_v49_$(date +%Y%m%d).log"

echo "========================================" >> "$LOG_FILE"
echo "ML-V49 Morning Check - $(date)" >> "$LOG_FILE"
echo "========================================" >> "$LOG_FILE"

# Wait for network
sleep 10

# Run the check
cd "$(dirname "$SCRIPT_PATH")"
$PYTHON_PATH "$SCRIPT_PATH" --once >> "$LOG_FILE" 2>&1

# Optional: Send notification if trades found
if grep -q "BUY SIGNALS" "$LOG_FILE"; then
    # Mac notification
    if [[ "$OSTYPE" == "darwin"* ]]; then
        osascript -e 'display notification "New ML-V49 signals detected!" with title "Trading Alert"'
    fi
    # Linux notification (requires libnotify)
    if command -v notify-send &> /dev/null; then
        notify-send "Trading Alert" "New ML-V49 signals detected!"
    fi
fi

echo "Check completed at $(date)" >> "$LOG_FILE"
EOF

chmod +x ~/ml_v49_morning_check.sh
echo "✅ Created ~/ml_v49_morning_check.sh"

# Create cron entries
echo
echo "========================================"
echo "CRON SCHEDULE OPTIONS"
echo "========================================"
echo
echo "Choose your schedule (copy ONE line to crontab):"
echo
echo "# ML-Adaptive V49 Trading Alerts"
echo "# Option 1: Every weekday at 9:00 AM Eastern"
echo "0 9 * * 1-5 $HOME/ml_v49_morning_check.sh"
echo
echo "# Option 2: Every weekday at 9:30 AM Eastern (market open)"
echo "30 9 * * 1-5 $HOME/ml_v49_morning_check.sh"
echo
echo "# Option 3: Multiple checks per day"
echo "0 9 * * 1-5 $HOME/ml_v49_morning_check.sh   # Pre-market"
echo "30 9 * * 1-5 $HOME/ml_v49_morning_check.sh  # Market open"
echo "0 12 * * 1-5 $HOME/ml_v49_morning_check.sh  # Midday"
echo "30 15 * * 1-5 $HOME/ml_v49_morning_check.sh # Near close"
echo
echo "========================================"
echo "SETUP INSTRUCTIONS"
echo "========================================"
echo
echo "1. EDIT the script:"
echo "   nano ~/ml_v49_morning_check.sh"
echo "   - Update PYTHON_PATH if using virtual environment"
echo "   - Update SCRIPT_PATH to your ml_adaptive_v49_alerts_fixed.py location"
echo
echo "2. ADD to crontab:"
echo "   crontab -e"
echo "   - Add one of the schedule options above"
echo "   - Save and exit"
echo
echo "3. VERIFY it's scheduled:"
echo "   crontab -l"
echo
echo "4. CHECK logs at:"
echo "   ~/trading/logs/"
echo
echo "========================================"
echo "TIMEZONE NOTE"
echo "========================================"
echo "Cron uses system time. Check your timezone:"
echo "Current system time: $(date)"
echo "Timezone: $(date +%Z)"
echo
echo "For Eastern Time, you may need to adjust hours:"
echo "EST = UTC-5, EDT = UTC-4"
echo
echo "========================================"

# Create systemd service as alternative (for Linux)
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo
    echo "ALTERNATIVE: SYSTEMD TIMER (Linux only)"
    echo "========================================"
    
    cat > ml-v49-alerts.service << EOF
[Unit]
Description=ML-Adaptive V49 Trading Alerts
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
ExecStart=$HOME/ml_v49_morning_check.sh
User=$USER

[Install]
WantedBy=multi-user.target
EOF

    cat > ml-v49-alerts.timer << EOF
[Unit]
Description=Run ML-V49 Alerts every weekday morning
Requires=ml-v49-alerts.service

[Timer]
OnCalendar=Mon-Fri 09:00:00
Persistent=true

[Install]
WantedBy=timers.target
EOF

    echo "Created systemd service and timer files"
    echo
    echo "To use systemd instead of cron:"
    echo "  sudo cp ml-v49-alerts.* /etc/systemd/system/"
    echo "  sudo systemctl daemon-reload"
    echo "  sudo systemctl enable ml-v49-alerts.timer"
    echo "  sudo systemctl start ml-v49-alerts.timer"
    echo "  sudo systemctl status ml-v49-alerts.timer"
fi

echo
echo "✅ Setup complete!"
echo "========================================"
