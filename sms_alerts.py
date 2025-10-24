"""
SMS Trade Alerts Module
Send trade notifications to: 804-345-8049

Setup:
1. pip install twilio
2. Sign up at https://www.twilio.com/try-twilio
3. Get your Account SID, Auth Token, and Twilio phone number
4. Update credentials below
"""
from twilio.rest import Client
from typing import Dict, Optional
from datetime import datetime

class SMSAlerts:
    """Send SMS alerts for trading activities"""
    
    def __init__(
        self, 
        account_sid: str, 
        auth_token: str, 
        from_phone: str, 
        to_phone: str = "+18043458049"
    ):
        """
        Initialize SMS alerts
        
        Args:
            account_sid: Twilio Account SID
            auth_token: Twilio Auth Token
            from_phone: Your Twilio phone number (format: +1234567890)
            to_phone: Destination phone (default: 804-345-8049)
        """
        self.client = Client(account_sid, auth_token)
        self.from_phone = from_phone
        self.to_phone = to_phone
        print(f"✓ SMS Alerts initialized to {to_phone}")
    
    def send_trade_alert(self, trade: Dict):
        """
        Send SMS alert for new trade entry
        
        Args:
            trade: Dict with keys: symbol, price, shares, stop_loss, 
                   take_profit, signal_type, quality, risk_amount, risk_pct
        """
        message = f"""
🔔 TRADE SIGNAL - V4.8

Action: BUY
Symbol: {trade['symbol']}
Price: ${trade['price']:.2f}
Shares: {trade['shares']:.1f}
Stop Loss: ${trade['stop_loss']:.2f}
Take Profit: ${trade['take_profit']:.2f}

Signal: {trade['signal_type'].upper()}
Quality: {trade['quality']}/3 ⭐

Risk: ${trade['risk_amount']:.0f} ({trade['risk_pct']*100:.1f}%)
Time: {datetime.now().strftime('%I:%M %p')}
        """.strip()
        
        try:
            self.client.messages.create(
                body=message,
                from_=self.from_phone,
                to=self.to_phone
            )
            print(f"✓ Entry SMS sent for {trade['symbol']}")
        except Exception as e:
            print(f"✗ SMS failed: {e}")
    
    def send_exit_alert(self, exit: Dict):
        """
        Send SMS alert for trade exit
        
        Args:
            exit: Dict with keys: symbol, exit_price, pnl, return_pct, 
                  days_held, exit_reason
        """
        emoji = "✅" if exit['pnl'] > 0 else "❌"
        win_loss = "WIN" if exit['pnl'] > 0 else "LOSS"
        
        message = f"""
{emoji} TRADE EXIT - {win_loss}

Symbol: {exit['symbol']}
Exit Price: ${exit['exit_price']:.2f}
P&L: ${exit['pnl']:.2f}
Return: {exit['return_pct']:.1f}%

Days Held: {exit['days_held']}
Reason: {exit['exit_reason']}
Time: {datetime.now().strftime('%I:%M %p')}
        """.strip()
        
        try:
            self.client.messages.create(
                body=message,
                from_=self.from_phone,
                to=self.to_phone
            )
            print(f"✓ Exit SMS sent for {exit['symbol']}")
        except Exception as e:
            print(f"✗ SMS failed: {e}")
    
    def send_daily_summary(self, summary: Dict):
        """
        Send daily portfolio summary
        
        Args:
            summary: Dict with keys: equity, daily_pnl, daily_return, 
                     open_positions, cash, signals_today
        """
        emoji = "📈" if summary.get('daily_pnl', 0) >= 0 else "📉"
        
        message = f"""
{emoji} DAILY SUMMARY - V4.8

Portfolio: ${summary['equity']:,.0f}
Today P&L: ${summary['daily_pnl']:,.0f} ({summary['daily_return']:.2f}%)

Open Positions: {summary['open_positions']}
Available Cash: ${summary['cash']:,.0f}
Signals Today: {summary['signals_today']}

Date: {datetime.now().strftime('%m/%d/%Y')}
        """.strip()
        
        try:
            self.client.messages.create(
                body=message,
                from_=self.from_phone,
                to=self.to_phone
            )
            print("✓ Daily summary SMS sent")
        except Exception as e:
            print(f"✗ SMS failed: {e}")
    
    def send_alert(self, title: str, message: str):
        """
        Send custom alert
        
        Args:
            title: Alert title
            message: Alert message
        """
        full_message = f"""
🚨 {title}

{message}

Time: {datetime.now().strftime('%I:%M %p')}
        """.strip()
        
        try:
            self.client.messages.create(
                body=full_message,
                from_=self.from_phone,
                to=self.to_phone
            )
            print(f"✓ Alert SMS sent: {title}")
        except Exception as e:
            print(f"✗ SMS failed: {e}")
    
    def send_error_alert(self, error: str):
        """Send error alert"""
        self.send_alert("ERROR", error)
    
    def send_startup_alert(self):
        """Send alert when system starts"""
        self.send_alert(
            "SYSTEM STARTED",
            "V4.8 Trading System is now active and monitoring for signals."
        )
    
    def send_shutdown_alert(self):
        """Send alert when system stops"""
        self.send_alert(
            "SYSTEM STOPPED",
            "V4.8 Trading System has been stopped."
        )


# ============================================================================
# EMAIL-TO-SMS ALTERNATIVE (FREE)
# ============================================================================

class EmailToSMS:
    """
    Free alternative using email-to-SMS gateway
    Requires Gmail account and app password
    """
    
    def __init__(self, gmail_address: str, gmail_app_password: str, phone: str = "8043458049"):
        """
        Args:
            gmail_address: Your Gmail address
            gmail_app_password: Gmail app password (not regular password)
            phone: Phone number (10 digits, no formatting)
        """
        import smtplib
        from email.mime.text import MIMEText
        
        self.smtp_server = "smtp.gmail.com"
        self.smtp_port = 587
        self.from_email = gmail_address
        self.password = gmail_app_password
        self.phone = phone
        
        # Carrier gateways (will try all)
        self.gateways = [
            f"{phone}@vtext.com",      # Verizon
            f"{phone}@txt.att.net",     # AT&T
            f"{phone}@tmomail.net",     # T-Mobile
            f"{phone}@messaging.sprintpcs.com",  # Sprint
        ]
        
        print(f"✓ Email-to-SMS initialized to {phone}")
    
    def send(self, message: str):
        """Send SMS via email gateway"""
        import smtplib
        from email.mime.text import MIMEText
        
        # Keep message under 160 characters for SMS
        if len(message) > 160:
            message = message[:157] + "..."
        
        for gateway in self.gateways:
            try:
                msg = MIMEText(message)
                msg['From'] = self.from_email
                msg['To'] = gateway
                msg['Subject'] = ""  # No subject for SMS
                
                with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                    server.starttls()
                    server.login(self.from_email, self.password)
                    server.send_message(msg)
                
                print(f"✓ SMS sent via {gateway.split('@')[1]}")
                return True
            except:
                continue
        
        print("✗ SMS failed on all gateways")
        return False


# ============================================================================
# TEST SCRIPT
# ============================================================================

def test_sms():
    """Test SMS alerts with sample data"""
    
    print("\n" + "="*60)
    print("SMS ALERTS TEST")
    print("="*60 + "\n")
    
    # OPTION 1: Twilio (Recommended)
    print("Testing Twilio SMS...")
    print("\n⚠️  Update credentials below before running!")
    
    # UPDATE THESE:
    ACCOUNT_SID = "YOUR_ACCOUNT_SID_HERE"
    AUTH_TOKEN = "YOUR_AUTH_TOKEN_HERE"
    TWILIO_PHONE = "+1234567890"  # Your Twilio number
    
    if ACCOUNT_SID == "YOUR_ACCOUNT_SID_HERE":
        print("\n❌ Please update Twilio credentials in the script first!")
        print("\nSteps:")
        print("1. Sign up at https://www.twilio.com/try-twilio")
        print("2. Get your Account SID and Auth Token")
        print("3. Get a Twilio phone number")
        print("4. Update the credentials above")
        print("5. Run this script again")
        return
    
    try:
        sms = SMSAlerts(
            account_sid=ACCOUNT_SID,
            auth_token=AUTH_TOKEN,
            from_phone=TWILIO_PHONE,
            to_phone="+18048458049"
        )
        
        # Test 1: Startup alert
        print("\n1. Testing startup alert...")
        sms.send_startup_alert()
        
        # Test 2: Entry alert
        print("\n2. Testing entry alert...")
        sms.send_trade_alert({
            'symbol': 'NVDA',
            'price': 875.50,
            'shares': 2.5,
            'stop_loss': 862.37,
            'take_profit': 945.54,
            'signal_type': 'momentum',
            'quality': 3,
            'risk_amount': 200,
            'risk_pct': 0.02
        })
        
        # Test 3: Exit alert
        print("\n3. Testing exit alert...")
        sms.send_exit_alert({
            'symbol': 'NVDA',
            'exit_price': 920.00,
            'pnl': 445.00,
            'return_pct': 5.1,
            'days_held': 12,
            'exit_reason': 'Take Profit'
        })
        
        # Test 4: Daily summary
        print("\n4. Testing daily summary...")
        sms.send_daily_summary({
            'equity': 12445,
            'daily_pnl': 445,
            'daily_return': 3.71,
            'open_positions': 5,
            'cash': 2500,
            'signals_today': 3
        })
        
        print("\n✅ All tests complete! Check your phone (804-345-8049)")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Verify Account SID and Auth Token are correct")
        print("2. Make sure phone number is verified in Twilio")
        print("3. Check Twilio account balance")


if __name__ == "__main__":
    test_sms()