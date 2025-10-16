"""
Improved Main Trading Loop
Integrates SPY options data and macro indicators for live trading signals
"""
import time
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from data.improved_data_fetcher import DataFetcher
from indicators.momentum import compute_momentum_indicators, MomentumParams
from indicators.alignment_checker import compute_alignment, AlignmentParams


class LiveTrader:
    """Live trading signal generator with options and macro integration"""
    
    def __init__(self, check_interval_minutes=15):
        self.data_fetcher = DataFetcher()
        self.check_interval = check_interval_minutes
        
        # Parameters tuned for 2+ trades per week
        self.mom_params = MomentumParams(
            fast=10,
            slow=30,
            rsi_len=7,
            rsi_buy=55.0,
            rsi_sell=45.0,
            z_window=10,
            z_entry=0.3,
            z_exit=-0.1,
        )
        
        self.align_params = AlignmentParams(
            min_aligned=5,
            lookback=5
        )
        
        self.last_signal = None
        self.signal_count = 0
        
    def analyze_market_conditions(self) -> dict:
        """
        Comprehensive market analysis including:
        - Price momentum
        - Macro conditions
        - Options sentiment
        """
        analysis = {}
        
        # 1. Fetch price data (last 5 days, hourly)
        print("\nFetching SPY price data...")
        price_df = self.data_fetcher.fetch_spy_price(period='5d', interval='1h')
        
        if price_df.empty:
            print("❌ Failed to fetch price data")
            return None
        
        # Standardize columns
        price_df.columns = [c.lower() for c in price_df.columns]
        if 'adj close' in price_df.columns:
            price_df['close'] = price_df['adj close']
        
        current_price = price_df['close'].iloc[-1]
        analysis['current_price'] = current_price
        
        # 2. Compute momentum indicators
        print("Computing momentum indicators...")
        mom_df = compute_momentum_indicators(price_df, self.mom_params)
        
        # Combine for alignment calculation
        combined = price_df[['close']].join(mom_df, how='inner')
        align_df = compute_alignment(combined, self.align_params)
        combined = combined.join(align_df, how='left')
        
        # Latest momentum values
        latest = combined.iloc[-1]
        analysis['ema_fast'] = latest['ema_fast']
        analysis['ema_slow'] = latest['ema_slow']
        analysis['rsi'] = latest['rsi']
        analysis['zscore'] = latest['zscore']
        analysis['aligned_count'] = latest['aligned_count']
        
        # Momentum signals
        analysis['price_momentum'] = 'BULLISH' if latest['ema_fast'] > latest['ema_slow'] else 'BEARISH'
        analysis['rsi_signal'] = 'OVERBOUGHT' if latest['rsi'] > 70 else 'OVERSOLD' if latest['rsi'] < 30 else 'NEUTRAL'
        
        # 3. Fetch macro indicators
        print("Fetching macro indicators...")
        macro_signal = self.data_fetcher.get_latest_macro_signal()
        analysis['macro'] = macro_signal
        
        # 4. Fetch options data and sentiment
        print("Analyzing options market...")
        options_signals = self.data_fetcher.fetch_options_signals()
        analysis['options'] = options_signals
        
        return analysis
    
    def generate_trade_signal(self, analysis: dict) -> dict:
        """
        Generate actionable trading signals based on complete analysis
        Returns: dict with signal details
        """
        if not analysis:
            return {'signal': 'NO_SIGNAL', 'reason': 'Insufficient data'}
        
        signal = {
            'timestamp': datetime.now(),
            'signal': 'NO_SIGNAL',
            'direction': None,
            'strategy': None,
            'confidence': 0,
            'reasons': [],
        }
        
        # Extract key metrics
        price_bullish = analysis['price_momentum'] == 'BULLISH'
        rsi = analysis['rsi']
        zscore = analysis['zscore']
        aligned = analysis['aligned_count'] >= self.align_params.min_aligned
        
        macro_bullish = analysis['macro']['macro_bullish']
        macro_positive_count = analysis['macro']['positive_count']
        
        options_sentiment = analysis['options'].get('options_sentiment', 'neutral')
        pc_ratio = analysis['options'].get('pc_ratio_volume', 1.0)
        
        # Score calculation
        confidence_score = 0
        
        # Price momentum component (30 points)
        if price_bullish:
            confidence_score += 15
            signal['reasons'].append(f"Bullish price momentum (EMA fast > slow)")
        else:
            confidence_score += 15
            signal['reasons'].append(f"Bearish price momentum (EMA fast < slow)")
        
        if aligned:
            confidence_score += 15
            signal['reasons'].append(f"Strong alignment ({analysis['aligned_count']:.0f})")
        
        # RSI component (20 points)
        if rsi > self.mom_params.rsi_buy and price_bullish:
            confidence_score += 20
            signal['reasons'].append(f"Strong RSI signal ({rsi:.1f})")
        elif rsi < self.mom_params.rsi_sell and not price_bullish:
            confidence_score += 20
            signal['reasons'].append(f"Strong RSI signal ({rsi:.1f})")
        elif 45 < rsi < 55:
            confidence_score += 10  # Neutral RSI
        
        # Z-score component (20 points)
        if abs(zscore) > self.mom_params.z_entry:
            confidence_score += 20
            signal['reasons'].append(f"Strong momentum z-score ({zscore:.2f})")
        elif abs(zscore) > 0.1:
            confidence_score += 10
        
        # Macro component (15 points)
        if macro_bullish and macro_positive_count >= 3:
            confidence_score += 15
            signal['reasons'].append(f"Positive macro ({macro_positive_count} indicators)")
        elif macro_positive_count >= 2:
            confidence_score += 10
            signal['reasons'].append(f"Neutral macro ({macro_positive_count} indicators)")
        
        # Options sentiment component (15 points)
        if options_sentiment == 'bullish' and price_bullish:
            confidence_score += 15
            signal['reasons'].append(f"Bullish options sentiment (PC ratio: {pc_ratio:.2f})")
        elif options_sentiment == 'bearish' and not price_bullish:
            confidence_score += 15
            signal['reasons'].append(f"Bearish options sentiment (PC ratio: {pc_ratio:.2f})")
        elif options_sentiment == 'neutral':
            confidence_score += 7
        
        signal['confidence'] = confidence_score
        
        # Generate signal based on confidence and direction
        MIN_CONFIDENCE = 65  # Threshold for signal generation
        
        if confidence_score >= MIN_CONFIDENCE:
            if price_bullish:
                signal['signal'] = 'BUY'
                signal['direction'] = 'LONG'
                
                # Determine strategy
                if options_sentiment == 'bullish' and rsi < 60:
                    signal['strategy'] = 'CALL_SPREAD'
                    signal['reasons'].append("Recommended: Bull Call Spread")
                else:
                    signal['strategy'] = 'LONG_STOCK'
                    signal['reasons'].append("Recommended: Long Stock/Calls")
                    
            else:  # Bearish
                signal['signal'] = 'SELL'
                signal['direction'] = 'SHORT'
                
                # Determine strategy
                if options_sentiment == 'bearish' and rsi > 40:
                    signal['strategy'] = 'PUT_SPREAD'
                    signal['reasons'].append("Recommended: Bear Put Spread")
                else:
                    signal['strategy'] = 'SHORT_STOCK'
                    signal['reasons'].append("Recommended: Short Stock/Puts")
        
        else:
            signal['reasons'].append(f"Confidence too low ({confidence_score:.0f} < {MIN_CONFIDENCE})")
        
        return signal
    
    def print_analysis(self, analysis: dict, signal: dict):
        """Print formatted analysis and signal"""
        print(f"\n{'='*70}")
        print(f"  MARKET ANALYSIS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*70}\n")
        
        # Price and momentum
        print(f"PRICE DATA:")
        print(f"  SPY Price:           ${analysis['current_price']:.2f}")
        print(f"  Momentum:            {analysis['price_momentum']}")
        print(f"  RSI:                 {analysis['rsi']:.1f} ({analysis['rsi_signal']})")
        print(f"  Z-Score:             {analysis['zscore']:.2f}")
        print(f"  Alignment:           {analysis['aligned_count']:.0f}")
        
        # Macro
        macro = analysis['macro']
        print(f"\nMACRO CONDITIONS:")
        print(f"  Overall:             {'✓ BULLISH' if macro['macro_bullish'] else '✗ BEARISH'}")
        print(f"  Positive Signals:    {macro['positive_count']}/6")
        if macro['vix']:
            print(f"  VIX:                 {macro['vix']:.1f}")
        if macro['yield_curve']:
            print(f"  Yield Curve:         {macro['yield_curve']:.2f}%")
        
        # Options
        opts = analysis['options']
        if opts:
            print(f"\nOPTIONS SENTIMENT:")
            print(f"  Sentiment:           {opts.get('options_sentiment', 'N/A').upper()}")
            print(f"  P/C Ratio (Vol):     {opts.get('pc_ratio_volume', 0):.2f}")
            print(f"  P/C Ratio (OI):      {opts.get('pc_ratio_oi', 0):.2f}")
            if 'iv_skew' in opts:
                print(f"  IV Skew:             {opts['iv_skew']:.4f}")
            if 'max_pain' in opts:
                print(f"  Max Pain:            ${opts['max_pain']:.2f}")
        
        # Signal
        print(f"\n{'='*70}")
        print(f"TRADING SIGNAL:")
        print(f"  Signal:              {signal['signal']}")
        print(f"  Direction:           {signal['direction'] or 'N/A'}")
        print(f"  Strategy:            {signal['strategy'] or 'N/A'}")
        print(f"  Confidence:          {signal['confidence']:.0f}%")
        
        if signal['reasons']:
            print(f"\nReasoning:")
            for reason in signal['reasons']:
                print(f"  • {reason}")
        
        print(f"\n{'='*70}\n")
    
    def run_live_loop(self):
        """Main live trading loop"""
        print(f"\n🚀 Starting Live Trading Signal Generator")
        print(f"Check Interval: Every {self.check_interval} minutes")
        print(f"Press Ctrl+C to stop\n")
        
        while True:
            try:
                # Analyze market
                analysis = self.analyze_market_conditions()
                
                if analysis:
                    # Generate signal
                    signal = self.generate_trade_signal(analysis)
                    
                    # Print results
                    self.print_analysis(analysis, signal)
                    
                    # Track signals
                    if signal['signal'] != 'NO_SIGNAL':
                        self.signal_count += 1
                        print(f"📊 Total signals generated this session: {self.signal_count}")
                    
                    self.last_signal = signal
                else:
                    print("⚠️ Could not complete market analysis")
                
                # Wait for next check
                print(f"\n💤 Waiting {self.check_interval} minutes until next check...")
                time.sleep(self.check_interval * 60)
                
            except KeyboardInterrupt:
                print("\n\n👋 Shutting down gracefully...")
                print(f"Total signals generated: {self.signal_count}")
                break
            except Exception as e:
                print(f"\n❌ Error in main loop: {e}")
                print("Retrying in 1 minute...")
                time.sleep(60)


def main():
    """Entry point"""
    trader = LiveTrader(check_interval_minutes=15)
    trader.run_live_loop()


if __name__ == "__main__":
    main()
