#!/usr/bin/env python3
"""
BATCH STOCK EVALUATION FOR V4.8/V4.9 STRATEGY
Tests multiple candidate stocks to find which ones would improve your strategy
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class StockEvaluator:
    def __init__(self, initial_capital=7000):
        self.initial_capital = initial_capital
        
        # V4 Strategy Parameters
        self.rsi_period = 14
        self.rsi_buy = 55
        self.rsi_sell = 45
        self.ma_fast = 10
        self.ma_slow = 30
        self.stop_loss_pct = 0.08
        self.take_profit_pct = 0.25
        self.leverage = 2.0
        
        # Candidate stocks to evaluate
        self.candidates = {
            # AI/Cloud Infrastructure
            'SMCI': 'Super Micro Computer - AI servers',
            'MSTR': 'MicroStrategy - Bitcoin proxy',
            'CRWD': 'CrowdStrike - Cybersecurity',
            'SNOW': 'Snowflake - Data analytics',
            'NET': 'Cloudflare - Edge computing',
            
            # High-Beta Tech
            'SHOP': 'Shopify - E-commerce',
            'SQ': 'Block (Square) - Fintech',
            'ROKU': 'Roku - Streaming',
            
            # Semiconductors
            'MRVL': 'Marvell - Semiconductors',
            'AVGO': 'Broadcom - Semiconductors',
            
            # Crypto/Blockchain
            'RIOT': 'Riot Platforms - Bitcoin mining',
            'MARA': 'Marathon Digital - Bitcoin mining',
            
            # Growth Tech
            'DDOG': 'Datadog - Monitoring',
            'MDB': 'MongoDB - Database',
            'ZS': 'Zscaler - Cloud security',
            
            # Nuclear/Energy (per previous discussion)
            'CEG': 'Constellation Energy - Nuclear'
        }
    
    def calculate_rsi(self, prices, period=14):
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def check_trend_follow(self, row, prev_row):
        """TREND_FOLLOW strategy"""
        try:
            rsi = float(row['RSI']) if hasattr(row['RSI'], 'item') else row['RSI']
            close = float(row['Close']) if hasattr(row['Close'], 'item') else row['Close']
            ma_fast = float(row['MA_Fast']) if hasattr(row['MA_Fast'], 'item') else row['MA_Fast']
            ma_slow = float(row['MA_Slow']) if hasattr(row['MA_Slow'], 'item') else row['MA_Slow']
            prev_rsi = float(prev_row['RSI']) if hasattr(prev_row['RSI'], 'item') else prev_row['RSI']
            
            if pd.isna(prev_rsi):
                return False
            return (rsi > 55 and close > ma_fast and close > ma_slow and ma_fast > ma_slow)
        except:
            return False
    
    def check_pullback(self, row, prev_row):
        """PULLBACK strategy"""
        try:
            rsi = float(row['RSI']) if hasattr(row['RSI'], 'item') else row['RSI']
            close = float(row['Close']) if hasattr(row['Close'], 'item') else row['Close']
            ma_fast = float(row['MA_Fast']) if hasattr(row['MA_Fast'], 'item') else row['MA_Fast']
            ma_slow = float(row['MA_Slow']) if hasattr(row['MA_Slow'], 'item') else row['MA_Slow']
            prev_rsi = float(prev_row['RSI']) if hasattr(prev_row['RSI'], 'item') else prev_row['RSI']
            
            if pd.isna(prev_rsi):
                return False
            return (45 <= rsi <= 55 and close > ma_slow and ma_fast > ma_slow and prev_rsi > 55)
        except:
            return False
    
    def evaluate_stock(self, symbol, description, start_date='2021-01-01'):
        """Evaluate a single stock"""
        try:
            # Download data
            df = yf.download(symbol, start=start_date, progress=False)
            
            if len(df) < 100:
                return None
            
            # Calculate indicators
            df['RSI'] = self.calculate_rsi(df['Close'], self.rsi_period)
            df['MA_Fast'] = df['Close'].rolling(self.ma_fast).mean()
            df['MA_Slow'] = df['Close'].rolling(self.ma_slow).mean()
            df = df.dropna()
            
            # Run backtest
            capital = self.initial_capital
            position = None
            trades = []
            
            for idx in range(1, len(df)):
                date = df.index[idx]
                row = df.iloc[idx]
                prev_row = df.iloc[idx-1]
                
                # Exit logic
                if position:
                    entry_price = position['entry_price']
                    current_price = float(row['Close']) if hasattr(row['Close'], 'item') else row['Close']
                    pnl_pct = (current_price - entry_price) / entry_price
                    rsi = float(row['RSI']) if hasattr(row['RSI'], 'item') else row['RSI']
                    
                    exit_signal = False
                    exit_reason = None
                    
                    if pnl_pct <= -self.stop_loss_pct:
                        exit_signal = True
                        exit_reason = 'STOP'
                    elif pnl_pct >= self.take_profit_pct:
                        exit_signal = True
                        exit_reason = 'PROFIT'
                    elif rsi < self.rsi_sell:
                        exit_signal = True
                        exit_reason = 'RSI'
                    
                    if exit_signal:
                        shares = position['shares']
                        exit_value = shares * current_price
                        pnl = exit_value - position['entry_value']
                        capital += pnl
                        
                        trades.append({
                            'pnl': pnl,
                            'pnl_pct': pnl_pct * 100,
                            'exit_reason': exit_reason
                        })
                        position = None
                
                # Entry logic (V4.9 winners only)
                if not position:
                    signal = False
                    
                    if self.check_trend_follow(row, prev_row):
                        signal = True
                    elif self.check_pullback(row, prev_row):
                        signal = True
                    
                    if signal:
                        current_price = float(row['Close']) if hasattr(row['Close'], 'item') else row['Close']
                        position_value = capital * self.leverage
                        shares = int(position_value / current_price)
                        
                        if shares > 0:
                            position = {
                                'entry_price': current_price,
                                'entry_value': shares * current_price,
                                'shares': shares
                            }
            
            # Close final position
            if position:
                final_price = float(df['Close'].iloc[-1])
                shares = position['shares']
                pnl = (shares * final_price) - position['entry_value']
                pnl_pct = (final_price - position['entry_price']) / position['entry_price']
                capital += pnl
                trades.append({'pnl': pnl, 'pnl_pct': pnl_pct * 100, 'exit_reason': 'END'})
            
            if len(trades) == 0:
                return None
            
            # Calculate metrics
            trades_df = pd.DataFrame(trades)
            total_return = ((capital - self.initial_capital) / self.initial_capital) * 100
            wins = len(trades_df[trades_df['pnl'] > 0])
            win_rate = (wins / len(trades_df)) * 100
            avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if wins > 0 else 0
            avg_loss = trades_df[trades_df['pnl'] <= 0]['pnl'].mean() if wins < len(trades_df) else 0
            
            # Get price stats
            start_price = float(df['Close'].iloc[0])
            end_price = float(df['Close'].iloc[-1])
            price_return = ((end_price - start_price) / start_price) * 100
            
            return {
                'symbol': symbol,
                'description': description,
                'trades': len(trades_df),
                'win_rate': win_rate,
                'total_return': total_return,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'final_capital': capital,
                'price_return': price_return,
                'data_days': len(df)
            }
            
        except Exception as e:
            print(f"  ✗ {symbol}: {str(e)[:50]}")
            return None
    
    def run_evaluation(self):
        """Evaluate all candidates"""
        print("\n" + "="*90)
        print("EVALUATING CANDIDATE STOCKS FOR V4.9 MOMENTUM STRATEGY")
        print("Testing with V4.9 winners (TREND_FOLLOW + PULLBACK only)")
        print("="*90)
        
        results = []
        
        print(f"\n📊 Testing {len(self.candidates)} candidate stocks...\n")
        
        for symbol, description in self.candidates.items():
            print(f"  Testing {symbol:6} - {description[:40]:40}...", end=' ')
            result = self.evaluate_stock(symbol, description)
            
            if result:
                print(f"✓ {result['trades']:3d} trades, {result['total_return']:+6.1f}% return")
                results.append(result)
            else:
                print("✗ Insufficient data/trades")
        
        if len(results) == 0:
            print("\n❌ No stocks passed evaluation")
            return
        
        # Sort by total return
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('total_return', ascending=False)
        
        # Print results
        print("\n" + "="*90)
        print("RESULTS - RANKED BY TOTAL RETURN")
        print("="*90)
        print(f"\n{'Rank':<6} {'Symbol':<8} {'Trades':<8} {'Win%':<8} {'Return':<12} "
              f"{'Avg Win':<12} {'Avg Loss':<12} {'Rating':<10}")
        print("-"*90)
        
        for idx, row in results_df.iterrows():
            # Calculate rating
            rating = "🔥 EXCELLENT" if row['total_return'] > 50 and row['win_rate'] > 60 else \
                     "✅ GOOD" if row['total_return'] > 20 and row['win_rate'] > 50 else \
                     "⚠️  MARGINAL" if row['total_return'] > 0 else \
                     "❌ POOR"
            
            rank = results_df['total_return'].rank(ascending=False).loc[idx]
            
            print(f"{int(rank):<6} {row['symbol']:<8} {row['trades']:<8} "
                  f"{row['win_rate']:<7.1f}% {row['total_return']:>+10.1f}%  "
                  f"${row['avg_win']:>+9.0f}   ${row['avg_loss']:>+9.0f}   {rating}")
        
        # Top recommendations
        print("\n" + "="*90)
        print("TOP 5 RECOMMENDATIONS")
        print("="*90 + "\n")
        
        top_5 = results_df.head(5)
        
        for idx, row in top_5.iterrows():
            print(f"#{int(results_df['total_return'].rank(ascending=False).loc[idx])}: {row['symbol']} - {row['description']}")
            print(f"    Return: {row['total_return']:+.1f}% | Win Rate: {row['win_rate']:.1f}% | "
                  f"{row['trades']} trades")
            print(f"    Avg Win: ${row['avg_win']:+,.0f} | Avg Loss: ${row['avg_loss']:+,.0f}")
            
            # Comparison to buy-and-hold
            outperformance = row['total_return'] - row['price_return']
            if outperformance > 0:
                print(f"    💡 Strategy outperformed buy-and-hold by {outperformance:+.1f}%")
            else:
                print(f"    ⚠️  Buy-and-hold was better by {abs(outperformance):.1f}%")
            print()
        
        # Summary
        positive_returns = len(results_df[results_df['total_return'] > 0])
        avg_return = results_df['total_return'].mean()
        
        print("="*90)
        print("SUMMARY")
        print("="*90)
        print(f"  Stocks tested:      {len(results_df)}")
        print(f"  Positive returns:   {positive_returns} ({positive_returns/len(results_df)*100:.0f}%)")
        print(f"  Average return:     {avg_return:+.1f}%")
        print(f"  Best performer:     {results_df.iloc[0]['symbol']} ({results_df.iloc[0]['total_return']:+.1f}%)")
        print(f"  Worst performer:    {results_df.iloc[-1]['symbol']} ({results_df.iloc[-1]['total_return']:+.1f}%)")
        
        print("\n💡 RECOMMENDATION:")
        excellent_picks = results_df[(results_df['total_return'] > 50) & (results_df['win_rate'] > 60)]
        
        if len(excellent_picks) > 0:
            print(f"\n   ✅ Add these {len(excellent_picks)} stocks to your strategy:")
            for idx, row in excellent_picks.iterrows():
                print(f"      • {row['symbol']} ({row['description'][:40]})")
            print(f"\n   These showed >50% returns with >60% win rates")
        else:
            good_picks = results_df[(results_df['total_return'] > 20) & (results_df['win_rate'] > 50)]
            if len(good_picks) > 0:
                print(f"\n   ✅ Consider adding these {len(good_picks)} stocks:")
                for idx, row in good_picks.iterrows():
                    print(f"      • {row['symbol']} ({row['description'][:40]})")
            else:
                print("\n   ⚠️  No strong candidates found in this batch")
                print("      Current holdings (NVDA, TSLA, PLTR, AMD, COIN) may be optimal")
        
        print("\n" + "="*90 + "\n")

if __name__ == "__main__":
    evaluator = StockEvaluator()
    evaluator.run_evaluation()
