EA GlobalFlow Pro v8.0 - Complete 34 Entry Conditions
Reference

📋 Overview

This document contains all 34 entry conditions (17 Buy + 17 Sell) for EA GlobalFlow Pro v8.0, based on

the Ichimoku-TDI-PA-BBsqueeze-Kumocross strategy targeting 90%+ win rate.

🎯 Strategy Framework

Target Win Rate: 90%+ (95% with Triple Enhancement)

Markets: Indian F&O, Forex, International Commodities, CFDs

Entry Logic: Sequential execution (CT first, then PB)

One trade per condition per bar close

Minimum 10 conditions required for entry

📊 Indicator Settings (Confirmed)

1. Ichimoku Kinko Hyo: (9, 26, 52)

2. TDI (Ichi-Trader Dynamic Index):

RSI Period: 13

Volatility Band Period: 34

Price Line Period: 2

Signal Line Period: 7

Levels: 68 (overbought), 32 (oversold)

3. SMMA: Period 50

4. Bollinger Bands: (20, 2.0)

5. STR-ENTRY: ATR Period 20, Multiplier 1.0

6. STR-EXIT: ATR Period 20, Multiplier 1.5

🔄 Timeframe Cascade

Major Trend: Daily OR 4H

Middle Trend:

CT: 30min OR 1H

PB: 1H OR 30min

Entry:

CT: 5min OR 15min

PB: 15min OR 5min

📝 Notation System

SSB(26) = SSB 26 bars ago

rsig(1) = TDI RSI value 1 bar ago

MB(1) = TDI Main Band 1 bar ago

TSL(1) = TDI Signal Line 1 bar ago

VB_High/Low = TDI Volatility Bands

A. CONTINUATION TREND PATH (CT)

🔴 CT-SELL CONDITIONS (S1-S13)

BB_Squeeze Sell Group (S1-S4)

S1 - CT_BBSqueeze_Bear Complex

Conditions:

- (rsig(1) < MB(1) OR rsig(1) < TSL(1)) AND

- CSpan(26) < Low(26) AND

- Close(1) < SSB(1) AND

- SMMA50(1) <= SMMA50(3) AND

- Close(1) < SSA(1) AND

- Close(1) < SMMA50(1)

Action: Margin SELL Order S1

S2 - CT_BBBreak_Lower

Condition:

- Low(1) < LowerBoll(1)

Action: Margin SELL Order S2

S3 - CT_KumoBreak_Bear

Conditions:

- CSpan(26) < Close(26) AND

- (Close(1) < SSA(1) OR Close(1) < SSB(1)) AND

- Close(1) <= LowerBoll(2) AND

- Low(1) < Low(2) AND

- rsig(1) < TSL(1) AND

- Low(1) < LowerBoll(1)

Action: Margin SELL Order S3

S4 - CT_SMMA_Bear

Condition:

- Close(1) < SMMA50(1)

Action: Margin SELL Order S4

STR-ENTRY-1 Sell Group (S5-S8)

Prerequisite: MB(1) <= 50 AND MB(1) > 32 AND VB_Low(1) > 32 AND rsig(1) < 50

S5 - S8 Conditions

Base conditions for STR-ENTRY-1:

- MB(1) < MB(3) AND

- (Open(1) < SMMA50(1) OR Close(1) < SMMA50(1)) AND

- rsig(2) > rsig(1) AND

- STR_ENTRY(0) > Open(1)

Then execute based on:

S5: rsig(1) < TSL(1)

S6: rsig(1) < MB(1)

S7: TSL(1) < MB(1)

S8: rsig(1) < VB_Low(1)

STR-ENTRY-2 Sell Group (S9-S13)

Prerequisite: MB(1) >= 50 AND MB(1) < 68 AND VB_High(1) > 68 AND rsig(1) > 50

S9 - S13 Conditions

Base conditions for STR-ENTRY-2:

- MB(1) < MB(3) AND

- (Open(1) > SMMA50(1) OR Close(1) < SMMA50(1)) AND

- rsig(2) > rsig(1) AND

- STR_ENTRY(0) > Open(1)

Then execute based on:

S9: rsig(1) < TSL(1)

S10: rsig(1) < MB(1)

S11: TSL(1) < MB(1)

S12: rsig(1) < VB_Low(1)

S13: rsig(1) < VB_High(1)

🟢 CT-BUY CONDITIONS (B1-B13)

BB_Squeeze Buy Group (B1-B4)

B1 - CT_BBSqueeze_Bull Complex

Conditions:

- (rsig(1) > MB(1) OR rsig(1) > TSL(1)) AND

- CSpan(26) > Low(26) AND

- Close(1) > SSB(1) AND

- Close(1) > SSA(1) AND

- SMMA50(1) >= SMMA50(3) AND

- Close(1) > SMMA50(1)

Action: Margin BUY Order B1

B2 - CT_BBBreak_Upper

Condition:

- Low(1) > UpperBoll(1)

Action: Margin BUY Order B2

B3 - CT_KumoBreak_Bull

Conditions:

- CSpan(26) > Close(26) AND

- (Close(1) > SSA(1) OR Close(1) > SSB(1)) AND

- Close(1) >= UpperBoll(2) AND

- Low(1) > Low(2) AND

- rsig(1) > TSL(1) AND

- High(1) > UpperBoll(1)

Action: Margin BUY Order B3

B4 - CT_SMMA_Bull

Condition:

- Close(1) < SMMA50(1)  [Note: This appears to be an error in original, should verify]

Action: Margin BUY Order B4

STR-ENTRY-1 Buy Group (B5-B8)

Prerequisite: MB(1) >= 50 AND MB(1) < 68 AND VB_High(1) < 68 AND rsig(1) > 50

B5 - B8 Conditions

Base conditions for STR-ENTRY-1:

- MB(1) > MB(3) AND

- (Open(1) > SMMA50(1) OR Close(1) > SMMA50(1)) AND

- rsig(2) < rsig(1) AND

- STR_ENTRY(0) < Open(1)

Then execute based on:

B5: rsig(1) > TSL(1)

B6: rsig(1) > MB(1)

B7: TSL(1) > MB(1)

B8: rsig(1) > VB_High(1)

STR-ENTRY-2 Buy Group (B9-B13)

Prerequisite: MB(1) <= 50 AND MB(1) > 32 AND VB_Low(1) < 32 AND rsig(1) < 50

B9 - B13 Conditions

Base conditions for STR-ENTRY-2:

- MB(1) > MB(3) AND

- (Open(1) < SMMA50(1) OR Close(1) < SMMA50(1)) AND

- rsig(2) < rsig(1) AND

- STR_ENTRY(0) < Open(1)

Then execute based on:

B9: rsig(1) > MB(1)

B10: rsig(1) > VB_Low(1)

B11: TSL(1) > MB(1)

B12: rsig(1) > TSL(1)

B13: rsig(1) > VB_High(1)

B. PULLBACK PATH (PB)

🔴 PB-SELL CONDITIONS (S14-S17)

PB-Sell Major Trend Prerequisites (Daily/4H)

- SMMA50(1) > SSA(1) AND SMMA50(1) > SSB(1) AND

- High(2) > High(1) AND High(1) > High(0) AND

- High(1) > SSA(1) AND High(1) > SSB(1) AND

- High(2) > SSA(2) AND High(2) > SSB(2) AND

- TSL(2) > MB(1) AND rsig(2) > MB(2) AND

- KS(1) < TS(1) AND Close(1) < TS(1) AND

- (rsig(1) < MB(1) OR rsig(1) < VB_Low(1)) AND

- TSL(1) < MB(1)

PB-Sell Middle Trend Prerequisites (1H/30min)

- (Close(1) < TS(1) OR Close(1) < KS(1) OR

   Close(1) < SMMA50(1) OR TS(1) < SMMA50(1) OR

   TS(1) < KS(1)) AND

- (rsig(1) < MB(1) OR rsig(1) < TSL(1) OR rsig(1) < VB_Low(1))

PB-Sell Entry Prerequisites (15min/5min)

- (Close(1) > SSA(1) OR Close(1) > SSB(1)) AND

- High(2) > High(1) AND High(1) > High(0) AND

- (Close(1) < TS(1) OR Close(1) < KS(1) OR Close(1) < SMMA50(1))

S14-S17 Entry Triggers

S14: rsig(1) < VB_High(1)

S15: rsig(1) < TSL(1)

S16: rsig(1) < MB(1)

S17: rsig(1) < VB_Low(1)

🟢 PB-BUY CONDITIONS (B14-B17)

PB-Buy Major Trend Prerequisites (Daily/4H)

- SMMA50(1) < SSA(1) AND SMMA50(1) < SSB(1) AND

- Low(2) < Low(1) AND Low(1) < Low(0) AND

- Low(1) < SSA(1) AND Low(1) < SSB(1) AND

- Low(2) < SSA(2) AND Low(2) < SSB(2) AND

- KS(1) > TS(1) AND Close(1) > TS(1) AND

- TSL(1) < MB(1) AND rsig(2) < MB(2) AND

- (rsig(1) > MB(1) OR rsig(1) > VB_High(1))

PB-Buy Middle Trend Prerequisites (1H/30min)

- (Close(1) > TS(1) OR Close(1) > KS(1) OR

   Close(1) > SMMA50(1) OR TS(1) > SMMA50(1) OR

   TS(1) > KS(1)) AND

- (rsig(1) > MB(1) OR rsig(1) > TSL(1) OR rsig(1) > VB_High(1))

PB-Buy Entry Prerequisites (15min/5min)

- (Close(1) < SSA(1) OR Close(1) < SSB(1)) AND

- Low(2) < Low(1) AND Low(1) < Low(0) AND

- (Close(1) > TS(1) OR Close(1) > KS(1) OR Close(1) > SMMA50(1))

B14-B17 Entry Triggers

B14: rsig(1) > VB_High(1)

B15: rsig(1) > TSL(1)

B16: rsig(1) > MB(1)

B17: rsig(1) > VB_Low(1)

🚀 Triple Enhancement System

Layer 1: 34 Entry Conditions (65-70% win rate)

Foundation technical signals

Minimum 10 conditions required

Sequential execution system

Layer 2: ML Validation (+10-15% win rate)

Statistical filtering

Confidence scoring

85% minimum score required

Layer 3: Candlestick + Volume (+10-15% win rate)

Real-time sentiment confirmation

Volume validation

Pattern recognition

Total Target: 90-95% win rate

📌 Implementation Notes

1. Sequential Execution: Check conditions in order at each bar close

2. One Trade Per Condition: Each condition can trigger only once per strike per day

3. Path Priority: Always check CT path first, then PB if CT doesn't trigger

4. Exit Management: All positions exit via STR-EXIT on 1H timeframe

5. Stop Loss: Fixed at 0.25% below STR-EXIT line

6. F&O Specific:

Use hybrid OI approach for strike selection

ATM only on expiry day

Maximum 3 trades per secondary chart

🔧 Next Steps

1. Implement trend detection functions for Major/Middle levels

2. Create entry condition checking functions

3. Integrate with TrueData and Fyers APIs

4. Add ML enhancement layer

5. Implement candlestick pattern recognition

6. Create dashboard visualization

7. Add risk management controls

8. Test with paper trading mode