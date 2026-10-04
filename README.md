# Genetic algorithmic trading

An adaptive quantitative trading system that continuously re-selects assets and re-tunes strategy parameters as market conditions change.

### Motivation

A central challenge in quantitative finance is dealing with a constantly changing market environment. Market regimes shift, correlations evolve, and strategies that worked in one period can suddenly stop working. The difficulty lies not only in developing strategies, but in fine-tuning their parameters for specific market conditions, asset groups, and more.

### Strategies
The system starts from two primitive base strategies:
- **Momentum**
- **Mean reversion**

On each periodic rebalance, it selects new ideal stocks for each strategy and new ideal strategy parameters for those stocks.

### Parameter Optimization
Parameters are chosen with a genetic framework, allowing the system to adapt to shifting market conditions.

### Portfolio Allocation
Capital is allocated across strategies using **Hierarchical Risk Parity (HRP)**.

### Risk Management
HRP is combined with several layers of risk controls, including:
- Drawdown circuit breaker
- Individual strategy drawdown controls
- Portfolio-level drawdown controls
- Falling knife guard
- Regime detection based on SPY volatility
