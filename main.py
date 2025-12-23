# region imports
from AlgorithmImports import *
import numpy as np
import pandas as pd
from QuantConnect.Indicators import SimpleMovingAverage, StandardDeviation
# endregion

class HipsterApricotHamster(QCAlgorithm):
    def initialize(self):
        self.set_start_date(2025, 12, 7)
        self.set_end_date(2025, 12, 14)
        self.set_cash(10000000)
        self.universe_settings.resolution = Resolution.DAILY
        self.set_universe_selection(QC500UniverseSelectionModel())

        self._spy = self.add_equity("SPY", Resolution.DAILY).symbol
        self.set_benchmark(self._spy)

        # SPY rolling window for regime detection
        self._spy_window = RollingWindow[float](22)

        # Rebalance settings
        self._rebalance_period = 35
        self._last_rebalance = None

        # Lookback periods
        self._lookback_mom = 100
        self._min_non_na_count = self._lookback_mom - 10
        self._lookback_ga = 160

        # GA parameter distributions (adaptive)
        self._mom_upper_mean = 0.8
        self._mom_lower_mean = 0.0
        self._mom_upper_sd = 0.15
        self._mom_lower_sd = 0.15
        self._rev_lower_mean = -0.8
        self._rev_lower_sd = 0.2

        # Stock lists and optimized parameters
        self.momentum = []
        self.reversion = []
        self.momentum_params = []
        self.reversion_params = []
        self.momentum_upper_k = []
        self.momentum_lower_k = []
        self.reversion_lower_k = []

        self._prices_arr: Dict[Symbol, list] = {}

        # Volatility targeting
        self._target_portfolio_vol = 0.12
        self._target_daily_vol = self._target_portfolio_vol / np.sqrt(252)

        # Trailing stops
        self._entry_prices: Dict[Symbol, float] = {}
        self._peak_prices: Dict[Symbol, float] = {}
        self._trailing_stop_pct = 0.10

        # Portfolio drawdown control
        self._portfolio_peak = 10000000
        self._max_portfolio_dd = 0.10
        self._critical_portfolio_dd = 0.16
        self._dd_recovery_threshold = 0.05
        self._in_drawdown_mode = False
        self._portfolio_trough = 10000000

        # Position-level hard stop
        self._hard_stop_pct = 0.08

        # Dynamic leverage bounds
        self._base_leverage = 7
        self._min_leverage = 1.0
        self._max_leverage = 4.0

        self.schedule.on(self.date_rules.every_day(self._spy),
                         self.time_rules.after_market_close(self._spy, 1),
                         self.update_regime_metrics)

    def rebalance(self) -> None:
        all_syms = [s for s in self.active_securities.keys() if s in self.securities and self.securities[s].type == SecurityType.EQUITY]
        symbols = [s for s in all_syms if s != self._spy][:400]

        if len(symbols) == 0: return

        hist = self.history(symbols, self._lookback_mom, Resolution.DAILY)
        if hist.empty: return
        prices_df = hist['close'].unstack(level=0)

        non_na_counts = prices_df.count()
        cols_to_keep = non_na_counts[non_na_counts >= self._min_non_na_count].index.tolist()
        prices_df = prices_df[cols_to_keep]
        if prices_df.empty: return

        cols = list(prices_df.columns)
        col_to_symbol = {i: cols[i] for i in range(len(cols))}

        # Select momentum stocks by autocorrelation
        num_stocks = 35
        momentum_idx = self.find_stocks_momentum(prices_df)[:num_stocks]
        self.momentum = [col_to_symbol[i] for i in momentum_idx if i in col_to_symbol]

        # Optimize parameters for each momentum stock
        if len(self.momentum) > 0:
            h_1 = self.history(self.momentum, self._lookback_ga, Resolution.DAILY)
            if not h_1.empty:
                self.prices_for_ga = h_1["close"].unstack(level=0).dropna(how='all').dropna(axis=1)
                self.momentum_params = []
                self.momentum_upper_k = []
                self.momentum_lower_k = []

                for i in range(min(num_stocks, len(self.momentum))):
                    param_scores = []

                    # Random search over 200 parameter samples
                    for _ in range(200):
                        upper_k = np.random.normal(self._mom_upper_mean, self._mom_upper_sd)
                        lower_k = np.random.normal(self._mom_lower_mean, self._mom_lower_sd)
                        if upper_k < lower_k: upper_k += 0.5

                        score = self.test_parameters_momentum_validated(upper_k, lower_k, i, self.prices_for_ga)
                        param_scores.append((upper_k, lower_k, score))

                    param_scores.sort(key=lambda x: x[2], reverse=True)
                    top_3 = param_scores[:3]
                    self.momentum_params.append(top_3)

                    best_u, best_l, _ = top_3[0]
                    self.momentum_upper_k.append(best_u)
                    self.momentum_lower_k.append(best_l)

            self.build_indicators(self.prices_for_ga)

            # Update adaptive distributions
            self._mom_upper_mean = np.mean(np.array(self.momentum_upper_k))
            self._mom_lower_mean = np.mean(np.array(self.momentum_lower_k))
            self._mom_upper_sd = np.std(np.array(self.momentum_upper_k))
            self._mom_lower_sd = np.std(np.array(self.momentum_lower_k))

        # Select mean reversion stocks by autocorrelation
        reversion_idx = self.find_stocks_reversion(prices_df)[:num_stocks]
        self.reversion = [col_to_symbol[i] for i in reversion_idx if i in col_to_symbol]

        # Optimize parameters for each reversion stock
        if len(self.reversion) > 0:
            h_2 = self.history(self.reversion, self._lookback_ga, Resolution.DAILY)
            if not h_2.empty:
                self.prices_for_ga_2 = h_2["close"].unstack(level=0).dropna(how='all').dropna(axis=1)
                self.reversion_params = []
                self.reversion_lower_k = []

                for i in range(min(num_stocks, len(self.reversion))):
                    param_scores = []

                    for _ in range(200):
                        lower_k = np.random.normal(self._rev_lower_mean, self._rev_lower_sd)

                        score = self.test_parameters_reversion_validated(lower_k, i, self.prices_for_ga_2)
                        param_scores.append((lower_k, score))

                    param_scores.sort(key=lambda x: x[1], reverse=True)
                    top_3 = param_scores[:3]
                    self.reversion_params.append(top_3)

                    best_l, _ = top_3[0]
                    self.reversion_lower_k.append(best_l)

                self.build_indicators_2(self.prices_for_ga_2)

                self._rev_lower_mean = np.mean(np.array(self.reversion_lower_k))
                self._rev_lower_sd = np.std(np.array(self.reversion_lower_k))

        self._last_rebalance = self.time

        # Liquidate positions no longer in either sleeve
        for held_symbol in list(self.portfolio.keys()):
            if held_symbol not in self.momentum and held_symbol not in self.reversion and self.portfolio[held_symbol].invested:
                self.liquidate(held_symbol)


    def on_securities_changed(self, changes: SecurityChanges) -> None:
        if self._last_rebalance is None:
            self.rebalance()

    def _psr_score(self, daily_returns: np.ndarray, directions: np.ndarray) -> float:
        """Composite fitness score: Sortino, Calmar, turnover penalty, stability bonus."""
        if len(daily_returns) < 60: return 0.0

        strat_rets = directions[:-1] * daily_returns[1:]

        # Sortino ratio
        downside_rets = strat_rets[strat_rets < 0]
        downside_std = np.std(downside_rets) if len(downside_rets) > 0 else 1e-6
        sortino = (np.mean(strat_rets) / downside_std) * np.sqrt(252)

        # Calmar ratio
        cum_rets = np.cumprod(1 + strat_rets)
        running_max = np.maximum.accumulate(cum_rets)
        drawdowns = (cum_rets - running_max) / running_max
        max_dd = abs(np.min(drawdowns)) if len(drawdowns) > 0 else 1.0
        calmar = (np.mean(strat_rets) * 252) / (max_dd + 0.01)

        # Turnover penalty (adaptive)
        turnover = np.sum(np.abs(np.diff(directions))) / len(directions)
        turnover_penalty = turnover * np.maximum(0.3, 1.0 - sortino/2.0)

        # Stability bonus (reward consistent rolling Sharpe)
        stability_bonus = 0.0
        if len(strat_rets) >= 60:
            window = 30
            rolling_sharpes = []
            for i in range(window, len(strat_rets)):
                window_rets = strat_rets[i-window:i]
                mu = np.mean(window_rets)
                std = np.std(window_rets)
                if std > 1e-6:
                    rolling_sharpe = (mu / std) * np.sqrt(252)
                    rolling_sharpes.append(rolling_sharpe)

            if len(rolling_sharpes) > 5:
                sharpe_volatility = np.std(rolling_sharpes)
                stability_bonus = 1.0 / (sharpe_volatility + 0.1)

        score = 0.45 * sortino + 0.25 * calmar - 0.20 * turnover_penalty + 0.1 * stability_bonus
        return score

    def test_parameters_momentum(self, upper_k: float, lower_k: float, index: int, prices_df: pd.DataFrame) -> float:
        """Test momentum parameters on historical data, return fitness score."""
        if index >= len(self.momentum): return 0.0
        s = self.momentum[index]
        if s not in prices_df.columns: return 0.0

        series = prices_df[s].values
        returns = pd.Series(series).pct_change().dropna()

        rolling_std = returns.rolling(30).std().shift(1).dropna()
        if rolling_std.empty: return 0.0

        start_index = returns.index.get_loc(rolling_std.index[0])
        aligned_returns = returns.iloc[start_index:].values
        aligned_vol = rolling_std.values

        # Z-score: return normalized by rolling volatility
        z = np.divide(aligned_returns, aligned_vol, out=np.zeros_like(aligned_returns), where=aligned_vol!=0)

        directions = np.zeros(len(z))
        hold = 0.0
        for i, score in enumerate(z):
            if score > upper_k: hold = 1.0   # Strong up move -> enter
            elif score < lower_k: hold = 0.0  # Weak move -> exit
            directions[i] = hold

        return self._psr_score(aligned_returns, directions)

    def test_parameters_reversion(self, lower_k: float, index: int, prices_df: pd.DataFrame) -> float:
        """Test mean reversion parameters on historical data, return fitness score."""
        if index >= len(self.reversion): return 0.0
        s = self.reversion[index]
        if s not in prices_df.columns: return 0.0

        series = prices_df[s].values
        returns = pd.Series(series).pct_change().dropna()
        rolling_std = returns.rolling(30).std().shift(1).dropna()
        if rolling_std.empty: return 0.0

        start_index = returns.index.get_loc(rolling_std.index[0])
        aligned_returns = returns.iloc[start_index:].values
        aligned_vol = rolling_std.values

        z = np.divide(aligned_returns, aligned_vol, out=np.zeros_like(aligned_returns), where=aligned_vol!=0)

        directions = np.zeros(len(z))
        for i, score in enumerate(z):
            if score < lower_k: directions[i] = 1.0

        return self._psr_score(aligned_returns, directions)

    def test_parameters_momentum_validated(self, upper_k: float, lower_k: float, index: int, prices_df: pd.DataFrame) -> float:
        """Walk-forward validation with 60/40 train/test split."""
        if index >= len(self.momentum): return 0.0
        s = self.momentum[index]
        if s not in prices_df.columns: return 0.0
        if len(prices_df) < 60: return 0.0

        train_size = int(len(prices_df) * 0.6)
        train_prices = prices_df.iloc[:train_size]
        test_prices = prices_df.iloc[train_size:]

        train_score = self.test_parameters_momentum(upper_k, lower_k, index, train_prices)
        test_score = self.test_parameters_momentum(upper_k, lower_k, index, test_prices)

        # Reject params that perform terribly out-of-sample
        if test_score < -0.5:
            return -999.0

        combined_score = 0.4 * train_score + 0.6 * test_score
        return combined_score

    def test_parameters_reversion_validated(self, lower_k: float, index: int, prices_df: pd.DataFrame) -> float:
        """Walk-forward validation with 60/40 train/test split."""
        if index >= len(self.reversion): return 0.0
        s = self.reversion[index]
        if s not in prices_df.columns: return 0.0
        if len(prices_df) < 60: return 0.0

        train_size = int(len(prices_df) * 0.6)
        train_prices = prices_df.iloc[:train_size]
        test_prices = prices_df.iloc[train_size:]

        train_score = self.test_parameters_reversion(lower_k, index, train_prices)
        test_score = self.test_parameters_reversion(lower_k, index, test_prices)

        if test_score < -0.5:
            return -999.0

        combined_score = 0.4 * train_score + 0.6 * test_score
        return combined_score

    def find_stocks_momentum(self, prices: pd.DataFrame) -> List[int]:
        """Select stocks with high autocorrelation (trending)."""
        valid = prices.iloc[-1] >= prices.iloc[0]
        prices = prices.loc[:, valid]
        returns = prices.pct_change()
        autocorr = returns.apply(lambda x: x.autocorr()).dropna()
        return [prices.columns.get_loc(c) for c in autocorr.sort_values(ascending=False).index[:50]]

    def find_stocks_reversion(self, prices: pd.DataFrame) -> List[int]:
        """Select stocks with low/negative autocorrelation (mean-reverting)."""
        returns = prices.pct_change()
        autocorr = returns.apply(lambda x: x.autocorr()).dropna()
        return [prices.columns.get_loc(c) for c in autocorr.sort_values(ascending=True).index[:50]]

    def build_indicators(self, prices: pd.DataFrame) -> None:
        self._prices_arr = {s: list(prices[s].values) for s in self.momentum if s in prices.columns}

    def build_indicators_2(self, prices: pd.DataFrame) -> None:
        for s in self.reversion:
            if s in prices.columns:
                self._prices_arr[s] = list(prices[s].values)

    def calculate_volatility_data(self, s: Symbol) -> Tuple[float, float, float]:
        """Returns (20d vol, autocorr, vol expansion ratio)."""
        if s not in self._prices_arr or len(self._prices_arr[s]) < 60:
            return 0.01, 0.0, 1.0

        prices = pd.Series(self._prices_arr[s])
        rets = prices.pct_change().dropna()
        if len(rets) < 30: return 0.01, 0.0, 1.0

        current_vol = rets.iloc[-20:].std()
        long_vol = rets.iloc[-60:].std()
        vol_ratio = current_vol / (long_vol + 1e-6)  # >1.5 = vol expanding (danger)

        autocorr = rets.autocorr()
        return current_vol, autocorr, vol_ratio

    def calculate_hrp_weights(self, symbols: List[Symbol]) -> Dict[Symbol, float]:
        """Hierarchical Risk Parity portfolio weights."""
        if len(symbols) == 0:
            return {}

        hist = self.history(symbols, 63, Resolution.DAILY)
        if hist.empty:
            equal_weight = 1.0 / len(symbols)
            return {s: equal_weight for s in symbols}

        try:
            returns = hist['close'].unstack(level=0).pct_change().dropna()

            if len(returns) < 20:
                equal_weight = 1.0 / len(symbols)
                return {s: equal_weight for s in symbols}

            valid_symbols = [s for s in symbols if s in returns.columns]
            if len(valid_symbols) < 2:
                equal_weight = 1.0 / len(valid_symbols) if valid_symbols else 0.0
                return {s: equal_weight for s in valid_symbols}

            returns = returns[valid_symbols]

            corr = returns.corr().fillna(0)
            dist = np.sqrt((1 - corr) / 2)

            from scipy.cluster.hierarchy import linkage
            from scipy.spatial.distance import squareform

            dist_condensed = squareform(dist.values, checks=False)
            link = linkage(dist_condensed, method='single')

            def get_cluster_var(cluster_items):
                if len(cluster_items) == 1:
                    return returns[cluster_items].var().values[0]
                cov = returns[cluster_items].cov()
                inv_diag = 1 / (np.diag(cov.values) + 1e-6)
                w = inv_diag / inv_diag.sum()
                return np.dot(w, np.dot(cov.values, w))

            weights = np.ones(len(valid_symbols)) / len(valid_symbols)

            def recursive_bisection(node_idx, current_weight):
                if node_idx < len(valid_symbols):
                    weights[node_idx] = current_weight
                else:
                    left_child = int(link[node_idx - len(valid_symbols), 0])
                    right_child = int(link[node_idx - len(valid_symbols), 1])

                    def get_cluster_symbols(idx):
                        if idx < len(valid_symbols):
                            return [valid_symbols[idx]]
                        left = int(link[idx - len(valid_symbols), 0])
                        right = int(link[idx - len(valid_symbols), 1])
                        return get_cluster_symbols(left) + get_cluster_symbols(right)

                    left_symbols = get_cluster_symbols(left_child)
                    right_symbols = get_cluster_symbols(right_child)

                    left_var = get_cluster_var(left_symbols)
                    right_var = get_cluster_var(right_symbols)

                    total_inv_var = (1.0 / (left_var + 1e-6)) + (1.0 / (right_var + 1e-6))
                    left_weight = current_weight * (1.0 / (left_var + 1e-6)) / total_inv_var
                    right_weight = current_weight * (1.0 / (right_var + 1e-6)) / total_inv_var

                    recursive_bisection(left_child, left_weight)
                    recursive_bisection(right_child, right_weight)

            if len(link) > 0:
                root_idx = len(valid_symbols) + len(link) - 1
                recursive_bisection(root_idx, 1.0)

            result = {valid_symbols[i]: max(0.005, min(0.04, weights[i])) for i in range(len(valid_symbols))}

            total_weight = sum(result.values())
            if total_weight > 0:
                result = {s: w / total_weight for s, w in result.items()}

            return result

        except Exception as e:
            self.debug(f"HRP calculation failed: {e}, falling back to equal weight")
            equal_weight = 1.0 / len(symbols)
            return {s: equal_weight for s in symbols}

    def calculate_signal_strength(self, s: Symbol) -> float:
        """Multi-timeframe volatility-adjusted momentum signal, bounded [-1, 1]."""
        if s not in self._prices_arr or len(self._prices_arr[s]) < 30:
            return 0.0

        prices = pd.Series(self._prices_arr[s])
        returns = prices.pct_change().dropna()

        if len(prices) < 20:
            return 0.0

        mom_5d = (prices.iloc[-1] / prices.iloc[-5] - 1.0) if len(prices) >= 5 else 0.0
        mom_10d = (prices.iloc[-1] / prices.iloc[-10] - 1.0) if len(prices) >= 10 else 0.0
        mom_20d = (prices.iloc[-1] / prices.iloc[-20] - 1.0) if len(prices) >= 20 else 0.0

        vol = returns.iloc[-20:].std() if len(returns) >= 20 else 0.01
        signal_5 = mom_5d / (vol + 1e-6)
        signal_10 = mom_10d / (vol + 1e-6)
        signal_20 = mom_20d / (vol + 1e-6)

        combined = 0.5 * signal_5 + 0.3 * signal_10 + 0.2 * signal_20
        return np.tanh(combined / 2.0)

    def get_ensemble_signal_momentum(self, symbol_idx: int) -> float:
        """Weighted ensemble signal from top-3 optimized parameter sets."""
        if symbol_idx >= len(self.momentum_params) or len(self.momentum_params[symbol_idx]) == 0:
            if symbol_idx < len(self.momentum):
                return self.calculate_signal_strength(self.momentum[symbol_idx])
            return 0.0

        s = self.momentum[symbol_idx]
        top_params = self.momentum_params[symbol_idx]

        signals = []
        weights = []

        for upper_k, lower_k, score in top_params:
            continuous_signal = self.calculate_signal_strength(s)

            if continuous_signal > 0:
                scaled_signal = continuous_signal * (1.0 + max(0, continuous_signal - lower_k) / (upper_k - lower_k + 1e-6))
            else:
                scaled_signal = continuous_signal * 0.5

            signals.append(scaled_signal)
            weights.append(max(score, 0.1))

        total_weight = sum(weights)
        if total_weight > 0:
            ensemble_signal = sum(s * w for s, w in zip(signals, weights)) / total_weight
            return np.clip(ensemble_signal, -1.0, 1.0)

        return 0.0

    def get_ensemble_signal_reversion(self, symbol_idx: int) -> float:
        """Weighted ensemble signal from top-3 optimized parameter sets for mean reversion."""
        if symbol_idx >= len(self.reversion_params) or len(self.reversion_params[symbol_idx]) == 0:
            if symbol_idx < len(self.reversion):
                return -self.calculate_signal_strength(self.reversion[symbol_idx])
            return 0.0

        s = self.reversion[symbol_idx]
        top_params = self.reversion_params[symbol_idx]

        signals = []
        weights = []

        for lower_k, score in top_params:
            continuous_signal = -self.calculate_signal_strength(s)

            if continuous_signal > 0:
                scaled_signal = continuous_signal * (1.0 + max(0, -lower_k - continuous_signal) / (abs(lower_k) + 1e-6))
            else:
                scaled_signal = continuous_signal * 0.5

            signals.append(scaled_signal)
            weights.append(max(score, 0.1))

        total_weight = sum(weights)
        if total_weight > 0:
            ensemble_signal = sum(s * w for s, w in zip(signals, weights)) / total_weight
            return np.clip(ensemble_signal, -1.0, 1.0)

        return 0.0



    def update_regime_metrics(self) -> None:
        if self.securities.contains_key(self._spy):
            self._spy_window.add(float(self.securities[self._spy].price))

    def get_portfolio_drawdown_multiplier(self) -> float:
        """Returns exposure multiplier based on portfolio drawdown."""
        current_value = self.portfolio.total_portfolio_value

        if current_value > self._portfolio_peak:
            self._portfolio_peak = current_value
            self._in_drawdown_mode = False

        current_dd = (self._portfolio_peak - current_value) / self._portfolio_peak

        if current_value < self._portfolio_trough:
            self._portfolio_trough = current_value

        if self._in_drawdown_mode:
            recovery = (current_value - self._portfolio_trough) / self._portfolio_trough
            if recovery > self._dd_recovery_threshold:
                self._in_drawdown_mode = False
                self._portfolio_trough = current_value

        if current_dd >= self._critical_portfolio_dd:
            self._in_drawdown_mode = True
            return 0.0

        if current_dd >= self._max_portfolio_dd:
            self._in_drawdown_mode = True
            scale = 1.0 - (current_dd - self._max_portfolio_dd) / (self._critical_portfolio_dd - self._max_portfolio_dd)
            return max(0.25, scale)

        if self._in_drawdown_mode:
            return 0.5

        return 1.0

    def get_dynamic_leverage(self) -> float:
        """Returns leverage scaled inversely to realized volatility."""
        if not self._spy_window.is_ready:
            return self._base_leverage

        prices = [x for x in self._spy_window]
        rets = np.diff(prices) / prices[1:]
        realized_vol = np.std(rets) * np.sqrt(252)

        vol_scalar = self._target_portfolio_vol / (realized_vol + 0.01)
        dynamic_lev = self._base_leverage * vol_scalar
        return np.clip(dynamic_lev, self._min_leverage, self._max_leverage)

    def check_hard_stop(self, symbol: Symbol, current_price: float) -> bool:
        if symbol not in self._entry_prices:
            return False
        entry_price = self._entry_prices[symbol]
        loss_pct = (current_price - entry_price) / entry_price
        return loss_pct < -self._hard_stop_pct

    def get_market_regime_multiplier(self) -> float:
        """Returns exposure multiplier based on SPY volatility and trend."""
        if not self._spy_window.is_ready:
            return 1.0

        prices = [x for x in self._spy_window]
        rets = np.diff(prices) / prices[1:]
        realized_vol = np.std(rets) * np.sqrt(252)

        sma_short = np.mean(prices[:5])
        sma_long = np.mean(prices[:22])
        trend_positive = sma_short > sma_long

        if realized_vol > 0.35 and not trend_positive:
            return 0.0
        elif realized_vol > 0.35:
            return 0.4
        elif realized_vol > 0.22:
            return 0.7 if trend_positive else 0.5
        elif realized_vol > 0.15:
            return 0.85 if trend_positive else 0.65
        return 1.0

    def on_data(self, data: Slice) -> None:
        if self._last_rebalance is None: return

        if (self.time - self._last_rebalance).days >= self._rebalance_period:
            self.rebalance()

        regime_mult = self.get_market_regime_multiplier()
        dd_mult = self.get_portfolio_drawdown_multiplier()
        dynamic_lev = self.get_dynamic_leverage()
        total_mult = regime_mult * dd_mult

        if total_mult == 0.0:
            self.liquidate()
            self._entry_prices.clear()
            self._peak_prices.clear()
            return

        all_active_symbols = list(set(self.momentum + self.reversion))
        hrp_weights = self.calculate_hrp_weights(all_active_symbols) if len(all_active_symbols) > 0 else {}

        # Momentum sleeve
        for i, s in enumerate(self.momentum):
            if not self.securities.contains_key(s) or not self.securities[s].has_data: continue

            price = float(self.securities[s].price)
            if s not in self._prices_arr: self._prices_arr[s] = []
            self._prices_arr[s].append(price)

            vol, corr, vol_expansion = self.calculate_volatility_data(s)
            current_holdings = self.securities[s].holdings.quantity

            if current_holdings > 0 and self.check_hard_stop(s, price):
                self.liquidate(s)
                self.log(f"Hard stop triggered for {s}: >{self._hard_stop_pct:.0%} loss from entry")
                if s in self._entry_prices: del self._entry_prices[s]
                if s in self._peak_prices: del self._peak_prices[s]
                continue

            if current_holdings > 0:
                if s not in self._peak_prices:
                    self._peak_prices[s] = price
                else:
                    self._peak_prices[s] = max(self._peak_prices[s], price)

                drawdown_from_peak = (price - self._peak_prices[s]) / self._peak_prices[s]
                if drawdown_from_peak < -self._trailing_stop_pct:
                    self.liquidate(s)
                    self.log(f"Trailing stop triggered for {s}: {drawdown_from_peak:.1%} from peak")
                    if s in self._entry_prices: del self._entry_prices[s]
                    if s in self._peak_prices: del self._peak_prices[s]
                    continue

            if corr < -0.1:
                if current_holdings > 0:
                    self.liquidate(s)
                    if s in self._entry_prices: del self._entry_prices[s]
                    if s in self._peak_prices: del self._peak_prices[s]
                continue

            signal_strength = self.get_ensemble_signal_momentum(i)
            base_weight = hrp_weights.get(s, 0.04)

            if signal_strength > 0.08:
                target_weight = signal_strength * base_weight * total_mult * dynamic_lev
                if current_holdings <= 0:
                    self.set_holdings(s, target_weight)
                    self._entry_prices[s] = price
                    self._peak_prices[s] = price
                else:
                    self.set_holdings(s, target_weight)
            elif signal_strength < -0.1 and current_holdings > 0:
                self.liquidate(s)
                if s in self._entry_prices: del self._entry_prices[s]
                if s in self._peak_prices: del self._peak_prices[s]

        # Reversion sleeve
        for i, s in enumerate(self.reversion):
            if not self.securities.contains_key(s) or not self.securities[s].has_data: continue

            price = float(self.securities[s].price)
            if s not in self._prices_arr: self._prices_arr[s] = []
            self._prices_arr[s].append(price)

            vol, corr, vol_expansion = self.calculate_volatility_data(s)
            current_holdings = self.securities[s].holdings.quantity

            if current_holdings > 0 and self.check_hard_stop(s, price):
                self.liquidate(s)
                self.log(f"Hard stop triggered for {s}: >{self._hard_stop_pct:.0%} loss from entry")
                if s in self._entry_prices: del self._entry_prices[s]
                if s in self._peak_prices: del self._peak_prices[s]
                continue

            if current_holdings > 0:
                if s not in self._peak_prices:
                    self._peak_prices[s] = price
                else:
                    self._peak_prices[s] = max(self._peak_prices[s], price)

                drawdown_from_peak = (price - self._peak_prices[s]) / self._peak_prices[s]
                if drawdown_from_peak < -self._trailing_stop_pct:
                    self.liquidate(s)
                    self.log(f"Trailing stop triggered for {s}: {drawdown_from_peak:.1%} from peak")
                    if s in self._entry_prices: del self._entry_prices[s]
                    if s in self._peak_prices: del self._peak_prices[s]
                    continue

            falling_knife = vol_expansion > 1.5
            mr_signal = self.get_ensemble_signal_reversion(i)
            base_weight = hrp_weights.get(s, 0.04)

            if mr_signal > 0.08 and not falling_knife:
                target_weight = mr_signal * base_weight * total_mult * dynamic_lev
                if current_holdings <= 0:
                    self.set_holdings(s, target_weight)
                    self._entry_prices[s] = price
                    self._peak_prices[s] = price
                else:
                    self.set_holdings(s, target_weight)
            elif mr_signal < -0.05 and current_holdings > 0:
                self.liquidate(s)
                if s in self._entry_prices: del self._entry_prices[s]
                if s in self._peak_prices: del self._peak_prices[s]