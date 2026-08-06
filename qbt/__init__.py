"""qbt -- a signal-engine / backtester pair with a single shared interface.

The organising idea: a strategy is a pure function from a *sliced* price panel
to target weights. The backtester and the live runner both call that same
function, so there is exactly one implementation of your logic.

    from qbt import (
        SyntheticRepository, CrossSectionalMomentum, RiskGate,
        Backtester, CostModel, ExecutionConfig,
    )

    panel = SyntheticRepository().fetch()
    result = Backtester(
        panel=panel,
        strategy=CrossSectionalMomentum(lookback=126, top_n=10),
        risk_gate=RiskGate(target_vol=0.12),
    ).run()
    print(result.summary())

Swap ``SyntheticRepository`` for ``OpenBBRepository`` to run on real data;
nothing else changes.
"""

from .broker import (
    BrokerAccount,
    BrokerAdapter,
    BrokerOrder,
    MockBroker,
    RobinhoodMCPBroker,
    ToolBinding,
)
from .corporate import CorpsPanel, CorpsRepository
from .data import (
    OpenBBRepository,
    PricePanel,
    PriceRepository,
    SyntheticRepository,
    align_panels,
)
from .engine import (
    Backtester,
    BacktestResult,
    CostModel,
    ExecutionConfig,
    compare,
    drawdown_series,
    rebalance_dates,
    summary_stats,
)
from .fundamentals import FundamentalsPanel, FundamentalsRepository
from .live import LivePlan, LiveSignalRunner, OrderIntent, PortfolioState
from .macro import MacrosPanel, MacrosRepository
from .options import OptionsPanel, OptionsRepository, derive_indicators as derive_options_indicators
from .orders import AuditLog, ExecutionPolicy, ExecutionReport, OrderManager
from .research import (
    ParameterSweep,
    expected_max_sharpe,
    forward_returns,
    ic_grid,
    ic_summary,
    information_coefficient,
    return_autocorrelation,
    sharpe_haircut,
    trailing_signal,
    walk_forward_splits,
)
from .risk import DayTradeLedger, RiskContext, RiskDecision, RiskGate
from .signals import (
    CalendarSeasonality,
    Composite,
    CrossSectionalMomentum,
    EqualWeightBuyHold,
    FundamentalsValueFilter,
    InsiderEventDrift,
    InverseVolWeighted,
    MacroRegimeFilter,
    MultiFactorCrossSectional,
    OptionsMeanReversion,
    PairsTrading,
    RiskParityAllocation,
    ShortHorizonReversal,
    Strategy,
    TimeSeriesMomentum,
    TrendFilter,
)

__version__ = "0.1.0"

__all__ = [
    "PricePanel", "PriceRepository", "OpenBBRepository", "SyntheticRepository",
    "align_panels",
    "FundamentalsPanel", "FundamentalsRepository",
    "MacrosPanel", "MacrosRepository",
    "CorpsPanel", "CorpsRepository",
    "OptionsPanel", "OptionsRepository", "derive_options_indicators",
    "Strategy", "EqualWeightBuyHold", "CrossSectionalMomentum",
    "TimeSeriesMomentum", "ShortHorizonReversal", "TrendFilter", "Composite",
    "FundamentalsValueFilter", "MacroRegimeFilter",
    "InverseVolWeighted", "PairsTrading", "MultiFactorCrossSectional",
    "CalendarSeasonality", "RiskParityAllocation", "OptionsMeanReversion",
    "InsiderEventDrift",
    "RiskGate", "RiskContext", "RiskDecision", "DayTradeLedger",
    "Backtester", "BacktestResult", "CostModel", "ExecutionConfig",
    "rebalance_dates", "summary_stats", "drawdown_series", "compare",
    "forward_returns", "trailing_signal", "information_coefficient",
    "ic_summary", "ic_grid", "return_autocorrelation", "walk_forward_splits",
    "expected_max_sharpe", "sharpe_haircut", "ParameterSweep",
    "LiveSignalRunner", "LivePlan", "OrderIntent", "PortfolioState",
    "BrokerAdapter", "BrokerAccount", "BrokerOrder", "MockBroker",
    "RobinhoodMCPBroker", "ToolBinding",
    "OrderManager", "ExecutionPolicy", "ExecutionReport", "AuditLog",
]
