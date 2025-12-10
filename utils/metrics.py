import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from configs.config_revised import ALPHA

def calculate_cumulative_return(returns: np.ndarray) -> float:
    """Calculate cumulative return"""
    return np.prod(1 + returns) - 1

def calculate_annualized_return(returns: np.ndarray, periods_per_year: int = 252) -> float:
    """Calculate annualized return"""
    cum_return = calculate_cumulative_return(returns)
    n_periods = len(returns)
    return (1 + cum_return) ** (periods_per_year / n_periods) - 1

def calculate_volatility(returns: np.ndarray, periods_per_year: int = 252) -> float:
    """Calculate annualized volatility"""
    return np.std(returns, ddof=1) * np.sqrt(periods_per_year)

def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0, 
                          periods_per_year: int = 252) -> float:
    """Calculate Sharpe ratio"""
    excess_returns = returns - risk_free_rate / periods_per_year
    if np.std(excess_returns, ddof=1) == 0:
        return 0.0
    return np.mean(excess_returns) / np.std(excess_returns, ddof=1) * np.sqrt(periods_per_year)

def calculate_sortino_ratio(returns: np.ndarray, risk_free_rate: float = 0.0, 
                            periods_per_year: int = 252) -> float:
    """
    Calculate annualized Sortino ratio
    
    Args:
        returns: Array of periodic returns
        risk_free_rate: Annual risk-free rate (target return)
        periods_per_year: Number of periods per year
        
    Returns:
        Annualized Sortino ratio
    """
    
    excess_returns = returns - risk_free_rate / periods_per_year
    avg_periodic_excess_return = np.mean(excess_returns)
    
    downside_squares = np.where(excess_returns < 0, excess_returns**2, 0.0)
    mean_downside_squares = np.mean(downside_squares)
    
    periodic_downside_deviation = np.sqrt(mean_downside_squares)
    
    if periodic_downside_deviation == 0:
        return 0.0

    sortino_ratio = (avg_periodic_excess_return / periodic_downside_deviation) * np.sqrt(periods_per_year)
    
    return sortino_ratio

def calculate_max_drawdown(returns: np.ndarray) -> float:
    """Calculate maximum drawdown"""
    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    return np.min(drawdown)

def calculate_calmar_ratio(returns: np.ndarray, periods_per_year: int = 252) -> float:
    """Calculate Calmar ratio"""
    annualized_return = calculate_annualized_return(returns, periods_per_year)
    max_drawdown = calculate_max_drawdown(returns)
    
    if max_drawdown == 0.0:
        return np.nan
        
    return annualized_return / abs(max_drawdown)

def calculate_turnover(weights_history) -> List:
    """
    Calculate average portfolio turnover
    
    Args:
        weights_history: List or array of weight vectors
        
    Returns:
        Average turnover across rebalancing periods
    """
    if len(weights_history) < 2:
        return 0.0
    
    # Convert to numpy array if it's a list
    if isinstance(weights_history, list):
        weights_history = np.array(weights_history)
    
    turnovers = []
    for i in range(1, len(weights_history)):
        turnover = np.sum(np.abs(weights_history[i] - weights_history[i-1]))
        turnovers.append(turnover)
    
    return turnovers

def calculate_var_cvar(returns: np.ndarray, confidence_level: float = 0.95) -> Tuple[float, float]:
    """
    Calculate Value at Risk (VaR) and Conditional VaR (CVaR)
    
    Args:
        returns: Array of periodic returns
        confidence_level: Confidence level (e.g., 0.95, 0.99)
        
    Returns:
        (var, cvar): Tuple of VaR and CVaR
    """
    
    alpha = (1 - confidence_level) * 100
    
    var = np.percentile(returns, alpha)
    cvar = returns[returns <= var].mean()
    
    return var, cvar


def calculate_transaction_cost(weights_history, fee_rate: float = 0.0015):
    turnovers = []
    prev_w = None
    
    for w in weights_history:
        w = np.asarray(w, dtype=float)
        if prev_w is None:
            turnovers.append(0.0)
        else:
            turnovers.append(np.sum(np.abs(w - prev_w)))
        prev_w = w
    turnovers = np.asarray(turnovers, dtype=float)  # shape: (T,)
    transaction_cost = turnovers * float(fee_rate)  # shape: (T,)

    return transaction_cost

def calculate_net_returns(returns: np.ndarray, weights_history) -> np.ndarray:
    transaction_cost = np.asarray(calculate_transaction_cost(weights_history), dtype=float)
    returns = np.asarray(returns, dtype=float)
    
    net_returns = (1.0 + returns) * (1.0- transaction_cost) - 1.0
    
    return net_returns
    
    ### Use net returns for calculating portfolio metrics which includes transaction cost.

def calculate_portfolio_metrics(portfolio, 
                                periods_per_year: int = 252,
                                risk_free_rate: float = 0.0) -> Dict:
    """
    Calculate all performance metrics for a Portfolio object
    
    Args:
        portfolio: Portfolio object with returns, weights, etc.
        periods_per_year: Number of periods per year (252 for daily, 52 for weekly)
        risk_free_rate: Annual risk-free rate
        
    Returns:
        metrics: Dictionary of performance metrics
    """
    returns = portfolio.get_returns_array()
    weights = portfolio.get_weights_array()
    returns = calculate_net_returns(returns, weights)
    
    if len(returns) == 0:
        return {
            'n_periods': 0,
            'cumulative_return': 0.0,
            'annualized_return': 0.0,
            'volatility': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'sortino_ratio': 0.0,
            'calmar ratio': 0.0,
            'turnover': 0.0,
            'VaR': 0.0,
            'CVaR': 0.0,
            'avg_solve_time': 0.0,
            'total_solve_time': 0.0
        }
    
    metrics = {}
    metrics['n_periods'] = len(returns)
    metrics['cumulative_return'] = calculate_cumulative_return(returns)
    metrics['annualized_return'] = calculate_annualized_return(returns, periods_per_year)
    metrics['volatility'] = calculate_volatility(returns, periods_per_year)
    metrics['sharpe_ratio'] = calculate_sharpe_ratio(returns, risk_free_rate, periods_per_year)
    metrics['max_drawdown'] = calculate_max_drawdown(returns)
    metrics['sortino_ratio'] = calculate_sortino_ratio(returns, risk_free_rate, periods_per_year)
    metrics['calmar_ratio'] = calculate_calmar_ratio(returns, periods_per_year)
    metrics['turnover'] = np.mean(calculate_turnover(weights))
    metrics['VaR'], metrics['CVaR'] = calculate_var_cvar(returns, confidence_level=ALPHA)
    
    # Computational metrics
    solve_times = [t for t in portfolio.solve_times if t > 0]  # Only non-zero solve times
    metrics['avg_solve_time'] = np.mean(solve_times) if solve_times else 0.0
    metrics['total_solve_time'] = np.sum(solve_times)
    
    # CPP-specific metrics
    if portfolio.thresholds_post:
        thresholds = np.array(portfolio.thresholds_post)
        # For CPP, we stored threshold per rebalancing period, not per day
        # Need to match returns with thresholds properly
        # Simplified: just report average threshold
        metrics['avg_threshold'] = np.mean(thresholds)
        metrics['min_threshold'] = np.min(thresholds)
        metrics['max_threshold'] = np.max(thresholds)
        
        # Calculate Coverage Rate
        min_len = min(len(returns), len(thresholds))
        curr_returns = returns[:min_len]
        curr_thresholds = thresholds[:min_len]
        
        satisfied_count = np.sum(curr_returns >= curr_thresholds)
        coverage_rate = satisfied_count / min_len if min_len > 0 else 0.0
        
        metrics['coverage_rate'] = coverage_rate
    
    else:
        metrics['avg_threshold'] = None
        metrics['min_threshold'] = None
        metrics['max_threshold'] = None
        metrics['coverage_rate'] = None
    
    return metrics


def compare_methods(portfolios: Dict[str, Any], 
                   periods_per_year: int = 252) -> pd.DataFrame:
    """
    Compare performance metrics across multiple portfolio methods
    
    Args:
        portfolios: Dictionary of {method_name: Portfolio object}
        periods_per_year: Number of periods per year
        
    Returns:
        comparison_df: DataFrame with methods as rows and metrics as columns
    """
    all_metrics = {}
    
    for method_name, portfolio in portfolios.items():
        all_metrics[method_name] = calculate_portfolio_metrics(
            portfolio, 
            periods_per_year=periods_per_year
        )
    
    df = pd.DataFrame(all_metrics).T
    
    # Reorder columns for better readability
    column_order = [
        'n_periods', 'cumulative_return', 'annualized_return', 'volatility',
        'sharpe_ratio', 'max_drawdown', 'sortino_raio', 'calmar_ratio', 'turnover', 
        'VaR', 'CVaR', 'avg_solve_time', 'total_solve_time'
    ]
    
    # Add CPP-specific columns if they exist
    if 'avg_threshold' in df.columns:
        column_order.extend(['avg_threshold', 'min_threshold', 'max_threshold', 'coverage_rate'])
    
    # Only keep columns that exist
    column_order = [col for col in column_order if col in df.columns]
    df = df[column_order]
    
    return df


def print_portfolio_metrics(metrics: Dict, portfolio_name: str = "Portfolio"):
    """Print metrics in a formatted way"""
    print(f"\n{'='*60}")
    print(f"{portfolio_name} Performance Metrics")
    print(f"{'='*60}")
    
    print(f"\n📊 Return Metrics:")
    print(f"  Periods:              {metrics['n_periods']:>10}")
    print(f"  Cumulative Return:    {metrics['cumulative_return']:>10.2%}")
    print(f"  Annualized Return:    {metrics['annualized_return']:>10.2%}")
    
    print(f"\n⚠️  Risk Metrics:")
    print(f"  Volatility (Ann.):    {metrics['volatility']:>10.2%}")
    print(f"  Sharpe Ratio:         {metrics['sharpe_ratio']:>10.4f}")
    print(f"  Max Drawdown:         {metrics['max_drawdown']:>10.2%}")
    print(f"  Sortino Ratio:        {metrics['sortino_ratio']:>10.4f}")
    print(f"  Calmar Ratio:         {metrics['calmar_ratio']:>10.4f}")
    print(f"  VaR:        {metrics['VaR']:>10.4f}")
    print(f"  Conditional VaR:         {metrics['CVaR']:>10.4f}")    
    
    print(f"\n💼 Trading Metrics:")
    print(f"  Avg Turnover:         {metrics['turnover']:>10.2%}")
    
    print(f"\n⏱️  Computational Metrics:")
    print(f"  Avg Solve Time:       {metrics['avg_solve_time']:>10.4f}s")
    print(f"  Total Solve Time:     {metrics['total_solve_time']:>10.2f}s")
    
    # CPP-specific metrics
    if 'avg_threshold' in metrics and metrics['avg_threshold'] is not None:
        print(f"\n🎯 CPP Threshold Metrics:")
        print(f"  Avg Threshold:        {metrics['avg_threshold']:>10.6f}")
        print(f"  Min Threshold:        {metrics['min_threshold']:>10.6f}")
        print(f"  Max Threshold:        {metrics['max_threshold']:>10.6f}")

        if'coverage_rate' in metrics and metrics['coverage_rate'] is not None:
            print(f"  Coverage Rate:        {metrics['coverage_rate']:>10.6%}")
    print(f"{'='*60}\n")