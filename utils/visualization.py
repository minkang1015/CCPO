"""
Visualization utilities for portfolio performance comparison
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
import os
from utils.metrics import calculate_net_returns

def plot_cumulative_returns(
    portfolios: Dict[str, any],
    save_path: Optional[str] = None,
    title: str = "Cumulative Returns Comparison",
    figsize: tuple = (12, 6)
):
    """
    Plot cumulative returns for multiple portfolios
    
    Args:
        portfolios: Dictionary of {method_name: Portfolio object}
        save_path: Path to save the plot (if None, only display)
        title: Plot title
        figsize: Figure size (width, height)
    """
    plt.figure(figsize=figsize)
    
    # Plot each portfolio's cumulative returns
    for method_name, portfolio in portfolios.items():
        if len(portfolio) == 0:
            continue
        
        # Get returns and dates
        weights = portfolio.get_weights_array()
        returns = portfolio.get_returns_array()
        
        dates = portfolio.dates
        returns = calculate_net_returns(returns, weights)
        
        # Calculate cumulative returns
        cumulative_returns = (1 + returns).cumprod()
        
        # Plot
        plt.plot(dates, cumulative_returns, label=method_name, linewidth=2)
    
    # Formatting
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Cumulative Return (Base = 1)', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Cumulative returns plot saved to '{save_path}'")
    
    plt.close()


def plot_drawdown(
    portfolios: Dict[str, any],
    save_path: Optional[str] = None,
    title: str = "Drawdown Comparison",
    figsize: tuple = (12, 6)
):
    """
    Plot drawdown for multiple portfolios
    
    Args:
        portfolios: Dictionary of {method_name: Portfolio object}
        save_path: Path to save the plot (if None, only display)
        title: Plot title
        figsize: Figure size (width, height)
    """
    plt.figure(figsize=figsize)
    
    # Plot each portfolio's drawdown
    for method_name, portfolio in portfolios.items():
        if len(portfolio) == 0:
            continue
        
        # Get returns and dates
        returns = portfolio.get_returns_array()
        dates = portfolio.dates
        
        # Calculate cumulative returns and drawdown
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        
        # Plot
        plt.plot(dates, drawdown * 100, label=method_name, linewidth=2)
    
    # Formatting
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Drawdown (%)', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    plt.tight_layout()
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Drawdown plot saved to '{save_path}'")
    
    plt.close()


def plot_rolling_returns(
    portfolios: Dict[str, any],
    window: int = 21,
    save_path: Optional[str] = None,
    title: str = "Rolling Returns Comparison",
    figsize: tuple = (12, 6)
):
    """
    Plot rolling average returns for multiple portfolios
    
    Args:
        portfolios: Dictionary of {method_name: Portfolio object}
        window: Rolling window size
        save_path: Path to save the plot (if None, only display)
        title: Plot title
        figsize: Figure size (width, height)
    """
    plt.figure(figsize=figsize)
    
    # Plot each portfolio's rolling returns
    for method_name, portfolio in portfolios.items():
        if len(portfolio) == 0:
            continue
        
        # Get returns and dates
        returns_series = portfolio.get_returns_series()
        
        # Calculate rolling mean
        rolling_mean = returns_series.rolling(window=window).mean() * 100
        
        # Plot
        plt.plot(rolling_mean.index, rolling_mean.values, label=method_name, linewidth=2)
    
    # Formatting
    plt.xlabel('Date', fontsize=12)
    plt.ylabel(f'{window}-Period Rolling Return (%)', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    plt.tight_layout()
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Rolling returns plot saved to '{save_path}'")
    
    plt.close()


def create_all_plots(
    portfolios: Dict[str, any],
    result_folder: str,
    prefix: str = "cpp"
):
    """
    Create all standard plots for portfolio comparison
    
    Args:
        portfolios: Dictionary of {method_name: Portfolio object}
        result_folder: Folder to save plots
        prefix: Prefix for plot filenames
    """
    print(f"\n{'='*60}")
    print("📊 Creating Visualization Plots")
    print(f"{'='*60}")
    
    # Cumulative returns - shows overall performance over time
    cumret_path = os.path.join(result_folder, f'{prefix}_cumulative_returns.png')
    plot_cumulative_returns(portfolios, save_path=cumret_path)
    
    # Drawdown - shows risk and recovery patterns
    # drawdown_path = os.path.join(result_folder, f'{prefix}_drawdown.png')
    # plot_drawdown(portfolios, save_path=drawdown_path)
    
    # Note: Rolling returns not needed because our validation periods don't overlap
    # Each split's V-period is independent, so cumulative return already shows the full picture
    
    print(f"{'='*60}\n")
