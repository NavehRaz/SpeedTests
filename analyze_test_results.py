#!/usr/bin/env python3
"""
Analysis script for smurf distance test results.

This script analyzes the combined CSV results to provide:
1. Summary of pass/fail rates for each configuration
2. Ranking of configurations by performance
3. Analysis of parameter sensitivity
4. Comparison of variation sizes
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import argparse

def load_and_clean_data(csv_file):
    """
    Load and clean the combined results CSV.
    
    Args:
        csv_file (str): Path to the combined CSV file
        
    Returns:
        pandas.DataFrame: Cleaned DataFrame
    """
    print(f"Loading data from: {csv_file}")
    df = pd.read_csv(csv_file)
    
    # Create configuration ID based on n, sigma, and grid_factor
    df['config_id'] = df['sigma'].astype(str) + '_' + df['grid_factor'].astype(str) + '_' + df['n'].astype(str)
    
    # Convert theta from string back to list for analysis
    df['theta_list'] = df['theta'].apply(eval)
    
    print(f"Loaded {len(df)} test results")
    print(f"Unique configurations (n, sigma, grid_factor): {df['config_id'].nunique()}")
    print(f"Unique presets: {df['preset_name'].unique()}")
    print(f"Unique sigma values: {sorted(df['sigma'].unique())}")
    print(f"Unique grid_factor values: {sorted(df['grid_factor'].unique())}")
    print(f"Unique n values: {sorted(df['n'].unique())}")
    
    return df

def configuration_summary(df):
    """
    Create a summary of pass/fail rates for each configuration (n, sigma, grid_factor).
    
    Args:
        df (pandas.DataFrame): Cleaned DataFrame
        
    Returns:
        pandas.DataFrame: Summary DataFrame
    """
    print("\n" + "="*80)
    print("CONFIGURATION SUMMARY (n, sigma, grid_factor)")
    print("="*80)
    
    # Group by configuration (n, sigma, grid_factor)
    config_summary = df.groupby('config_id').agg({
        'sigma': 'first',
        'grid_factor': 'first',
        'n': 'first',
        'pass_smurf': ['count', 'sum', 'mean'],
        'pass_baysian': ['count', 'sum', 'mean']
    }).round(3)
    
    # Check if old pass columns exist
    has_old_pass = 'pass_smurf_old' in df.columns
    if has_old_pass:
        old_pass_agg = df.groupby('config_id').agg({
            'pass_smurf_old': ['count', 'sum', 'mean']
        }).round(3)
        print("Found old pass columns - including in analysis")
    
    # Flatten column names
    config_summary.columns = ['_'.join(col).strip() for col in config_summary.columns]
    
    # Rename columns for clarity
    config_summary = config_summary.rename(columns={
        'sigma_first': 'sigma',
        'grid_factor_first': 'grid_factor',
        'n_first': 'n',
        'pass_smurf_count': 'total_tests',
        'pass_smurf_sum': 'smurf_passed',
        'pass_smurf_mean': 'smurf_pass_rate',
        'pass_baysian_sum': 'baysian_passed',
        'pass_baysian_mean': 'baysian_pass_rate'
    })
    
    # Add old pass columns if they exist
    if has_old_pass:
        old_pass_agg.columns = ['_'.join(col).strip() for col in old_pass_agg.columns]
        config_summary['smurf_old_passed'] = old_pass_agg['pass_smurf_old_sum']
        config_summary['smurf_old_pass_rate'] = old_pass_agg['pass_smurf_old_mean']
        config_summary['smurf_old_total_tests'] = old_pass_agg['pass_smurf_old_count']
    
    # Calculate Bayesian pass rate excluding xs variations
    df_no_xs = df[df['parameter_varied'] != 'xs']
    baysian_summary = df_no_xs.groupby('config_id').agg({
        'pass_baysian': ['count', 'sum', 'mean']
    }).round(3)
    
    # Update Bayesian columns with corrected values
    if not baysian_summary.empty:
        config_summary['baysian_passed'] = baysian_summary[('pass_baysian', 'sum')]
        config_summary['baysian_pass_rate'] = baysian_summary[('pass_baysian', 'mean')]
        config_summary['baysian_total_tests'] = baysian_summary[('pass_baysian', 'count')]
    else:
        config_summary['baysian_passed'] = 0
        config_summary['baysian_pass_rate'] = 0
        config_summary['baysian_total_tests'] = 0
    
    # Calculate overall pass rate (both tests must pass, excluding xs for Bayesian)
    config_summary['both_passed'] = df.groupby('config_id').apply(
        lambda x: ((x['pass_smurf'] == 1) & 
                  ((x['parameter_varied'] != 'xs') & (x['pass_baysian'] == 1))).sum()
    )
    config_summary['both_pass_rate'] = (config_summary['both_passed'] / config_summary['total_tests']).round(3)
    
    # Add configuration details
    config_summary['config_details'] = (
        f"σ={config_summary['sigma']}, "
        f"grid={config_summary['grid_factor']}, "
        f"n={config_summary['n']}"
    )
    
    # Sort by overall pass rate for better readability
    config_summary = config_summary.sort_values('both_pass_rate', ascending=False)
    
    print("Configuration Performance (sorted by overall pass rate):")
    print("Note: Bayesian pass rates exclude xs parameter variations")
    display_cols = ['config_details', 'total_tests', 
                   'smurf_passed', 'smurf_pass_rate', 
                   'baysian_passed', 'baysian_pass_rate',
                   'both_passed', 'both_pass_rate']
    
    # Add old pass columns to display if they exist
    if has_old_pass:
        display_cols.extend(['smurf_old_passed', 'smurf_old_pass_rate'])
        print("Note: Old smurf pass rates are included for comparison")
    
    print(config_summary[display_cols].to_string())
    
    return config_summary

def rank_configurations(config_summary, df):
    """
    Rank configurations (n, sigma, grid_factor) by performance.
    
    Args:
        config_summary (pandas.DataFrame): Configuration summary
        df (pandas.DataFrame): Original DataFrame with preset information
        
    Returns:
        pandas.DataFrame: Ranked configurations
    """
    print("\n" + "="*80)
    print("CONFIGURATION RANKING (n, sigma, grid_factor)")
    print("="*80)
    
    # Create ranking based on multiple criteria
    ranking = config_summary.copy()
    
    # Sort by overall pass rate (both tests), then by individual test rates
    ranking = ranking.sort_values(['both_pass_rate', 'smurf_pass_rate', 'baysian_pass_rate'], 
                                 ascending=[False, False, False])
    
    ranking['rank'] = range(1, len(ranking) + 1)
    
    print("Top 10 Configurations:")
    print(ranking[['rank', 'config_details', 'both_pass_rate', 
                  'smurf_pass_rate', 'baysian_pass_rate']].head(10).to_string(index=False))
    
    print("\nBottom 10 Configurations:")
    print(ranking[['rank', 'config_details', 'both_pass_rate', 
                  'smurf_pass_rate', 'baysian_pass_rate']].tail(10).to_string(index=False))
    
    return ranking

def best_configuration_per_preset(df):
    """
    Find the best configuration (sigma, n, grid_factor) for each preset.
    
    Args:
        df (pandas.DataFrame): Cleaned DataFrame
        
    Returns:
        pandas.DataFrame: Best configuration for each preset
    """
    print("\n" + "="*80)
    print("BEST CONFIGURATION PER PRESET")
    print("="*80)
    
    # Group by preset and configuration, calculate pass rates
    preset_config_summary = df.groupby(['preset_name', 'config_id']).agg({
        'sigma': 'first',
        'grid_factor': 'first',
        'n': 'first',
        'pass_smurf': 'mean',
        'pass_baysian': 'mean'
    }).round(3)
    
    # Check if old pass columns exist
    has_old_pass = 'pass_smurf_old' in df.columns
    if has_old_pass:
        old_pass_agg = df.groupby(['preset_name', 'config_id']).agg({
            'pass_smurf_old': 'mean'
        }).round(3)
        preset_config_summary['pass_smurf_old'] = old_pass_agg['pass_smurf_old']
        print("Found old pass columns - including in preset analysis")
    
    # Calculate overall pass rate (excluding xs for Bayesian)
    preset_config_summary['overall_pass_rate'] = (
        df.groupby(['preset_name', 'config_id']).apply(
            lambda x: ((x['pass_smurf'] == 1) & 
                      ((x['parameter_varied'] != 'xs') & (x['pass_baysian'] == 1))).mean()
        )
    ).round(3)
    
    # Find best configuration for each preset
    best_configs = []
    for preset in df['preset_name'].unique():
        preset_data = preset_config_summary.loc[preset]
        
        # Get the best configuration for this preset
        best_config = preset_data.loc[preset_data['overall_pass_rate'].idxmax()]
        
        best_configs.append({
            'preset': preset,
            'best_config_id': best_config.name,
            'sigma': best_config['sigma'],
            'grid_factor': best_config['grid_factor'],
            'n': best_config['n'],
            'smurf_pass_rate': best_config['pass_smurf'],
            'baysian_pass_rate': best_config['pass_baysian'],
            'overall_pass_rate': best_config['overall_pass_rate']
        })
        
        # Add old pass rate if it exists
        if has_old_pass:
            best_configs[-1]['smurf_old_pass_rate'] = best_config['pass_smurf_old']
    
    best_configs_df = pd.DataFrame(best_configs)
    best_configs_df = best_configs_df.sort_values('overall_pass_rate', ascending=False)
    
    print("Best Configuration for Each Preset:")
    print("Note: Bayesian pass rates exclude xs parameter variations")
    display_cols = ['preset', 'sigma', 'grid_factor', 'n', 
                   'smurf_pass_rate', 'baysian_pass_rate', 'overall_pass_rate']
    
    # Add old pass rate to display if it exists
    if has_old_pass:
        display_cols.append('smurf_old_pass_rate')
        print("Note: Old smurf pass rates are included for comparison")
    
    print(best_configs_df[display_cols].to_string(index=False))
    
    return best_configs_df

def parameter_sensitivity_analysis(df):
    """
    Analyze which parameters are most problematic when varied.
    
    Args:
        df (pandas.DataFrame): Cleaned DataFrame
    """
    print("\n" + "="*80)
    print("PARAMETER SENSITIVITY ANALYSIS")
    print("="*80)
    
    # Analyze by parameter and preset
    param_analysis = df.groupby(['preset_name', 'parameter_varied']).agg({
        'pass_smurf': 'mean',
        'pass_baysian': 'mean',
        'variation_value': 'count'
    }).round(3)
    
    param_analysis.columns = ['smurf_pass_rate', 'baysian_pass_rate', 'test_count']
    
    # Check if old pass columns exist
    has_old_pass = 'pass_smurf_old' in df.columns
    if has_old_pass:
        old_pass_agg = df.groupby(['preset_name', 'parameter_varied']).agg({
            'pass_smurf_old': 'mean'
        }).round(3)
        param_analysis['smurf_old_pass_rate'] = old_pass_agg['pass_smurf_old']
        print("Found old pass columns - including in parameter sensitivity analysis")
    
    # Calculate overall pass rate for each parameter
    param_analysis['overall_pass_rate'] = (
        df.groupby(['preset_name', 'parameter_varied']).apply(
            lambda x: ((x['pass_smurf'] == 1) & (x['pass_baysian'] == 1)).mean()
        )
    ).round(3)
    
    print("Parameter Sensitivity by Preset:")
    print(param_analysis.to_string())
    
    # Find problematic parameters (low pass rates)
    problematic = param_analysis[param_analysis['overall_pass_rate'] < 0.5]
    if not problematic.empty:
        print("\n⚠️  PROBLEMATIC PARAMETERS (pass rate < 50%):")
        print(problematic.to_string())
    else:
        print("\n✅ No parameters with pass rate below 50%")
    
    return param_analysis

def variation_size_analysis(df):
    """
    Analyze how variation size affects test results.
    
    Args:
        df (pandas.DataFrame): Cleaned DataFrame
    """
    print("\n" + "="*80)
    print("VARIATION SIZE ANALYSIS")
    print("="*80)
    
    # Analyze by variation value
    variation_analysis = df.groupby('variation_value').agg({
        'pass_smurf': 'mean',
        'pass_baysian': 'mean',
        'parameter_varied': 'count'
    }).round(3)
    
    variation_analysis.columns = ['smurf_pass_rate', 'baysian_pass_rate', 'test_count']
    
    # Check if old pass columns exist
    has_old_pass = 'pass_smurf_old' in df.columns
    if has_old_pass:
        old_pass_agg = df.groupby('variation_value').agg({
            'pass_smurf_old': 'mean'
        }).round(3)
        variation_analysis['smurf_old_pass_rate'] = old_pass_agg['pass_smurf_old']
        print("Found old pass columns - including in variation size analysis")
    
    # Calculate overall pass rate
    variation_analysis['overall_pass_rate'] = (
        df.groupby('variation_value').apply(
            lambda x: ((x['pass_smurf'] == 1) & (x['pass_baysian'] == 1)).mean()
        )
    ).round(3)
    
    print("Test Results by Variation Size:")
    print(variation_analysis.to_string())
    
    # Analyze by parameter and variation
    param_variation = df.groupby(['parameter_varied', 'variation_value']).agg({
        'pass_smurf': 'mean',
        'pass_baysian': 'mean'
    }).round(3)
    
    param_variation.columns = ['smurf_pass_rate', 'baysian_pass_rate']
    
    # Add old pass rate if it exists
    if has_old_pass:
        old_pass_agg = df.groupby(['parameter_varied', 'variation_value']).agg({
            'pass_smurf_old': 'mean'
        }).round(3)
        param_variation['smurf_old_pass_rate'] = old_pass_agg['pass_smurf_old']
    
    param_variation['overall_pass_rate'] = (
        df.groupby(['parameter_varied', 'variation_value']).apply(
            lambda x: ((x['pass_smurf'] == 1) & (x['pass_baysian'] == 1)).mean()
        )
    ).round(3)
    
    print("\nTest Results by Parameter and Variation Size:")
    print(param_variation.to_string())
    
    return variation_analysis, param_variation

def parameter_effect_analysis(df):
    """
    Analyze the effect of sigma, n, and grid_factor on test results.
    
    Args:
        df (pandas.DataFrame): Cleaned DataFrame
        
    Returns:
        tuple: Three DataFrames for sigma, n, and grid_factor effects
    """
    print("\n" + "="*80)
    print("PARAMETER EFFECT ANALYSIS")
    print("="*80)
    
    # 1. Sigma effect analysis
    print("\n1. SIGMA EFFECT ANALYSIS:")
    print("-" * 50)
    sigma_effect = df.groupby('sigma').agg({
        'pass_smurf': 'mean',
        'pass_baysian': 'mean',
        'parameter_varied': 'count'
    }).round(3)
    
    sigma_effect.columns = ['smurf_pass_rate', 'baysian_pass_rate', 'total_tests']
    
    # Check if old pass columns exist
    has_old_pass = 'pass_smurf_old' in df.columns
    if has_old_pass:
        old_pass_agg = df.groupby('sigma').agg({
            'pass_smurf_old': 'mean'
        }).round(3)
        sigma_effect['smurf_old_pass_rate'] = old_pass_agg['pass_smurf_old']
        print("Found old pass columns - including in sigma effect analysis")
    
    # Calculate overall pass rate (excluding xs for Bayesian)
    sigma_effect['overall_pass_rate'] = (
        df.groupby('sigma').apply(
            lambda x: ((x['pass_smurf'] == 1) & 
                      ((x['parameter_varied'] != 'xs') & (x['pass_baysian'] == 1))).mean()
        )
    ).round(3)
    
    print("Effect of Sigma on Test Results:")
    print(sigma_effect.to_string())
    
    # 2. N effect analysis
    print("\n2. N EFFECT ANALYSIS:")
    print("-" * 50)
    n_effect = df.groupby('n').agg({
        'pass_smurf': 'mean',
        'pass_baysian': 'mean',
        'parameter_varied': 'count'
    }).round(3)
    
    n_effect.columns = ['smurf_pass_rate', 'baysian_pass_rate', 'total_tests']
    
    # Add old pass rate if it exists
    if has_old_pass:
        old_pass_agg = df.groupby('n').agg({
            'pass_smurf_old': 'mean'
        }).round(3)
        n_effect['smurf_old_pass_rate'] = old_pass_agg['pass_smurf_old']
    
    # Calculate overall pass rate (excluding xs for Bayesian)
    n_effect['overall_pass_rate'] = (
        df.groupby('n').apply(
            lambda x: ((x['pass_smurf'] == 1) & 
                      ((x['parameter_varied'] != 'xs') & (x['pass_baysian'] == 1))).mean()
        )
    ).round(3)
    
    print("Effect of N on Test Results:")
    print(n_effect.to_string())
    
    # 3. Grid factor effect analysis
    print("\n3. GRID FACTOR EFFECT ANALYSIS:")
    print("-" * 50)
    grid_effect = df.groupby('grid_factor').agg({
        'pass_smurf': 'mean',
        'pass_baysian': 'mean',
        'parameter_varied': 'count'
    }).round(3)
    
    grid_effect.columns = ['smurf_pass_rate', 'baysian_pass_rate', 'total_tests']
    
    # Add old pass rate if it exists
    if has_old_pass:
        old_pass_agg = df.groupby('grid_factor').agg({
            'pass_smurf_old': 'mean'
        }).round(3)
        grid_effect['smurf_old_pass_rate'] = old_pass_agg['pass_smurf_old']
    
    # Calculate overall pass rate (excluding xs for Bayesian)
    grid_effect['overall_pass_rate'] = (
        df.groupby('grid_factor').apply(
            lambda x: ((x['pass_smurf'] == 1) & 
                      ((x['parameter_varied'] != 'xs') & (x['pass_baysian'] == 1))).mean()
        )
    ).round(3)
    
    print("Effect of Grid Factor on Test Results:")
    print(grid_effect.to_string())
    
    return sigma_effect, n_effect, grid_effect

def aggregate_test_results(df_or_csv, output_csv=None, include_config_summary=False, clean=True):
    """
    Aggregate test results by configuration, calculating means, stds, and pass percentages.
    
    Args:
        df_or_csv: Either a pandas DataFrame or path to CSV file
        output_csv (str): Optional path to save the aggregated results CSV
        include_config_summary (bool): Whether to include the config_summary column (default: True)
        clean (bool): Whether to remove entries where reference values are -inf (default: False)
        
    Returns:
        pandas.DataFrame: Aggregated DataFrame with means, stds, and pass percentages
    """
    print("\n" + "="*80)
    print("AGGREGATING TEST RESULTS BY CONFIGURATION")
    print("="*80)
    
    # Load data if CSV path is provided
    if isinstance(df_or_csv, str):
        print(f"Loading data from: {df_or_csv}")
        df = pd.read_csv(df_or_csv)
    else:
        df = df_or_csv.copy()
    
    print(f"Original data shape: {df.shape}")
    
    # Show unique values in key columns for debugging
    print(f"Unique presets: {df['preset_name'].unique()}")
    print(f"Unique variation_values: {sorted(df['variation_value'].unique())}")
    print(f"Unique parameter_varied: {df['parameter_varied'].unique()}")
    
    # Define aggregation columns
    agg_columns = ['preset_name', 'grid_factor', 'sigma', 'n', 'parameter_varied', 'variation_value']
    
    # Check if all required columns exist
    missing_cols = [col for col in agg_columns if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing columns: {missing_cols}")
        return pd.DataFrame()
    
    # Show data distribution before filtering
    print(f"\nData distribution before filtering:")
    print(f"Rows with 'ERROR' in smurf_distance: {(df['smurf_distance'] == 'ERROR').sum()}")
    print(f"Rows with -inf in reference_distance: {(df['reference_distance'] == -np.inf).sum()}")
    print(f"Rows with -inf in reference_baysian_distance: {(df['reference_baysian_distance'] == -np.inf).sum()}")
    
    # Filter out error rows for numerical calculations
    df_numeric = df[df['smurf_distance'] != 'ERROR'].copy()
    df_numeric['smurf_distance'] = pd.to_numeric(df_numeric['smurf_distance'], errors='coerce')
    df_numeric['baysian_distance'] = pd.to_numeric(df_numeric['baysian_distance'], errors='coerce')
    
    print(f"Rows after removing 'ERROR': {len(df_numeric)}")
    
    # Handle old distance columns if they exist
    has_old_columns = 'smurf_distance_old' in df_numeric.columns and 'reference_distance_old' in df_numeric.columns
    if has_old_columns:
        df_numeric['smurf_distance_old'] = pd.to_numeric(df_numeric['smurf_distance_old'], errors='coerce')
        print("Found old distance columns - including in analysis")
    
    # Handle -inf reference values intelligently: keep rows for tests where reference is valid
    initial_count = len(df_numeric)
    
    # Create masks for valid reference values
    valid_smurf_ref = df_numeric['reference_distance'] != -np.inf
    valid_baysian_ref = df_numeric['reference_baysian_distance'] != -np.inf
    valid_old_ref = None
    if has_old_columns:
        valid_old_ref = df_numeric['reference_distance_old'] != -np.inf
    
    # Count how many rows have valid references for each test
    print(f"Rows with valid smurf reference: {valid_smurf_ref.sum()}")
    print(f"Rows with valid baysian reference: {valid_baysian_ref.sum()}")
    if has_old_columns:
        print(f"Rows with valid old reference: {valid_old_ref.sum()}")
    
    # Calculate differences only for valid reference values
    df_numeric['smurf_distance_diff'] = np.where(valid_smurf_ref, 
                                                 df_numeric['reference_distance'] - df_numeric['smurf_distance'], 
                                                 np.nan)
    df_numeric['baysian_distance_diff'] = np.where(valid_baysian_ref, 
                                                   df_numeric['reference_baysian_distance'] - df_numeric['baysian_distance'], 
                                                   np.nan)
    
    # Calculate differences for old columns if they exist
    if has_old_columns:
        df_numeric['smurf_distance_old_diff'] = np.where(valid_old_ref, 
                                                         df_numeric['reference_distance_old'] - df_numeric['smurf_distance_old'], 
                                                         np.nan)
        print("Calculated differences for old distance columns")
    
    print(f"Rows after handling -inf references: {len(df_numeric)}")
    
    print(f"Valid numeric rows: {len(df_numeric)}")
    
    # Show grouping information
    print(f"\nGrouping by: {agg_columns}")
    print(f"Number of unique groups: {df_numeric.groupby(agg_columns).ngroups}")
    
    # Show sample of groups for debugging
    sample_groups = df_numeric.groupby(agg_columns).size().head(10)
    print(f"Sample group sizes:")
    print(sample_groups)
    
    # Prepare aggregation dictionary with separate handling for each test type
    agg_dict = {
        'smurf_distance': ['mean', 'std'],
        'baysian_distance': ['mean', 'std'],
        'reference_distance': ['first', 'mean', 'std'],
        'reference_baysian_distance': ['first', 'mean', 'std'],
        'smurf_distance_diff': ['mean', 'std'],
        'baysian_distance_diff': ['mean', 'std'],
        'theta': 'first'
    }
    
    # Add old columns to aggregation if they exist
    if has_old_columns:
        agg_dict.update({
            'smurf_distance_old': ['mean', 'std'],
            'reference_distance_old': ['first', 'mean', 'std'],
            'smurf_distance_old_diff': ['mean', 'std']
        })
    
    # Aggregate by configuration
    aggregated = df_numeric.groupby(agg_columns).agg(agg_dict).round(4)
    
    # Flatten column names first
    aggregated.columns = ['_'.join(col).strip() for col in aggregated.columns]
    
    # Add separate count columns for each test type
    # Count valid smurf tests (where reference is not -inf)
    smurf_counts = df_numeric[valid_smurf_ref].groupby(agg_columns).size()
    aggregated['smurf_distance_count'] = smurf_counts
    
    # Count valid baysian tests (where reference is not -inf)
    baysian_counts = df_numeric[valid_baysian_ref].groupby(agg_columns).size()
    aggregated['baysian_distance_count'] = baysian_counts
    
    # Count valid old tests if they exist
    if has_old_columns:
        old_counts = df_numeric[valid_old_ref].groupby(agg_columns).size()
        aggregated['smurf_distance_old_count'] = old_counts
    
    # Calculate pass rates only for valid tests
    # Smurf pass rate (only for rows with valid smurf reference)
    smurf_pass_rates = df_numeric[valid_smurf_ref].groupby(agg_columns)['pass_smurf'].mean()
    aggregated['pass_smurf_mean'] = smurf_pass_rates
    
    # Bayesian pass rate (only for rows with valid baysian reference)
    baysian_pass_rates = df_numeric[valid_baysian_ref].groupby(agg_columns)['pass_baysian'].mean()
    aggregated['pass_baysian_mean'] = baysian_pass_rates
    
    # Old pass rate (only for rows with valid old reference)
    if has_old_columns and 'pass_smurf_old' in df_numeric.columns:
        old_pass_rates = df_numeric[valid_old_ref].groupby(agg_columns)['pass_smurf_old'].mean()
        aggregated['pass_smurf_old_mean'] = old_pass_rates
        print("Found old pass columns - including in aggregation")
    
    # Rename columns for clarity
    column_mapping = {
        'smurf_distance_count': 'n_tests_smurf',
        'smurf_distance_mean': 'mean_smurf_distance',
        'smurf_distance_std': 'std_smurf_distance',
        'baysian_distance_count': 'n_tests_baysian',
        'baysian_distance_mean': 'mean_baysian_distance',
        'baysian_distance_std': 'std_baysian_distance',
        'pass_smurf_mean': 'smurf_pass_percentage',
        'pass_baysian_mean': 'baysian_pass_percentage',
        'reference_distance_first': 'reference_distance',
        'reference_distance_mean': 'mean_reference_distance',
        'reference_distance_std': 'std_reference_distance',
        'reference_baysian_distance_first': 'reference_baysian_distance',
        'reference_baysian_distance_mean': 'mean_reference_baysian_distance',
        'reference_baysian_distance_std': 'std_reference_baysian_distance',
        'smurf_distance_diff_mean': 'mean_smurf_diff',
        'smurf_distance_diff_std': 'std_smurf_diff',
        'baysian_distance_diff_mean': 'mean_baysian_diff',
        'baysian_distance_diff_std': 'std_baysian_diff',
        'theta_first': 'theta'
    }
    
    # Add old column mappings if they exist
    if has_old_columns:
        column_mapping.update({
            'smurf_distance_old_count': 'n_tests_old',
            'smurf_distance_old_mean': 'mean_smurf_distance_old',
            'smurf_distance_old_std': 'std_smurf_distance_old',
            'reference_distance_old_first': 'reference_distance_old',
            'reference_distance_old_mean': 'mean_reference_distance_old',
            'reference_distance_old_std': 'std_reference_distance_old',
            'smurf_distance_old_diff_mean': 'mean_smurf_old_diff',
            'smurf_distance_old_diff_std': 'std_smurf_old_diff'
        })
        
        # Add old pass percentage mapping if it exists
        if 'pass_smurf_old_mean' in aggregated.columns:
            column_mapping['pass_smurf_old_mean'] = 'smurf_old_pass_percentage'
    
    aggregated = aggregated.rename(columns=column_mapping)
    
    # Convert percentages to actual percentages (multiply by 100)
    aggregated['smurf_pass_percentage'] = (aggregated['smurf_pass_percentage'] * 100).round(2)
    aggregated['baysian_pass_percentage'] = (aggregated['baysian_pass_percentage'] * 100).round(2)
    
    # Convert old pass percentage if it exists
    if 'smurf_old_pass_percentage' in aggregated.columns:
        aggregated['smurf_old_pass_percentage'] = (aggregated['smurf_old_pass_percentage'] * 100).round(2)
    
    # Reset index to make configuration columns regular columns
    aggregated = aggregated.reset_index()
    
    # Add configuration summary column if requested
    if include_config_summary:
        aggregated['config_summary'] = (
            f"σ={aggregated['sigma']}, "
            f"grid={aggregated['grid_factor']}, "
            f"n={aggregated['n']}, "
            f"{aggregated['parameter_varied']}={aggregated['variation_value']}"
        )
    
    # Sort by preset and then by pass percentage
    aggregated = aggregated.sort_values(['preset_name', 'smurf_pass_percentage'], ascending=[True, False])
    
    print(f"Aggregated data shape: {aggregated.shape}")
    print(f"Unique configurations: {len(aggregated)}")
    
    # Display summary statistics
    print("\nSummary Statistics:")
    print(f"Mean smurf pass percentage: {aggregated['smurf_pass_percentage'].mean():.2f}%")
    print(f"Mean baysian pass percentage: {aggregated['baysian_pass_percentage'].mean():.2f}%")
    print(f"Configurations with 100% smurf pass rate: {(aggregated['smurf_pass_percentage'] == 100).sum()}")
    print(f"Configurations with 100% baysian pass rate: {(aggregated['baysian_pass_percentage'] == 100).sum()}")
    
    if has_old_columns:
        print(f"Mean smurf old distance: {aggregated['mean_smurf_distance_old'].mean():.2f}")
        print(f"Mean reference old distance: {aggregated['mean_reference_distance_old'].mean():.2f}")
        if 'smurf_old_pass_percentage' in aggregated.columns:
            print(f"Mean smurf old pass percentage: {aggregated['smurf_old_pass_percentage'].mean():.2f}%")
            print(f"Configurations with 100% smurf old pass rate: {(aggregated['smurf_old_pass_percentage'] == 100).sum()}")
    
    # Show top 10 configurations by smurf pass percentage
    print("\nTop 10 Configurations by Smurf Pass Percentage:")
    top_configs = aggregated.nlargest(10, 'smurf_pass_percentage')
    display_cols = ['preset_name', 'n_tests_smurf', 'n_tests_baysian', 'smurf_pass_percentage', 
                   'baysian_pass_percentage', 'mean_smurf_distance', 'std_smurf_distance',
                   'mean_reference_distance', 'std_reference_distance',
                   'mean_smurf_diff', 'std_smurf_diff']
    
    # Add old columns to display if they exist
    if has_old_columns:
        display_cols.extend(['n_tests_old', 'mean_smurf_distance_old', 'std_smurf_distance_old',
                           'mean_reference_distance_old', 'std_reference_distance_old',
                           'mean_smurf_old_diff', 'std_smurf_old_diff'])
        if 'smurf_old_pass_percentage' in aggregated.columns:
            display_cols.append('smurf_old_pass_percentage')
    
    if include_config_summary:
        display_cols.insert(1, 'config_summary')
    print(top_configs[display_cols].to_string(index=False))
    
    # Save to CSV if requested
    if output_csv:
        aggregated.to_csv(output_csv, index=False)
        print(f"\nAggregated results saved to: {output_csv}")
    
    return aggregated

def create_visualizations(df, config_summary, ranking, best_configs_df):
    """
    Create visualizations of the results.
    
    Args:
        df (pandas.DataFrame): Cleaned DataFrame
        config_summary (pandas.DataFrame): Configuration summary
        ranking (pandas.DataFrame): Ranked configurations
        best_configs_df (pandas.DataFrame): Best configuration per preset
    """
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)
    
    # Check if old columns exist
    has_old_columns = 'pass_smurf_old' in df.columns
    
    # Set up the plotting style
    plt.style.use('default')
    
    # Create figure with proper spacing for title
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Smurf Distance Test Results Analysis', fontsize=16, y=0.95)
    
    # 1. Best configuration per preset
    if has_old_columns and 'smurf_old_pass_rate' in best_configs_df.columns:
        # Include old pass rate if available
        plot_columns = ['smurf_pass_rate', 'baysian_pass_rate', 'overall_pass_rate', 'smurf_old_pass_rate']
        plot_labels = ['Smurf', 'Baysian', 'Overall', 'Smurf (Old)']
    else:
        plot_columns = ['smurf_pass_rate', 'baysian_pass_rate', 'overall_pass_rate']
        plot_labels = ['Smurf', 'Baysian', 'Overall']
    
    best_configs_df.plot(x='preset', y=plot_columns, kind='bar', ax=axes[0,0])
    axes[0,0].set_title('Best Configuration Performance per Preset')
    axes[0,0].set_ylabel('Pass Rate')
    axes[0,0].tick_params(axis='x', rotation=45)
    axes[0,0].legend(plot_labels)
    
    # 2. Parameter sensitivity
    if has_old_columns:
        # Compare current vs old parameter sensitivity
        param_sensitivity_current = df.groupby('parameter_varied')['pass_smurf'].mean()
        param_sensitivity_old = df.groupby('parameter_varied')['pass_smurf_old'].mean()
        
        # Create comparison plot
        x_pos = np.arange(len(param_sensitivity_current))
        width = 0.35
        
        axes[0,1].bar(x_pos - width/2, param_sensitivity_current.values, width, 
                     label='Current', color='skyblue', alpha=0.8)
        axes[0,1].bar(x_pos + width/2, param_sensitivity_old.values, width, 
                     label='Old', color='lightcoral', alpha=0.8)
        
        axes[0,1].set_xlabel('Parameter Varied')
        axes[0,1].set_ylabel('Pass Rate')
        axes[0,1].set_title('Parameter Sensitivity Comparison')
        axes[0,1].set_xticks(x_pos)
        axes[0,1].set_xticklabels(param_sensitivity_current.index, rotation=45)
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
    else:
        # Original parameter sensitivity plot
        param_sensitivity = df.groupby('parameter_varied')['pass_smurf'].mean()
        param_sensitivity.plot(kind='bar', ax=axes[0,1], color='skyblue')
        axes[0,1].set_title('Parameter Sensitivity (Smurf Tests)')
        axes[0,1].set_ylabel('Pass Rate')
        axes[0,1].tick_params(axis='x', rotation=45)
    
    # 3. Variation size comparison
    if has_old_columns:
        # Compare current vs old variation sensitivity
        variation_comparison = df.groupby('variation_value')[['pass_smurf', 'pass_baysian', 'pass_smurf_old']].mean()
        variation_comparison.plot(kind='bar', ax=axes[1,0])
        axes[1,0].set_title('Test Results by Variation Size')
        axes[1,0].set_ylabel('Pass Rate')
        axes[1,0].tick_params(axis='x', rotation=45)
        axes[1,0].legend(['Smurf', 'Baysian', 'Smurf (Old)'])
    else:
        # Original variation comparison
        variation_comparison = df.groupby('variation_value')[['pass_smurf', 'pass_baysian']].mean()
        variation_comparison.plot(kind='bar', ax=axes[1,0])
        axes[1,0].set_title('Test Results by Variation Size')
        axes[1,0].set_ylabel('Pass Rate')
        axes[1,0].tick_params(axis='x', rotation=45)
        axes[1,0].legend(['Smurf', 'Baysian'])
    
    # 4. Top configurations
    if has_old_columns and 'smurf_old_pass_rate' in ranking.columns:
        # Include old pass rate in top configurations
        top_configs = ranking.head(10)
        plot_columns = ['smurf_pass_rate', 'baysian_pass_rate', 'smurf_old_pass_rate']
        plot_labels = ['Smurf', 'Baysian', 'Smurf (Old)']
        
        top_configs.plot(x='config_details', y=plot_columns, kind='bar', ax=axes[1,1])
        axes[1,1].set_title('Top 10 Configurations (n, σ, grid_factor)')
        axes[1,1].set_ylabel('Pass Rate')
        axes[1,1].tick_params(axis='x', rotation=45)
        axes[1,1].legend(plot_labels)
    else:
        # Original top configurations plot
        top_configs = ranking.head(10)
        top_configs.plot(x='config_details', y=['smurf_pass_rate', 'baysian_pass_rate'], 
                        kind='bar', ax=axes[1,1])
        axes[1,1].set_title('Top 10 Configurations (n, σ, grid_factor)')
        axes[1,1].set_ylabel('Pass Rate')
        axes[1,1].tick_params(axis='x', rotation=45)
        axes[1,1].legend(['Smurf', 'Baysian'])
    
    # Fix padding between title and panels
    plt.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.9, hspace=0.3, wspace=0.3)
    
    plt.savefig('test_results_analysis.png', dpi=300, bbox_inches='tight')
    print("Visualizations saved to: test_results_analysis.png")
    
    return fig

def plot_parameter_dependency(df_or_csv, x_param, y_metric, fixed_params=None, output_file=None):
    """
    Plot dependency of y_metric on x_param for different parameter variations and presets.
    
    Args:
        df_or_csv: Either a pandas DataFrame or path to CSV file
        x_param (str): Parameter to plot on x-axis ('n', 'sigma', 'grid_factor', 'variation_value')
        y_metric (str): Metric to plot on y-axis. Can be:
                       - Specific smurf/baysian metrics: 'smurf_pass_percentage', 'baysian_pass_percentage', 
                         'mean_reference_distance', 'mean_reference_baysian_distance', 
                         'mean_smurf_diff', 'mean_baysian_diff', 'mean_diff', 'pass_percentage'
                       - When using smurf/baysian specific metrics, separate figures will be created automatically
        fixed_params (dict): Dictionary with fixed values for the other parameters
                            e.g., {'sigma': 0.1, 'grid_factor': 0.5} when x_param='n'
                            When x_param='variation_value', fixed_params should include sigma, grid_factor, and optionally n
        output_file (str): Optional path to save the plot. If creating separate figures,
                          will save as base_name_smurf.png and base_name_baysian.png
        
    Returns:
        list: List of created figures. For metrics that have both smurf and baysian versions
              (mean_diff, pass_percentage, mean_reference_distance), returns two figures:
              [smurf_figure, baysian_figure]. For other metrics, returns [single_figure].
    """
    print(f"\n" + "="*80)
    print(f"PLOTTING {y_metric} vs {x_param}")
    print("="*80)
    
    # Load data if CSV path is provided
    if isinstance(df_or_csv, str):
        print(f"Loading data from: {df_or_csv}")
        df = pd.read_csv(df_or_csv)
    else:
        df = df_or_csv.copy()
    
    # Validate x_param
    valid_x_params = ['n', 'sigma', 'grid_factor', 'variation_value']
    if x_param not in valid_x_params:
        raise ValueError(f"x_param must be one of {valid_x_params}")
    
    # Validate y_metric
    valid_y_metrics = ['smurf_pass_percentage', 'baysian_pass_percentage', 
                      'mean_reference_distance', 'mean_reference_baysian_distance',
                      'mean_smurf_diff', 'mean_baysian_diff',
                      'mean_diff', 'pass_percentage',
                      'mean_smurf_distance_old', 'mean_reference_distance_old',
                      'mean_smurf_old_diff']
    if y_metric not in valid_y_metrics:
        raise ValueError(f"y_metric must be one of {valid_y_metrics}")
    
    # Determine if we need to create separate figures for smurf and baysian
    create_separate_figures = False
    smurf_metric = None
    baysian_metric = None
    
    # Check if old columns exist for comparison
    has_old_columns = any(col in df.columns for col in ['mean_smurf_distance_old', 'mean_reference_distance_old', 'mean_smurf_old_diff', 'smurf_old_pass_percentage'])
    
    if y_metric in ['mean_smurf_diff', 'mean_baysian_diff']:
        create_separate_figures = True
        smurf_metric = 'mean_smurf_diff'
        baysian_metric = 'mean_baysian_diff'
    elif y_metric in ['smurf_pass_percentage', 'baysian_pass_percentage']:
        create_separate_figures = True
        smurf_metric = 'smurf_pass_percentage'
        baysian_metric = 'baysian_pass_percentage'
    elif y_metric in ['mean_reference_distance', 'mean_reference_baysian_distance']:
        create_separate_figures = True
        smurf_metric = 'mean_reference_distance'
        baysian_metric = 'mean_reference_baysian_distance'
    elif y_metric == 'mean_diff':
        create_separate_figures = True
        smurf_metric = 'mean_smurf_diff'
        baysian_metric = 'mean_baysian_diff'
    elif y_metric == 'pass_percentage':
        create_separate_figures = True
        smurf_metric = 'smurf_pass_percentage'
        baysian_metric = 'baysian_pass_percentage'
    elif y_metric == 'mean_reference_distance':
        create_separate_figures = True
        smurf_metric = 'mean_reference_distance'
        baysian_metric = 'mean_reference_baysian_distance'
    elif y_metric in ['mean_smurf_distance_old', 'mean_reference_distance_old', 'mean_smurf_old_diff']:
        # Old metrics don't have baysian counterparts, so no separate figures
        create_separate_figures = False
    
    # Set default fixed_params if not provided
    if fixed_params is None:
        fixed_params = {}
    
    # Handle different x_param cases
    if x_param == 'variation_value':
        # For variation_value as x-axis, we need to fix sigma and grid_factor, and optionally n
        required_fixed = ['sigma', 'grid_factor']
        optional_fixed = ['n']
        
        # Check required fixed parameters
        missing_required = [param for param in required_fixed if param not in fixed_params]
        if missing_required:
            # Use most common values for missing required parameters
            for param in missing_required:
                most_common = df[param].mode().iloc[0]
                fixed_params[param] = most_common
                print(f"Using most common value for {param}: {most_common}")
        
        # Set up subplot structure: parameters vs n values
        param_variations = sorted(df['parameter_varied'].unique())
        n_values = sorted(df['n'].unique())
        
        # Filter data based on fixed parameters
        filtered_df = df.copy()
        for param, value in fixed_params.items():
            if param in ['sigma', 'grid_factor']:
                # Handle sigma and grid_factor filtering
                if str(value).replace('.', '').replace('-', '').isdigit():
                    try:
                        numeric_value = float(value)
                        filtered_df = filtered_df[filtered_df[param] == numeric_value]
                    except (ValueError, TypeError):
                        filtered_df = filtered_df[filtered_df[param].astype(str) == str(value)]
                else:
                    filtered_df = filtered_df[filtered_df[param].astype(str) == str(value)]
        
        # Get unique variation values for x-axis
        variation_values = sorted(filtered_df['variation_value'].unique())
        presets = sorted(filtered_df['preset_name'].unique())
        
        print(f"Fixed parameters: {fixed_params}")
        print(f"Parameter variations: {param_variations}")
        print(f"N values: {n_values}")
        print(f"Variation values (x-axis): {variation_values}")
        print(f"Presets: {presets}")
        
        # Determine if we need error bars
        use_error_bars = 'mean' in y_metric
        if use_error_bars:
            # Map y_metric to corresponding std column
            std_mapping = {
                'mean_reference_distance': 'std_reference_distance',
                'mean_reference_baysian_distance': 'std_reference_baysian_distance',
                'mean_smurf_diff': 'std_smurf_diff',
                'mean_baysian_diff': 'std_baysian_diff',
                'mean_smurf_distance_old': 'std_smurf_distance_old',
                'mean_reference_distance_old': 'std_reference_distance_old',
                'mean_smurf_old_diff': 'std_smurf_old_diff'
            }
            std_column = std_mapping.get(y_metric, None)
        
        # Define metrics to plot
        if create_separate_figures:
            metrics_to_plot = [(smurf_metric, 'Smurf'), (baysian_metric, 'Baysian')]
            
            # Add old metric if available and not already an old metric
            if has_old_columns and not any('old' in metric for metric in [smurf_metric, baysian_metric]):
                # Map current metric to old metric
                old_metric_mapping = {
                    'smurf_pass_percentage': 'smurf_old_pass_percentage',
                    'mean_smurf_diff': 'mean_smurf_old_diff',
                    'mean_reference_distance': 'mean_reference_distance_old'
                }
                old_metric = old_metric_mapping.get(smurf_metric)
                if old_metric and old_metric in df.columns:
                    metrics_to_plot.append((old_metric, 'Old'))
                    print(f"Adding old metric comparison: {old_metric}")
        else:
            metrics_to_plot = [(y_metric, '')]
            
            # Add old metric if available and not already an old metric
            if has_old_columns and 'old' not in y_metric:
                # Map current metric to old metric
                old_metric_mapping = {
                    'smurf_pass_percentage': 'smurf_old_pass_percentage',
                    'mean_smurf_diff': 'mean_smurf_old_diff',
                    'mean_reference_distance': 'mean_reference_distance_old'
                }
                old_metric = old_metric_mapping.get(y_metric)
                if old_metric and old_metric in df.columns:
                    metrics_to_plot.append((old_metric, 'Old'))
                    print(f"Adding old metric comparison: {old_metric}")
        
        # Create subplots: parameters (rows) vs n values (columns)
        n_variations = len(param_variations)
        n_n_values = len(n_values)
        n_panels = n_variations * n_n_values
        
        if n_panels == 0:
            print("No panels to create - no data available")
            return None
        
        figures = []
        
        # Create separate figures for each metric
        for metric, metric_label in metrics_to_plot:
            # Add extra space for legend
            fig, axes = plt.subplots(n_variations, n_n_values, figsize=(5*n_n_values, 4*n_variations + 1))
            if n_panels == 1:
                axes = np.array([axes])
            axes = axes.flatten()
            
            # Create title with metric label if creating separate figures
            if create_separate_figures:
                title = f'{metric_label} {metric} vs {x_param} (Fixed: {fixed_params})'
            else:
                title = f'{metric} vs {x_param} (Fixed: {fixed_params})'
            
            fig.suptitle(title, fontsize=16, y=0.95)
            
            panel_idx = 0
            for param_idx, param_varied in enumerate(param_variations):
                for n_idx, n_value in enumerate(n_values):
                    ax = axes[panel_idx]
                    
                    # Filter data for this specific parameter and n value
                    panel_data = filtered_df[
                        (filtered_df['parameter_varied'] == param_varied) & 
                        (filtered_df['n'] == n_value)
                    ]
                    
                    if not panel_data.empty:
                        # Plot for each preset
                        for preset in presets:
                            preset_data = panel_data[panel_data['preset_name'] == preset]
                            
                            if not preset_data.empty:
                                # Use variation_value as x-axis
                                x_values = preset_data['variation_value'].values
                                y_values = preset_data[metric].values
                                
                                # Handle inf and -inf values
                                y_values_plot = y_values.copy()
                                if np.any(np.isinf(y_values)) or np.any(np.isneginf(y_values)):
                                    # Get all y values in this panel for reference
                                    all_y_in_panel = panel_data[metric].values
                                    all_y_finite = all_y_in_panel[np.isfinite(all_y_in_panel)]
                                    
                                    if len(all_y_finite) > 0:
                                        y_max = np.max(all_y_finite)
                                        y_min = np.min(all_y_finite)
                                        
                                        # Replace inf with max*10, -inf with min*10
                                        y_values_plot[np.isinf(y_values)] = y_max 
                                        y_values_plot[np.isneginf(y_values)] = y_min
                                
                                # Sort data by x-values for proper plotting order
                                sort_indices = np.argsort(x_values)
                                x_values_sorted = x_values[sort_indices]
                                y_values_sorted = y_values_plot[sort_indices]
                                
                                # Get corresponding std column for error bars
                                if use_error_bars:
                                    std_mapping = {
                                        'mean_reference_distance': 'std_reference_distance',
                                        'mean_reference_baysian_distance': 'std_reference_baysian_distance',
                                        'mean_smurf_diff': 'std_smurf_diff',
                                        'mean_baysian_diff': 'std_baysian_diff',
                                        'mean_smurf_distance_old': 'std_smurf_distance_old',
                                        'mean_reference_distance_old': 'std_reference_distance_old',
                                        'mean_smurf_old_diff': 'std_smurf_old_diff'
                                    }
                                    std_column = std_mapping.get(metric, None)
                                    
                                    if std_column and std_column in preset_data.columns:
                                        y_errors = preset_data[std_column].values
                                        y_errors_sorted = y_errors[sort_indices]
                                        ax.errorbar(x_values_sorted, y_values_sorted, yerr=y_errors_sorted, 
                                                  marker='o', label=preset, capsize=3)
                                    else:
                                        ax.plot(x_values_sorted, y_values_sorted, marker='o', label=preset)
                                else:
                                    ax.plot(x_values_sorted, y_values_sorted, marker='o', label=preset)
                    
                    # Customize subplot
                    ax.set_xlabel('Variation Value')
                    ax.set_ylabel(metric)
                    ax.set_title(f'{param_varied}, n={n_value}')
                    ax.grid(True, alpha=0.3)
                    
                    panel_idx += 1
            
            # Create a single legend for all panels
            handles, labels = axes[0].get_legend_handles_labels()
            if handles:  # Only create legend if there are handles
                # Place legend at the bottom of the figure, outside the subplots
                fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.02), 
                          ncol=min(len(presets), 4), fancybox=True, shadow=True)
            
            # Fix padding between title and panels
            plt.subplots_adjust(top=0.9, bottom=0.15, left=0.1, right=0.9, hspace=0.3, wspace=0.3)
            
            # Save plot if output file is specified
            if output_file and create_separate_figures:
                # Create separate filenames for each metric
                base_name = output_file.replace('.png', '')
                metric_filename = f"{base_name}_{metric_label.lower()}.png"
                plt.savefig(metric_filename, dpi=300, bbox_inches='tight')
                print(f"Plot saved to: {metric_filename}")
            elif output_file:
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                print(f"Plot saved to: {output_file}")
            
            figures.append(fig)
        
        return figures
        
    else:
        # Original logic for n, sigma, grid_factor as x-axis
        # Determine which parameters to fix
        other_params = [p for p in ['n', 'sigma', 'grid_factor'] if p != x_param]
        for param in other_params:
            if param not in fixed_params:
                # Use the most common value for this parameter
                most_common = df[param].mode().iloc[0]
                fixed_params[param] = most_common
                print(f"Using most common value for {param}: {most_common}")
        
        print(f"Fixed parameters: {fixed_params}")
        print(f"Data types in DataFrame:")
        for param in ['n', 'sigma', 'grid_factor']:
            if param in df.columns:
                print(f"  {param}: {df[param].dtype}, unique values: {sorted(df[param].unique())[:5]}")
        
        # Filter data based on fixed parameters
        filtered_df = df.copy()
        for param, value in fixed_params.items():
            # Convert value to the appropriate data type for comparison
            if param in ['n']:
                # n should be numeric
                try:
                    numeric_value = int(value) if '.' not in str(value) else float(value)
                    filtered_df = filtered_df[filtered_df[param] == numeric_value]
                except (ValueError, TypeError):
                    print(f"Warning: Could not convert {value} to numeric for parameter {param}")
                    filtered_df = filtered_df[filtered_df[param].astype(str) == str(value)]
            elif param in ['sigma', 'grid_factor']:
                # sigma and grid_factor can be numeric or string (like 'scale_120')
                if str(value).replace('.', '').replace('-', '').isdigit():
                    # It's a numeric value
                    try:
                        numeric_value = float(value)
                        filtered_df = filtered_df[filtered_df[param] == numeric_value]
                    except (ValueError, TypeError):
                        filtered_df = filtered_df[filtered_df[param].astype(str) == str(value)]
                else:
                    # It's a string value (like 'scale_120')
                    filtered_df = filtered_df[filtered_df[param].astype(str) == str(value)]
            else:
                # For other parameters, use string comparison
                filtered_df = filtered_df[filtered_df[param].astype(str) == str(value)]
        
        # Convert string grid_factor values to numeric for plotting
        if 'grid_factor' in filtered_df.columns:
            filtered_df = filtered_df.copy()
            # Create a mapping for string to numeric values
            grid_factor_mapping = {'scale_60': 10, 'scale_120': 5}
            # Apply mapping only to string values
            mask = filtered_df['grid_factor'].astype(str).isin(grid_factor_mapping.keys())
            filtered_df.loc[mask, 'grid_factor_plot'] = filtered_df.loc[mask, 'grid_factor'].map(grid_factor_mapping)
            # Keep original values for non-mapped entries
            filtered_df.loc[~mask, 'grid_factor_plot'] = filtered_df.loc[~mask, 'grid_factor']
            # Convert to numeric
            filtered_df['grid_factor_plot'] = pd.to_numeric(filtered_df['grid_factor_plot'], errors='coerce')
        
        print(f"Filtered data shape: {filtered_df.shape}")
        
        # Check if we have any data after filtering
        if filtered_df.empty:
            print("Warning: No data found with the specified fixed parameters!")
            print(f"Available values for each parameter:")
            for param in ['n', 'sigma', 'grid_factor']:
                if param in df.columns:
                    unique_vals = sorted(df[param].unique())
                    print(f"  {param}: {unique_vals[:10]}{'...' if len(unique_vals) > 10 else ''}")
            return None
        
        # Get unique values for parameter variations
        param_variations = filtered_df['parameter_varied'].unique()
        variation_values = sorted(filtered_df['variation_value'].unique())
        presets = sorted(filtered_df['preset_name'].unique())
        
        # Get unique n values for subplot structure
        n_values = sorted(filtered_df['n'].unique())
        
        print(f"Parameter variations: {param_variations}")
        print(f"Variation values: {variation_values}")
        print(f"Presets: {presets}")
        print(f"N values: {n_values}")
        
        # Determine if we need error bars
        use_error_bars = 'mean' in y_metric
        if use_error_bars:
            # Map y_metric to corresponding std column
            std_mapping = {
                'mean_reference_distance': 'std_reference_distance',
                'mean_reference_baysian_distance': 'std_reference_baysian_distance',
                'mean_smurf_diff': 'std_smurf_diff',
                'mean_baysian_diff': 'std_baysian_diff',
                'mean_smurf_distance_old': 'std_smurf_distance_old',
                'mean_reference_distance_old': 'std_reference_distance_old',
                'mean_smurf_old_diff': 'std_smurf_old_diff'
            }
            std_column = std_mapping.get(y_metric, None)
        
        # Define metrics to plot
        if create_separate_figures:
            metrics_to_plot = [(smurf_metric, 'Smurf'), (baysian_metric, 'Baysian')]
        else:
            metrics_to_plot = [(y_metric, '')]
        
        # Create subplots: parameters (rows) vs n values (columns)
        n_variations = len(param_variations)
        n_n_values = len(n_values)
        n_panels = n_variations * n_n_values
        
        # Create figure with subplots
        if n_panels == 0:
            print("No panels to create - no data available")
            return None
        
        figures = []
        
        # Create separate figures for each metric
        for metric, metric_label in metrics_to_plot:
            # Add extra space for legend
            fig, axes = plt.subplots(len(n_values), len(param_variations), figsize=(5*len(param_variations), 4*len(n_values) + 1))
            if n_panels == 1:
                axes = np.array([axes])
            axes = axes.flatten()
            
            # Create title with metric label if creating separate figures
            if create_separate_figures:
                title = f'{metric_label} {metric} vs {x_param} (Fixed: {fixed_params})'
            else:
                title = f'{metric} vs {x_param} (Fixed: {fixed_params})'
            
            fig.suptitle(title, fontsize=16, y=0.95)
            
            panel_idx = 0
            for var_idx, param_varied in enumerate(param_variations):
                for val_idx, variation_value in enumerate(variation_values):
                    ax = axes[panel_idx]
                    
                    # Filter data for this specific variation
                    panel_data = filtered_df[
                        (filtered_df['parameter_varied'] == param_varied) & 
                        (filtered_df['variation_value'] == variation_value)
                    ]
                    
                    if not panel_data.empty:
                        # Plot for each preset
                        for preset in presets:
                            preset_data = panel_data[panel_data['preset_name'] == preset]
                            
                            if not preset_data.empty:
                                # Use grid_factor_plot if x_param is grid_factor
                                if x_param == 'grid_factor' and 'grid_factor_plot' in preset_data.columns:
                                    x_values = preset_data['grid_factor_plot'].values
                                else:
                                    x_values = preset_data[x_param].values
                                y_values = preset_data[metric].values
                                
                                # Handle inf and -inf values
                                y_values_plot = y_values.copy()
                                if np.any(np.isinf(y_values)) or np.any(np.isneginf(y_values)):
                                    # Get all y values in this panel for reference
                                    all_y_in_panel = panel_data[metric].values
                                    all_y_finite = all_y_in_panel[np.isfinite(all_y_in_panel)]
                                    
                                    if len(all_y_finite) > 0:
                                        y_max = np.max(all_y_finite)
                                        y_min = np.min(all_y_finite)
                                        
                                        # Replace inf with max*10, -inf with min*10
                                        y_values_plot[np.isinf(y_values)] = y_max 
                                        y_values_plot[np.isneginf(y_values)] = y_min
                                
                                # Sort data by x-values for proper plotting order
                                sort_indices = np.argsort(x_values)
                                x_values_sorted = x_values[sort_indices]
                                y_values_sorted = y_values_plot[sort_indices]
                                
                                # Get corresponding std column for error bars
                                if use_error_bars:
                                    std_mapping = {
                                        'mean_reference_distance': 'std_reference_distance',
                                        'mean_reference_baysian_distance': 'std_reference_baysian_distance',
                                        'mean_smurf_diff': 'std_smurf_diff',
                                        'mean_baysian_diff': 'std_baysian_diff',
                                        'mean_smurf_distance_old': 'std_smurf_distance_old',
                                        'mean_reference_distance_old': 'std_reference_distance_old',
                                        'mean_smurf_old_diff': 'std_smurf_old_diff'
                                    }
                                    std_column = std_mapping.get(metric, None)
                                    
                                    if std_column and std_column in preset_data.columns:
                                        y_errors = preset_data[std_column].values
                                        y_errors_sorted = y_errors[sort_indices]
                                        ax.errorbar(x_values_sorted, y_values_sorted, yerr=y_errors_sorted, 
                                                  marker='o', label=preset, capsize=3)
                                    else:
                                        ax.plot(x_values_sorted, y_values_sorted, marker='o', label=preset)
                                else:
                                    ax.plot(x_values_sorted, y_values_sorted, marker='o', label=preset)
                    
                    # Customize subplot
                    ax.set_xlabel(x_param)
                    ax.set_ylabel(metric)
                    ax.set_title(f'{param_varied} = {variation_value}')
                    ax.grid(True, alpha=0.3)
                    
                    panel_idx += 1
            
            # Create a single legend for all panels
            handles, labels = axes[0].get_legend_handles_labels()
            if handles:  # Only create legend if there are handles
                # Place legend at the bottom of the figure, outside the subplots
                fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.02), 
                          ncol=min(len(presets), 4), fancybox=True, shadow=True)
            
            # Fix padding between title and panels
            plt.subplots_adjust(top=0.9, bottom=0.15, left=0.1, right=0.9, hspace=0.3, wspace=0.3)
            
            # Save plot if output file is specified
            if output_file:
                if create_separate_figures or metric_label == 'Old':
                    # Create separate filenames for each metric
                    base_name = output_file.replace('.png', '')
                    metric_filename = f"{base_name}_{metric_label.lower()}.png"
                    plt.savefig(metric_filename, dpi=300, bbox_inches='tight')
                    print(f"Plot saved to: {metric_filename}")
                else:
                    plt.savefig(output_file, dpi=300, bbox_inches='tight')
                    print(f"Plot saved to: {output_file}")
            
            figures.append(fig)
        
        return figures

def plot_by_preset(df_or_csv, y_metric, sigma=1, grid_factor='scale_120', n=20000, output_file=None):
    """
    Plot dependency of y_metric on variation_value for each parameter, organized by preset.
    
    Creates a 5-panel plot (one for each parameter) showing the dependency of y_metric 
    on variation_value for each preset, with all 3 metrics (Bayesian, Smurf, and Old) 
    displayed in each panel.
    
    Note: Color intensity and text annotations show the number of tests used for each data point.
          - Darker/more opaque points indicate more tests (higher statistical reliability)
          - Text annotations show exact test counts (n=X) for every other point to avoid clutter
          - Both visual cues help assess the statistical reliability of each measurement.
    
    Args:
        df_or_csv: Either a pandas DataFrame or path to CSV file
        y_metric (str): Metric to plot on y-axis. Can be:
                       - 'pass_percentage': Shows pass percentages for all metrics
                       - 'mean_diff': Shows mean differences for all metrics
                       - 'mean_reference_distance': Shows mean reference distances for all metrics
        sigma (float): Sigma parameter to filter data (default: 1)
        grid_factor (str or float): Grid factor parameter to filter data (default: 'scale_120')
        n (int): N parameter to filter data (default: 20000)
        output_file (str): Optional path to save the plot
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    print(f"\n" + "="*80)
    print(f"PLOTTING {y_metric} BY PRESET")
    print(f"Parameters: sigma={sigma}, grid_factor={grid_factor}, n={n}")
    print("="*80)
    
    # Load data if CSV path is provided
    if isinstance(df_or_csv, str):
        print(f"Loading data from: {df_or_csv}")
        df = pd.read_csv(df_or_csv)
    else:
        df = df_or_csv.copy()
    
    # Validate y_metric
    valid_y_metrics = ['pass_percentage', 'mean_diff', 'mean_reference_distance']
    if y_metric not in valid_y_metrics:
        raise ValueError(f"y_metric must be one of {valid_y_metrics}")
    
    # Check if old columns exist for comparison
    has_old_columns = any(col in df.columns for col in ['mean_smurf_distance_old', 'mean_reference_distance_old', 'mean_smurf_old_diff', 'smurf_old_pass_percentage'])
    
    if not has_old_columns:
        print("Warning: No old columns found. Will only plot Bayesian and Smurf metrics.")
    
    # Map y_metric to specific columns
    metric_mapping = {
        'pass_percentage': {
            'smurf': 'smurf_pass_percentage',
            'baysian': 'baysian_pass_percentage',
            'old': 'smurf_old_pass_percentage'
        },
        'mean_diff': {
            'smurf': 'mean_smurf_diff',
            'baysian': 'mean_baysian_diff',
            'old': 'mean_smurf_old_diff'
        },
        'mean_reference_distance': {
            'smurf': 'mean_reference_distance',
            'baysian': 'mean_reference_baysian_distance',
            'old': 'mean_reference_distance_old'
        }
    }
    
    # Filter data based on specified parameters
    filtered_df = df.copy()
    
    # Filter by sigma
    if str(sigma).replace('.', '').replace('-', '').isdigit():
        try:
            numeric_sigma = float(sigma)
            filtered_df = filtered_df[filtered_df['sigma'] == numeric_sigma]
        except (ValueError, TypeError):
            filtered_df = filtered_df[filtered_df['sigma'].astype(str) == str(sigma)]
    else:
        filtered_df = filtered_df[filtered_df['sigma'].astype(str) == str(sigma)]
    
    # Filter by grid_factor
    # Handle both numeric and string grid_factor values
    if str(grid_factor).replace('.', '').replace('-', '').isdigit():
        try:
            numeric_grid_factor = float(grid_factor)
            # Try numeric comparison first
            numeric_match = filtered_df[filtered_df['grid_factor'] == numeric_grid_factor]
            if not numeric_match.empty:
                filtered_df = numeric_match
            else:
                # If no numeric match, try string comparison
                filtered_df = filtered_df[filtered_df['grid_factor'].astype(str) == str(grid_factor)]
        except (ValueError, TypeError):
            filtered_df = filtered_df[filtered_df['grid_factor'].astype(str) == str(grid_factor)]
    else:
        filtered_df = filtered_df[filtered_df['grid_factor'].astype(str) == str(grid_factor)]
    
    # Filter by n
    try:
        numeric_n = int(n) if '.' not in str(n) else float(n)
        filtered_df = filtered_df[filtered_df['n'] == numeric_n]
    except (ValueError, TypeError):
        filtered_df = filtered_df[filtered_df['n'].astype(str) == str(n)]
    
    print(f"Filtered data shape: {filtered_df.shape}")
    
    # Check if we have any data after filtering
    if filtered_df.empty:
        print("Warning: No data found with the specified parameters!")
        print(f"Available values:")
        print(f"  sigma: {sorted(df['sigma'].unique())}")
        print(f"  grid_factor: {sorted(df['grid_factor'].unique())}")
        print(f"  n: {sorted(df['n'].unique())}")
        return None
    
    # Get unique values
    presets = sorted(filtered_df['preset_name'].unique())
    param_variations = sorted(filtered_df['parameter_varied'].unique())
    variation_values = sorted(filtered_df['variation_value'].unique())
    
    print(f"Presets: {presets}")
    print(f"Parameter variations: {param_variations}")
    print(f"Variation values: {variation_values}")
    
    # Create figure with subplots (n_presets rows, 5 columns - one for each parameter)
    n_presets = len(presets)
    n_params = len(param_variations)
    fig, axes = plt.subplots(n_presets, n_params, figsize=(4*n_params, 3*n_presets))
    
    # Handle single preset case
    if n_presets == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle(f'{y_metric} vs Variation Value by Preset and Parameter\n(σ={sigma}, grid_factor={grid_factor}, n={n})', 
                 fontsize=16, y=0.95)
    
    # Define colors and markers for different metrics
    colors = {'smurf': 'blue', 'baysian': 'red', 'old': 'green'}
    markers = {'smurf': 'o', 'baysian': 's', 'old': '^'}
    labels = {'smurf': 'Smurf', 'baysian': 'Bayesian', 'old': 'Smurf (Old)'}
    
    # Store all legend handles and labels for a single legend
    all_handles = []
    all_labels = []
    
    # Plot for each preset (row)
    for preset_idx, preset in enumerate(presets):
        # Filter data for this preset
        preset_data = filtered_df[filtered_df['preset_name'] == preset]
        
        # Plot for each parameter (column)
        for param_idx, param_varied in enumerate(param_variations):
            ax = axes[preset_idx, param_idx]
            
            # Create secondary y-axis for test counts
            ax2 = ax.twinx()
            
            # Filter data for this parameter
            param_data = preset_data[preset_data['parameter_varied'] == param_varied]
            
            if not param_data.empty:
                # Plot each metric (skip Bayesian for xs parameter)
                metrics_to_plot = ['smurf', 'baysian'] if param_varied != 'xs' else ['smurf']
                
                for metric_type in metrics_to_plot:
                    metric_col = metric_mapping[y_metric][metric_type]
                    
                    if metric_col in param_data.columns:
                        x_values = param_data['variation_value'].values
                        y_values = param_data[metric_col].values
                        
                        # Handle inf and -inf values
                        y_values_plot = y_values.copy()
                        if np.any(np.isinf(y_values)) or np.any(np.isneginf(y_values)):
                            all_y_finite = y_values[np.isfinite(y_values)]
                            if len(all_y_finite) > 0:
                                y_max = np.max(all_y_finite)
                                y_min = np.min(all_y_finite)
                                y_values_plot[np.isinf(y_values)] = y_max * 1.1
                                y_values_plot[np.isneginf(y_values)] = y_min * 1.1
                        
                        # Sort data by x-values
                        sort_indices = np.argsort(x_values)
                        x_values_sorted = x_values[sort_indices]
                        y_values_sorted = y_values_plot[sort_indices]
                        
                        # Get test counts for this metric
                        test_count_col = f'n_tests_{metric_type}' if metric_type != 'old' else 'n_tests_old'
                        if test_count_col in param_data.columns:
                            test_counts = param_data[test_count_col].values
                            test_counts_sorted = test_counts[sort_indices]
                            # Replace NaN with 0 for test count plotting
                            test_counts_sorted = np.nan_to_num(test_counts_sorted, nan=0)
                            
                            # Filter out NaN test counts for main plot
                            valid_mask = ~np.isnan(test_counts_sorted)
                            x_valid = x_values_sorted[valid_mask]
                            y_valid = y_values_sorted[valid_mask]
                            counts_valid = test_counts_sorted[valid_mask]
                            
                            # Option 4: Color intensity based on number of tests
                            max_count = counts_valid.max() if len(counts_valid) > 0 else 1
                            if max_count > 0:
                                alphas = np.clip(0.3 + 0.7 * (counts_valid / max_count), 0.3, 1.0)
                            else:
                                alphas = np.full_like(counts_valid, 0.5, dtype=float)
                            alphas = np.nan_to_num(alphas, nan=0.5)
                            
                            # Plot each point with its own alpha value
                            # Use larger markers for earlier metrics (lower z-order) so they can be seen behind later ones
                            base_marker_size = 8  # Base size for the first metric
                            if metric_type == 'smurf':
                                marker_size = base_marker_size + 6  # Largest for smurf (drawn first)
                            elif metric_type == 'baysian':
                                marker_size = base_marker_size  # Medium for baysian (drawn second)
                            else:  # old
                                marker_size = base_marker_size - 2  # Smallest for old (drawn last)
                            
                            for i, (x, y, alpha) in enumerate(zip(x_valid, y_valid, alphas)):
                                ax.plot(x, y, marker=markers[metric_type], color=colors[metric_type], 
                                       alpha=alpha, markersize=marker_size, linestyle='')
                            
                            # Connect points with line (using average alpha)
                            if len(x_valid) > 0:
                                avg_alpha = np.mean(alphas)
                                line, = ax.plot(x_valid, y_valid, 
                                               marker='', color=colors[metric_type], 
                                               linestyle='-', alpha=avg_alpha, linewidth=2)
                            
                            # Plot test counts on secondary y-axis (faded dashed lines)
                            ax2.plot(x_values_sorted, test_counts_sorted, 
                                    color=colors[metric_type], linestyle='--', alpha=0.3, linewidth=1,
                                    label=f'{labels[metric_type]} Tests')
                        else:
                            # Fallback if test count column doesn't exist
                            # Use larger markers for earlier metrics (lower z-order) so they can be seen behind later ones
                            base_marker_size = 8  # Base size for the first metric
                            if metric_type == 'smurf':
                                marker_size = base_marker_size + 6 # Largest for smurf (drawn first)
                            elif metric_type == 'baysian':
                                marker_size = base_marker_size  # Medium for baysian (drawn second)
                            else:  # old
                                marker_size = base_marker_size - 2  # Smallest for old (drawn last)
                            
                            line, = ax.plot(x_values_sorted, y_values_sorted, 
                                           marker=markers[metric_type], color=colors[metric_type], 
                                           linestyle='-', label=f'{labels[metric_type]}',
                                           alpha=0.8, markersize=marker_size)
                        
                        # Store handle and label for legend (only once per metric)
                        legend_label = f'{labels[metric_type]}'
                        if legend_label not in all_labels:
                            all_handles.append(line)
                            all_labels.append(legend_label)
                
                # Plot old metric if available
                if has_old_columns:
                    old_metric_col = metric_mapping[y_metric]['old']
                    if old_metric_col in param_data.columns:
                        x_values = param_data['variation_value'].values
                        y_values = param_data[old_metric_col].values
                        
                        # Handle inf and -inf values
                        y_values_plot = y_values.copy()
                        if np.any(np.isinf(y_values)) or np.any(np.isneginf(y_values)):
                            all_y_finite = y_values[np.isfinite(y_values)]
                            if len(all_y_finite) > 0:
                                y_max = np.max(all_y_finite)
                                y_min = np.min(all_y_finite)
                                y_values_plot[np.isinf(y_values)] = y_max * 1.1
                                y_values_plot[np.isneginf(y_values)] = y_min * 1.1
                        
                        # Sort data by x-values
                        sort_indices = np.argsort(x_values)
                        x_values_sorted = x_values[sort_indices]
                        y_values_sorted = y_values_plot[sort_indices]
                        
                        # Get test counts for old metric
                        test_count_col = 'n_tests_old'
                        if test_count_col in param_data.columns:
                            test_counts = param_data[test_count_col].values
                            test_counts_sorted = test_counts[sort_indices]
                            # Replace NaN with 0 for test count plotting
                            test_counts_sorted = np.nan_to_num(test_counts_sorted, nan=0)
                            
                            # Filter out NaN test counts for main plot
                            valid_mask = ~np.isnan(test_counts_sorted)
                            x_valid = x_values_sorted[valid_mask]
                            y_valid = y_values_sorted[valid_mask]
                            counts_valid = test_counts_sorted[valid_mask]
                            
                            # Option 4: Color intensity based on number of tests
                            max_count = counts_valid.max() if len(counts_valid) > 0 else 1
                            if max_count > 0:
                                alphas = np.clip(0.3 + 0.7 * (counts_valid / max_count), 0.3, 1.0)
                            else:
                                alphas = np.full_like(counts_valid, 0.5, dtype=float)
                            alphas = np.nan_to_num(alphas, nan=0.5)
                            
                            # Plot each point with its own alpha value
                            # Use larger markers for earlier metrics (lower z-order) so they can be seen behind later ones
                            base_marker_size = 8  # Base size for the first metric
                            marker_size = base_marker_size - 2  # Smallest for old (drawn last)
                            
                            for i, (x, y, alpha) in enumerate(zip(x_valid, y_valid, alphas)):
                                ax.plot(x, y, marker=markers['old'], color=colors['old'], 
                                       alpha=alpha, markersize=marker_size, linestyle='')
                            
                            # Connect points with line (using average alpha)
                            if len(x_valid) > 0:
                                avg_alpha = np.mean(alphas)
                                line, = ax.plot(x_valid, y_valid, 
                                               marker='', color=colors['old'], 
                                               linestyle='-', alpha=avg_alpha, linewidth=2)
                            
                            # Plot test counts on secondary y-axis (faded dashed lines)
                            ax2.plot(x_values_sorted, test_counts_sorted, 
                                    color=colors['old'], linestyle='--', alpha=0.3, linewidth=1,
                                    label=f'{labels["old"]} Tests')
                        else:
                            # Fallback if test count column doesn't exist
                            # Use larger markers for earlier metrics (lower z-order) so they can be seen behind later ones
                            base_marker_size = 8  # Base size for the first metric
                            marker_size = base_marker_size - 2  # Smallest for old (drawn last)
                            
                            line, = ax.plot(x_values_sorted, y_values_sorted, 
                                           marker=markers['old'], color=colors['old'], 
                                           linestyle='-', label=f'{labels["old"]}',
                                           alpha=0.8, markersize=marker_size)
                        
                        # Store handle and label for legend (only once per metric)
                        legend_label = f'{labels["old"]}'
                        if legend_label not in all_labels:
                            all_handles.append(line)
                            all_labels.append(legend_label)
            
            # Customize subplot
            ax.set_xlabel('Variation Value')
            if param_idx == 0:  # Only show y-label for leftmost panels
                ax.set_ylabel(y_metric.replace('_', ' ').title())
            ax.set_title(f'{preset} - {param_varied}')
            ax.grid(True, alpha=0.3)
            
            # Customize secondary y-axis
            ax2.set_ylabel('Number of Tests', color='gray', fontsize=8)
            ax2.tick_params(axis='y', labelcolor='gray', labelsize=8)
            ax2.grid(False)  # No grid for secondary axis
    
    # Create a single legend for all panels
    if all_handles:
        fig.legend(all_handles, all_labels, loc='upper center', bbox_to_anchor=(0.5, 0.02), 
                  ncol=len(all_labels), fancybox=True, shadow=True, fontsize=10)
    
    # Adjust layout
    plt.subplots_adjust(top=0.9, bottom=0.15, left=0.05, right=0.95, hspace=0.4, wspace=0.3)
    
    # Save plot if output file is specified
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_file}")
    
    return fig

# def main():
#     parser = argparse.ArgumentParser(description="Analyze smurf distance test results")
#     parser.add_argument("csv_file", help="Path to the combined results CSV file")
#     parser.add_argument("--no-viz", action="store_true", help="Skip creating visualizations")
#     parser.add_argument("--aggregate", action="store_true", help="Create aggregated results by configuration")
#     parser.add_argument("--clean", action="store_true", help="Remove entries with -inf reference values in aggregation")
#     parser.add_argument("--plot", choices=['n', 'sigma', 'grid_factor'], 
#                        help="Create dependency plots for specified parameter")
#     parser.add_argument("--y-metric", choices=['smurf_pass_percentage', 'baysian_pass_percentage', 
#                                               'mean_reference_distance', 'mean_reference_baysian_distance',
#                                               'mean_smurf_diff', 'mean_baysian_diff'],
#                        help="Y-axis metric for dependency plots")
#     parser.add_argument("--fixed-params", type=str, 
#                        help="Fixed parameters as JSON string, e.g., '{\"sigma\": 0.1, \"grid_factor\": 0.5}'")
    
#     args = parser.parse_args()
    
#     # Load and analyze data
#     df = load_and_clean_data(args.csv_file)
#     config_summary = configuration_summary(df)
#     ranking = rank_configurations(config_summary, df)
#     best_configs_df = best_configuration_per_preset(df)
#     param_analysis = parameter_sensitivity_analysis(df)
#     variation_analysis, param_variation = variation_size_analysis(df)
#     sigma_effect, n_effect, grid_effect = parameter_effect_analysis(df)
    
#     # Create aggregated results if requested
#     if args.aggregate:
#         aggregated_df = aggregate_test_results(df, 'aggregated_test_results.csv', include_config_summary=True, clean=args.clean)
    
#     # Create dependency plots if requested
#     if args.plot and args.y_metric:
#         import json
#         fixed_params = {}
#         if args.fixed_params:
#             try:
#                 fixed_params = json.loads(args.fixed_params)
#             except json.JSONDecodeError:
#                 print("Warning: Invalid JSON in --fixed-params, using defaults")
        
#         plot_filename = f"dependency_plot_{args.plot}_vs_{args.y_metric}.png"
#         plot_parameter_dependency(aggregated_df if args.aggregate else df, 
#                                 args.plot, args.y_metric, fixed_params, plot_filename)
    
#     # Create visualizations
#     if not args.no_viz:
#         create_visualizations(df, config_summary, ranking, best_configs_df)
    
#     # Save summary to CSV
#     config_summary.to_csv('configuration_summary.csv')
#     ranking.to_csv('configuration_ranking.csv')
#     best_configs_df.to_csv('best_configuration_per_preset.csv')
#     param_analysis.to_csv('parameter_sensitivity.csv')
#     variation_analysis.to_csv('variation_analysis.csv')
#     sigma_effect.to_csv('sigma_effect.csv')
#     n_effect.to_csv('n_effect.csv')
#     grid_effect.to_csv('grid_factor_effect.csv')
    
#     print("\n" + "="*80)
#     print("SUMMARY FILES SAVED")
#     print("="*80)
#     print("configuration_summary.csv - Overall configuration performance")
#     print("configuration_ranking.csv - Ranked configurations")
#     print("best_configuration_per_preset.csv - Best configuration for each preset")
#     print("parameter_sensitivity.csv - Parameter sensitivity analysis")
#     print("variation_analysis.csv - Variation size analysis")
#     print("sigma_effect.csv - Effect of sigma on test results")
#     print("n_effect.csv - Effect of n on test results")
#     print("grid_factor_effect.csv - Effect of grid_factor on test results")
#     if args.aggregate:
#         print("aggregated_test_results.csv - Aggregated results by configuration")
#     if not args.no_viz:
#         print("test_results_analysis.png - Visualizations")

# if __name__ == "__main__":
#     main() 