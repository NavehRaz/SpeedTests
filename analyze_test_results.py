#!/usr/bin/env python3
"""
Analysis script for adaptive vs non-adaptive test results.

This script analyzes the combined CSV results to provide:
1. Summary of pass/fail rates for each configuration
2. Ranking of configurations by performance
3. Analysis of parameter sensitivity
4. Comparison of variation sizes
5. Runtime analysis
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
    
    # Create configuration ID based on n, step_size, adaptive_step_divisor, and time_unit
    df['config_id'] = (df['step_size'].astype(str) + '_' + 
                      df['adaptive_step_divisor'].astype(str) + '_' + 
                      df['n'].astype(str))
    
    # Convert theta from string back to list for analysis
    df['theta_list'] = df['theta'].apply(eval)
    
    print(f"Loaded {len(df)} test results")
    print(f"Unique configurations (step_size, adaptive_step_divisor, n): {df['config_id'].nunique()}")
    print(f"Unique presets: {df['preset_name'].unique()}")
    print(f"Unique step_size values: {sorted(df['step_size'].unique())}")
    print(f"Unique adaptive_step_divisor values: {sorted(df['adaptive_step_divisor'].unique())}")
    print(f"Unique n values: {sorted(df['n'].unique())}")
    
    return df

def configuration_summary(df):
    """
    Create a summary of pass/fail rates for each configuration (n, step_size, adaptive_step_divisor).
    
    Args:
        df (pandas.DataFrame): Cleaned DataFrame
        
    Returns:
        pandas.DataFrame: Summary DataFrame
    """
    print("\n" + "="*80)
    print("CONFIGURATION SUMMARY (step_size, adaptive_step_divisor, n)")
    print("="*80)
    
    # Group by configuration (step_size, adaptive_step_divisor, n)
    config_summary = df.groupby('config_id').agg({
        'step_size': 'first',
        'adaptive_step_divisor': 'first',
        'n': 'first',
        'pass_adaptive': ['count', 'sum', 'mean'],
        'pass_nonadaptive': ['count', 'sum', 'mean']
    }).round(3)
    
    # Flatten column names
    config_summary.columns = ['_'.join(col).strip() for col in config_summary.columns]
    
    # Rename columns for clarity
    config_summary = config_summary.rename(columns={
        'step_size_first': 'step_size',
        'adaptive_step_divisor_first': 'adaptive_step_divisor',
        'n_first': 'n',
        'pass_adaptive_count': 'total_tests',
        'pass_adaptive_sum': 'adaptive_passed',
        'pass_adaptive_mean': 'adaptive_pass_rate',
        'pass_nonadaptive_sum': 'nonadaptive_passed',
        'pass_nonadaptive_mean': 'nonadaptive_pass_rate'
    })
    
    # Calculate overall pass rate (both tests must pass)
    config_summary['both_passed'] = df.groupby('config_id').apply(
        lambda x: ((x['pass_adaptive'] == 1) & (x['pass_nonadaptive'] == 1)).sum()
    )
    config_summary['both_pass_rate'] = (config_summary['both_passed'] / config_summary['total_tests']).round(3)
    
    # Add configuration details
    config_summary['config_details'] = (
        f"step={config_summary['step_size']}, "
        f"div={config_summary['adaptive_step_divisor']}, "
        f"n={config_summary['n']}"
    )
    
    # Sort by overall pass rate for better readability
    config_summary = config_summary.sort_values('both_pass_rate', ascending=False)
    
    print("Configuration Performance (sorted by overall pass rate):")
    display_cols = ['config_details', 'total_tests', 
                   'adaptive_passed', 'adaptive_pass_rate', 
                   'nonadaptive_passed', 'nonadaptive_pass_rate',
                   'both_passed', 'both_pass_rate']
    
    print(config_summary[display_cols].to_string())
    
    return config_summary

def rank_configurations(config_summary, df):
    """
    Rank configurations (step_size, adaptive_step_divisor, n) by performance.
    
    Args:
        config_summary (pandas.DataFrame): Configuration summary
        df (pandas.DataFrame): Original DataFrame with preset information
        
    Returns:
        pandas.DataFrame: Ranked configurations
    """
    print("\n" + "="*80)
    print("CONFIGURATION RANKING (step_size, adaptive_step_divisor, n)")
    print("="*80)
    
    # Create ranking based on multiple criteria
    ranking = config_summary.copy()
    
    # Sort by overall pass rate (both tests), then by individual test rates
    ranking = ranking.sort_values(['both_pass_rate', 'adaptive_pass_rate', 'nonadaptive_pass_rate'], 
                                 ascending=[False, False, False])
    
    ranking['rank'] = range(1, len(ranking) + 1)
    
    print("Top 10 Configurations:")
    print(ranking[['rank', 'config_details', 'both_pass_rate', 
                  'adaptive_pass_rate', 'nonadaptive_pass_rate']].head(10).to_string(index=False))
    
    print("\nBottom 10 Configurations:")
    print(ranking[['rank', 'config_details', 'both_pass_rate', 
                  'adaptive_pass_rate', 'nonadaptive_pass_rate']].tail(10).to_string(index=False))
    
    return ranking

def best_configuration_per_preset(df):
    """
    Find the best configuration (step_size, adaptive_step_divisor, n) for each preset.
    
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
        'step_size': 'first',
        'adaptive_step_divisor': 'first',
        'n': 'first',
        'pass_adaptive': 'mean',
        'pass_nonadaptive': 'mean'
    }).round(3)
    
    # Calculate overall pass rate
    preset_config_summary['overall_pass_rate'] = (
        df.groupby(['preset_name', 'config_id']).apply(
            lambda x: ((x['pass_adaptive'] == 1) & (x['pass_nonadaptive'] == 1)).mean()
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
            'step_size': best_config['step_size'],
            'adaptive_step_divisor': best_config['adaptive_step_divisor'],
            'n': best_config['n'],
            'adaptive_pass_rate': best_config['pass_adaptive'],
            'nonadaptive_pass_rate': best_config['pass_nonadaptive'],
            'overall_pass_rate': best_config['overall_pass_rate']
        })
    
    best_configs_df = pd.DataFrame(best_configs)
    best_configs_df = best_configs_df.sort_values('overall_pass_rate', ascending=False)
    
    print("Best Configuration for Each Preset:")
    display_cols = ['preset', 'step_size', 'adaptive_step_divisor', 'n', 
                   'adaptive_pass_rate', 'nonadaptive_pass_rate', 'overall_pass_rate']
    
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
        'pass_adaptive': 'mean',
        'pass_nonadaptive': 'mean',
        'variation_value': 'count'
    }).round(3)
    
    param_analysis.columns = ['adaptive_pass_rate', 'nonadaptive_pass_rate', 'test_count']
    
    # Calculate overall pass rate for each parameter
    param_analysis['overall_pass_rate'] = (
        df.groupby(['preset_name', 'parameter_varied']).apply(
            lambda x: ((x['pass_adaptive'] == 1) & (x['pass_nonadaptive'] == 1)).mean()
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
        'pass_adaptive': 'mean',
        'pass_nonadaptive': 'mean',
        'parameter_varied': 'count'
    }).round(3)
    
    variation_analysis.columns = ['adaptive_pass_rate', 'nonadaptive_pass_rate', 'test_count']
    
    # Calculate overall pass rate
    variation_analysis['overall_pass_rate'] = (
        df.groupby('variation_value').apply(
            lambda x: ((x['pass_adaptive'] == 1) & (x['pass_nonadaptive'] == 1)).mean()
        )
    ).round(3)
    
    print("Test Results by Variation Size:")
    print(variation_analysis.to_string())
    
    # Analyze by parameter and variation
    param_variation = df.groupby(['parameter_varied', 'variation_value']).agg({
        'pass_adaptive': 'mean',
        'pass_nonadaptive': 'mean'
    }).round(3)
    
    param_variation.columns = ['adaptive_pass_rate', 'nonadaptive_pass_rate']
    
    param_variation['overall_pass_rate'] = (
        df.groupby(['parameter_varied', 'variation_value']).apply(
            lambda x: ((x['pass_adaptive'] == 1) & (x['pass_nonadaptive'] == 1)).mean()
        )
    ).round(3)
    
    print("\nTest Results by Parameter and Variation Size:")
    print(param_variation.to_string())
    
    return variation_analysis, param_variation

def parameter_effect_analysis(df):
    """
    Analyze the effect of step_size, n, and adaptive_step_divisor on test results.
    
    Args:
        df (pandas.DataFrame): Cleaned DataFrame
        
    Returns:
        tuple: Three DataFrames for step_size, n, and adaptive_step_divisor effects
    """
    print("\n" + "="*80)
    print("PARAMETER EFFECT ANALYSIS")
    print("="*80)
    
    # 1. Step size effect analysis
    print("\n1. STEP SIZE EFFECT ANALYSIS:")
    print("-" * 50)
    step_size_effect = df.groupby('step_size').agg({
        'pass_adaptive': 'mean',
        'pass_nonadaptive': 'mean',
        'parameter_varied': 'count'
    }).round(3)
    
    step_size_effect.columns = ['adaptive_pass_rate', 'nonadaptive_pass_rate', 'total_tests']
    
    # Calculate overall pass rate
    step_size_effect['overall_pass_rate'] = (
        df.groupby('step_size').apply(
            lambda x: ((x['pass_adaptive'] == 1) & (x['pass_nonadaptive'] == 1)).mean()
        )
    ).round(3)
    
    print("Effect of Step Size on Test Results:")
    print(step_size_effect.to_string())
    
    # 2. N effect analysis
    print("\n2. N EFFECT ANALYSIS:")
    print("-" * 50)
    n_effect = df.groupby('n').agg({
        'pass_adaptive': 'mean',
        'pass_nonadaptive': 'mean',
        'parameter_varied': 'count'
    }).round(3)
    
    n_effect.columns = ['adaptive_pass_rate', 'nonadaptive_pass_rate', 'total_tests']
    
    # Calculate overall pass rate
    n_effect['overall_pass_rate'] = (
        df.groupby('n').apply(
            lambda x: ((x['pass_adaptive'] == 1) & (x['pass_nonadaptive'] == 1)).mean()
        )
    ).round(3)
    
    print("Effect of N on Test Results:")
    print(n_effect.to_string())
    
    # 3. Adaptive step divisor effect analysis
    print("\n3. ADAPTIVE STEP DIVISOR EFFECT ANALYSIS:")
    print("-" * 50)
    divisor_effect = df.groupby('adaptive_step_divisor').agg({
        'pass_adaptive': 'mean',
        'pass_nonadaptive': 'mean',
        'parameter_varied': 'count'
    }).round(3)
    
    divisor_effect.columns = ['adaptive_pass_rate', 'nonadaptive_pass_rate', 'total_tests']
    
    # Calculate overall pass rate
    divisor_effect['overall_pass_rate'] = (
        df.groupby('adaptive_step_divisor').apply(
            lambda x: ((x['pass_adaptive'] == 1) & (x['pass_nonadaptive'] == 1)).mean()
        )
    ).round(3)
    
    print("Effect of Adaptive Step Divisor on Test Results:")
    print(divisor_effect.to_string())
    
    return step_size_effect, n_effect, divisor_effect

def runtime_analysis(df):
    """
    Analyze runtime performance of adaptive vs non-adaptive methods.
    
    Args:
        df (pandas.DataFrame): Cleaned DataFrame
        
    Returns:
        pandas.DataFrame: Runtime analysis summary
    """
    print("\n" + "="*80)
    print("RUNTIME ANALYSIS")
    print("="*80)
    
    # Filter out error rows
    df_runtime = df[df['runtime_adaptive'] != 'ERROR'].copy()
    df_runtime = df_runtime[df_runtime['runtime_nonadaptive'] != 'ERROR'].copy()
    
    if df_runtime.empty:
        print("No valid runtime data found")
        return pd.DataFrame()
    
    # Convert to numeric
    df_runtime['runtime_adaptive'] = pd.to_numeric(df_runtime['runtime_adaptive'], errors='coerce')
    df_runtime['runtime_nonadaptive'] = pd.to_numeric(df_runtime['runtime_nonadaptive'], errors='coerce')
    
    # Calculate speedup
    df_runtime['speedup'] = df_runtime['runtime_nonadaptive'] / df_runtime['runtime_adaptive']
    
    # Group by configuration
    runtime_summary = df_runtime.groupby('config_id').agg({
        'step_size': 'first',
        'adaptive_step_divisor': 'first',
        'n': 'first',
        'runtime_adaptive': ['mean', 'std'],
        'runtime_nonadaptive': ['mean', 'std'],
        'speedup': ['mean', 'std']
    }).round(4)
    
    # Flatten column names
    runtime_summary.columns = ['_'.join(col).strip() for col in runtime_summary.columns]
    
    # Rename columns
    runtime_summary = runtime_summary.rename(columns={
        'step_size_first': 'step_size',
        'adaptive_step_divisor_first': 'adaptive_step_divisor',
        'n_first': 'n',
        'runtime_adaptive_mean': 'mean_adaptive_runtime',
        'runtime_adaptive_std': 'std_adaptive_runtime',
        'runtime_nonadaptive_mean': 'mean_nonadaptive_runtime',
        'runtime_nonadaptive_std': 'std_nonadaptive_runtime',
        'speedup_mean': 'mean_speedup',
        'speedup_std': 'std_speedup'
    })
    
    # Add configuration details
    runtime_summary['config_details'] = (
        f"step={runtime_summary['step_size']}, "
        f"div={runtime_summary['adaptive_step_divisor']}, "
        f"n={runtime_summary['n']}"
    )
    
    # Sort by speedup
    runtime_summary = runtime_summary.sort_values('mean_speedup', ascending=False)
    
    print("Runtime Performance by Configuration:")
    display_cols = ['config_details', 'mean_adaptive_runtime', 'std_adaptive_runtime',
                   'mean_nonadaptive_runtime', 'std_nonadaptive_runtime',
                   'mean_speedup', 'std_speedup']
    
    print(runtime_summary[display_cols].to_string())
    
    # Overall statistics
    print(f"\nOverall Runtime Statistics:")
    print(f"Mean adaptive runtime: {df_runtime['runtime_adaptive'].mean():.4f}s ± {df_runtime['runtime_adaptive'].std():.4f}s")
    print(f"Mean non-adaptive runtime: {df_runtime['runtime_nonadaptive'].mean():.4f}s ± {df_runtime['runtime_nonadaptive'].std():.4f}s")
    print(f"Mean speedup: {df_runtime['speedup'].mean():.2f}x ± {df_runtime['speedup'].std():.2f}x")
    print(f"Max speedup: {df_runtime['speedup'].max():.2f}x")
    print(f"Min speedup: {df_runtime['speedup'].min():.2f}x")
    
    return runtime_summary

def aggregate_test_results(df_or_csv, output_csv=None, include_config_summary=False):
    """
    Aggregate test results by configuration, calculating means, stds, and pass percentages.
    
    Args:
        df_or_csv: Either a pandas DataFrame or path to CSV file
        output_csv (str): Optional path to save the aggregated results CSV
        include_config_summary (bool): Whether to include the config_summary column
        
    Returns:
        pandas.DataFrame: Aggregated DataFrame with means, stds, and pass percentages
    """
    import numpy as np
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
    agg_columns = ['preset_name', 'step_size', 'adaptive_step_divisor', 'n', 'parameter_varied', 'variation_value']
    
    # Check if all required columns exist
    missing_cols = [col for col in agg_columns if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing columns: {missing_cols}")
        return pd.DataFrame()
    
    # Show data distribution before filtering
    print(f"\nData distribution before filtering:")
    print(f"Rows with 'ERROR' in baysian_distance_adaptive: {(df['baysian_distance_adaptive'] == 'ERROR').sum()}")
    print(f"Rows with 'ERROR' in baysian_distance_nonadaptive: {(df['baysian_distance_nonadaptive'] == 'ERROR').sum()}")
    print(f"Rows with -inf in reference_baysian_distance_adaptive: {(df['reference_baysian_distance_adaptive'] == -np.inf).sum()}")
    print(f"Rows with -inf in reference_baysian_distance_nonadaptive: {(df['reference_baysian_distance_nonadaptive'] == -np.inf).sum()}")
    
    # Create separate dataframes for adaptive and non-adaptive tests
    # Filter out error rows and -inf reference values for each method separately
    df_adaptive = df[df['baysian_distance_adaptive'] != 'ERROR'].copy()
    df_adaptive = df_adaptive[~df_adaptive['reference_baysian_distance_adaptive'].isin([-np.inf, '-inf'])].copy()
    
    df_nonadaptive = df[df['baysian_distance_nonadaptive'] != 'ERROR'].copy()
    df_nonadaptive = df_nonadaptive[~df_nonadaptive['reference_baysian_distance_nonadaptive'].isin([-np.inf, '-inf'])].copy()
    
    # For overall counting, we need tests where both methods are valid
    df_both_valid = df[df['baysian_distance_adaptive'] != 'ERROR'].copy()
    df_both_valid = df_both_valid[df_both_valid['baysian_distance_nonadaptive'] != 'ERROR'].copy()
    df_both_valid = df_both_valid[~df_both_valid['reference_baysian_distance_adaptive'].isin([-np.inf, '-inf'])].copy()
    df_both_valid = df_both_valid[~df_both_valid['reference_baysian_distance_nonadaptive'].isin([-np.inf, '-inf'])].copy()
    
    # Convert to numeric for each dataframe separately
    df_adaptive['baysian_distance_adaptive'] = pd.to_numeric(df_adaptive['baysian_distance_adaptive'], errors='coerce')
    df_adaptive['reference_baysian_distance_adaptive'] = pd.to_numeric(df_adaptive['reference_baysian_distance_adaptive'], errors='coerce')
    df_adaptive['runtime_adaptive'] = pd.to_numeric(df_adaptive['runtime_adaptive'], errors='coerce')
    
    df_nonadaptive['baysian_distance_nonadaptive'] = pd.to_numeric(df_nonadaptive['baysian_distance_nonadaptive'], errors='coerce')
    df_nonadaptive['reference_baysian_distance_nonadaptive'] = pd.to_numeric(df_nonadaptive['reference_baysian_distance_nonadaptive'], errors='coerce')
    df_nonadaptive['runtime_nonadaptive'] = pd.to_numeric(df_nonadaptive['runtime_nonadaptive'], errors='coerce')
    
    # Convert both_valid dataframe for overall statistics
    df_both_valid['baysian_distance_adaptive'] = pd.to_numeric(df_both_valid['baysian_distance_adaptive'], errors='coerce')
    df_both_valid['baysian_distance_nonadaptive'] = pd.to_numeric(df_both_valid['baysian_distance_nonadaptive'], errors='coerce')
    df_both_valid['reference_baysian_distance_adaptive'] = pd.to_numeric(df_both_valid['reference_baysian_distance_adaptive'], errors='coerce')
    df_both_valid['reference_baysian_distance_nonadaptive'] = pd.to_numeric(df_both_valid['reference_baysian_distance_nonadaptive'], errors='coerce')
    df_both_valid['runtime_adaptive'] = pd.to_numeric(df_both_valid['runtime_adaptive'], errors='coerce')
    df_both_valid['runtime_nonadaptive'] = pd.to_numeric(df_both_valid['runtime_nonadaptive'], errors='coerce')
    
    print(f"Valid adaptive tests: {len(df_adaptive)}")
    print(f"Valid non-adaptive tests: {len(df_nonadaptive)}")
    print(f"Tests where both methods are valid: {len(df_both_valid)}")
    
    # Calculate differences from reference for both_valid dataframe
    df_both_valid['baysian_distance_adaptive_diff'] = (
        df_both_valid['reference_baysian_distance_adaptive'] - df_both_valid['baysian_distance_adaptive']
    )
    df_both_valid['baysian_distance_nonadaptive_diff'] = (
        df_both_valid['reference_baysian_distance_nonadaptive'] - df_both_valid['baysian_distance_nonadaptive']
    )
    
    # Show grouping information
    print(f"\nGrouping by: {agg_columns}")
    print(f"Number of unique groups: {df_both_valid.groupby(agg_columns).ngroups}")
    
    # Show sample of groups for debugging
    sample_groups = df_both_valid.groupby(agg_columns).size().head(10)
    print(f"Sample group sizes:")
    print(sample_groups)
    
    # Aggregate adaptive statistics (only on valid adaptive tests)
    adaptive_agg_dict = {
        'baysian_distance_adaptive': ['mean', 'std'],
        'reference_baysian_distance_adaptive': ['first', 'mean', 'std'],
        'runtime_adaptive': ['mean', 'std'],
        'theta': 'first'
    }
    
    adaptive_aggregated = df_adaptive.groupby(agg_columns).agg(adaptive_agg_dict).round(4)
    adaptive_aggregated.columns = ['_'.join(col).strip() for col in adaptive_aggregated.columns]
    
    # Aggregate non-adaptive statistics (only on valid non-adaptive tests)
    nonadaptive_agg_dict = {
        'baysian_distance_nonadaptive': ['mean', 'std'],
        'reference_baysian_distance_nonadaptive': ['first', 'mean', 'std'],
        'runtime_nonadaptive': ['mean', 'std'],
        'theta': 'first'
    }
    
    nonadaptive_aggregated = df_nonadaptive.groupby(agg_columns).agg(nonadaptive_agg_dict).round(4)
    nonadaptive_aggregated.columns = ['_'.join(col).strip() for col in nonadaptive_aggregated.columns]
    
    # Aggregate both-valid statistics (for differences and overall counts)
    both_valid_agg_dict = {
        'baysian_distance_adaptive_diff': ['mean', 'std'],
        'baysian_distance_nonadaptive_diff': ['mean', 'std'],
        'theta': 'first'
    }
    
    both_valid_aggregated = df_both_valid.groupby(agg_columns).agg(both_valid_agg_dict).round(4)
    both_valid_aggregated.columns = ['_'.join(col).strip() for col in both_valid_aggregated.columns]
    
    # Add count columns
    adaptive_test_counts = df_adaptive.groupby(agg_columns).size()
    nonadaptive_test_counts = df_nonadaptive.groupby(agg_columns).size()
    both_valid_test_counts = df_both_valid.groupby(agg_columns).size()
    
    # Calculate pass rates (only on valid tests for each method)
    adaptive_pass_rates = df_adaptive.groupby(agg_columns)['pass_adaptive'].mean()
    nonadaptive_pass_rates = df_nonadaptive.groupby(agg_columns)['pass_nonadaptive'].mean()
    
    # Combine all aggregations
    aggregated = pd.concat([
        adaptive_aggregated,
        nonadaptive_aggregated,
        both_valid_aggregated,
        adaptive_test_counts.rename('n_tests_adaptive'),
        nonadaptive_test_counts.rename('n_tests_nonadaptive'),
        both_valid_test_counts.rename('n_tests_both_valid'),
        adaptive_pass_rates.rename('pass_adaptive_mean'),
        nonadaptive_pass_rates.rename('pass_nonadaptive_mean')
    ], axis=1)
    
    # Rename columns for clarity
    column_mapping = {
        'n_tests_adaptive': 'n_tests_adaptive',
        'n_tests_nonadaptive': 'n_tests_nonadaptive',
        'n_tests_both_valid': 'n_tests_both_valid',
        'baysian_distance_adaptive_mean': 'mean_baysian_distance_adaptive',
        'baysian_distance_adaptive_std': 'std_baysian_distance_adaptive',
        'baysian_distance_nonadaptive_mean': 'mean_baysian_distance_nonadaptive',
        'baysian_distance_nonadaptive_std': 'std_baysian_distance_nonadaptive',
        'pass_adaptive_mean': 'adaptive_pass_percentage',
        'pass_nonadaptive_mean': 'nonadaptive_pass_percentage',
        'reference_baysian_distance_adaptive_first': 'reference_baysian_distance_adaptive',
        'reference_baysian_distance_adaptive_mean': 'mean_reference_baysian_distance_adaptive',
        'reference_baysian_distance_adaptive_std': 'std_reference_baysian_distance_adaptive',
        'reference_baysian_distance_nonadaptive_first': 'reference_baysian_distance_nonadaptive',
        'reference_baysian_distance_nonadaptive_mean': 'mean_reference_baysian_distance_nonadaptive',
        'reference_baysian_distance_nonadaptive_std': 'std_reference_baysian_distance_nonadaptive',
        'baysian_distance_adaptive_diff_mean': 'mean_adaptive_diff',
        'baysian_distance_adaptive_diff_std': 'std_adaptive_diff',
        'baysian_distance_nonadaptive_diff_mean': 'mean_nonadaptive_diff',
        'baysian_distance_nonadaptive_diff_std': 'std_nonadaptive_diff',
        'runtime_adaptive_mean': 'mean_adaptive_runtime',
        'runtime_adaptive_std': 'std_adaptive_runtime',
        'runtime_nonadaptive_mean': 'mean_nonadaptive_runtime',
        'runtime_nonadaptive_std': 'std_nonadaptive_runtime',
        'theta_first': 'theta'
    }
    
    aggregated = aggregated.rename(columns=column_mapping)
    
    # Convert percentages to actual percentages (multiply by 100)
    aggregated['adaptive_pass_percentage'] = (aggregated['adaptive_pass_percentage'] * 100).round(2)
    aggregated['nonadaptive_pass_percentage'] = (aggregated['nonadaptive_pass_percentage'] * 100).round(2)
    
    # Reset index to make configuration columns regular columns
    aggregated = aggregated.reset_index()
    
    # Add configuration summary column if requested
    if include_config_summary:
        aggregated['config_summary'] = (
            f"step={aggregated['step_size']}, "
            f"div={aggregated['adaptive_step_divisor']}, "
            f"n={aggregated['n']}, "
            f"{aggregated['parameter_varied']}={aggregated['variation_value']}"
        )
    
    # Sort by preset and then by adaptive pass percentage
    aggregated = aggregated.sort_values(['preset_name', 'adaptive_pass_percentage'], ascending=[True, False])
    
    print(f"Aggregated data shape: {aggregated.shape}")
    print(f"Unique configurations: {len(aggregated)}")
    
    # Display summary statistics
    print("\nSummary Statistics:")
    print(f"Mean adaptive pass percentage: {aggregated['adaptive_pass_percentage'].mean():.2f}%")
    print(f"Mean non-adaptive pass percentage: {aggregated['nonadaptive_pass_percentage'].mean():.2f}%")
    print(f"Configurations with 100% adaptive pass rate: {(aggregated['adaptive_pass_percentage'] == 100).sum()}")
    print(f"Configurations with 100% non-adaptive pass rate: {(aggregated['nonadaptive_pass_percentage'] == 100).sum()}")
    
    # Show top 10 configurations by adaptive pass percentage
    print("\nTop 10 Configurations by Adaptive Pass Percentage:")
    top_configs = aggregated.nlargest(10, 'adaptive_pass_percentage')
    display_cols = ['preset_name', 'n_tests_adaptive', 'n_tests_nonadaptive', 'n_tests_both_valid', 
                   'adaptive_pass_percentage', 'nonadaptive_pass_percentage', 
                   'mean_baysian_distance_adaptive', 'std_baysian_distance_adaptive',
                   'mean_reference_baysian_distance_adaptive', 'std_reference_baysian_distance_adaptive',
                   'mean_adaptive_diff', 'std_adaptive_diff', 'mean_adaptive_runtime', 'mean_nonadaptive_runtime']
    
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
    
    # Set up the plotting style
    plt.style.use('default')
    
    # Create figure with proper spacing for title
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Adaptive vs Non-Adaptive Test Results Analysis', fontsize=16, y=0.95)
    
    # 1. Best configuration per preset
    plot_columns = ['adaptive_pass_rate', 'nonadaptive_pass_rate', 'overall_pass_rate']
    plot_labels = ['Adaptive', 'Non-Adaptive', 'Overall']
    
    best_configs_df.plot(x='preset', y=plot_columns, kind='bar', ax=axes[0,0])
    axes[0,0].set_title('Best Configuration Performance per Preset')
    axes[0,0].set_ylabel('Pass Rate')
    axes[0,0].tick_params(axis='x', rotation=45)
    axes[0,0].legend(plot_labels)
    
    # 2. Parameter sensitivity
    param_sensitivity = df.groupby('parameter_varied')['pass_adaptive'].mean()
    param_sensitivity.plot(kind='bar', ax=axes[0,1], color='skyblue')
    axes[0,1].set_title('Parameter Sensitivity (Adaptive Tests)')
    axes[0,1].set_ylabel('Pass Rate')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # 3. Variation size comparison
    variation_comparison = df.groupby('variation_value')[['pass_adaptive', 'pass_nonadaptive']].mean()
    variation_comparison.plot(kind='bar', ax=axes[1,0])
    axes[1,0].set_title('Test Results by Variation Size')
    axes[1,0].set_ylabel('Pass Rate')
    axes[1,0].tick_params(axis='x', rotation=45)
    axes[1,0].legend(['Adaptive', 'Non-Adaptive'])
    
    # 4. Top configurations
    top_configs = ranking.head(10)
    top_configs.plot(x='config_details', y=['adaptive_pass_rate', 'nonadaptive_pass_rate'], 
                    kind='bar', ax=axes[1,1])
    axes[1,1].set_title('Top 10 Configurations (step_size, adaptive_step_divisor, n)')
    axes[1,1].set_ylabel('Pass Rate')
    axes[1,1].tick_params(axis='x', rotation=45)
    axes[1,1].legend(['Adaptive', 'Non-Adaptive'])
    
    # Fix padding between title and panels
    plt.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.9, hspace=0.3, wspace=0.3)
    
    plt.savefig('test_results_analysis.png', dpi=300, bbox_inches='tight')
    print("Visualizations saved to: test_results_analysis.png")
    
    return fig

def plot_by_preset(df_or_csv, y_metric, step_size=0.1, adaptive_step_divisor=0.1, n=40000, output_file=None):
    """
    Plot dependency of y_metric on variation_value for each parameter, organized by preset.
    
    Creates a multi-panel plot showing the dependency of y_metric on variation_value 
    for each preset and parameter, comparing adaptive vs non-adaptive methods.
    
    Note: Color intensity and text annotations show the number of tests used for each data point.
          - Darker/more opaque points indicate more tests (higher statistical reliability)
          - Text annotations show exact test counts (n=X) for every other point to avoid clutter
          - Both visual cues help assess the statistical reliability of each measurement.
    
    Args:
        df_or_csv: Either a pandas DataFrame or path to CSV file
        y_metric (str): Metric to plot on y-axis. Can be:
                       - 'pass_percentage': Shows pass percentages for adaptive vs non-adaptive
                       - 'mean_distance': Shows mean Bayesian distances for adaptive vs non-adaptive
                       - 'mean_runtime': Shows mean runtime for adaptive vs non-adaptive
                       - 'mean_diff': Shows mean differences from reference for adaptive vs non-adaptive
        step_size (float): Step size parameter to filter data (default: 0.1)
        adaptive_step_divisor (float): Adaptive step divisor parameter to filter data (default: 0.1)
        n (int): N parameter to filter data (default: 40000)
        output_file (str): Optional path to save the plot
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    print(f"\n" + "="*80)
    print(f"PLOTTING {y_metric} BY PRESET")
    print(f"Parameters: step_size={step_size}, adaptive_step_divisor={adaptive_step_divisor}, n={n}")
    print("="*80)
    
    # Load data if CSV path is provided
    if isinstance(df_or_csv, str):
        print(f"Loading data from: {df_or_csv}")
        df = pd.read_csv(df_or_csv)
    else:
        df = df_or_csv.copy()
    
    # Validate y_metric
    valid_y_metrics = ['pass_percentage', 'mean_distance', 'mean_runtime', 'mean_diff']
    if y_metric not in valid_y_metrics:
        raise ValueError(f"y_metric must be one of {valid_y_metrics}")
    
    # Map y_metric to specific columns
    metric_mapping = {
        'pass_percentage': {
            'adaptive': 'adaptive_pass_percentage',
            'nonadaptive': 'nonadaptive_pass_percentage'
        },
        'mean_distance': {
            'adaptive': 'mean_baysian_distance_adaptive',
            'nonadaptive': 'mean_baysian_distance_nonadaptive'
        },
        'mean_runtime': {
            'adaptive': 'mean_adaptive_runtime',
            'nonadaptive': 'mean_nonadaptive_runtime'
        },
        'mean_diff': {
            'adaptive': 'mean_adaptive_diff',
            'nonadaptive': 'mean_nonadaptive_diff'
        }
    }
    
    # Filter data based on specified parameters
    filtered_df = df.copy()
    
    # Filter by step_size
    try:
        numeric_step_size = float(step_size)
        filtered_df = filtered_df[filtered_df['step_size'] == numeric_step_size]
    except (ValueError, TypeError):
        filtered_df = filtered_df[filtered_df['step_size'].astype(str) == str(step_size)]
    
    # Filter by adaptive_step_divisor
    try:
        numeric_divisor = float(adaptive_step_divisor)
        filtered_df = filtered_df[filtered_df['adaptive_step_divisor'] == numeric_divisor]
    except (ValueError, TypeError):
        filtered_df = filtered_df[filtered_df['adaptive_step_divisor'].astype(str) == str(adaptive_step_divisor)]
    
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
        print(f"  step_size: {sorted(df['step_size'].unique())}")
        print(f"  adaptive_step_divisor: {sorted(df['adaptive_step_divisor'].unique())}")
        print(f"  n: {sorted(df['n'].unique())}")
        return None
    
    # Get unique values
    presets = sorted(filtered_df['preset_name'].unique())
    param_variations = sorted(filtered_df['parameter_varied'].unique())
    variation_values = sorted(filtered_df['variation_value'].unique())
    
    print(f"Presets: {presets}")
    print(f"Parameter variations: {param_variations}")
    print(f"Variation values: {variation_values}")
    
    # Create figure with subplots (n_presets rows, n_params columns - one for each parameter)
    n_presets = len(presets)
    n_params = len(param_variations)
    fig, axes = plt.subplots(n_presets, n_params, figsize=(4*n_params, 3*n_presets))
    
    # Handle single preset case
    if n_presets == 1:
        axes = axes.reshape(1, -1)
    
    # Handle single parameter case
    if n_params == 1:
        axes = axes.reshape(-1, 1)
    
    fig.suptitle(f'{y_metric.replace("_", " ").title()} vs Variation Value by Preset and Parameter\n(step_size={step_size}, adaptive_step_divisor={adaptive_step_divisor}, n={n})', 
                 fontsize=16, y=0.95)
    
    # Define colors and markers for different methods
    colors = {'adaptive': 'blue', 'nonadaptive': 'red'}
    markers = {'adaptive': 'o', 'nonadaptive': 's'}
    labels = {'adaptive': 'Adaptive', 'nonadaptive': 'Non-Adaptive'}
    
    # Store all legend handles and labels for a single legend
    all_handles = []
    all_labels = []
    test_handles = []
    test_labels = []
    
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
            
            # Debug: Print data info for this subplot
            print(f"Subplot [{preset_idx}, {param_idx}]: {preset} - {param_varied}")
            print(f"  Data points: {len(param_data)}")
            if len(param_data) > 0:
                print(f"  Variation values: {sorted(param_data['variation_value'].unique())}")
                print(f"  Available columns: {[col for col in param_data.columns if 'pass' in col or 'distance' in col or 'runtime' in col]}")
            else:
                print(f"  No data found for this combination")
            
            if not param_data.empty:
                # Plot each method (non-adaptive first, then adaptive so adaptive appears on top)
                for method_type in ['nonadaptive', 'adaptive']:
                    metric_col = metric_mapping[y_metric][method_type]
                    
                    print(f"    Method: {method_type}, Metric column: {metric_col}")
                    print(f"    Column exists: {metric_col in param_data.columns}")
                    
                    if metric_col in param_data.columns:
                        x_values = param_data['variation_value'].values
                        y_values = param_data[metric_col].values
                        
                        # Handle 'ERROR' values and convert to numeric
                        if y_metric in ['mean_distance', 'mean_runtime', 'mean_diff']:
                            # Convert to numeric, replacing 'ERROR' with NaN
                            y_values = pd.to_numeric(y_values, errors='coerce')
                        
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
                        
                        # Filter out NaN values
                        valid_mask = ~np.isnan(y_values_sorted)
                        x_valid = x_values_sorted[valid_mask]
                        y_valid = y_values_sorted[valid_mask]
                        
                        print(f"      Valid data points: {len(x_valid)}")
                        if len(x_valid) > 0:
                            print(f"      X values: {x_valid}")
                            print(f"      Y values: {y_valid}")
                        
                        if len(x_valid) > 0:
                            # Plot points and line
                            line, = ax.plot(x_valid, y_valid, 
                                           marker=markers[method_type], color=colors[method_type], 
                                           linestyle='-', label=f'{labels[method_type]}',
                                           alpha=0.8, markersize=8, linewidth=2)
                            
                            # Plot test counts on secondary y-axis
                            test_count_col = f'n_tests_{method_type}'
                            if test_count_col in param_data.columns:
                                test_counts = param_data[test_count_col].values
                                test_counts_sorted = test_counts[sort_indices]
                                test_counts_valid = test_counts_sorted[valid_mask]
                                
                                # Plot test counts as dashed lines on secondary axis
                                test_line, = ax2.plot(x_valid, test_counts_valid, 
                                                     color=colors[method_type], linestyle='--', 
                                                     alpha=0.3, linewidth=1,
                                                     label=f'{labels[method_type]} Tests')
                                
                                # Store test count handles for legend
                                test_label = f'{labels[method_type]} Tests'
                                if test_label not in test_labels:
                                    test_handles.append(test_line)
                                    test_labels.append(test_label)
                            
                            # Store handle and label for legend (only once per method)
                            legend_label = f'{labels[method_type]}'
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
    if all_handles or test_handles:
        # Combine main metrics and test counts
        combined_handles = all_handles + test_handles
        combined_labels = all_labels + test_labels
        
        fig.legend(combined_handles, combined_labels, loc='upper center', bbox_to_anchor=(0.5, 0.02), 
                  ncol=len(combined_labels), fancybox=True, shadow=True, fontsize=10)
    
    # Adjust layout
    plt.subplots_adjust(top=0.9, bottom=0.15, left=0.05, right=0.95, hspace=0.4, wspace=0.3)
    
    # Save plot if output file is specified
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_file}")
    
    return fig

def example_usage():
    """
    Example usage of the analysis functions.
    """
    print("Example usage of analyze_test_results.py functions:")
    print("\n1. Load and analyze data:")
    print("   df = load_and_clean_data('combined_results.csv')")
    print("   config_summary = configuration_summary(df)")
    print("   ranking = rank_configurations(config_summary, df)")
    
    print("\n2. Create aggregated results:")
    print("   aggregated_df = aggregate_test_results(df, 'aggregated_results.csv', include_config_summary=True)")
    
    print("\n3. Create plots by preset:")
    print("   # Plot pass percentages")
    print("   fig1 = plot_by_preset(df, 'pass_percentage', step_size=0.1, adaptive_step_divisor=0.1, n=40000)")
    print("   ")
    print("   # Plot mean Bayesian distances")
    print("   fig2 = plot_by_preset(df, 'mean_distance', step_size=0.1, adaptive_step_divisor=0.1, n=40000)")
    print("   ")
    print("   # Plot mean runtime")
    print("   fig3 = plot_by_preset(df, 'mean_runtime', step_size=0.1, adaptive_step_divisor=0.1, n=40000)")
    print("   ")
    print("   # Plot mean differences from reference")
    print("   fig4 = plot_by_preset(df, 'mean_diff', step_size=0.1, adaptive_step_divisor=0.1, n=40000)")
    
    print("\n4. Available y_metric options:")
    print("   - 'pass_percentage': Shows pass percentages for adaptive vs non-adaptive")
    print("   - 'mean_distance': Shows mean Bayesian distances for adaptive vs non-adaptive")
    print("   - 'mean_runtime': Shows mean runtime for adaptive vs non-adaptive")
    print("   - 'mean_diff': Shows mean differences from reference for adaptive vs non-adaptive")


# def main():
#     parser = argparse.ArgumentParser(description="Analyze adaptive vs non-adaptive test results")
#     parser.add_argument("csv_file", help="Path to the combined results CSV file")
#     parser.add_argument("--no-viz", action="store_true", help="Skip creating visualizations")
#     parser.add_argument("--aggregate", action="store_true", help="Create aggregated results by configuration")
#     parser.add_argument("--plot", choices=['n', 'step_size', 'adaptive_step_divisor'], 
#                        help="Create dependency plots for specified parameter")
#     parser.add_argument("--y-metric", choices=['adaptive_pass_percentage', 'nonadaptive_pass_percentage', 
#                                               'mean_baysian_distance_adaptive', 'mean_baysian_distance_nonadaptive',
#                                               'mean_adaptive_diff', 'mean_nonadaptive_diff'],
#                        help="Y-axis metric for dependency plots")
#     parser.add_argument("--fixed-params", type=str, 
#                        help="Fixed parameters as JSON string, e.g., '{\"step_size\": 0.1, \"adaptive_step_divisor\": 10}'")
    
#     args = parser.parse_args()
    
#     # Load and analyze data
#     df = load_and_clean_data(args.csv_file)
#     config_summary = configuration_summary(df)
#     ranking = rank_configurations(config_summary, df)
#     best_configs_df = best_configuration_per_preset(df)
#     param_analysis = parameter_sensitivity_analysis(df)
#     variation_analysis, param_variation = variation_size_analysis(df)
#     step_size_effect, n_effect, divisor_effect = parameter_effect_analysis(df)
#     runtime_summary = runtime_analysis(df)
    
#     # Create aggregated results if requested
#     if args.aggregate:
#         aggregated_df = aggregate_test_results(df, 'aggregated_test_results.csv', include_config_summary=True)
    
#     # Create visualizations
#     if not args.no_viz:
#         create_visualizations(df, config_summary, ranking, best_configs_df)
    
#     # Save summary to CSV
#     config_summary.to_csv('configuration_summary.csv')
#     ranking.to_csv('configuration_ranking.csv')
#     best_configs_df.to_csv('best_configuration_per_preset.csv')
#     param_analysis.to_csv('parameter_sensitivity.csv')
#     variation_analysis.to_csv('variation_analysis.csv')
#     step_size_effect.to_csv('step_size_effect.csv')
#     n_effect.to_csv('n_effect.csv')
#     divisor_effect.to_csv('adaptive_step_divisor_effect.csv')
#     runtime_summary.to_csv('runtime_analysis.csv')
    
#     print("\n" + "="*80)
#     print("SUMMARY FILES SAVED")
#     print("="*80)
#     print("configuration_summary.csv - Overall configuration performance")
#     print("configuration_ranking.csv - Ranked configurations")
#     print("best_configuration_per_preset.csv - Best configuration for each preset")
#     print("parameter_sensitivity.csv - Parameter sensitivity analysis")
#     print("variation_analysis.csv - Variation size analysis")
#     print("step_size_effect.csv - Effect of step_size on test results")
#     print("n_effect.csv - Effect of n on test results")
#     print("adaptive_step_divisor_effect.csv - Effect of adaptive_step_divisor on test results")
#     print("runtime_analysis.csv - Runtime performance analysis")
#     if args.aggregate:
#         print("aggregated_test_results.csv - Aggregated results by configuration")
#     if not args.no_viz:
#         print("test_results_analysis.png - Visualizations")

# if __name__ == "__main__":
#     main()