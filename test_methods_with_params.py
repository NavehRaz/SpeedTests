#!/usr/bin/env python3
"""
Test script for speedTestSRclass with adaptive=True vs adaptive=False comparison.

This script provides functions to test the SpeedTestSR class with different adaptive settings,
measuring both accuracy (using Bayesian metric) and runtime performance.
Both methods are compared against the same non-adaptive blue simulation reference.
"""

import numpy as np
import pandas as pd
import time
import warnings
import traceback
from SRtools import presets
import speedTestSRclass as stsr

def test_adaptive_vs_nonadaptive_with_params(preset_name, n=40000, variations=[0.95, 0.9, 0.8, 0.7], 
                                            verbose=False, save_results=False, index=None, 
                                            results_path=None, parallel=False, step_size=0.1, 
                                            adaptive_step_divisor=10, time_unit='days'):
    """
    Test SpeedTestSR class with adaptive=True vs adaptive=False.
    Both methods are compared against the same non-adaptive blue simulation reference.
    
    Args:
        preset_name (str): Name of the preset to test
        n (int): Number of people for simulation (default: 40000)
        variations (float or list): Variation factor(s) to test. Can be a single float or list/array of floats (default: [0.95, 0.9, 0.8, 0.7])
        verbose (bool): If True, print detailed output during testing
        save_results (bool): If True, save results to CSV file
        index (int): Index for filename when saving results
        results_path (str): Path to save results
        parallel (bool): Whether to use parallel processing
        step_size (float): Step size for simulations (default: 0.1)
        adaptive_step_divisor (int): Divisor for adaptive step size (default: 10)
        time_unit (str): Time unit for config params (default: 'days')
        
    Returns:
        pandas.DataFrame: DataFrame with columns: preset_name, n, step_size, adaptive_step_divisor, theta, parameter_varied, 
                         variation_value, baysian_distance_adaptive, baysian_distance_nonadaptive, 
                         runtime_adaptive, runtime_nonadaptive, pass_adaptive, pass_nonadaptive
    """
    warnings.filterwarnings('ignore')
    
    # Import for file operations
    import os
    
    if verbose:
        print(f"Testing preset: {preset_name}")
        print(f"Parameters: n={n}, step_size={step_size}, adaptive_step_divisor={adaptive_step_divisor}, time_unit={time_unit}")
    
    # Parameter names mapping
    param_names = ['eta', 'beta', 'epsilon', 'xc']
    
    # Initialize results list
    results_list = []
    
    try:
        # Get theta
        theta = presets.getTheta(preset_name)
        if verbose:
            print(f"Original theta: {theta}")
        

        # Get configuration parameters
        config_params = presets.get_config_params(preset_name, time_unit=time_unit, verbose=verbose)

        nsteps = config_params['nsteps']
        time_step_multiplier = config_params['time_step_multiplier']
        npeople = config_params['npeople']
        t_end = config_params['t_end']
        
        if verbose:
            print(f"Config params: nsteps={nsteps}, time_step_multiplier={time_step_multiplier}, npeople={npeople}, t_end={t_end}")
        
        # Create "truth" simulation (blue) using non-adaptive method only
        if verbose:
            print("Creating truth simulation (blue) with adaptive=False...")
        
        start_time = time.time()
        blue_reference = stsr.getSpeedTestSR(
            theta, 
            n=1100,  # Use n=1100 for blue
            nsteps=nsteps, 
            t_end=t_end,
            time_step_multiplier=time_step_multiplier, 
            parallel=parallel,
            adaptive=False,
            # step_size=step_size
        )
        blue_reference_runtime = time.time() - start_time
        
        # Test with same theta first (should be best) - Adaptive
        if verbose:
            print("Creating simulation with same theta (adaptive=True)...")
        
        start_time = time.time()
        sim_same_adaptive = stsr.getSpeedTestSR(
            theta, 
            n=n,  # Use provided n parameter
            nsteps=nsteps, 
            t_end=t_end,
            time_step_multiplier=time_step_multiplier, 
            parallel=parallel,
            adaptive=True,
            # step_size=step_size/adaptive_step_divisor
        )
        sim_adaptive_runtime = time.time() - start_time
        
        # Test with same theta first (should be best) - Non-adaptive
        if verbose:
            print("Creating simulation with same theta (adaptive=False)...")
        
        start_time = time.time()
        sim_same_nonadaptive = stsr.getSpeedTestSR(
            theta, 
            n=n,  # Use provided n parameter
            nsteps=nsteps, 
            t_end=t_end,
            time_step_multiplier=time_step_multiplier, 
            parallel=parallel,
            adaptive=False,
            # step_size=step_size
        )
        sim_nonadaptive_runtime = time.time() - start_time
        
        # Calculate reference distances using non-adaptive blue as reference for both
        reference_baysian_distance_adaptive = stsr.baysianDistance(blue_reference, sim_same_adaptive)
        reference_baysian_distance_nonadaptive = stsr.baysianDistance(blue_reference, sim_same_nonadaptive)
        
        if verbose:
            print(f"Reference baysian distance (adaptive=True): {reference_baysian_distance_adaptive}")
            print(f"Reference baysian distance (adaptive=False): {reference_baysian_distance_nonadaptive}")
            print(f"Blue simulation runtime (non-adaptive reference): {blue_reference_runtime:.4f}s")
            print(f"Simulation runtime (adaptive=True): {sim_adaptive_runtime:.4f}s")
            print(f"Simulation runtime (adaptive=False): {sim_nonadaptive_runtime:.4f}s")
        
        # Convert variations to list if it's a single float
        if isinstance(variations, (int, float)):
            variations = [variations]
        variations = np.array(variations)
        
        for i in range(len(theta)):
            for var in variations:
                try:
                    theta_modified = theta.copy()
                    theta_modified[i] = var * theta[i]
                    
                    if verbose:
                        print(f"Testing {param_names[i]} = {var} * {param_names[i]} ({theta_modified[i]:.6f})...")
                    
                    # Test with adaptive=True
                    start_time = time.time()
                    sim_adaptive = stsr.getSpeedTestSR(
                        theta_modified, 
                        n=n,  # Use provided n parameter
                        nsteps=nsteps, 
                        t_end=t_end,
                        time_step_multiplier=time_step_multiplier, 
                        parallel=parallel,
                        adaptive=True,
                        # step_size=step_size/adaptive_step_divisor
                    )
                    runtime_adaptive = time.time() - start_time
                    
                    baysian_distance_adaptive = stsr.baysianDistance(blue_reference, sim_adaptive)
                    
                    # Test with adaptive=False
                    start_time = time.time()
                    sim_nonadaptive = stsr.getSpeedTestSR(
                        theta_modified, 
                        n=n,  # Use provided n parameter
                        nsteps=nsteps, 
                        t_end=t_end,
                        time_step_multiplier=time_step_multiplier, 
                        parallel=parallel,
                        adaptive=False,
                        #step_size=step_size
                    )
                    runtime_nonadaptive = time.time() - start_time
                    
                    baysian_distance_nonadaptive = stsr.baysianDistance(blue_reference, sim_nonadaptive)
                    
                    # Check if distances are better than or equal to reference (lower is better)
                    pass_adaptive = 1 if baysian_distance_adaptive <= reference_baysian_distance_adaptive else 0
                    pass_nonadaptive = 1 if baysian_distance_nonadaptive <= reference_baysian_distance_nonadaptive else 0
                    
                    # Store test result
                    test_result = {
                        'preset_name': preset_name,
                        'n': n,
                        'step_size': step_size,
                        'adaptive_step_divisor': adaptive_step_divisor,
                        'theta': str(theta.tolist()),  # Convert to string for DataFrame
                        'parameter_varied': param_names[i],
                        'variation_value': var,
                        'baysian_distance_adaptive': baysian_distance_adaptive,
                        'baysian_distance_nonadaptive': baysian_distance_nonadaptive,
                        'reference_baysian_distance_adaptive': reference_baysian_distance_adaptive,
                        'reference_baysian_distance_nonadaptive': reference_baysian_distance_nonadaptive,
                        'runtime_adaptive': runtime_adaptive,
                        'runtime_nonadaptive': runtime_nonadaptive,
                        'blue_reference_runtime': blue_reference_runtime,
                        'sim_same_runtime_adaptive': sim_adaptive_runtime,
                        'sim_same_runtime_nonadaptive': sim_nonadaptive_runtime,
                        'pass_adaptive': pass_adaptive,
                        'pass_nonadaptive': pass_nonadaptive
                    }
                    
                    results_list.append(test_result)
                    
                    if verbose:
                        adaptive_status = "✅ PASSED" if pass_adaptive else "❌ FAILED"
                        nonadaptive_status = "✅ PASSED" if pass_nonadaptive else "❌ FAILED"
                        print(f"  Adaptive=True: {adaptive_status} - Distance {baysian_distance_adaptive:.6f} vs reference {reference_baysian_distance_adaptive:.6f} (Runtime: {runtime_adaptive:.4f}s)")
                        print(f"  Adaptive=False: {nonadaptive_status} - Distance {baysian_distance_nonadaptive:.6f} vs reference {reference_baysian_distance_nonadaptive:.6f} (Runtime: {runtime_nonadaptive:.4f}s)")
                
                except Exception as e:
                    if verbose:
                        print(f"  ❌ ERROR: {e}")
                        print(f"  Full traceback:")
                        print(traceback.format_exc())
                    
                    # Store error result
                    test_result = {
                        'preset_name': preset_name,
                        'n': n,
                        'step_size': step_size,
                        'adaptive_step_divisor': adaptive_step_divisor,
                        'theta': str(theta.tolist()),
                        'parameter_varied': param_names[i],
                        'variation_value': var,
                        'baysian_distance_adaptive': 'ERROR',
                        'baysian_distance_nonadaptive': 'ERROR',
                        'reference_baysian_distance_adaptive': reference_baysian_distance_adaptive,
                        'reference_baysian_distance_nonadaptive': reference_baysian_distance_nonadaptive,
                        'runtime_adaptive': 'ERROR',
                        'runtime_nonadaptive': 'ERROR',
                        'blue_reference_runtime': blue_reference_runtime,
                        'sim_same_runtime_adaptive': sim_adaptive_runtime,
                        'sim_same_runtime_nonadaptive': sim_nonadaptive_runtime,
                        'pass_adaptive': 0,
                        'pass_nonadaptive': 0
                    }
                    
                    results_list.append(test_result)
        
        if verbose:
            print(f"✅ Preset '{preset_name}' completed")

        # Convert results to DataFrame
        df = pd.DataFrame(results_list)
        
        # Save results if requested
        if save_results and index is not None:
            # Create filename with the specified format
            filename = f"{preset_name}_adaptive_test_{index}_{n}.csv"
            
            # If results_path is provided, save to that folder
            if results_path is not None:
                # Ensure the folder exists
                os.makedirs(results_path, exist_ok=True)
                filepath = os.path.join(results_path, filename)
            else:
                filepath = filename
                
            df.to_csv(filepath, index=False)
            if verbose:
                print(f"Results saved to: {filepath}")

        return df
    
    except Exception as e:
        if verbose:
            print(f"❌ Error testing preset '{preset_name}': {e}")
            print(f"Full traceback:")
            print(traceback.format_exc())
        
        # Return empty DataFrame with correct columns if there's an overall error
        return pd.DataFrame(columns=['preset_name', 'n', 'step_size', 'adaptive_step_divisor', 'theta', 
                                   'parameter_varied', 'variation_value', 'baysian_distance_adaptive', 
                                   'baysian_distance_nonadaptive', 'reference_baysian_distance_adaptive',
                                   'reference_baysian_distance_nonadaptive', 'runtime_adaptive', 
                                   'runtime_nonadaptive', 'blue_reference_runtime', 
                                   'sim_same_runtime_adaptive', 'sim_same_runtime_nonadaptive',
                                   'pass_adaptive', 'pass_nonadaptive'])


def create_theta_dict_from_preset(preset_name, n_blue=1100, time_unit='days'):
    """
    Create a theta dictionary from a preset for use with test_adaptive_vs_nonadaptive_with_custom_params.
    
    Args:
        preset_name (str): Name of the preset to use
        n_blue (int): Number of people for blue simulation (default: 1100)
        time_unit (str): Time unit for config params (default: 'days')
        
    Returns:
        dict: Dictionary containing theta and configuration parameters
    """
    # Get theta
    theta = presets.getTheta(preset_name)
    
    # Get configuration parameters
    config_params = presets.get_config_params(preset_name, time_unit=time_unit)
    
    # Create theta dictionary
    theta_dict = {
        'theta': theta,
        'nsteps': config_params['nsteps'],
        'time_step_multiplier': config_params['time_step_multiplier'],
        't_end': config_params['t_end'],
        'n_blue': n_blue,
        'preset_name': preset_name
    }
    
    return theta_dict


def test_adaptive_vs_nonadaptive_with_custom_params(theta_dict, n=40000, variations=[0.95, 0.9, 0.8, 0.7], 
                                                   verbose=False, save_results=False, index=None, 
                                                   results_path=None, parallel=False, step_size=0.1,
                                                   adaptive_step_divisor=10):
    """
    Test SpeedTestSR class with custom theta and parameters instead of a preset.
    Both methods are compared against the same non-adaptive blue simulation reference.
    
    Args:
        theta_dict (dict): Dictionary containing:
            - 'theta': numpy array of 4 parameters [eta, beta, epsilon, xc]
            - 'nsteps': number of time steps
            - 'time_step_multiplier': time step multiplier
            - 't_end': end time
            - 'n_blue': number of people for blue simulation (default: 1100)
        n (int): Number of people for simulation (default: 40000)
        variations (float or list): Variation factor(s) to test. Can be a single float or list/array of floats (default: [0.95, 0.9, 0.8, 0.7])
        verbose (bool): If True, print detailed output during testing
        save_results (bool): If True, save results to CSV file
        index (int): Index for filename when saving results
        results_path (str): Path to save results
        parallel (bool): Whether to use parallel processing
        step_size (float): Step size for simulations (default: 0.1)
        adaptive_step_divisor (int): Divisor for adaptive step size (default: 10)
        
    Returns:
        pandas.DataFrame: DataFrame with columns: preset_name, n, step_size, adaptive_step_divisor, theta, parameter_varied, 
                         variation_value, baysian_distance_adaptive, baysian_distance_nonadaptive, 
                         runtime_adaptive, runtime_nonadaptive, pass_adaptive, pass_nonadaptive
    """
    warnings.filterwarnings('ignore')
    
    # Import for file operations
    import os
    
    # Extract parameters from dictionary
    theta = np.array(theta_dict['theta'])
    nsteps = theta_dict['nsteps']
    time_step_multiplier = theta_dict['time_step_multiplier']
    t_end = theta_dict['t_end']
    n_blue = theta_dict.get('n_blue', 1100)  # Default to 1100 for blue simulation
    preset_name = theta_dict.get('preset_name', 'custom')  # For naming purposes
    
    if verbose:
        print(f"Testing custom parameters")
        print(f"Parameters: n={n}, step_size={step_size}, adaptive_step_divisor={adaptive_step_divisor}")
    
    # Parameter names mapping
    param_names = ['eta', 'beta', 'epsilon', 'xc']
    
    # Initialize results list
    results_list = []
    
    try:
        if verbose:
            print(f"Original theta: {theta}")
            print(f"Config params: nsteps={nsteps}, time_step_multiplier={time_step_multiplier}, t_end={t_end}")
        
        # Create "truth" simulation (blue) using non-adaptive method only
        if verbose:
            print("Creating truth simulation (blue) with adaptive=False...")
        
        start_time = time.time()
        blue_reference = stsr.getSpeedTestSR(
            theta, 
            n=n_blue,  # Use n_blue for blue simulation
            nsteps=nsteps, 
            t_end=t_end,
            time_step_multiplier=time_step_multiplier, 
            parallel=parallel,
            adaptive=False,
            step_size=step_size
        )
        blue_reference_runtime = time.time() - start_time
        
        # Test with same theta first (should be best) - Adaptive
        if verbose:
            print("Creating simulation with same theta (adaptive=True)...")
        
        start_time = time.time()
        sim_same_adaptive = stsr.getSpeedTestSR(
            theta, 
            n=n,  # Use provided n parameter
            nsteps=nsteps, 
            t_end=t_end,
            time_step_multiplier=time_step_multiplier, 
            parallel=parallel,
            adaptive=True,
            step_size=step_size/adaptive_step_divisor
        )
        sim_adaptive_runtime = time.time() - start_time
        
        # Test with same theta first (should be best) - Non-adaptive
        if verbose:
            print("Creating simulation with same theta (adaptive=False)...")
        
        start_time = time.time()
        sim_same_nonadaptive = stsr.getSpeedTestSR(
            theta, 
            n=n,  # Use provided n parameter
            nsteps=nsteps, 
            t_end=t_end,
            time_step_multiplier=time_step_multiplier, 
            parallel=parallel,
            adaptive=False,
            step_size=step_size
        )
        sim_nonadaptive_runtime = time.time() - start_time
        
        # Calculate reference distances using non-adaptive blue as reference for both
        reference_baysian_distance_adaptive = stsr.baysianDistance(blue_reference, sim_same_adaptive)
        reference_baysian_distance_nonadaptive = stsr.baysianDistance(blue_reference, sim_same_nonadaptive)
        
        if verbose:
            print(f"Reference baysian distance (adaptive=True): {reference_baysian_distance_adaptive}")
            print(f"Reference baysian distance (adaptive=False): {reference_baysian_distance_nonadaptive}")
            print(f"Blue simulation runtime (non-adaptive reference): {blue_reference_runtime:.4f}s")
            print(f"Simulation runtime (adaptive=True): {sim_adaptive_runtime:.4f}s")
            print(f"Simulation runtime (adaptive=False): {sim_nonadaptive_runtime:.4f}s")
        
        # Convert variations to list if it's a single float
        if isinstance(variations, (int, float)):
            variations = [variations]
        variations = np.array(variations)
        
        for i in range(len(theta)):
            for var in variations:
                try:
                    theta_modified = theta.copy()
                    theta_modified[i] = var * theta[i]
                    
                    if verbose:
                        print(f"Testing {param_names[i]} = {var} * {param_names[i]} ({theta_modified[i]:.6f})...")
                    
                    # Test with adaptive=True
                    start_time = time.time()
                    sim_adaptive = stsr.getSpeedTestSR(
                        theta_modified, 
                        n=n,  # Use provided n parameter
                        nsteps=nsteps, 
                        t_end=t_end,
                        time_step_multiplier=time_step_multiplier, 
                        parallel=parallel,
                        adaptive=True,
                        step_size=step_size/adaptive_step_divisor
                    )
                    runtime_adaptive = time.time() - start_time
                    
                    baysian_distance_adaptive = stsr.baysianDistance(blue_reference, sim_adaptive)
                    
                    # Test with adaptive=False
                    start_time = time.time()
                    sim_nonadaptive = stsr.getSpeedTestSR(
                        theta_modified, 
                        n=n,  # Use provided n parameter
                        nsteps=nsteps, 
                        t_end=t_end,
                        time_step_multiplier=time_step_multiplier, 
                        parallel=parallel,
                        adaptive=False,
                        step_size=step_size
                    )
                    runtime_nonadaptive = time.time() - start_time
                    
                    baysian_distance_nonadaptive = stsr.baysianDistance(blue_reference, sim_nonadaptive)
                    
                    # Check if distances are better than or equal to reference (lower is better)
                    pass_adaptive = 1 if baysian_distance_adaptive <= reference_baysian_distance_adaptive else 0
                    pass_nonadaptive = 1 if baysian_distance_nonadaptive <= reference_baysian_distance_nonadaptive else 0
                    
                    # Store test result
                    test_result = {
                        'preset_name': preset_name,
                        'n': n,
                        'step_size': step_size,
                        'adaptive_step_divisor': adaptive_step_divisor,
                        'theta': str(theta.tolist()),  # Convert to string for DataFrame
                        'parameter_varied': param_names[i],
                        'variation_value': var,
                        'baysian_distance_adaptive': baysian_distance_adaptive,
                        'baysian_distance_nonadaptive': baysian_distance_nonadaptive,
                        'reference_baysian_distance_adaptive': reference_baysian_distance_adaptive,
                        'reference_baysian_distance_nonadaptive': reference_baysian_distance_nonadaptive,
                        'runtime_adaptive': runtime_adaptive,
                        'runtime_nonadaptive': runtime_nonadaptive,
                        'blue_reference_runtime': blue_reference_runtime,
                        'sim_same_runtime_adaptive': sim_adaptive_runtime,
                        'sim_same_runtime_nonadaptive': sim_nonadaptive_runtime,
                        'pass_adaptive': pass_adaptive,
                        'pass_nonadaptive': pass_nonadaptive
                    }
                    
                    results_list.append(test_result)
                    
                    if verbose:
                        adaptive_status = "✅ PASSED" if pass_adaptive else "❌ FAILED"
                        nonadaptive_status = "✅ PASSED" if pass_nonadaptive else "❌ FAILED"
                        print(f"  Adaptive=True: {adaptive_status} - Distance {baysian_distance_adaptive:.6f} vs reference {reference_baysian_distance_adaptive:.6f} (Runtime: {runtime_adaptive:.4f}s)")
                        print(f"  Adaptive=False: {nonadaptive_status} - Distance {baysian_distance_nonadaptive:.6f} vs reference {reference_baysian_distance_nonadaptive:.6f} (Runtime: {runtime_nonadaptive:.4f}s)")
                
                except Exception as e:
                    if verbose:
                        print(f"  ❌ ERROR: {e}")
                        print(f"  Full traceback:")
                        print(traceback.format_exc())
                    
                    # Store error result
                    test_result = {
                        'preset_name': preset_name,
                        'n': n,
                        'step_size': step_size,
                        'adaptive_step_divisor': adaptive_step_divisor,
                        'theta': str(theta.tolist()),
                        'parameter_varied': param_names[i],
                        'variation_value': var,
                        'baysian_distance_adaptive': 'ERROR',
                        'baysian_distance_nonadaptive': 'ERROR',
                        'reference_baysian_distance_adaptive': reference_baysian_distance_adaptive,
                        'reference_baysian_distance_nonadaptive': reference_baysian_distance_nonadaptive,
                        'runtime_adaptive': 'ERROR',
                        'runtime_nonadaptive': 'ERROR',
                        'blue_reference_runtime': blue_reference_runtime,
                        'sim_same_runtime_adaptive': sim_adaptive_runtime,
                        'sim_same_runtime_nonadaptive': sim_nonadaptive_runtime,
                        'pass_adaptive': 0,
                        'pass_nonadaptive': 0
                    }
                    
                    results_list.append(test_result)
        
        if verbose:
            print(f"✅ Custom parameters test completed")

        # Convert results to DataFrame
        df = pd.DataFrame(results_list)
        
        # Save results if requested
        if save_results and index is not None:
            # Create filename with the specified format
            filename = f"{preset_name}_adaptive_test_{index}_{n}.csv"
            
            # If results_path is provided, save to that folder
            if results_path is not None:
                # Ensure the folder exists
                os.makedirs(results_path, exist_ok=True)
                filepath = os.path.join(results_path, filename)
            else:
                filepath = filename
                
            df.to_csv(filepath, index=False)
            if verbose:
                print(f"Results saved to: {filepath}")

        return df
    
    except Exception as e:
        if verbose:
            print(f"❌ Error testing custom parameters: {e}")
            print(f"Full traceback:")
            print(traceback.format_exc())
        
        # Return empty DataFrame with correct columns if there's an overall error
        return pd.DataFrame(columns=['preset_name', 'n', 'step_size', 'adaptive_step_divisor', 'theta', 
                                   'parameter_varied', 'variation_value', 'baysian_distance_adaptive', 
                                   'baysian_distance_nonadaptive', 'reference_baysian_distance_adaptive',
                                   'reference_baysian_distance_nonadaptive', 'runtime_adaptive', 
                                   'runtime_nonadaptive', 'blue_reference_runtime', 
                                   'sim_same_runtime_adaptive', 'sim_same_runtime_nonadaptive',
                                   'pass_adaptive', 'pass_nonadaptive'])


def print_test_results(df):
    """
    Print a formatted summary of test results.
    
    Args:
        df (pandas.DataFrame): DataFrame from test_adaptive_vs_nonadaptive_with_params
    """
    if df.empty:
        print("❌ No test results to display")
        return
    
    # Get unique preset name (should be the same for all rows)
    preset_name = df['preset_name'].iloc[0]
    n = df['n'].iloc[0]
    step_size = df['step_size'].iloc[0]
    adaptive_step_divisor = df['adaptive_step_divisor'].iloc[0]
    
    print(f"\n{'='*80}")
    print(f"ADAPTIVE VS NON-ADAPTIVE TEST RESULTS FOR PRESET: {preset_name}")
    print(f"Parameters: n={n}, step_size={step_size}, adaptive_step_divisor={adaptive_step_divisor}")
    print(f"Note: Both methods compared against same non-adaptive blue simulation")
    print(f"{'='*80}")
    
    total_tests = len(df)
    passed_adaptive_tests = len(df[df['pass_adaptive'] == 1])
    failed_adaptive_tests = len(df[df['pass_adaptive'] == 0])
    passed_nonadaptive_tests = len(df[df['pass_nonadaptive'] == 1])
    failed_nonadaptive_tests = len(df[df['pass_nonadaptive'] == 0])
    error_tests = len(df[df['baysian_distance_adaptive'] == 'ERROR'])
    
    print(f"Total tests: {total_tests}")
    print(f"Adaptive=True tests - Passed: {passed_adaptive_tests}, Failed: {failed_adaptive_tests}")
    print(f"Adaptive=False tests - Passed: {passed_nonadaptive_tests}, Failed: {failed_nonadaptive_tests}")
    print(f"Errors: {error_tests}")
    
    # Runtime analysis
    if not df.empty and 'runtime_adaptive' in df.columns and 'runtime_nonadaptive' in df.columns:
        valid_runtime_df = df[(df['runtime_adaptive'] != 'ERROR') & (df['runtime_nonadaptive'] != 'ERROR')]
        if not valid_runtime_df.empty:
            avg_runtime_adaptive = valid_runtime_df['runtime_adaptive'].mean()
            avg_runtime_nonadaptive = valid_runtime_df['runtime_nonadaptive'].mean()
            speedup = avg_runtime_nonadaptive / avg_runtime_adaptive if avg_runtime_adaptive > 0 else 0
            
            print(f"\nRuntime Analysis:")
            print(f"Average runtime (adaptive=True): {avg_runtime_adaptive:.4f}s")
            print(f"Average runtime (adaptive=False): {avg_runtime_nonadaptive:.4f}s")
            print(f"Speedup (non-adaptive/adaptive): {speedup:.2f}x")
    
    if failed_adaptive_tests > 0 or failed_nonadaptive_tests > 0 or error_tests > 0:
        print(f"\n❌ {failed_adaptive_tests + failed_nonadaptive_tests + error_tests} tests failed:")
        
        # Show failed tests
        failed_adaptive_tests_df = df[df['pass_adaptive'] == 0]
        failed_nonadaptive_tests_df = df[df['pass_nonadaptive'] == 0]
        
        if not failed_adaptive_tests_df.empty:
            print("\nFailed Adaptive=True Tests:")
            display_cols = ['parameter_varied', 'variation_value', 'baysian_distance_adaptive', 'reference_baysian_distance_adaptive', 'runtime_adaptive']
            print(failed_adaptive_tests_df[display_cols].to_string(index=False))
        
        if not failed_nonadaptive_tests_df.empty:
            print("\nFailed Adaptive=False Tests:")
            display_cols = ['parameter_varied', 'variation_value', 'baysian_distance_nonadaptive', 'reference_baysian_distance_nonadaptive', 'runtime_nonadaptive']
            print(failed_nonadaptive_tests_df[display_cols].to_string(index=False))
    else:
        print("\n✅ All tests passed!")


def run_multiple_preset_tests(preset_names, n=40000, variations=[0.95, 0.9, 0.8, 0.7], 
                             verbose=False, save_results=False, index=None, results_path=None, 
                             parallel=False, step_size=0.1, adaptive_step_divisor=10, time_unit='days'):
    """
    Run tests for multiple presets comparing adaptive=True vs adaptive=False.
    Both methods are compared against the same non-adaptive blue simulation reference.
    
    Args:
        preset_names (list): List of preset names to test
        n (int): Number of people for simulation (default: 40000)
        variations (float or list): Variation factor(s) to test. Can be a single float or list/array of floats (default: [0.95, 0.9, 0.8, 0.7])
        verbose (bool): If True, print detailed output during testing
        save_results (bool): If True, save results to CSV file
        index (int): Index for filename when saving results
        results_path (str): Path to save results
        parallel (bool): Whether to use parallel processing
        step_size (float): Step size for simulations (default: 0.1)
        adaptive_step_divisor (int): Divisor for adaptive step size (default: 10)
        time_unit (str): Time unit for config params (default: 'days')
        
    Returns:
        pandas.DataFrame: Combined DataFrame with results from all presets
    """
    all_results = []
    
    for preset in preset_names:
        if verbose:
            print(f"\n{'='*60}")
        
        result_df = test_adaptive_vs_nonadaptive_with_params(preset, n, variations, verbose, save_results, index, results_path, parallel, step_size, adaptive_step_divisor, time_unit)
        all_results.append(result_df)
        
        if verbose and not result_df.empty:
            print_test_results(result_df)
    
    # Combine all results into one DataFrame
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        return combined_df
    else:
        return pd.DataFrame(columns=['preset_name', 'n', 'step_size', 'adaptive_step_divisor', 'theta', 
                                   'parameter_varied', 'variation_value', 'baysian_distance_adaptive', 
                                   'baysian_distance_nonadaptive', 'reference_baysian_distance_adaptive',
                                   'reference_baysian_distance_nonadaptive', 'runtime_adaptive', 
                                   'runtime_nonadaptive', 'blue_reference_runtime', 
                                   'sim_same_runtime_adaptive', 'sim_same_runtime_nonadaptive',
                                   'pass_adaptive', 'pass_nonadaptive']) 

def merge_csv_files_in_folder(folder_path, output_file="combined_results.csv", output_folder=None, progress_bar=True):
    """
    Merge all CSV files in a specified folder into one combined CSV file.
    
    Args:
        folder_path (str): Path to the folder containing CSV files
        output_file (str): Name of the output combined CSV file
        output_folder (str): Optional folder to save the combined CSV file to
        progress_bar (bool): Whether to show progress bar during processing
        
    Returns:
        pandas.DataFrame: Combined DataFrame
    """
    import glob
    import os
    from tqdm import tqdm
    
    # Ensure folder path exists
    if not os.path.exists(folder_path):
        print(f"Error: Folder {folder_path} does not exist")
        return pd.DataFrame()
    
    # Find all CSV files in the folder
    csv_pattern = os.path.join(folder_path, "*.csv")
    csv_files = glob.glob(csv_pattern)
    
    if not csv_files:
        print(f"No CSV files found in folder: {folder_path}")
        return pd.DataFrame()
    
    print(f"Found {len(csv_files)} CSV files in {folder_path}:")
    for file in csv_files:
        print(f"  - {os.path.basename(file)}")
    
    # Read and combine all CSV files
    dfs = []
    if progress_bar:
        csv_files_iter = tqdm(csv_files, desc="Reading CSV files")
    else:
        csv_files_iter = csv_files
        
    for file in csv_files_iter:
        try:
            df = pd.read_csv(file)
            dfs.append(df)
            if not progress_bar:
                print(f"Successfully read: {os.path.basename(file)} ({len(df)} rows)")
        except Exception as e:
            print(f"Error reading {os.path.basename(file)}: {e}")
            print(f"Full traceback:")
            print(traceback.format_exc())
    
    if not dfs:
        print("No valid CSV files could be read")
        return pd.DataFrame()
    
    # Combine all DataFrames
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Save combined results
    if output_folder is not None:
        # Ensure output folder exists
        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, output_file)
    else:
        output_path = os.path.join(folder_path, output_file)
    
    combined_df.to_csv(output_path, index=False)
    print(f"Combined {len(combined_df)} rows into: {output_path}")
    
    return combined_df