import sys
import os

# Add error handling for imports
try:
    import numpy as np
    print("NumPy imported successfully")
except ImportError as e:
    print(f"Error importing NumPy: {e}")
    sys.exit(1)

try:
    from SRtools import presets as presets
    print("SRtools presets imported successfully")
except ImportError as e:
    print(f"Error importing SRtools: {e}")
    sys.exit(1)

try:
    import ast
    import argparse
    import pandas as pd
    print("Standard libraries imported successfully")
except ImportError as e:
    print(f"Error importing standard libraries: {e}")
    sys.exit(1)

try:
    import test_methods_with_params as t
    print("Test module imported successfully")
except ImportError as e:
    print(f"Error importing test module: {e}")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="A script that processes command line arguments for adaptive vs non-adaptive testing.")
    parser.add_argument("preset", type=str, help="the preset to use")
    parser.add_argument("results_path", type=str, help="The path of the output folder")
    parser.add_argument("index", type=int, help="The index of array job")

    args = parser.parse_args()
    preset = args.preset
    results_path = args.results_path
    index = args.index - 1

    # Updated parameters for the new test functions
    step_sizes = [0.1, 0.05, 0.2]
    adaptive_step_divisors = [0.1, 0.2, 0.5]
    time_units = ['days']
    n_values = [20000, 40000]
    variations = [0.8, 0.9, 0.95, 0.98, 1.02, 1.05, 1.1, 1.2]

    # Calculate which combination of parameters to use based on index
    # Each test is run 10 times (for variations), so we need to divide by 10 to get the combination index
    combination_index = index // 10

    # Calculate the total number of combinations
    total_combinations = len(step_sizes) * len(adaptive_step_divisors) * len(time_units) * len(n_values)

    # Check if index is valid
    if combination_index >= total_combinations:
        print(f"Error: Index {index} is out of range. Maximum valid index is {total_combinations * 10 - 1}")
        sys.exit(1)

    # Calculate which step_size, adaptive_step_divisor, time_unit, and n to use
    step_size_index = combination_index % len(step_sizes)
    remaining_combinations = combination_index // len(step_sizes)
    adaptive_step_divisor_index = remaining_combinations % len(adaptive_step_divisors)
    remaining_combinations = remaining_combinations // len(adaptive_step_divisors)
    time_unit_index = remaining_combinations % len(time_units)
    remaining_combinations = remaining_combinations // len(time_units)
    n_index = remaining_combinations % len(n_values)
    variation_index = remaining_combinations // len(n_values)

    # Get the actual parameter values
    step_size = step_sizes[step_size_index]
    adaptive_step_divisor = adaptive_step_divisors[adaptive_step_divisor_index]
    time_unit = time_units[time_unit_index]
    n_value = n_values[n_index]
    variation_value = variations[variation_index]

    print(f"Index {index} corresponds to:")
    print(f"  Step size: {step_size}")
    print(f"  Adaptive step divisor: {adaptive_step_divisor}")
    print(f"  Time unit: {time_unit}")
    print(f"  N: {n_value}")
    print(f"  Variation: {variation_value}")
    print(f"  Combination index: {combination_index}")

    # Run the test with the new parameters
    df = t.test_adaptive_vs_nonadaptive_with_params(
        preset, 
        n=n_value, 
        variations=variation_value, 
        verbose=False, 
        save_results=True, 
        index=index, 
        results_path=results_path,
        step_size=step_size,
        adaptive_step_divisor=adaptive_step_divisor,
        time_unit=time_unit
    )


if __name__ == "__main__":
    main()