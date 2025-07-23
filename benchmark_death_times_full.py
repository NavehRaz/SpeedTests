
import numpy as np
import matplotlib.pyplot as plt
import time
from numba import jit

# Simulation settings
npeople = 200000
s = 5000
dt = 0.2
sdt = 1.0
t = np.linspace(0, s * dt, s)

# Model parameters
eta0, eta_var = 25, 0
beta0, beta_var = 50.0, 0
kappa0, kappa_var = 0.5, 0
epsilon0, epsilon_var = 50, 0
xc0, xc_var = 17.0, 0
external_hazard = np.inf
base_multiplier = 100
time_step_multiplier = 1000

# Euler method (optimized)
@jit(nopython=True)
def death_times_optimized(s, dt, t, eta0, eta_var, beta0, beta_var, kappa0, kappa_var,
                          epsilon0, epsilon_var, xc0, xc_var, sdt, npeople,
                          external_hazard=np.inf, time_step_multiplier=1):
    death_times = np.zeros(npeople)
    events = np.zeros(npeople, dtype=np.int32)
    ndt = dt / time_step_multiplier
    nsdt = sdt / np.sqrt(time_step_multiplier)
    chance_to_die_externally = np.exp(-external_hazard) * ndt
    for person in range(npeople):
        x = 0.0
        j = 0
        eta = eta0 * np.random.normal(1.0, eta_var)
        beta = beta0 * np.random.normal(1.0, beta_var)
        kappa = kappa0 * np.random.normal(1.0, kappa_var)
        epsilon = epsilon0 * np.random.normal(1.0, epsilon_var)
        xc = xc0 * np.random.normal(1.0, xc_var)
        sqrt_2epsilon = np.sqrt(2 * epsilon)
        while j < s - 1 and x < xc:
            for _ in range(time_step_multiplier):
                drift = eta * t[j] - beta * x / (x + kappa)
                noise = sqrt_2epsilon * np.random.normal()
                x += ndt * drift + noise * nsdt
                x = max(x, 0.0)
                if np.random.rand() < chance_to_die_externally:
                    x = xc
                    break
                if x >= xc:
                    break
            j += 1
        death_times[person] = j * dt
        events[person] = int(x >= xc)
    return death_times, events

# Euler method (highly optimized)
@jit(nopython=True)
def death_times_euler_fast(s, dt, t, eta0, eta_var, beta0, beta_var, kappa0, kappa_var,
                          epsilon0, epsilon_var, xc0, xc_var, sdt, npeople,
                          external_hazard=np.inf, time_step_multiplier=1):
    death_times = []
    events = []
    ndt = dt / time_step_multiplier
    nsdt = sdt / np.sqrt(time_step_multiplier)
    chance_to_die_externally = np.exp(-external_hazard) * ndt
    
    for person in range(npeople):
        x = 0.0
        j = 0
        eta = eta0 * np.random.normal(1.0, eta_var)
        beta = beta0 * np.random.normal(1.0, beta_var)
        kappa = kappa0 * np.random.normal(1.0, kappa_var)
        epsilon = epsilon0 * np.random.normal(1.0, epsilon_var)
        xc = xc0 * np.random.normal(1.0, xc_var)
        sqrt_2epsilon = np.sqrt(2 * epsilon)
        
        while j < s - 1 and x < xc:
            for _ in range(time_step_multiplier):
                # Inline calculations to reduce function calls
                drift = eta * t[j] - beta * x / (x + kappa)
                x += ndt * drift + sqrt_2epsilon * nsdt * np.random.normal()
                x = max(x, 0.0)
                if np.random.rand() < chance_to_die_externally:
                    x = xc
                    break
                if x >= xc:
                    break
            j += 1
        death_times.append(j * dt)
        events.append(int(x >= xc))
    
    return np.array(death_times), np.array(events)

# Heun method (optimized)
@jit(nopython=True)
def death_times_heun_optimized(s, dt, t, eta0, eta_var, beta0, beta_var, kappa0, kappa_var,
                               epsilon0, epsilon_var, xc0, xc_var, sdt, npeople,
                               external_hazard=np.inf, time_step_multiplier=1):
    death_times = np.zeros(npeople)
    events = np.zeros(npeople, dtype=np.int32)
    ndt = dt / time_step_multiplier
    nsdt = sdt / np.sqrt(time_step_multiplier)
    constant_hazard = np.isfinite(external_hazard)
    if constant_hazard:
        chance_to_die_externally = np.exp(-external_hazard) * ndt
    for person in range(npeople):
        x = 0.0
        j = 0
        eta = eta0 * np.random.normal(1.0, eta_var)
        beta = beta0 * np.random.normal(1.0, beta_var)
        kappa = kappa0 * np.random.normal(1.0, kappa_var)
        epsilon = epsilon0 * np.random.normal(1.0, epsilon_var)
        xc = xc0 * np.random.normal(1.0, xc_var)
        sqrt_2epsilon = np.sqrt(2 * epsilon)
        while j < s - 1 and x < xc:
            for _ in range(time_step_multiplier):
                tj = t[j]
                drift1 = eta * tj - beta * x / (x + kappa)
                noise = sqrt_2epsilon * np.random.normal()
                x_predict = x + ndt * drift1 + noise * nsdt
                x_predict = max(x_predict, 0.0)
                drift2 = eta * (tj + ndt) - beta * x_predict / (x_predict + kappa)
                x += ndt * 0.5 * (drift1 + drift2) + noise * nsdt
                x = max(x, 0.0)
                if (constant_hazard and np.random.rand() < chance_to_die_externally) or x >= xc:
                    x = xc
                    break
            j += 1
        death_times[person] = j * dt
        events[person] = int(x >= xc)
    return death_times, events

# Runge-Kutta 4th order method (optimized)
@jit(nopython=True)
def death_times_rk4_optimized(s, dt, t, eta0, eta_var, beta0, beta_var, kappa0, kappa_var,
                              epsilon0, epsilon_var, xc0, xc_var, sdt, npeople,
                              external_hazard=np.inf, time_step_multiplier=1):
    death_times = np.zeros(npeople)
    events = np.zeros(npeople, dtype=np.int32)
    ndt = dt / time_step_multiplier
    nsdt = sdt / np.sqrt(time_step_multiplier)
    constant_hazard = np.isfinite(external_hazard)
    if constant_hazard:
        chance_to_die_externally = np.exp(-external_hazard) * ndt
    for person in range(npeople):
        x = 0.0
        j = 0
        eta = eta0 * np.random.normal(1.0, eta_var)
        beta = beta0 * np.random.normal(1.0, beta_var)
        kappa = kappa0 * np.random.normal(1.0, kappa_var)
        epsilon = epsilon0 * np.random.normal(1.0, epsilon_var)
        xc = xc0 * np.random.normal(1.0, xc_var)
        sqrt_2epsilon = np.sqrt(2 * epsilon)
        while j < s - 1 and x < xc:
            for _ in range(time_step_multiplier):
                tj = t[j]
                
                # Generate noise term once for this step
                noise = sqrt_2epsilon * np.random.normal()
                
                # RK4 coefficients for drift term
                k1 = eta * tj - beta * x / (x + kappa)
                x_temp = x + 0.5 * ndt * k1 + 0.5 * noise * nsdt
                x_temp = max(x_temp, 0.0)
                k2 = eta * (tj + 0.5 * ndt) - beta * x_temp / (x_temp + kappa)
                x_temp = x + 0.5 * ndt * k2 + 0.5 * noise * nsdt
                x_temp = max(x_temp, 0.0)
                k3 = eta * (tj + 0.5 * ndt) - beta * x_temp / (x_temp + kappa)
                x_temp = x + ndt * k3 + noise * nsdt
                x_temp = max(x_temp, 0.0)
                k4 = eta * (tj + ndt) - beta * x_temp / (x_temp + kappa)
                
                # Update position using RK4 for drift + proper noise integration
                drift_update = ndt * (k1 + 2*k2 + 2*k3 + k4) / 6.0
                x += drift_update + noise * nsdt
                x = max(x, 0.0)
                
                if (constant_hazard and np.random.rand() < chance_to_die_externally) or x >= xc:
                    x = xc
                    break
            j += 1
        death_times[person] = j * dt
        events[person] = int(x >= xc)
    return death_times, events

# Runge-Kutta 6th order method (optimized)
@jit(nopython=True)
def death_times_rk6_optimized(s, dt, t, eta0, eta_var, beta0, beta_var, kappa0, kappa_var,
                              epsilon0, epsilon_var, xc0, xc_var, sdt, npeople,
                              external_hazard=np.inf, time_step_multiplier=1):
    death_times = np.zeros(npeople)
    events = np.zeros(npeople, dtype=np.int32)
    ndt = dt / time_step_multiplier
    nsdt = sdt / np.sqrt(time_step_multiplier)
    constant_hazard = np.isfinite(external_hazard)
    if constant_hazard:
        chance_to_die_externally = np.exp(-external_hazard) * ndt
    for person in range(npeople):
        x = 0.0
        j = 0
        eta = eta0 * np.random.normal(1.0, eta_var)
        beta = beta0 * np.random.normal(1.0, beta_var)
        kappa = kappa0 * np.random.normal(1.0, kappa_var)
        epsilon = epsilon0 * np.random.normal(1.0, epsilon_var)
        xc = xc0 * np.random.normal(1.0, xc_var)
        sqrt_2epsilon = np.sqrt(2 * epsilon)
        while j < s - 1 and x < xc:
            for _ in range(time_step_multiplier):
                tj = t[j]
                
                # Generate noise term once for this step
                noise = sqrt_2epsilon * np.random.normal()
                
                # RK6 coefficients for drift term (Butcher tableau for 6th order)
                k1 = eta * tj - beta * x / (x + kappa)
                
                x_temp = x + ndt * k1 / 6.0 + noise * nsdt / 6.0
                x_temp = max(x_temp, 0.0)
                k2 = eta * (tj + ndt / 6.0) - beta * x_temp / (x_temp + kappa)
                
                x_temp = x + ndt * (4*k1 + k2) / 15.0 + noise * nsdt / 3.0
                x_temp = max(x_temp, 0.0)
                k3 = eta * (tj + ndt / 3.0) - beta * x_temp / (x_temp + kappa)
                
                x_temp = x + ndt * (2*k1 + 3*k2 + 4*k3) / 10.0 + noise * nsdt / 2.0
                x_temp = max(x_temp, 0.0)
                k4 = eta * (tj + ndt / 2.0) - beta * x_temp / (x_temp + kappa)
                
                x_temp = x + ndt * (3*k1 - 15*k2 + 20*k3 + 6*k4) / 6.0 + noise * nsdt * 2.0 / 3.0
                x_temp = max(x_temp, 0.0)
                k5 = eta * (tj + 2*ndt / 3.0) - beta * x_temp / (x_temp + kappa)
                
                x_temp = x + ndt * (-6*k1 + 36*k2 - 40*k3 - 12*k4 + 8*k5) / 15.0 + noise * nsdt * 5.0 / 6.0
                x_temp = max(x_temp, 0.0)
                k6 = eta * (tj + 5*ndt / 6.0) - beta * x_temp / (x_temp + kappa)
                
                x_temp = x + ndt * (9*k1 - 36*k2 + 40*k3 + 12*k4 - 8*k5 + 6*k6) / 10.0 + noise * nsdt
                x_temp = max(x_temp, 0.0)
                k7 = eta * (tj + ndt) - beta * x_temp / (x_temp + kappa)
                
                # Update position using RK6 for drift + proper noise integration
                drift_update = ndt * (41*k1 + 216*k3 + 27*k4 + 272*k5 + 27*k6 + 216*k7) / 840.0
                x += drift_update + noise * nsdt
                x = max(x, 0.0)
                
                if (constant_hazard and np.random.rand() < chance_to_die_externally) or x >= xc:
                    x = xc
                    break
            j += 1
        death_times[person] = j * dt
        events[person] = int(x >= xc)
    return death_times, events

# Benchmarking
methods = {
    "Euler": death_times_optimized,
    "Heun": death_times_heun_optimized
}

results = {}
timings = {}

# # Run all simulations
# for name, func in methods.items():
#     start = time.time()
#     deaths, _ = func(s, dt, t, eta0, eta_var, beta0, beta_var, kappa0, kappa_var,
#                      epsilon0, epsilon_var, xc0, xc_var, sdt, npeople,
#                      external_hazard, time_step_multiplier)
#     timings[name] = time.time() - start
#     results[name] = deaths

# # Plot
# plt.figure(figsize=(12, 6))
# bins = np.linspace(0, 10, 100)
# for label, data in results.items():
#     plt.hist(data, bins=bins, alpha=0.6, label=f"{label} (t={timings[label]:.2f}s)", density=True)
# plt.title("Death Time Distribution Comparison")
# plt.xlabel("Death Time")
# plt.ylabel("Density")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()


# Adaptive method (highly optimized)
@jit(nopython=True)
def death_times_adaptive_fast(s, dt, t, eta0, eta_var, beta0, beta_var, kappa0, kappa_var,
                              epsilon0, epsilon_var, xc0, xc_var, sdt, npeople,
                              external_hazard=np.inf, base_multiplier=1):
    death_times = []
    events = []
    chance_to_die_externally = np.exp(-external_hazard) * dt / base_multiplier
    
    for person in range(npeople):
        x = 0.0
        j = 0
        eta = eta0 * np.random.normal(1.0, eta_var)
        beta = beta0 * np.random.normal(1.0, beta_var)
        kappa = kappa0 * np.random.normal(1.0, kappa_var)
        epsilon = epsilon0 * np.random.normal(1.0, epsilon_var)
        xc = xc0 * np.random.normal(1.0, xc_var)
        sqrt_2epsilon = np.sqrt(2 * epsilon)
        
        while j < s - 1 and x < xc:
            proximity = x / xc
            multiplier = max(1, int(base_multiplier * (1.0 + 9.0 * proximity)))
            ndt = dt / multiplier
            nsdt = sdt / np.sqrt(multiplier)
            chance_to_die_externally = np.exp(-external_hazard) * ndt
            
            for i in range(multiplier):
                # Inline calculations to reduce function calls
                drift = eta * (t[j] + i * ndt) - beta * x / (x + kappa)
                x += ndt * drift + sqrt_2epsilon * nsdt * np.random.normal()
                x = max(x, 0.0)
                if np.random.rand() < chance_to_die_externally or x >= xc:
                    x = xc
                    break
            j += 1
        death_times.append(j * dt)
        events.append(int(x >= xc))
    
    return np.array(death_times), np.array(events)

# Heun + Adaptive method (optimized)
@jit(nopython=True)
def death_times_heun_adaptive_optimized(s, dt, t, eta0, eta_var, beta0, beta_var, kappa0, kappa_var,
                                        epsilon0, epsilon_var, xc0, xc_var, sdt, npeople,
                                        external_hazard=np.inf, base_multiplier=1):
    death_times = np.zeros(npeople)
    events = np.zeros(npeople, dtype=np.int32)
    constant_hazard = np.isfinite(external_hazard)
    for person in range(npeople):
        x = 0.0
        j = 0
        eta = eta0 * np.random.normal(1.0, eta_var)
        beta = beta0 * np.random.normal(1.0, beta_var)
        kappa = kappa0 * np.random.normal(1.0, kappa_var)
        epsilon = epsilon0 * np.random.normal(1.0, epsilon_var)
        xc = xc0 * np.random.normal(1.0, xc_var)
        sqrt_2epsilon = np.sqrt(2 * epsilon)
        while j < s - 1 and x < xc:
            proximity = x / xc
            multiplier = max(1, int(base_multiplier * (1.0 + 9.0 * proximity)))
            ndt = dt / multiplier
            nsdt = sdt / np.sqrt(multiplier)
            if constant_hazard:
                chance_to_die_externally = np.exp(-external_hazard) * ndt
            for i  in range(multiplier):
                tj = t[j] + i * ndt
                drift1 = eta * tj - beta * x / (x + kappa)
                noise = sqrt_2epsilon * np.random.normal()
                x_predict = x + ndt * drift1 + noise * nsdt
                x_predict = max(x_predict, 0.0)
                drift2 = eta * (tj + ndt) - beta * x_predict / (x_predict + kappa)
                x += ndt * 0.5 * (drift1 + drift2) + sqrt_2epsilon * nsdt * np.random.normal()
                x = max(x, 0.0)
                if (constant_hazard and np.random.rand() < chance_to_die_externally) or x >= xc:
                    x = xc
                    break
            j += 1
        death_times[person] = j * dt
        events[person] = int(x >= xc)
    return death_times, events

# Original method (unoptimized)
@jit(nopython=True)
def death_times_original(s, dt, t, eta0, eta_var, beta0, beta_var, kappa0, kappa_var,
                         epsilon0, epsilon_var, xc0, xc_var, sdt, npeople,
                         external_hazard=np.inf, time_step_multiplier=1):
    death_times = []
    events = []
    for i in range(npeople):
        x = 0.0
        j = 0
        ndt = dt / time_step_multiplier
        nsdt = sdt / np.sqrt(time_step_multiplier)
        chance_to_die_externally = np.exp(-external_hazard) * ndt
        eta = eta0 * np.random.normal(loc=1.0, scale=eta_var)
        beta = beta0 * np.random.normal(loc=1.0, scale=beta_var)
        kappa = kappa0 * np.random.normal(loc=1.0, scale=kappa_var)
        epsilon = epsilon0 * np.random.normal(loc=1.0, scale=epsilon_var)
        xc = xc0 * np.random.normal(loc=1.0, scale=xc_var)
        while j < (s - 1) and x < xc:
            for _ in range(time_step_multiplier):
                noise = np.sqrt(2 * epsilon) * np.random.normal(loc=0.0, scale=1.0)
                x += ndt * (eta * t[j] - beta * x / (x + kappa)) + noise * nsdt
                x = max(x, 0.0)
                if np.random.rand() < chance_to_die_externally:
                    x = xc
                if x >= xc:
                    break
            j += 1
        death_times.append(j * dt)
        events.append(int(x >= xc))
    return np.array(death_times), np.array(events)

# Add all methods to benchmark
methods.update({
    "Euler_Fast": death_times_euler_fast,
    "RK4": death_times_rk4_optimized,
    "RK6": death_times_rk6_optimized,
    "Adaptive": death_times_adaptive_fast,
    "Heun+Adaptive": death_times_heun_adaptive_optimized,
    "Original": death_times_original
})

# Run all and plot
#dictionary of methods to run with their time step multipliers
methods_to_run = {
    # "Euler": (death_times_optimized, 1000),
    "Euler_Fast": (death_times_euler_fast, 11),
    "Heun": (death_times_heun_optimized, 10),
    "RK4": (death_times_rk4_optimized, 10),
    "RK6": (death_times_rk6_optimized, 10),
    # "Adaptive": (death_times_adaptive_optimized, base_multiplier),
    # "Heun+Adaptive": (death_times_heun_adaptive_optimized, base_multiplier),
    "Original": (death_times_original, 11)
}

def run_all_and_plot(
    methods_to_run,
    s=s,
    dt=dt,
    t=t,
    eta0=eta0,
    eta_var=eta_var,
    beta0=beta0,
    beta_var=beta_var,
    kappa0=kappa0,
    kappa_var=kappa_var,
    epsilon0=epsilon0,
    epsilon_var=epsilon_var,
    xc0=xc0,
    xc_var=xc_var,
    sdt=sdt,
    npeople=npeople,
    external_hazard=external_hazard,
    timings={},
    results={}
):
    """
    Runs all specified methods and plots the death time distributions.

    Args:
        methods_to_run (dict): Dictionary of method names to (function, time_step_multiplier) tuples.
        s, dt, t, eta0, eta_var, beta0, beta_var, kappa0, kappa_var, epsilon0, epsilon_var, xc0, xc_var, sdt, npeople, external_hazard:
            Model parameters.
        timings (dict): Dictionary to store timing results.
        results (dict): Dictionary to store death time results.
    """
    for name, (func, multiplier) in methods_to_run.items():
        start = time.time()
        deaths, _ = func(s, dt, t, eta0, eta_var, beta0, beta_var, kappa0, kappa_var,
                         epsilon0, epsilon_var, xc0, xc_var, sdt, npeople,
                         external_hazard, multiplier)
        timings[name] = time.time() - start
        results[name] = deaths

    plt.figure(figsize=(12, 6))
    bins = np.linspace(0, 5, 100)
    for label, data in results.items():
        plt.hist(data, bins=bins, alpha=0.6, label=f"{label} (t={timings[label]:.2f}s)", density=True)
    plt.title("Death Time Distribution Comparison (All Methods)")
    plt.xlabel("Death Time")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
