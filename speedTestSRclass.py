from SRtools import SR_hetro as srh
import numpy as np
from numba import jit
from joblib import Parallel, delayed
from SRtools import deathTimesDataSet as dtds
import os
from SRtools import sr_mcmc as srmc


jit_nopython = True

"""
After implementing your class, change sr_mcmc.model so it calls your class instead of the default one and uses your metric function.
"""


class SpeedTestSR(srh.SR_Hetro):
    def __init__(self, eta, beta, kappa, epsilon, xc, npeople, nsteps, t_end,
                 eta_var = 0, beta_var = 0, kappa_var =0, epsilon_var =0, xc_var =0,
                   t_start=0, tscale='years', external_hazard=np.inf, time_step_multiplier=1, parallel=False, bandwidth=3, heun=False, adaptive=False):
        """
        If you want to add parameters to the __init__ method, you can do so here before the call to super().__init__. if you add beta2 as aparameter for example then
        add self.beta2=beta2 here.
        """
        self.adaptive = adaptive

        #this is the call to my class, do not modify it. also, do not earase any of the parameters I cal here unless you give them a default value or somehting
        super().__init__(eta, beta, kappa, epsilon, xc, npeople, nsteps, t_end, eta_var, beta_var, kappa_var, epsilon_var, xc_var, t_start, tscale, external_hazard, time_step_multiplier, parallel, bandwidth, heun)




    def calc_death_times(self):
        s = len(self.t)
        dt = self.t[1] - self.t[0]
        sdt = np.sqrt(dt)
        t = self.t

        if self.adaptive:
            if self.parallel:
                # Use the parallel adaptive method if available
                death_times, events = death_times_adaptive_parallel(
                    s, dt, t, self.eta, self.eta_var, self.beta, self.beta_var,
                    self.kappa, self.kappa_var, self.epsilon, self.epsilon_var,
                    self.xc, self.xc_var, sdt, self.npeople, self.external_hazard, self.time_step_multiplier
                )
            else:
                # Use the serial adaptive method
                death_times, events = death_times_adaptive(
                    s, dt, t, self.eta, self.eta_var, self.beta, self.beta_var,
                    self.kappa, self.kappa_var, self.epsilon, self.epsilon_var,
                    self.xc, self.xc_var, sdt, self.npeople, self.external_hazard, self.time_step_multiplier
                )
        else:
            if self.parallel:
                death_times, events = death_times_accelerator2(
                    s, dt, t, self.eta, self.eta_var, self.beta, self.beta_var,
                    self.kappa, self.kappa_var, self.epsilon, self.epsilon_var,
                    self.xc, self.xc_var, sdt, self.npeople, self.external_hazard, self.time_step_multiplier
                )
            else:
                death_times, events = death_times_accelerator(
                    s, dt, t, self.eta, self.eta_var, self.beta, self.beta_var,
                    self.kappa, self.kappa_var, self.epsilon, self.epsilon_var,
                    self.xc, self.xc_var, sdt, self.npeople, self.external_hazard, self.time_step_multiplier
                )

        return np.array(death_times), np.array(events)
    

def getSpeedTestSR(
    theta,
    n=25000,
    nsteps=6000,
    t_end=110,
    external_hazard=np.inf,
    time_step_multiplier=1,
    npeople=None,
    parallel=False,
    eta_var=0,
    beta_var=0,
    epsilon_var=0,
    xc_var=0.2,
    kappa_var=0,
    hetro=True,
    bandwidth=3,
    adaptive=False,
    step_size=None
):
    """
    Optionally specify step_size. If step_size is given, nsteps and time_step_multiplier are ignored and recalculated so that
    t_end/(nsteps*time_step_multiplier) = step_size. If nsteps*time_step_multiplier <= 6000, time_step_multiplier=1, else
    increase time_step_multiplier until nsteps <= 6000. Both nsteps and time_step_multiplier are integers.
    """

    if npeople is not None:
        n = npeople
    eta = theta[0]
    beta = theta[1]
    epsilon = theta[2]
    xc = theta[3]
    if not hetro:
        eta_var = 0
        beta_var = 0
        epsilon_var = 0
        xc_var = 0
        kappa_var = 0

    if external_hazard is None or external_hazard == 'None':
        external_hazard = np.inf

    # Handle step_size logic
    if step_size is not None:
        # Calculate total steps needed
        total_steps = int(np.ceil(t_end / step_size))
        # Try to keep nsteps <= 6000 by increasing time_step_multiplier
        time_step_multiplier = 1
        nsteps = total_steps
        if nsteps > 6000:
            time_step_multiplier = int(np.ceil(nsteps / 6000))
            nsteps = int(np.ceil(total_steps / time_step_multiplier))
        # Recalculate step_size to be exact
        step_size = t_end / (nsteps * time_step_multiplier)

    sim = SpeedTestSR(
        eta=eta,
        beta=beta,
        epsilon=epsilon,
        xc=xc,
        eta_var=eta_var,
        beta_var=beta_var,
        kappa_var=kappa_var,
        epsilon_var=epsilon_var,
        xc_var=xc_var,
        kappa=0.5,
        npeople=n,
        nsteps=nsteps,
        t_end=t_end,
        external_hazard=external_hazard,
        time_step_multiplier=time_step_multiplier,
        parallel=parallel,
        bandwidth=bandwidth,
        adaptive=adaptive
    )

    return sim


def My_metric(ds,sim):
    """"
    implement your own metric here, and call it from sr_mcmc.model function
    ds is the dataset you compare to, sim is the simulation instance
    needs to return log likelihood of data given simulation
    """
    return




#example metric function
def baysianDistance(sr1, sr2, time_range=None, dt =1, debug = False):
    """
    Calculate the likelihood that the death times of sr1 are generated by sr2.
    convention is that sr1 is the data and sr2 is the simulation
    """
    from statsmodels.nonparametric.kernel_density import KDEMultivariate
    
    #if number of deathtimes is too small that causes issues and anyways not probable as a legitimate parameter set
    if len(sr2.getDeathTimes()) <= 5:
        return -np.inf
    
    death_times2 = sr2.getDeathTimes()
    events2 = sr2.events
    #if time range is not None, use only deathtimes withtin the time range
    if time_range is not None:
        events2 = events2[(death_times2 >= time_range[0]) & (death_times2 <= time_range[1])]
        death_times2 = death_times2[(death_times2 >= time_range[0]) & (death_times2 <= time_range[1])]
    #check that there are enough events to calculate the likelihood meaningfully
    if np.sum(events2) <= 5:
        return -np.inf
    
    death_times2 = death_times2[events2==1] #only those who died
    #this would be a smooth distribution generatated from sr2(simulation) to sample sr1(data) from
    kde = KDEMultivariate(death_times2, var_type='c', bw='normal_reference')
    
    
    events = sr1.events
    death_times = sr1.getDeathTimes()
    if time_range is not None:
        events = events[(death_times >= time_range[0]) & (death_times <= time_range[1])]
        death_times = death_times[(death_times >= time_range[0]) & (death_times <= time_range[1])]
    died = death_times[events==1]
    censored = death_times[events==0]
    ndied = len(died)
    p_death_before_t_end =np.sum(events2)/len(events2)
    
    
    #this piece of code is to accelerate the loglikelihood calculation
    times = np.linspace(0, max(died)+dt, int(np.ceil(max(died)+dt/dt)+1))
    log_pdt = kde.cdf(times) #the log integral of the probability density function on the time grid
    log_pdt = np.log(log_pdt[1:]-log_pdt[:-1])
    times = times[:-1]

    #if logp has nan values then try again then raise an error to debug
    if np.any(np.isnan(log_pdt)):
        if debug:
            print('log_pdt has nan values')
        return np.NaN

    logcdf = np.log(1-kde.cdf(times)) 
    #for every time in death times, find the nearst index in times and get the logp
    logps = 0
    logps_censored = 0
    for t in died:
        idx = np.argmin(np.abs(times-t))
        logps+=(log_pdt[idx])
        if debug:
            if log_pdt[idx] == -np.inf or np.isnan(log_pdt[idx]):
                print(f'log_pdt[{idx}] is {log_pdt[idx]}')

    for t in censored:
        idx = np.argmin(np.abs(times-t))
        logps_censored+=(logcdf[idx])

    #the liklihod given by the kde is L= p(t_death=x_i)|died before t_end) so to correct we need to multiply by the number of ndied/n
    #in the loglikelihood this gives a term of ndied*np.log(p_death_before_t_end)
    sums = logps+ndied*np.log(p_death_before_t_end) +logps_censored
    #check if sums is nan if so then raise an error to debug.
    if np.isnan(sums):
        print('ndied:',ndied)
        print('p_death_before_t_end:',p_death_before_t_end)
        print('logps:',logps)
        print('censored:',censored)
        print('censored_logLiklihood(censored):',logps_censored)
        raise ValueError('sums is nan')
    if debug:
        print('ndied:',ndied)
        print('p_death_before_t_end:',p_death_before_t_end)
        print('logps:',logps)
        print('censored:',censored)
        print('censored_logLiklihood(censored):',logps_censored)
        print('sums:',sums)
    return sums


def model(theta , n, nsteps, t_end, dataSet, sim=None, metric = 'baysian', time_range=None, time_step_multiplier = 1,parallel = False, dt=1, set_params=None, kwargs=None):
    #implement your model here. extra parameters that are passed to the sampler can be used here with kwargs as dictionary
    #if you passed extra parameteers to sr_mcmc.getSampler, as in sr_mcmc.getSampler(.....,bob='men',cost=3), then kwargs 
    #will be a dictionary {'bob':'men','cost':3}
    #The function accepts the parameters of the SR model and returns score according to the metric.

    if set_params is None:
        set_params = {}
    # parse parameters
    pv = srmc.parse_theta(theta, set_params)
    eta = pv['eta']
    beta = pv['beta']
    epsilon = pv['epsilon']
    xc = pv['xc']
    external_hazard = pv['external_hazard']
    return



def example_model(theta , n, nsteps, t_end, dataSet, sim=None, metric = 'baysian', time_range=None, time_step_multiplier = 1,parallel = False, dt=1, set_params=None, kwargs=None):
    """
    The function accepts the parameters of the SR model and returns score according to the metric.
    """
    if set_params is None:
        set_params = {}
    # parse parameters
    pv = srmc.parse_theta(theta, set_params)
    eta = pv['eta']
    beta = pv['beta']
    epsilon = pv['epsilon']
    xc = pv['xc']
    external_hazard = pv['external_hazard']
    theta_sr = np.array([eta, beta, epsilon, xc])
    time_step_size = t_end/(nsteps*time_step_multiplier)
    if 1/beta < time_step_size:
        return -np.inf
    sim = srmc.getSr(theta_sr, n, nsteps, t_end, external_hazard = external_hazard, time_step_multiplier=time_step_multiplier,parallel=parallel) if sim is None else sim
    
    import SRmodellib as sr
    tprob =  sr.distance(dataSet,sim,metric=metric,time_range=time_range, dt=dt)
    if np.any(np.isnan(tprob)):
        return -np.inf

    return tprob


#method without parallelization (for cluster usage)
@jit(nopython=jit_nopython)
def death_times_accelerator(s,dt,t,eta0,eta_var,beta0,beta_var,kappa0,kappa_var,epsilon0,epsilon_var,xc0,xc_var,sdt,npeople,external_hazard = np.inf,time_step_multiplier = 1):
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



##method with parallelization (run on your computer)
def death_times_accelerator2(s, dt, t, eta, eta_var, beta, beta_var, kappa, kappa_var, epsilon, epsilon_var, xc, xc_var, sdt, npeople, external_hazard=np.inf, time_step_multiplier=1):
    @jit(nopython=jit_nopython)
    def calculate_death_times(npeople, s, dt, t, eta0, eta_var, beta0, beta_var, kappa0, kappa_var, epsilon0, epsilon_var, xc0, xc_var, sdt, external_hazard, time_step_multiplier):
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
        return death_times, events

    n_jobs = os.cpu_count()
    npeople_per_job = npeople // n_jobs
    npeople_remainder = npeople % n_jobs
    job_args = [npeople_per_job] * n_jobs
    for i in range(npeople_remainder):
        job_args[i] += 1

    results = Parallel(n_jobs=n_jobs)(
        delayed(calculate_death_times)(
            n, s, dt, t, eta, eta_var, beta, beta_var, kappa, kappa_var, epsilon, epsilon_var, xc, xc_var, sdt, external_hazard, time_step_multiplier
        ) for n in job_args
    )

    death_times = [dt for sublist in results for dt in sublist[0]]
    events = [event for sublist in results for event in sublist[1]]
    return death_times, events


@jit(nopython=True)
def death_times_adaptive(s, dt, t, eta0, eta_var, beta0, beta_var, kappa0, kappa_var,
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


    # INSERT_YOUR_CODE
def death_times_adaptive_parallel(s, dt, t, eta0, eta_var, beta0, beta_var, kappa0, kappa_var,
                                 epsilon0, epsilon_var, xc0, xc_var, sdt, npeople,
                                 external_hazard=np.inf, base_multiplier=1, n_jobs=-1, chunk_size=1000):
    """
    Parallel version of death_times_adaptive.
    Splits npeople into chunks and runs death_times_adaptive on each chunk in parallel.
    """
    from joblib import Parallel, delayed
    import numpy as np

    def worker(npeople_chunk, s, dt, t, eta0, eta_var, beta0, beta_var, kappa0, kappa_var,
               epsilon0, epsilon_var, xc0, xc_var, sdt, external_hazard, base_multiplier):
        # Call the numba-jitted function for this chunk
        return death_times_adaptive(
            s, dt, t, eta0, eta_var, beta0, beta_var, kappa0, kappa_var,
            epsilon0, epsilon_var, xc0, xc_var, sdt, npeople_chunk,
            external_hazard, base_multiplier
        )

    # Split npeople into chunks
    n_chunks = npeople // chunk_size
    remainder = npeople % chunk_size
    chunk_sizes = [chunk_size] * n_chunks
    if remainder > 0:
        chunk_sizes.append(remainder)

    results = Parallel(n_jobs=n_jobs)(
        delayed(worker)(
            n_chunk, s, dt, t, eta0, eta_var, beta0, beta_var, kappa0, kappa_var,
            epsilon0, epsilon_var, xc0, xc_var, sdt, external_hazard, base_multiplier
        ) for n_chunk in chunk_sizes if n_chunk > 0
    )

    # Concatenate results
    death_times = np.concatenate([res[0] for res in results])
    events = np.concatenate([res[1] for res in results])
    return death_times, events
