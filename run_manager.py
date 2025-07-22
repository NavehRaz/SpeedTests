
import numpy as np
from SRtools import sr_mcmc as srmc
import argparse
from SRtools import config_lib as cl
import os
import subprocess
import datetime
import re
from SRtools import cluster_utils as cu

def main():
    parser = argparse.ArgumentParser(description="A script that processes command line arguments.")
    parser.add_argument("config_path", type=str, help="the path to the config file to get params from.")

    
    args = parser.parse_args()
    config_path = args.config_path
    config = cl.read_configs(config_path)
    nsteps = int(config.get('DEFAULT', 'nsteps'))
    npeople = int(config.get('DEFAULT', 'npeople'))
    t_end = int(config.get('DEFAULT', 't_end'))
    n_jobs = int(config.get('DEFAULT', 'n_jobs'))
    nwalkers = int(config.get('DEFAULT', 'nwalkers'))
    nsteps = int(config.get('DEFAULT', 'nsteps'))
    job_name = config.get('DEFAULT', 'job_name')
    memory = config.get('DEFAULT', 'initial_memory')
    h5_file = config.get('DEFAULT', 'h5_file_name')
    folder = config.get('DEFAULT', 'folder')
    name = config.get('DEFAULT', 'name')
    run_file_mcmc = config.get('DEFAULT', 'run_file_mcmc')
    n_mcmc_steps = int(config.get('DEFAULT', 'n_mcmc_steps'))
    queue = config.get('DEFAULT', 'queue')
    mcmc = config.getboolean('DEFAULT', 'mcmc')



    # Create a subfolder for the current submission date
    submission_date = datetime.datetime.now().strftime("%d_%m_%Y")
    submission_folder = os.path.join(folder, f"{name}_submit_{submission_date}")
    cl.add_submition_folder(config, submission_folder,config_path)
    os.makedirs(submission_folder, exist_ok=True)
    
    # Create subfolders for output files
    out_folder = os.path.join(submission_folder, "out_files")
    os.makedirs(out_folder, exist_ok=True)
    e_folder = os.path.join(submission_folder, "e_files")
    os.makedirs(e_folder, exist_ok=True)
    
    log_folder = os.path.join(submission_folder, "log")
    os.makedirs(log_folder, exist_ok=True)


    if mcmc:
        #create MCMC specific folders
        h5_folder = os.path.join(submission_folder, "h5_files")
        os.makedirs(h5_folder, exist_ok=True)
        out_mcmc = os.path.join(out_folder, f"out_files_mcmc")
        os.makedirs(out_mcmc, exist_ok=True)
        e_mcmc = os.path.join(e_folder, f"e_files_mcmc")
        os.makedirs(e_mcmc, exist_ok=True)

        # Send array job to LSF cluster
        job = f"bsub -J \"{job_name}[1-{n_jobs}]\" -R 'rusage[mem={memory}GB]' -oo {out_mcmc}/%J_%I.o -eo {e_mcmc}/%J_%I.e -q {queue} {run_file_mcmc} {log_folder} {config_path} {h5_folder}" 
        output = subprocess.run(job, shell=True, capture_output=True, text=True)
        print('mcmc')
        print(output.stdout)
        job_id = re.search(r'Job <(\d+)>', output.stdout).group(1)
        subject = f"Job {job_name} {job_id} ended" 
        cu.send_email(subject=subject, message=f"stdout: {output.stdout}", when_a_jobe_ends=job_id)


    print("config_path: ",args.config_path) 
    print("nsteps: ",nsteps)
    print("npeople: ",npeople)
    print("nsteps: ",nsteps)
    print("t_end: ",t_end)
    print("n_jobs: ",n_jobs)
    print("nwalkers: ",nwalkers)
    print("n_mcmc_steps: ",n_mcmc_steps)
    print("job_name: ",job_name)
    print("memory: ",memory)
    print("h5_file: ",h5_file)
    print("folder: ",folder)
    


if __name__ == "__main__":
    main()