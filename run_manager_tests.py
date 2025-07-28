
import numpy as np
from SRtools import sr_mcmc as srmc
import argparse
from SRtools import config_lib as cl
import os
import subprocess
import datetime
import re
from SRtools import presets 

def main():
    parser = argparse.ArgumentParser(description="A script that processes command line arguments.")
    parser.add_argument("config_path", type=str, help="the path to the config file to get params from.")

    

    n_jobs = 1440
    job_name = 'speedTests'
    memory = '0.8'
    folder = 'SpeedTests/metric_tests'
    name = 'test_metrics'
    run_file = 'SpeedTests/run_tests.csh'
    queue = 'short'




    # Create a subfolder for the current submission date
    submission_date = datetime.datetime.now().strftime("%d_%m_%Y")
    submission_folder = os.path.join(folder, f"{name}_submit_{submission_date}")
    os.makedirs(submission_folder, exist_ok=True)
    
    # Create subfolders for output files
    out_folder = os.path.join(submission_folder, "out_files")
    os.makedirs(out_folder, exist_ok=True)
    e_folder = os.path.join(submission_folder, "e_files")
    os.makedirs(e_folder, exist_ok=True)
    
    log_folder = os.path.join(submission_folder, "log")
    os.makedirs(log_folder, exist_ok=True)


    #create MCMC specific folders
    test_results_folder = os.path.join(submission_folder, "test_results")
    os.makedirs(test_results_folder, exist_ok=True)

    for preset in presets.get_preset_names():
        # Send array job to LSF cluster
        job = f"bsub -J \"{job_name}_{preset}[1-{n_jobs}]\" -R 'rusage[mem={memory}GB]' -oo {out_folder}/%J_%I.o -eo {e_folder}/%J_%I.e -q {queue} {run_file} {log_folder} {preset} {test_results_folder}" 
        output = subprocess.run(job, shell=True, capture_output=True, text=True)
        print(output.stdout)
      


    
    


    print("n_jobs: ",n_jobs)
    print("job_name: ",job_name)
    print("memory: ",memory)
    print("folder: ",folder)
    


if __name__ == "__main__":
    main()