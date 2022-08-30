import subprocess
import time
import numpy as np

def count_running_jobs():
    out = subprocess.run("squeue -u barratt -h -t pending,running -r".split(' '), stdout=subprocess.PIPE, text=True)
    n_jobs = out.stdout.count('\n')
    return n_jobs

def time_since(t0):
    return time.time()-t0

def submit_job():
    out = subprocess.run('sbatch launcher_script.sh'.split(' '), stdout=subprocess.PIPE, text=True)
    return out

if __name__ == '__main__':
    t0 = time.time()
    log = 'logs/rsub.log'
    # all in units of seconds
    query_interval = 10*60 # query interval = 10 minutes. 
    status_interval = 2 # report job status every x s.
    total_run_time = 6*60*60 # run for a total of 6 hours = 6*60*60s
    max_subs = 10
    n_subs = 0
    t1 = t0
    t2 = t0
    while True:
        if time_since(t0) > total_run_time: break # check if we're out of total time
        if time_since(t1) > query_interval:
            # check if we can submit a new job
            if count_running_jobs() < 3:
                to_sleep = 5*np.random.rand()
                with open(log, 'a') as f:
                    f.write('\n'+f'sleeping {to_sleep}s' + '\n')
                time.sleep(to_sleep)
                with open(log, 'a') as f:
                    f.write('\n' + submit_job().stdout + '\n')

                n_subs += 1
                if n_subs >= max_subs:
                    break
            t1 = t1+query_interval

        if time_since(t2) > status_interval:
            # check the status
            with open(log, 'a') as f:
                f.write(str(int(time_since(t0)))+',')
            print(int(time_since(t0)), 's elapsed, ', n_subs, ' submitted', sep='', end='\r')

            t2 = t2+status_interval
