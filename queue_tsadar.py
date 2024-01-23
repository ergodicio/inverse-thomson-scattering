import argparse, os, time

from inverse_thomson_scattering.runner import load_and_make_folders

if "BASE_TEMPDIR" in os.environ:
    BASE_TEMPDIR = os.environ["BASE_TEMPDIR"]
else:
    BASE_TEMPDIR = None


def _queue_run_(machine, run_id):
    if "cpu" in machine:
        base_job_file = os.environ["CPU_BASE_JOB_FILE"]
    elif "gpu" in machine:
        base_job_file = os.environ["GPU_BASE_JOB_FILE"]
    else:
        raise NotImplementedError

    with open(base_job_file, "r") as fh:
        base_job = fh.read()

    with open(os.path.join(os.getcwd(), "new_job.sh"), "w") as job_file:
        job_file.write(base_job + "\n")
        job_file.writelines(f"srun python run_tsadar.py --mode fit --run_id {run_id}")

    os.system(f"sbatch new_job.sh")
    time.sleep(0.1)
    os.system("sqs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TSADAR")
    parser.add_argument("--cfg", help="enter path to cfg")
    args = parser.parse_args()

    run_id, all_configs = load_and_make_folders(args.cfg)
    machine = (
        all_configs["inputs"]["machine"] if "machine" in all_configs["inputs"] else all_configs["defaults"]["machine"]
    )
    _queue_run_(machine, run_id)
