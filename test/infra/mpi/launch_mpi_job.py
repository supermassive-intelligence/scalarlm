import subprocess
import os
import sys

def launch_mpi_job():
    sbatch_script = f"""#!/bin/bash
        #SBATCH --job-name=mpi_python_job          # Job name
        #SBATCH --output=slurm-%j.out              # Standard output and error log
        #SBATCH --ntasks=2                         # Number of MPI tasks (ranks)
        #SBATCH --cpus-per-task=1                  # Number of CPUs per task
        #SBATCH --time=00:05:00                    # Time limit (HH:MM:SS)
        #SBATCH --partition=compute                # Partition name (adjust as needed)

        # Run the MPI Python script
        srun mpirun --allow-run-as-root -n 4 --oversubscribe python test_mpi.py
        """

    # Write the sbatch script to a temporary file
    sbatch_file = "mpi_job.sbatch"
    with open(sbatch_file, "w") as f:
        f.write(sbatch_script)

    print(f"Batch script '{sbatch_file}' created.")

    # Submit the batch job using sbatch
    try:
        result = subprocess.run(["sbatch", sbatch_file], capture_output=True, text=True, check=True)
        print(f"Job submitted successfully. Job ID: {result.stdout.strip()}")
    except subprocess.CalledProcessError as e:
        print(f"Error submitting job: {e.stderr.strip()}")

    # Clean up temp batch script 
    os.remove(sbatch_file)
    print(f"Temporary batch script '{sbatch_file}' removed.")

if __name__ == "__main__":
    launch_mpi_job()
