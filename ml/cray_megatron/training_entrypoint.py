"""SLURM batch-script entrypoint for training jobs.

Why this lives in Python (not bash): the per-slice lifecycle has three
distinct responsibilities — (1) launch mpirun, (2) forward SIGTERM so
the python ranks can checkpoint before the slice's slurm time limit
hits, (3) consume the relaunch flag in status.json that TrainingLoop
sets when slurm cut the slice short, and queue the next slice via
resubmit.sh. Bash handles signals awkwardly (traps only fire between
foreground commands; `wait` returns 128+signum and may need re-waiting),
and the relaunch check is real branching code rather than a linear
pipeline. Keeping it in Python keeps the shell wrapper tiny and the
logic testable.

Path contract: invoked from train_job_entrypoint.sh after the env vars
are exported. CRAY_TRAINING_JOB_CONFIG_PATH points at <job_dir>/config.yaml,
which the training code reads via cray_infra.util.get_job_config.
"""

import json
import os
import signal
import subprocess
import sys
from pathlib import Path

# Must match TrainingLoop.RELAUNCH_REQUESTED_KEY. Job state lives in
# status.json — no separate sentinel files.
RELAUNCH_REQUESTED_KEY = "relaunch_requested"
STATUS_FILE = "status.json"
RESUBMIT_SCRIPT = "resubmit.sh"


def main() -> int:
    config_path = os.environ["CRAY_TRAINING_JOB_CONFIG_PATH"]
    job_dir = Path(config_path).resolve().parent

    mpirun_cmd = [
        "mpirun",
        "--allow-run-as-root",
        sys.executable,
        str(job_dir / "ml" / "cray_megatron" / "main.py"),
        *sys.argv[1:],
    ]

    print(f"[entrypoint] launching: {' '.join(mpirun_cmd)}", flush=True)
    proc = subprocess.Popen(mpirun_cmd)

    def forward(signum, _frame):
        # SLURM sends SIGTERM `signal_grace_seconds` before the slice's
        # --time runs out (see --signal=B:TERM@N in
        # launch_training_job.create_slurm_run_command). Without
        # forwarding the signal would die at this wrapper and the
        # python ranks would never get a chance to checkpoint.
        print(
            f"[entrypoint] caught signal {signum}, forwarding to mpirun "
            f"(pid={proc.pid})",
            flush=True,
        )
        try:
            proc.send_signal(signum)
        except ProcessLookupError:
            pass

    signal.signal(signal.SIGTERM, forward)

    # proc.wait() is interruptible by signals on Python 3, but it
    # re-enters and waits again. No need for a manual re-wait loop
    # like the bash version needed.
    exit_code = proc.wait()
    print(f"[entrypoint] mpirun exited with status {exit_code}", flush=True)

    handle_relaunch(job_dir)

    return exit_code


def handle_relaunch(job_dir: Path) -> None:
    status_path = job_dir / STATUS_FILE
    if not status_path.exists():
        return

    try:
        status = json.loads(status_path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        print(
            f"[entrypoint] could not read {status_path} ({e}); skipping "
            "relaunch check",
            flush=True,
        )
        return

    if not status.get(RELAUNCH_REQUESTED_KEY):
        return

    resubmit = job_dir / RESUBMIT_SCRIPT
    print(
        f"[entrypoint] status.json has {RELAUNCH_REQUESTED_KEY}=True, "
        "queuing next slice",
        flush=True,
    )

    if not resubmit.is_file():
        print(
            f"[entrypoint] WARNING: {resubmit} missing; cannot relaunch — "
            "the next slice will not be queued",
            flush=True,
        )
        return

    # sbatch from inside a dying batch step is fine; slurmctld queues
    # the new job independently of this shell exiting. Don't `exec` —
    # we still want to return cleanly so the parent bash can exit with
    # mpirun's exit code.
    result = subprocess.run(["bash", str(resubmit)])
    if result.returncode != 0:
        print(
            f"[entrypoint] resubmit failed with exit code {result.returncode}",
            flush=True,
        )


if __name__ == "__main__":
    sys.exit(main())
