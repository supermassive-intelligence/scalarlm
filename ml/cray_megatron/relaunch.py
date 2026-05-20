"""Post-training relaunch dispatch.

When TrainingLoop._finalize_slice sets `relaunch_requested: true` in
status.json (slurm cut the slice short before the user's total timeout
was used up — see docs/training-lifecycle.md §5.4), main.py calls
handle_relaunch_if_needed on the main rank to shell out to
{job_dir}/resubmit.sh. resubmit.sh was written by the API server at
initial submission time and contains the verbatim sbatch invocation.

This lives in its own module — and not in main.py — so the importer
doesn't execute main() at import time. Keeps the logic unit-testable.
"""

import json
import subprocess
from pathlib import Path

# Must match TrainingLoop.RELAUNCH_REQUESTED_KEY. Job state lives in
# status.json — no separate sentinel files.
RELAUNCH_REQUESTED_KEY = "relaunch_requested"
STATUS_FILE = "status.json"
RESUBMIT_SCRIPT = "resubmit.sh"


def handle_relaunch_if_needed(job_dir) -> None:
    """Read status.json, and if relaunch_requested is true, shell out
    to bash resubmit.sh. Tolerates missing / corrupt status.json so a
    bad state doesn't crash the trainer process on its way out — the
    slice has already finished, raising here would just confuse
    operators reading slurm-{id}.out.

    Caller must guard with is_main_rank() — sbatch from every rank
    would queue N relaunches per slice.
    """
    job_dir = Path(job_dir)
    status_path = job_dir / STATUS_FILE
    if not status_path.exists():
        return

    try:
        status = json.loads(status_path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        print(
            f"[relaunch] could not read {status_path} ({e}); skipping",
            flush=True,
        )
        return

    if not status.get(RELAUNCH_REQUESTED_KEY):
        return

    resubmit = job_dir / RESUBMIT_SCRIPT
    print(
        f"[relaunch] status.json has {RELAUNCH_REQUESTED_KEY}=True, "
        "queuing next slice",
        flush=True,
    )

    if not resubmit.is_file():
        print(
            f"[relaunch] WARNING: {resubmit} missing; cannot relaunch — "
            "the next slice will not be queued",
            flush=True,
        )
        return

    # sbatch from inside a dying slurm step is fine; slurmctld queues
    # the new job independently of this process exiting.
    result = subprocess.run(["bash", str(resubmit)])
    if result.returncode != 0:
        print(
            f"[relaunch] resubmit failed with exit code {result.returncode}",
            flush=True,
        )
