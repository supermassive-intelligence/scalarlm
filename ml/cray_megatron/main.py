from cray_infra.training.training_job_status import TrainingJobStatus
from cray_infra.huggingface.get_hf_token import get_hf_token
from cray_infra.util.get_job_config import get_job_config

from cray_megatron.megatron.training_harness import TrainingHarness
from cray_megatron.megatron import stop_flag
from cray_megatron.relaunch import handle_relaunch_if_needed

from cray_megatron.collectives.main_rank_only import is_main_rank

import traceback
import sys
import os
from gpu_aware_mpi import finalize_mpi, get_rank

def print_exception():
    print(f"Rank {get_rank()} hit exception")
    exc_type, exc_value, exc_traceback = sys.exc_info()
    traceback.print_exception(exc_type, exc_value, exc_traceback)


try:
    from cray_megatron.megatron.megatron_trainer import MegatronTrainer
except Exception as e:
    print_exception()

import signal
import logging

logger = logging.getLogger(__name__)

def main():

    harness = TrainingHarness()

    os.environ["HUGGING_FACE_HUB_TOKEN"] = get_hf_token()

    try:
        setup_logging()
        setup_signal_handler(harness)

        trainer = MegatronTrainer(training_harness=harness)
        trainer.train()
    except Exception as e:
        print_exception()
        harness.update_status(
            status=TrainingJobStatus.FAILED, metadata={"error": str(e)}
        )
        raise e

    # Auto-relaunch on slurm slice timeout (docs/training-lifecycle.md
    # §5.4). _finalize_slice has already decided and persisted the
    # answer in status.json; we just dispatch. Main rank only — sbatch
    # from every rank would queue N copies. Runs before finalize_mpi
    # so is_main_rank() can still query MPI state.
    if is_main_rank():
        handle_relaunch_if_needed(get_job_config()["job_directory"])

    finalize_mpi()

def setup_logging():
    logging.basicConfig(level=logging.DEBUG)

    # Central noisy-library list lives in infra/cray_infra/util/quiet_loggers.py.
    # Keep the inline "filelock" line too, because training jobs may run with
    # PYTHONPATH that resolves cray_infra differently; the import guard below
    # falls back gracefully if the helper isn't visible (e.g. unit tests).
    try:
        from cray_infra.util.quiet_loggers import quiet_noisy_loggers
        quiet_noisy_loggers()
    except Exception:
        logging.getLogger("filelock").setLevel(logging.WARNING)

    logging.getLogger("cray_megatron.megatron.distribution.fsdp").setLevel(
        logging.INFO
    )

def setup_signal_handler(harness):
    def signal_handler(sig, frame):
        # Don't sys.exit here — that aborts mid-step, skips the
        # post-loop checkpoint, and the next slice would redo work
        # back to the last steps_per_checkpoint boundary. Set the
        # stop flag and let TrainingLoop unwind: finish the current
        # step, checkpoint, then exit cleanly. The training loop
        # consults stop_flag.last_signal() to decide whether to
        # request a slurm-relaunch (SIGTERM = slice timeout, relaunch;
        # SIGCONT = preempt, no relaunch — slurm owns requeue).
        logger.warning("Received signal %s — requesting graceful stop", sig)
        stop_flag.request_stop(signal_number=sig)

    signal.signal(signal.SIGCONT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


main()
