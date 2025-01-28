from cray_infra.training.training_harness import TrainingHarness
from cray_infra.training.training_job_status import TrainingJobStatus

import torch

torch.backends.mkldnn.enabled = False

import traceback
import sys


def print_exception():
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
    import faulthandler

    faulthandler.enable()

    try:
        setup_logging()
        setup_signal_handler(harness)

        trainer = MegatronTrainer(training_harness=harness)
        trainer.train()
    except Exception as e:
        import traceback

        print(f"An error occurred: {e}")
        traceback.print_exc()  # This prints the full stack trace
        print_exception()
        harness.update_status(
            status=TrainingJobStatus.FAILED, metadata={"error": str(e)}
        )
        raise e


def setup_logging():
    logging.basicConfig(level=logging.DEBUG)

    logging.getLogger("filelock").setLevel(logging.WARNING)


def setup_signal_handler(harness):
    def signal_handler(sig, frame):
        logger.warning("Received signal: %s", sig)
        harness.update_status(status=TrainingJobStatus.QUEUED)

        sys.exit(0)

    signal.signal(signal.SIGCONT, signal_handler)


main()
