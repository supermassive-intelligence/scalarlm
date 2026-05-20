"""Module-level signal latch for graceful training shutdown.

A signal handler can't easily reach into the live TrainingState (handlers
are registered before the trainer is built, and even after, threading the
reference through every callback is awkward). Instead, signal handlers set
a module-level flag here; the training loop polls `was_stop_requested()`
at each step boundary and unwinds cleanly so the post-loop checkpoint
runs. `last_signal` lets the loop tell apart a slurm-timeout SIGTERM
(should write a relaunch sentinel) from a SIGCONT preempt (no relaunch
— slurm controls requeue).
"""

_stop_requested = False
_last_signal = None


def request_stop(signal_number=None):
    global _stop_requested, _last_signal
    _stop_requested = True
    if signal_number is not None:
        _last_signal = signal_number


def was_stop_requested():
    return _stop_requested


def last_signal():
    return _last_signal


def reset():
    global _stop_requested, _last_signal
    _stop_requested = False
    _last_signal = None
