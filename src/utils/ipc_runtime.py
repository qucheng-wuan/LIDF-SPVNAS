"""Startup-only IPC checks and coordinated rank-zero failures; no model math."""
import json
import os
from pathlib import Path
import tempfile
import time
import traceback


def check_temporary_ipc():
    expected = os.environ.get('LIDF_EXPECTED_TMPDIR')
    if not expected:
        raise RuntimeError('Launch with LIDF_EXPECTED_TMPDIR and TMPDIR/TEMP/TMP set')
    directory = Path(expected).resolve()
    if Path(tempfile.gettempdir()).resolve() != directory:
        raise RuntimeError('Temporary directory fallback/cache mismatch: ' + tempfile.gettempdir())
    if directory.stat().st_uid != os.getuid():
        raise RuntimeError('IPC temporary directory must belong to the current user')
    with tempfile.TemporaryFile(dir=directory) as handle:
        handle.write(b'IPC'); handle.flush()
    # Use the very same multiprocessing address allocation as tensor FD sharing.
    from multiprocessing.connection import Listener
    with Listener(family='AF_UNIX') as listener:
        address = listener.address
        if not Path(address).is_relative_to(directory):
            raise RuntimeError('multiprocessing cached an unexpected directory: ' + address)
    return dict(directory=str(directory),socket_path_bytes=len(os.fsencode(address)))


def loader_timeout(workers, seconds=120):
    if seconds <= 0:
        raise ValueError('Multiprocess DataLoader timeout must be positive')
    return float(seconds) if workers else 0


def initialize_data_worker(worker_id):
    import cv2
    cv2.setNumThreads(1)
    check_temporary_ipc()


def write_progress(directory, phase, **fields):
    path = Path(directory) / 'progress.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(dict(phase=phase,timestamp=time.time(),**fields)))
    temporary.replace(path)


def rank_zero_action(action, label, distributed):
    """All ranks call in the same order, before the next RNG gather/barrier."""
    import torch.distributed as dist
    error, result = [None], None
    if not distributed or dist.get_rank() == 0:
        try:
            result = action()
        except Exception as exc:
            traceback.print_exc()
            error[0] = '%s: %r' % (label, exc)
    if distributed:
        dist.broadcast_object_list(error, src=0)
    if error[0]:
        raise RuntimeError(error[0])
    return result
