"""Linux photo-level process pool; portable branch keeps serial execution."""
from concurrent.futures import ProcessPoolExecutor,as_completed
import math
import multiprocessing
import os
from pathlib import Path
import sys


def available_workers(requested,count):
    if requested<0:raise ValueError('--workers должен быть 0 или положительным числом.')
    if sys.platform!='linux':
        if requested>1:raise ValueError('Параллельная ветка предназначена для Linux; используйте --workers 1.')
        return 1
    if requested:return max(1,min(requested,count))
    cpu=len(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else os.cpu_count() or 1
    try:
        quota,period=Path('/sys/fs/cgroup/cpu.max').read_text().split()
        if quota!='max':cpu=min(cpu,max(1,math.ceil(int(quota)/int(period))))
    except (OSError,ValueError):pass
    memory=4
    try:
        limit=Path('/sys/fs/cgroup/memory.max').read_text().strip()
        if limit!='max':
            used=int(Path('/sys/fs/cgroup/memory.current').read_text())
            stats=dict(line.split() for line in Path('/sys/fs/cgroup/memory.stat').read_text().splitlines())
            reclaimable=int(stats.get('inactive_file',0))
            free=int(limit)-used+reclaimable
        else:
            info=dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
            free=int(info['MemAvailable'].split()[0])*1024
        # Conservative budget per working photo plus a parent/UI reserve.
        memory=max(1,(free-512*2**20)//(768*2**20))
    except (OSError,ValueError,KeyError):pass
    return max(1,min(count,cpu,memory,4))


def initialize_worker():
    import cv2
    cv2.setNumThreads(1)


def execute(jobs,workers):
    from .pipeline import process_photo
    print(f'Параллельная обработка: {workers} процессов, по одному фото на процесс',flush=True)
    variables=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
    previous={key:os.environ.get(key) for key in variables}
    try:
        for key in variables:os.environ[key]='1'
        # Fresh interpreters avoid inheriting decoder/OpenCV thread state.
        with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),initializer=initialize_worker) as pool:
            pending={pool.submit(process_photo,job):job for job in jobs}
            for completed,future in enumerate(as_completed(pending),1):
                job=pending[future]
                print(f'[{completed}/{len(jobs)}] {job[1]} · B',flush=True)
                try:yield future.result()
                except Exception as exc:yield {'error':{'photo':job[1],'error':f'Рабочий процесс: {exc}'}}
    finally:
        for key,value in previous.items():
            if value is None:os.environ.pop(key,None)
            else:os.environ[key]=value
