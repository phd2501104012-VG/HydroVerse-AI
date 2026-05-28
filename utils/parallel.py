import os
from typing import List, Callable, Any, Optional
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import pandas as pd
import numpy as np

from utils.logger import log


def parallel_map(
    func: Callable,
    items: List[Any],
    max_workers: Optional[int] = None,
    use_processes: bool = False,
    desc: str = "Processing",
    verbose: bool = True,
) -> List[Any]:
    max_workers = max_workers or os.cpu_count() or 4
    Executor = ProcessPoolExecutor if use_processes else ThreadPoolExecutor

    results = []
    with Executor(max_workers=max_workers) as executor:
        futures = {executor.submit(func, item): item for item in items}
        for i, future in enumerate(as_completed(futures)):
            try:
                results.append(future.result())
            except Exception as e:
                log.warning(f"{desc}: {futures[future]} failed: {e}")
                results.append(None)
            if verbose and (i + 1) % max(1, len(items) // 5) == 0:
                log.info(f"{desc}: {i + 1}/{len(items)}")
    return results


def parallel_dataframe_apply(
    df: pd.DataFrame,
    func: Callable,
    axis: int = 0,
    max_workers: Optional[int] = None,
) -> pd.DataFrame:
    if axis == 0:
        chunks = np.array_split(df, max_workers or os.cpu_count())
        results = parallel_map(func, chunks, desc="DataFrame apply")
        return pd.concat(results)
    else:
        indices = np.array_split(df.index, max_workers or os.cpu_count())
        def process(idx_range):
            return df.loc[idx_range].apply(func, axis=1)
        results = parallel_map(process, indices, desc="DataFrame apply")
        return pd.concat(results)
