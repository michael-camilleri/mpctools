from mpctools.parallel import parallel_progress, ProgressBar

import joblib as jl


def parallelise(func, params, n_jobs=-1, title=None):
    with parallel_progress(ProgressBar(len(params), prec=2, title=title)) as pbar:
        res = jl.Parallel(n_jobs=n_jobs, prefer='processes')(jl.delayed(func)(p) for p in params)
    return res
