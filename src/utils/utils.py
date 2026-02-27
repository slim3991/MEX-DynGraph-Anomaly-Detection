# import numpy as np
# import tensorly as tl
#
#
# def de_anomalize_tensor(
#     T,
#     low_rank,
#     keep_pecentile: int = 98,
#     alpha: float = 0.4,
#     cp_kwargs: dict | None = None,
# ):
#     if cp_kwargs is None:
#         cp = tl.decomposition.CP(
#             tol=5e-4, rank=low_rank, init="random", normalize_factors=True
#         )
#     else:
#         cp = tl.decomposition.CP(rank=low_rank, **cp_kwargs)
#
#     factors = cp.fit_transform(T)
#     T_reconstructed = tl.cp_to_tensor(factors)
#     resid = T - T_reconstructed
#     threshold = np.percentile(np.abs(resid), keep_pecentile)
#     resid_cleaned = np.where(np.abs(resid) < threshold, resid, 0)
#     noise = alpha * resid_cleaned
#     T_prepared = T_reconstructed + noise
#     return T_prepared
