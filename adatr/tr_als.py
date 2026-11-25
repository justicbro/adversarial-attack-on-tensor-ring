"""Tensor ring (TR) decomposition by alternating least squares (TR-ALS).

A TR decomposition represents an N-th order tensor X of size I_1 x ... x I_N by N third-order
cores G_n of size R_n x I_n x R_{n+1}, with R_{N+1} = R_1:

    X(i_1, ..., i_N) = trace( G_1(:, i_1, :) G_2(:, i_2, :) ... G_N(:, i_N, :) ).

TR-ALS updates one core at a time by solving a linear least-squares problem,

    G_n  <-  X_[n] * pinv(G_{!=n}),

where X_[n] is the mode-n unfolding of X and G_{!=n} is the subchain matrix formed by all
other cores.
"""
import math

import torch


def init_cores(shape, ranks, device="cpu", dtype=torch.float32):
    """Random Gaussian TR cores.

    shape: sizes (I_1, ..., I_N) of the tensor.
    ranks: TR-ranks (R_1, ..., R_N, R_{N+1}) with R_{N+1} = R_1, or one integer for all ranks.
    Uses the global torch random generator (set torch.manual_seed before the call).
    """
    if isinstance(ranks, int):
        ranks = [ranks] * (len(shape) + 1)
    assert len(ranks) == len(shape) + 1 and ranks[0] == ranks[-1], "ranks must be (R_1, ..., R_N, R_1)"
    return [torch.randn((ranks[n], shape[n], ranks[n + 1]), device=device, dtype=dtype) for n in range(len(shape))]


def tr_to_tensor(cores):
    """Full tensor of a TR decomposition (same contraction order as tensorly.tr_to_tensor)."""
    shape = [c.shape[1] for c in cores]
    full = cores[0].reshape(-1, cores[0].shape[2])
    for c in cores[1:-1]:
        r_prev, _, r_next = c.shape
        full = (full @ c.reshape(r_prev, -1)).reshape(-1, r_next)
    last = cores[-1]
    full = full.reshape(last.shape[2], -1, last.shape[0])
    full = torch.movedim(full, 0, -1).reshape(-1, last.shape[0] * last.shape[2])
    return (full @ torch.movedim(last, -1, 1).reshape(-1, shape[-1])).reshape(shape)


def _unfold(T, order, k):
    """Permute the modes of T to `order`; the first k+1 modes index the rows, the rest the columns."""
    P = T.permute(*order)
    return P.reshape(math.prod(P.shape[:k + 1]), -1)


def unfold(X, n):
    """Mode-n unfolding X_[n] of size I_n x (I_{n+1} ... I_N I_1 ... I_{n-1})."""
    return _unfold(X, mode_order(X.ndim, n), 0)


def fold(M, n, shape):
    """Inverse of `unfold`: matrix of size I_n x (...) back to a tensor of size `shape`."""
    order = mode_order(len(shape), n)
    T = M.reshape(*[shape[i] for i in order])
    return T.permute(*[order.index(i) for i in range(len(shape))])


def mode_order(N, n):
    """Cyclic order of the modes that starts at mode n: n, n+1, ..., N-1, 0, ..., n-1."""
    return [n] + list(range(n + 1, N)) + list(range(0, n))


def subchain(cores, n):
    """Subchain matrix G_{!=n} of size (R_n R_{n+1}) x (I_{n+1} ... I_N I_1 ... I_{n-1}).

    All cores except the n-th are contracted in the order n+1, ..., N, 1, ..., n-1.
    """
    order = mode_order(len(cores), n)[1:]
    chain = cores[order[0]]
    for i in order[1:]:
        last_first = [chain.ndim - 1] + list(range(chain.ndim - 1))
        left = _unfold(chain, last_first, 0).T
        right = _unfold(cores[i], [0, 1, 2], 0)
        chain = (left @ right).reshape(*chain.shape[:-1], cores[i].shape[1], cores[i].shape[2])
    return _unfold(chain, [chain.ndim - 1] + list(range(chain.ndim - 1)), 1)


def als_update(X, cores, n, rtol=1e-6):
    """Least-squares update of the n-th core with the other cores fixed."""
    new = unfold(X, n) @ torch.linalg.pinv(subchain(cores, n), rtol=rtol)
    r_prev, size, r_next = cores[n].shape
    return new.reshape(size, r_prev, r_next).permute(1, 0, 2)


def tr_als(X, cores_init, n_iter=100, tol=1e-10):
    """TR-ALS on X, started from `cores_init`; returns the list of cores.

    One iteration updates all N cores once. The loop also stops when the relative error
    ||X - TR(G)||_F / ||X||_F changes by less than `tol` between two iterations.
    The function is differentiable with respect to X (used by AdaTR).
    """
    cores = [c.clone() for c in cores_init]
    prev = float("inf")
    for _ in range(n_iter):
        for n in range(len(cores)):
            cores[n] = als_update(X, cores, n)
        err = torch.norm(X - tr_to_tensor(cores)) / torch.norm(X)
        if abs(prev - err) < tol:
            break
        prev = err
    return cores
