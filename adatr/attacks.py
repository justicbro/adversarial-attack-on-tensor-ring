"""Adversarial attacks on tensor ring decomposition.

All attacks look for a perturbation E with ||E||_F <= sqrt_eps that is added to a tensor X
before the user decomposes X + E by TR-ALS.

    adatr       AdaTR (Algorithm 1 of the paper): bilevel attack, gradient by back-propagation
                through all inner TR-ALS iterations.
    fag_adatr   FAG-AdaTR (Algorithm 2): closed-form approximate gradient, no back-propagation.
    attr        ATTR baseline: the extension of adversarially-trained NMF to TR decomposition,
                in which the attacker and the decomposition share the same fitting loss.
    gaussian_noise   Gaussian noise scaled to the budget.

AdaTR and FAG-AdaTR maximize the error of the decomposition with respect to the clean tensor,

    g(E) = || X - TR(G(E)) ||_F^2 ,   G(E) = result of TR-ALS on X + E,

whereas ATTR maximizes the fitting error || X + E - TR(G) ||_F^2 of the perturbed tensor.
"""
import torch

from .tr_als import fold, subchain, tr_als, tr_to_tensor, unfold


def project(E, sqrt_eps):
    """Projection onto the Frobenius ball ||E||_F <= sqrt_eps."""
    norm = torch.norm(E)
    return E * (sqrt_eps / norm) if norm ** 2 > sqrt_eps ** 2 else E


def gaussian_noise(X, sqrt_eps):
    """Gaussian noise with ||E||_F = sqrt_eps. It is also the starting point of the attacks."""
    E = torch.randn_like(X)
    return E / torch.norm(E) * sqrt_eps


def _start(X, sqrt_eps, E_init):
    E = gaussian_noise(X, sqrt_eps) if E_init is None else E_init.clone().detach().to(X.device)
    return project(E, sqrt_eps)


def _small_change(E, E_old, tol):
    return torch.norm(E - E_old) / (torch.norm(E_old) + 1e-12) < tol


def adatr(X, cores_init, sqrt_eps, n_outer=100, n_inner=100, lr=0.1, E_init=None, tol=1e-6, inner_tol=1e-10,
          verbose=False):
    """AdaTR (Algorithm 1).

    In every outer iteration, TR-ALS is run on X + E for `n_inner` iterations from `cores_init`,
    and E is moved along the gradient of g(E) = ||X - TR(G(E))||_F^2, obtained by automatic
    differentiation through all inner iterations. E is then projected onto the budget.
    The ascent step is taken by Adam with learning rate `lr`.

    X: clean tensor. cores_init: initial TR cores of TR-ALS (list of tensors).
    Returns the perturbation E (same size as X).
    """
    E = _start(X, sqrt_eps, E_init).requires_grad_(True)
    optimizer = torch.optim.Adam([E], lr=lr)
    for step in range(n_outer):
        optimizer.zero_grad()
        E_old = E.detach().clone()
        cores = tr_als(X + E, cores_init, n_inner, inner_tol)
        loss = -torch.norm(X - tr_to_tensor(cores)) ** 2
        loss.backward()
        if torch.isnan(E.grad).any():
            break
        optimizer.step()
        with torch.no_grad():
            E.copy_(project(E, sqrt_eps))
        if verbose:
            print(f"[AdaTR] {step + 1}/{n_outer}  ||X - TR(G)||_F = {(-loss.item()) ** 0.5:.4f}")
        if _small_change(E.detach(), E_old, tol):
            break
    return E.detach()


@torch.no_grad()
def fag_adatr(X, cores_init, sqrt_eps, n_outer=100, n_inner=100, eta=0.1, E_init=None, tol=1e-6, inner_tol=1e-10,
              verbose=False):
    """FAG-AdaTR (Algorithm 2).

    The subchain matrices G_{!=n} obtained by TR-ALS on X + E are treated as independent of E.
    The gradient of the mode-n objective is then E_[n] P_n, where P_n = pinv(G_{!=n}) G_{!=n} is the
    projector onto the row space of G_{!=n}, and the update is

        E  <-  E + eta * sum_n w_n Fold_n( E_[n] P_n ),     w_n = I_n / sum_j I_j,

    followed by the projection onto the budget. All modes use the same E (parallel update).
    Returns the perturbation E.
    """
    E = _start(X, sqrt_eps, E_init)
    weights = [size / sum(X.shape) for size in X.shape]
    for step in range(n_outer):
        cores = tr_als(X + E, cores_init, n_inner, inner_tol)
        E_old = E
        E_new = torch.zeros_like(E)
        for n in range(X.ndim):
            G = subchain(cores, n)
            E_n = unfold(E_old, n)
            E_n = E_n + eta * (E_n @ torch.linalg.pinv(G) @ G)
            E_new += weights[n] * fold(E_n, n, X.shape)
        E = project(E_new, sqrt_eps)
        if verbose:
            print(f"[FAG-AdaTR] {step + 1}/{n_outer}  ||X - TR(G)||_F = {torch.norm(X - tr_to_tensor(cores)).item():.4f}")
        if _small_change(E, E_old, tol):
            break
    return E


def attr(X, cores_init, sqrt_eps, n_outer=100, n_inner=100, lr=0.1, E_init=None, tol=1e-6, inner_tol=1e-10,
         verbose=False):
    """ATTR baseline (symmetric min-max objective).

    Alternates between TR-ALS on X + E (the cores are kept from one outer iteration to the next)
    and an ascent step of E on the fitting error ||X + E - TR(G)||_F^2 with the cores fixed.
    Returns the perturbation E.
    """
    E = _start(X, sqrt_eps, E_init).requires_grad_(True)
    optimizer = torch.optim.Adam([E], lr=lr)
    cores = [c.clone().detach() for c in cores_init]
    for step in range(n_outer):
        optimizer.zero_grad()
        E_old = E.detach().clone()
        cores_old = cores
        with torch.no_grad():
            cores = tr_als(X + E, cores, n_inner, inner_tol)
            fitted = tr_to_tensor(cores)
        loss = -torch.norm(X + E - fitted) ** 2
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            E.copy_(project(E, sqrt_eps))
        if verbose:
            print(f"[ATTR] {step + 1}/{n_outer}  ||X - TR(G)||_F = {torch.norm(X - fitted).item():.4f}")
        change = max([_change(E.detach(), E_old)] + [_change(a, b) for a, b in zip(cores, cores_old)])
        if change < tol:
            break
    return E.detach()


def _change(new, old):
    return (torch.norm(new - old) / (torch.norm(old) + 1e-12)).item()
