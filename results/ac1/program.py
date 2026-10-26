"""
Propose a step‑function minimising ``evaluate_sequence``.

The optimiser works on a hierarchy of problem sizes (≈2k → 32k) using
Nesterov‑accelerated smooth‑max gradient descent on the probability simplex.
The main novelty is the **gap‑based step rule**

    η = min( s / ||m|| ,  gap / ||m||² ),

where s = step_scale·T (temperature‑scaled step length), m is the momentum,
and gap = |current cheap bound – best cheap bound seen so far* (computed
before a possible improvement).  This replaces the original pure‑temperature
step size and typically yields a finer, more stable descent.

After each phase a cheap LP refinement with a dense golden‑section line‑search
is performed, followed by several deterministic / Adam polishing stages.
The final vector is stored in the global ``height_sequence_1`` and returned
as a plain list of floats.
"""

import time
import math
import numpy as np

# optional SciPy (LP support)
try:
    from scipy.optimize import linprog
    from scipy.linalg import toeplitz
except Exception:  # pragma: no cover
    linprog = None
    toeplitz = None


# ----------------------------------------------------------------------
# FFT utilities
# ----------------------------------------------------------------------
def _next_fast_len(n: int) -> int:
    """Return a length ≥ n that is fast for FFTs (fallback to power‑of‑2)."""
    try:
        return int(np.fft.next_fast_len(int(n)))  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover
        return 1 << (int(n) - 1).bit_length()


# ----------------------------------------------------------------------
# Probability simplex projection (Euclidean)
# ----------------------------------------------------------------------
def simplex_projection(v: np.ndarray) -> np.ndarray:
    """Project a vector onto the simplex {x≥0, Σx = 1}."""
    if np.all(v <= 0):
        return np.full_like(v, 1.0 / v.shape[0])
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1.0
    rho = np.where(u - cssv / (np.arange(v.shape[0]) + 1) > 0)[0][-1]
    theta = cssv[rho] / (rho + 1.0)
    return np.maximum(v - theta, 0.0)


# ----------------------------------------------------------------------
# Seed generators (already on the simplex)
# ----------------------------------------------------------------------
def make_uniform(n: int) -> np.ndarray:
    return np.full(n, 1.0 / n)


def make_triangular(n: int) -> np.ndarray:
    i = np.arange(1, n + 1, dtype=float)
    rev = np.arange(n, 0, -1, dtype=float)
    a = i * rev
    return a / a.sum()


def make_u_shaped(n: int) -> np.ndarray:
    i = np.arange(1, n + 1, dtype=float)
    a = 1.0 / i + 1.0 / (n - i + 1)
    return a / a.sum()


def make_exponential(n: int) -> np.ndarray:
    decay = np.exp(-np.arange(n) / (n / 10.0))
    return decay / decay.sum()


def make_linear_decreasing(n: int) -> np.ndarray:
    a = np.arange(n, 0, -1, dtype=float)
    return a / a.sum()


def make_random_dirichlet(n: int, alpha: float = 1.0) -> np.ndarray:
    v = np.random.gamma(alpha, 1.0, size=n)
    return v / v.sum()


# ----------------------------------------------------------------------
# Upscaling (linear interpolation + simplex projection)
# ----------------------------------------------------------------------
def upscale_sequence(seq: np.ndarray, new_n: int) -> np.ndarray:
    old_n = seq.shape[0]
    if new_n == old_n:
        return seq.copy()
    x_old = np.arange(old_n, dtype=float)
    x_new = np.linspace(0, old_n - 1, new_n)
    seq_new = np.interp(x_new, x_old, seq)
    seq_new = np.maximum(seq_new, 0.0)
    seq_new /= seq_new.sum() + 1e-16
    return simplex_projection(seq_new)


# ----------------------------------------------------------------------
# Smooth‑max gradient (exact max if τ≈0)
# ----------------------------------------------------------------------
def _smoothmax_gradient(
    a_vec: np.ndarray,
    fft_a: np.ndarray,
    phase_factor: np.ndarray,
    tau: float,
    rel_tol: float,
    n: int,
    fft_len: int,
) -> tuple[float, np.ndarray]:
    """
    Return (max convolution value, gradient) for the smooth‑max of the
    self‑convolution. ``fft_a`` is rfft(a_vec) and ``phase_factor`` satisfies
    FFT(a_vec[::-1]) = conj(fft_a) * phase_factor.
    """
    conv_full = np.fft.irfft(fft_a * fft_a, n=fft_len)[: 2 * n - 1]
    max_val = float(conv_full.max())

    # Hard‑max regime – exact gradient via indexing
    if tau < 1e-7:
        idx = int(conv_full.argmax())
        j = np.arange(n)
        kmj = idx - j
        valid = (kmj >= 0) & (kmj < n)
        g = np.zeros(n, dtype=float)
        g[valid] = 2.0 * a_vec[kmj[valid]]
        return max_val, g

    # Smooth‑max regime
    thresh = max_val * (1.0 - rel_tol)
    active = np.where(conv_full >= thresh)[0]

    max_active = min(2000, max(400, int(0.125 * n)))
    if active.size > max_active:
        topk = np.argpartition(-conv_full[active], max_active - 1)[:max_active]
        active = active[topk]

    diffs = conv_full[active] - max_val      # ≤ 0
    tau_eff = max(tau, 1e-12)
    weights = np.exp(diffs / tau_eff)
    wsum = weights.sum()
    if wsum > 0:
        weights /= wsum
    else:
        weights = np.full_like(weights, 1.0 / active.size)

    w_full = np.zeros(2 * n - 1, dtype=float)
    w_full[active] = weights

    fft_w = np.fft.rfft(w_full, n=fft_len)
    fft_rev = np.conj(fft_a) * phase_factor
    conv_g_full = np.fft.irfft(fft_w * fft_rev, n=fft_len)[: 2 * n - 1]
    g = 2.0 * conv_g_full[n - 1 : n - 1 + n]
    return max_val, g


# ----------------------------------------------------------------------
# Cheap bound (exact max of self‑convolution)
# ----------------------------------------------------------------------
def _cheap_score(seq: np.ndarray) -> float:
    n = seq.shape[0]
    fft_len = _next_fast_len(2 * n - 1)
    fft_seq = np.fft.rfft(seq, n=fft_len)
    conv = np.fft.irfft(fft_seq * fft_seq, n=fft_len)[: 2 * n - 1]
    return float(2 * n * conv.max())


# ----------------------------------------------------------------------
# Optimiser for a fixed dimension (Nesterov‑accelerated smooth‑max
# with the *gap‑based* step rule)
# ----------------------------------------------------------------------
def _run_optimizer(
    n: int,
    start_global: float,
    deadline: float,
    seed: int = 0,
    extra_seeds: list[np.ndarray] | None = None,
    prune_frac: float = 0.01,
    safeguard_interval: int = 1_000_000,
) -> tuple[np.ndarray, float]:
    """Return best vector and its cheap bound for dimension ``n``."""
    np.random.seed(seed)

    # ----- initial seeds (already on simplex) -----
    seeds = [
        make_uniform(n),
        make_triangular(n),
        make_u_shaped(n),
        make_exponential(n),
        make_linear_decreasing(n),
        make_random_dirichlet(n, alpha=0.7),
    ]

    if extra_seeds is not None:
        for s in extra_seeds:
            if s.shape[0] == n:
                seeds.append(simplex_projection(s / s.sum()))

    # keep previous global result if it matches dimension
    try:
        prev = np.asarray(globals()["height_sequence_1"], dtype=float)
        if prev.size == n:
            seeds.append(simplex_projection(prev / prev.sum()))
    except Exception:
        pass

    # keep only the three most promising seeds (cheap score)
    scores = [_cheap_score(s) for s in seeds]
    best_idxs = np.argsort(scores)[:3]
    seeds = [seeds[i] for i in best_idxs]

    cur_vecs = [s.copy() for s in seeds]
    prev_vecs = [s.copy() for s in seeds]
    momenta = [np.zeros(n, dtype=float) for _ in seeds]
    prev_grads = [np.zeros(n, dtype=float) for _ in seeds]
    stagnation_counts = [0] * len(seeds)

    fft_len = _next_fast_len(2 * n - 1)
    fft_cur = [np.fft.rfft(v, n=fft_len) for v in cur_vecs]
    fft_prev = [np.fft.rfft(v, n=fft_len) for v in prev_vecs]

    # Pre‑compute the phase factor for fast reversed FFT
    phase_factor = np.exp(
        -2j * np.pi * (n - 1) * np.arange(fft_len // 2 + 1) / fft_len
    )

    # ----- hyper‑parameters (deepened) -----
    T_start = 0.5
    T_end = 1e-8
    beta_start = 0.80
    beta_end = 0.99
    step_scale_small = 1.0
    step_scale_large = 1.5
    jitter_factor = 0.02          # small stochastic perturbations
    JITTER_INTERVAL = 5000
    JITTER_T_MAX = 0.2
    _T_LOW_THR = 1e-3
    _STAGNATION_THR = 2000

    best_vec = None
    best_score = float("inf")    # cheapest bound seen so far (lower is better)
    iteration = 0
    num_seeds = len(seeds)
    pruned = False
    prune_cutoff = start_global + prune_frac * (deadline - start_global)

    while time.time() < deadline:
        # ----- temperature schedule (sqrt‑scaled) -----
        elapsed = time.time() - start_global
        frac = min(elapsed / (deadline - start_global), 1.0)
        T = T_start * ((T_end / T_start) ** (frac ** 0.5))
        rel_tol = max(1e-6, 0.1 * (T / T_start))

        # ----- momentum coeff (linear) -----
        beta = beta_start + (beta_end - beta_start) * frac

        # ----- round‑robin over seeds -----
        idx = iteration % num_seeds
        a_cur = cur_vecs[idx]
        a_prev = prev_vecs[idx]
        fft_a_cur = fft_cur[idx]
        fft_a_prev = fft_prev[idx]
        m_cur = momenta[idx]

        # ----- Nesterov look‑ahead -----
        gamma = iteration / (iteration + 3.0) if iteration > 0 else 0.0
        a_look = a_cur + gamma * (a_cur - a_prev)
        fft_look = (1.0 + gamma) * fft_a_cur - gamma * fft_a_prev

        # ----- smooth‑max gradient (fast reversed FFT) -----
        max_val, grad = _smoothmax_gradient(
            a_look,
            fft_look,
            phase_factor,
            tau=T,
            rel_tol=rel_tol,
            n=n,
            fft_len=fft_len,
        )

        # ----- Barzilai‑Borwein scaling (optional) -----
        if iteration > 0:
            s_vec = a_cur - a_prev
            y_vec = grad - prev_grads[idx]
            denom = np.dot(s_vec, y_vec)
            denom = max(denom, 1e-12)
            bb_factor = np.dot(s_vec, s_vec) / denom
            bb_factor = np.clip(bb_factor, 0.2, 10.0)
        else:
            bb_factor = 1.0
        prev_grads[idx] = grad.copy()

        # ----- momentum update -----
        m_cur = beta * m_cur + (1.0 - beta) * grad

        # ----- cheap bound for this look‑ahead point -----
        cheap_score = 2.0 * n * max_val

        # ----- gap‑based step size -----
        # current best_score may be inf for the very first iteration
        if not np.isfinite(best_score):
            eta2 = float("inf")
        else:
            gap = abs(cheap_score - best_score)    # always non‑negative
            eta2 = gap / (np.linalg.norm(m_cur) ** 2 + 1e-12)

        # eta1 – original temperature‑scaled step
        step_scale = (step_scale_small if T > _T_LOW_THR else step_scale_large) * bb_factor
        s = step_scale * T
        norm_m = np.linalg.norm(m_cur) + 1e-12
        eta1 = max(1e-12, s / max(1.0, norm_m))

        eta = min(eta1, eta2)

        # ----- projected gradient step -----
        a_step = simplex_projection(a_look - eta * m_cur)

        # ----- restart condition -----
        if np.dot(a_cur - a_prev, a_step - a_cur) > 0:
            a_prev = a_cur.copy()
            m_cur = np.zeros_like(m_cur)

        # ----- occasional jitter (helps escape plateaus) -----
        if jitter_factor > 0 and T > JITTER_T_MAX and (iteration % JITTER_INTERVAL) == 0:
            sigma = jitter_factor * math.sqrt(T)
            noise = sigma * np.random.randn(n)
            a_step = simplex_projection(a_step + noise)
            m_cur = np.zeros_like(m_cur)

        # ----- best‑so‑far handling (cheap bound) -----
        if cheap_score < best_score - 1e-14:
            best_score = cheap_score
            best_vec = a_step.copy()
            stagnation_counts[idx] = 0
        else:
            stagnation_counts[idx] += 1

        # ----- safeguard (rarely executed) -----
        if (iteration % safeguard_interval) == 0:
            conv_full = np.fft.irfft(fft_a_cur * fft_a_cur, n=fft_len)[: 2 * n - 1]
            actual_score = 2.0 * n * float(conv_full.max())
            if actual_score < best_score - 1e-14:
                best_score = actual_score
                best_vec = a_cur.copy()
                print(f"[{elapsed:6.1f}s] n={n} safeguard new best = {best_score:.6f}")
                stagnation_counts[idx] = 0

        # ----- stagnation handling -----
        if stagnation_counts[idx] > _STAGNATION_THR:
            m_cur = np.zeros_like(m_cur)
            stagnation_counts[idx] = 0

        # ----- store next state -----
        prev_vecs[idx] = a_cur
        cur_vecs[idx] = a_step
        fft_prev[idx] = fft_a_cur
        fft_cur[idx] = np.fft.rfft(a_step, n=fft_len)
        momenta[idx] = m_cur

        iteration += 1

        # ----- seed pruning (once) -----
        if not pruned and time.time() > prune_cutoff and num_seeds > 1:
            cur_scores = []
            for fft_vec in fft_cur:
                conv = np.fft.irfft(fft_vec * fft_vec, n=fft_len)[: 2 * n - 1]
                cur_scores.append(2.0 * n * float(conv.max()))
            best_idx = int(np.argmin(cur_scores))
            # keep only the best seed
            cur_vecs = [cur_vecs[best_idx]]
            prev_vecs = [prev_vecs[best_idx]]
            momenta = [momenta[best_idx]]
            fft_cur = [fft_cur[best_idx]]
            fft_prev = [fft_prev[best_idx]]
            prev_grads = [prev_grads[best_idx]]
            stagnation_counts = [stagnation_counts[best_idx]]
            num_seeds = 1
            pruned = True
            print(
                f"[{time.time() - start_global:6.1f}s] pruning seeds, keep seed {best_idx}"
                f" with score {cur_scores[best_idx]:.6f}"
            )

    # ----- final cheap sweep (in case no safeguard hit) -----
    for v in cur_vecs:
        scr = _cheap_score(v)
        if scr < best_score:
            best_score = scr
            best_vec = v.copy()

    if best_vec is None:  # safety fallback
        best_vec = cur_vecs[0].copy()
    return best_vec, best_score


# ----------------------------------------------------------------------
# Deterministic polishing (hard‑max gradient + linear decay)
# ----------------------------------------------------------------------
def _deterministic_polish(
    a: np.ndarray,
    n: int,
    start_step: float,
    deadline: float,
) -> tuple[np.ndarray, float]:
    """Simple projected‑gradient polishing with a linear step‑size decay."""
    best_vec = a.copy()
    cur_step = start_step
    total_time = max(deadline - time.time(), 1e-12)

    fft_len = _next_fast_len(2 * n - 1)
    phase_factor = np.exp(
        -2j * np.pi * (n - 1) * np.arange(fft_len // 2 + 1) / fft_len
    )

    while time.time() < deadline:
        fft_cur = np.fft.rfft(best_vec, n=fft_len)
        max_conv, grad = _smoothmax_gradient(
            best_vec,
            fft_cur,
            phase_factor,
            tau=1e-12,
            rel_tol=0.0,
            n=n,
            fft_len=fft_len,
        )
        cur_score = 2.0 * n * max_conv

        remaining = max(0.0, deadline - time.time())
        fraction = remaining / total_time
        step = cur_step * fraction
        if step < 1e-12:
            break

        cand = simplex_projection(best_vec - step * grad)
        cand_score = _cheap_score(cand)

        if cand_score + 1e-14 < cur_score:
            best_vec = cand
            cur_step = min(start_step, cur_step * 1.05)
        else:
            cur_step *= 0.5
            if cur_step < 1e-12:
                break

    return best_vec, _cheap_score(best_vec)


# ----------------------------------------------------------------------
# Adam‑style polishing (adaptive moments + back‑tracking)
# ----------------------------------------------------------------------
def _adam_polish(
    a: np.ndarray,
    n: int,
    deadline: float,
    init_lr: float = 0.1,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    max_backtrack: int = 5,
) -> tuple[np.ndarray, float]:
    """Adam‑style adaptive polishing with line‑search."""
    cur = a.copy()
    best_vec = cur.copy()
    best_score = _cheap_score(best_vec)

    m = np.zeros(n, dtype=float)
    v = np.zeros(n, dtype=float)
    t_iter = 0
    lr = init_lr

    fft_len = _next_fast_len(2 * n - 1)
    phase_factor = np.exp(
        -2j * np.pi * (n - 1) * np.arange(fft_len // 2 + 1) / fft_len
    )

    while time.time() < deadline:
        fft_cur = np.fft.rfft(cur, n=fft_len)
        _, grad = _smoothmax_gradient(
            cur,
            fft_cur,
            phase_factor,
            tau=1e-12,
            rel_tol=0.0,
            n=n,
            fft_len=fft_len,
        )
        t_iter += 1
        m = beta1 * m + (1.0 - beta1) * grad
        v = beta2 * v + (1.0 - beta2) * (grad ** 2)
        m_hat = m / (1.0 - beta1 ** t_iter)
        v_hat = v / (1.0 - beta2 ** t_iter)

        direction = m_hat / (np.sqrt(v_hat) + eps)

        # back‑tracking line search
        alpha = 1.0
        improved = False
        cur_score = _cheap_score(cur)
        while alpha > 1e-6:
            cand = simplex_projection(cur - lr * alpha * direction)
            cand_score = _cheap_score(cand)
            if cand_score + 1e-14 < cur_score:
                cur = cand
                cur_score = cand_score
                improved = True
                break
            alpha *= 0.5

        if not improved:
            lr *= 0.5
            if lr < 1e-8:
                break
        else:
            lr = min(init_lr, lr * 1.02)

        if cur_score < best_score:
            best_vec = cur.copy()
            best_score = cur_score

        if time.time() >= deadline - 0.05:
            break

    return best_vec, best_score


# ----------------------------------------------------------------------
# Small random mass‑transfer refinement (uses evaluate_sequence)
# ----------------------------------------------------------------------
def _post_refine(
    a: np.ndarray,
    current_score: float,
    deadline: float,
    init_delta: float = 0.015,
    decay: float = 0.9995,
    max_reject: int = 30,
) -> tuple[np.ndarray, float]:
    """Deterministic max‑to‑min transfer followed by cheap random swaps."""
    cur = a.copy()
    cur_score = current_score
    delta = init_delta
    reject_cnt = 0

    # deterministic max‑to‑min transfer
    while time.time() < deadline:
        i_max = int(np.argmax(cur))
        i_min = int(np.argmin(cur))
        if i_max == i_min:
            break
        move = min(delta, cur[i_max])
        if move <= 0:
            break
        cand = cur.copy()
        cand[i_max] -= move
        cand[i_min] += move
        cand = simplex_projection(cand)

        cand_score = evaluate_sequence(cand.tolist())
        if cand_score + 1e-14 < cur_score:
            cur = cand
            cur_score = cand_score
            delta *= decay
            reject_cnt = 0
        else:
            reject_cnt += 1
            if reject_cnt >= max_reject:
                delta = init_delta
                reject_cnt = 0

        if delta < 1e-12:
            break

    # random pairwise exchange refinement (cheap, uses evaluate_sequence)
    rand_delta = init_delta
    rand_reject = 0
    n = cur.shape[0]

    while time.time() < deadline:
        i = np.random.randint(n)
        j = np.random.randint(n)
        if i == j:
            continue
        move = min(rand_delta, cur[i])
        if move <= 0:
            continue
        cand = cur.copy()
        cand[i] -= move
        cand[j] += move
        cand = simplex_projection(cand)

        cand_score = evaluate_sequence(cand.tolist())
        if cand_score + 1e-14 < cur_score:
            cur = cand
            cur_score = cand_score
            rand_delta *= decay
            rand_reject = 0
        else:
            rand_reject += 1
            if rand_reject >= max_reject:
                rand_delta = init_delta
                rand_reject = 0

        if rand_delta < 1e-12:
            break

    return cur, cur_score


# ----------------------------------------------------------------------
# LP direction (maximise ℓ₁ under convolution bound)
# ----------------------------------------------------------------------
def _lp_direction(
    a: np.ndarray,
    n: int,
    deadline: float,
    downsample_target: int = 2000,
) -> np.ndarray:
    """Solve the LP used in cheap refinement and return the up‑sampled solution."""
    if linprog is None or toeplitz is None:
        return a.copy()

    if time.time() >= deadline - 0.5:
        return a.copy()

    d = max(500, min(downsample_target, n // 8))
    x_new = np.linspace(0, n - 1, d)
    a_down = np.interp(x_new, np.arange(n), a)
    a_down = np.maximum(a_down, 0.0)
    a_down = simplex_projection(a_down)

    fft_len = _next_fast_len(2 * d - 1)
    fft_a = np.fft.rfft(a_down, n=fft_len)
    conv = np.fft.irfft(fft_a * fft_a, n=fft_len)[: 2 * d - 1]
    M = float(conv.max())

    # convolution matrix (Toeplitz)
    col = np.concatenate([a_down, np.zeros(d - 1)])
    row = np.concatenate([a_down[:1], np.zeros(d - 1)])
    C = toeplitz(col, row)  # (2*d-1, d)

    c = -np.ones(d)
    bounds = [(0.0, None)] * d
    rhs = M * np.ones(C.shape[0])

    try:
        res = linprog(
            c,
            A_ub=C,
            b_ub=rhs,
            bounds=bounds,
            method="highs",
            options={"presolve": True, "time_limit": max(0.0, deadline - time.time())},
        )
    except Exception:
        return a.copy()

    if not res.success:
        return a.copy()

    g_opt = np.maximum(res.x, 0.0)
    g_up = upscale_sequence(g_opt, n)
    return g_up


# ----------------------------------------------------------------------
# LP‑based cheap refinement (keeps improvement if any)
# ----------------------------------------------------------------------
def _lp_refine(
    a: np.ndarray,
    n: int,
    deadline: float,
    downsample_target: int = 2000,
) -> tuple[np.ndarray, float]:
    """Perform the cheap LP refinement; keep it only if it improves the bound."""
    if linprog is None or toeplitz is None:
        return a, _cheap_score(a)

    if time.time() >= deadline - 0.5:
        return a, _cheap_score(a)

    d = max(500, min(downsample_target, n // 8))
    x_new = np.linspace(0, n - 1, d)
    a_down = np.interp(x_new, np.arange(n), a)
    a_down = np.maximum(a_down, 0.0)
    a_down = simplex_projection(a_down)

    fft_len = _next_fast_len(2 * d - 1)
    fft_a = np.fft.rfft(a_down, n=fft_len)
    conv = np.fft.irfft(fft_a * fft_a, n=fft_len)[: 2 * d - 1]
    M = float(conv.max())

    col = np.concatenate([a_down, np.zeros(d - 1)])
    row = np.concatenate([a_down[:1], np.zeros(d - 1)])
    C = toeplitz(col, row)

    c = -np.ones(d)
    bounds = [(0.0, None)] * d
    rhs = M * np.ones(C.shape[0])

    try:
        res = linprog(
            c,
            A_ub=C,
            b_ub=rhs,
            bounds=bounds,
            method="highs",
            options={"presolve": True, "time_limit": max(0.0, deadline - time.time())},
        )
    except Exception:
        return a, _cheap_score(a)

    if not res.success:
        return a, _cheap_score(a)

    g_opt = np.maximum(res.x, 0.0)
    g_up = upscale_sequence(g_opt, n)

    cand_score = _cheap_score(g_up)
    cur_score = _cheap_score(a)
    if cand_score + 1e-14 < cur_score:
        print(f"[lp refine] improvement {cur_score:.6f} → {cand_score:.6f}")
        return g_up, cand_score
    else:
        return a, cur_score


# ----------------------------------------------------------------------
# Dense LP line‑search (101‑point + golden‑section)
# ----------------------------------------------------------------------
def _lp_dense_line_search(
    a: np.ndarray,
    g: np.ndarray,
    a_score: float,
    deadline: float,
    n: int,
) -> tuple[np.ndarray, float]:
    """
    Coarse 101‑point sweep followed by a golden‑section refinement
    on the convex combination (1‑t)·a + t·g, t∈[0,1].
    """
    best_vec = a.copy()
    best_score = a_score
    best_t = 0.0

    # evaluate t=1 (pure LP direction) first
    cand1 = simplex_projection(g)
    cand1_score = _cheap_score(cand1)
    if cand1_score + 1e-14 < best_score:
        best_score = cand1_score
        best_vec = cand1.copy()
        best_t = 1.0

    # ----- coarse grid (101 points) -----
    for t in np.linspace(0.0, 1.0, 101)[1:-1]:  # skip 0 and 1 (already evaluated)
        if time.time() > deadline:
            break
        cand = simplex_projection((1.0 - t) * a + t * g)
        cand_score = _cheap_score(cand)
        if cand_score + 1e-14 < best_score:
            best_score = cand_score
            best_vec = cand.copy()
            best_t = t
            print(f"[lp line‑search] t={t:.3f} improved score to {best_score:.6f}")

    # ----- golden‑section refinement around best coarse point -----
    if time.time() < deadline:
        window = 0.02
        lo = max(0.0, best_t - window)
        hi = min(1.0, best_t + window)
        phi = (1.0 + math.sqrt(5.0)) / 2.0
        tol = 1e-5

        # initial interior points
        c = hi - (hi - lo) / phi
        d = lo + (hi - lo) / phi
        fc = _cheap_score(simplex_projection((1.0 - c) * a + c * g))
        fd = _cheap_score(simplex_projection((1.0 - d) * a + d * g))

        while time.time() < deadline and (hi - lo) > tol:
            if fc < fd:
                hi, fd = d, fc
                d = c
                fd = fc
                c = hi - (hi - lo) / phi
                fc = _cheap_score(simplex_projection((1.0 - c) * a + c * g))
                if fc < best_score:
                    best_score = fc
                    best_vec = simplex_projection((1.0 - c) * a + c * g)
                    best_t = c
                    print(f"[lp line‑search golden] t={c:.5f} improved score to {best_score:.6f}")
            else:
                lo, fc = c, fd
                c = d
                fc = fd
                d = lo + (hi - lo) / phi
                fd = _cheap_score(simplex_projection((1.0 - d) * a + d * g))
                if fd < best_score:
                    best_score = fd
                    best_vec = simplex_projection((1.0 - d) * a + d * g)
                    best_t = d
                    print(f"[lp line‑search golden] t={d:.5f} improved score to {best_score:.6f}")

    return best_vec, best_score


# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------
def propose_candidate(budget_s: int = 1000, seed: int = 0) -> list[float]:
    """
    Search for a non‑negative step‑function that minimises ``evaluate_sequence``.
    The algorithm follows a multi‑scale Nesterov optimisation with the new
    gap‑based step rule, a cheap LP refinement, a dense LP line‑search, and
    several polishing stages.
    """
    np.random.seed(seed)

    start_global = time.time()
    overall_deadline = start_global + budget_s - 2.0  # safety margin

    # time allocation (fractions of the total budget)
    phase1_frac = 0.02   # ~20 s, n≈2000
    phase2_frac = 0.05   # ~50 s, n≈8000
    phase3_frac = 0.10   # ~100 s, n≈16000
    phase4_frac = 0.70   # ~700 s, n≈32000
    polish_frac = 0.12   # ~120 s for polishing / LP

    # ------------------------------------------------------------------
    # Phase 1 (n≈2000)
    # ------------------------------------------------------------------
    n1 = 2000
    phase1_deadline = min(start_global + phase1_frac * budget_s, overall_deadline)
    best1, score1 = _run_optimizer(
        n=n1,
        start_global=start_global,
        deadline=phase1_deadline,
        seed=seed,
        extra_seeds=None,
        prune_frac=0.40,
    )
    print(f"[phase‑1] best cheap score = {score1:.6f}")

    # ------------------------------------------------------------------
    # Phase 2 (n≈8000)
    # ------------------------------------------------------------------
    n2 = 8000
    upscaled12 = upscale_sequence(best1, n2)
    extra2 = [
        upscaled12,
        make_random_dirichlet(n2, alpha=0.7),
        make_u_shaped(n2),
        make_triangular(n2),
    ]
    phase2_deadline = min(
        start_global + (phase1_frac + phase2_frac) * budget_s, overall_deadline
    )
    best2, score2 = _run_optimizer(
        n=n2,
        start_global=start_global,
        deadline=phase2_deadline,
        seed=seed + 1,
        extra_seeds=extra2,
        prune_frac=0.20,
    )
    print(f"[phase‑2] best cheap score = {score2:.6f}")

    # ------------------------------------------------------------------
    # Phase 3 (n≈16000)
    # ------------------------------------------------------------------
    n3 = 16000
    upscaled23 = upscale_sequence(best2, n3)
    extra3 = [
        upscaled23,
        make_random_dirichlet(n3, alpha=0.7),
        make_u_shaped(n3),
        make_triangular(n3),
    ]
    phase3_deadline = min(
        start_global + (phase1_frac + phase2_frac + phase3_frac) * budget_s,
        overall_deadline,
    )
    best3, score3 = _run_optimizer(
        n=n3,
        start_global=start_global,
        deadline=phase3_deadline,
        seed=seed + 2,
        extra_seeds=extra3,
        prune_frac=0.20,
    )
    print(f"[phase‑3] best cheap score = {score3:.6f}")

    # ------------------------------------------------------------------
    # Phase 4 (n≈32000) – deeper (later pruning)
    # ------------------------------------------------------------------
    n4 = 32000
    upscaled34 = upscale_sequence(best3, n4)
    extra4 = [
        upscaled34,
        make_random_dirichlet(n4, alpha=0.7),
    ]
    phase4_deadline = min(
        start_global + (phase1_frac + phase2_frac + phase3_frac + phase4_frac) * budget_s,
        overall_deadline,
    )
    best4, score4 = _run_optimizer(
        n=n4,
        start_global=start_global,
        deadline=phase4_deadline,
        seed=seed + 3,
        extra_seeds=extra4,
        prune_frac=0.05,
    )
    print(f"[phase‑4] best cheap score = {score4:.6f}")

    # ------------------------------------------------------------------
    # Choose the best intermediate result
    # ------------------------------------------------------------------
    best_vec, best_score = best1, score1
    if score2 < best_score:
        best_vec, best_score = best2, score2
    if score3 < best_score:
        best_vec, best_score = best3, score3
    if score4 < best_score:
        best_vec, best_score = best4, score4

    # ------------------------------------------------------------------
    # LP‑based cheap refinement (once)
    # ------------------------------------------------------------------
    lp_deadline = overall_deadline - 0.5
    if lp_deadline > time.time() + 0.5:
        best_vec, best_score = _lp_refine(
            a=best_vec,
            n=best_vec.shape[0],
            deadline=lp_deadline,
            downsample_target=2000,
        )
        print(f"[post‑lp] cheap LP refined score = {best_score:.6f}")

    # ------------------------------------------------------------------
    # Dense LP line‑search (golden‑section refinement)
    # ------------------------------------------------------------------
    mix_deadline = overall_deadline - 0.3
    if linprog is not None and toeplitz is not None and mix_deadline > time.time() + 0.2:
        g_dir = _lp_direction(best_vec, best_vec.shape[0], mix_deadline)
        if g_dir is not None and not np.allclose(g_dir, best_vec):
            best_vec, best_score = _lp_dense_line_search(
                a=best_vec,
                g=g_dir,
                a_score=best_score,
                deadline=mix_deadline,
                n=best_vec.shape[0],
            )

    # ------------------------------------------------------------------
    # Extra LP‑direction + fine line‑search iterations (deepening)
    # ------------------------------------------------------------------
    if linprog is not None and toeplitz is not None:
        extra_lp_iters = 0
        while time.time() < overall_deadline - 30.0 and extra_lp_iters < 5:
            g_dir = _lp_direction(best_vec, best_vec.shape[0], overall_deadline - 5.0)
            if g_dir is None or np.allclose(g_dir, best_vec):
                break
            best_vec, best_score = _lp_dense_line_search(
                a=best_vec,
                g=g_dir,
                a_score=best_score,
                deadline=overall_deadline - 5.0,
                n=best_vec.shape[0],
            )
            extra_lp_iters += 1
            print(f"[extra‑lp‑iter {extra_lp_iters}] score = {best_score:.6f}")

    # ------------------------------------------------------------------
    # Deterministic polishing
    # ------------------------------------------------------------------
    polish_deadline = min(
        start_global
        + (phase1_frac + phase2_frac + phase3_frac + phase4_frac + polish_frac)
        * budget_s,
        overall_deadline,
    )
    best_vec, best_score = _deterministic_polish(
        a=best_vec,
        n=best_vec.shape[0],
        start_step=0.05,
        deadline=polish_deadline,
    )
    print(f"[deterministic polish] score = {best_score:.6f}")

    # ------------------------------------------------------------------
    # Second deterministic polish (optional)
    # ------------------------------------------------------------------
    if time.time() < overall_deadline - 1.0:
        best_vec, best_score = _deterministic_polish(
            a=best_vec,
            n=best_vec.shape[0],
            start_step=0.07,
            deadline=overall_deadline - 0.5,
        )
        print(f"[second polish] score = {best_score:.6f}")

    # ------------------------------------------------------------------
    # Post‑refine: tiny mass‑transfer + random pairwise swaps
    # ------------------------------------------------------------------
    if time.time() < overall_deadline - 0.1:
        best_vec, best_score = _post_refine(
            a=best_vec,
            current_score=best_score,
            deadline=overall_deadline,
            init_delta=0.015,
            decay=0.9995,
            max_reject=30,
        )
        print(f"[post‑refine] score = {best_score:.6f}")

    # ------------------------------------------------------------------
    # Final Adam polishing (uses any remaining budget)
    # ------------------------------------------------------------------
    if time.time() < overall_deadline - 0.05:
        best_vec, best_score = _adam_polish(
            a=best_vec,
            n=best_vec.shape[0],
            deadline=overall_deadline,
            init_lr=0.12,
        )
        print(f"[adam polish] final score = {best_score:.6f}")

    total_elapsed = time.time() - start_global
    print(f"[done] elapsed {total_elapsed:.1f}s, final cheap score = {best_score:.6f}")

    globals()["height_sequence_1"] = best_vec.tolist()
    return best_vec.tolist()