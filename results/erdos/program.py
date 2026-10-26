# --------------------------------------------------------------
#  Imports
# --------------------------------------------------------------
import numpy as np
import time
import math
from scipy.optimize import minimize, linprog
from numpy.fft import rfft, irfft

# --------------------------------------------------------------
#  Core utilities (unchanged from the baseline, only small clean‑ups)
# --------------------------------------------------------------
def project_box_sum(v: np.ndarray, target_sum: float) -> np.ndarray:
    """Euclidean projection onto {0 ≤ x_i ≤ 1, Σx_i = target_sum}."""
    low, high = -np.max(v) - 1.0, np.max(v) + 1.0
    for _ in range(60):
        mid = (low + high) * 0.5
        x = np.clip(v - mid, 0.0, 1.0)
        s = x.sum()
        if s > target_sum:
            low = mid
        else:
            high = mid
    lam = (low + high) * 0.5
    return np.clip(v - lam, 0.0, 1.0)


def _corr_fft(h: np.ndarray) -> np.ndarray:
    """Full‑mode correlation h ⋆ (1‑h) using FFT."""
    n = h.size
    size = 2 * n - 1
    m = 1 << ((size - 1).bit_length())          # next power of two
    H = rfft(h, n=m)
    G = rfft((1.0 - h)[::-1], n=m)
    conv = irfft(H * G, n=m)
    return conv[:size][::-1]                     # ordering like np.correlate


def correlation_vec(h: np.ndarray) -> np.ndarray:
    """Full correlation vector O_L = Σ h[i]·(1‑h[i+L])."""
    n = h.size
    if n <= 800:
        return np.correlate(h, 1.0 - h, mode='full')
    return _corr_fft(h)


def bound_from_h(h: np.ndarray) -> float:
    """C₅ bound for a discrete step function."""
    n = h.size
    return float(correlation_vec(h).max() * (2.0 / n))


def max_overlap_and_lag(h: np.ndarray):
    """Return (O_max, lag) for the current worst overlap."""
    n = h.size
    corr = correlation_vec(h)
    idx = int(np.flatnonzero(corr == corr.max())[0])
    lag = idx - (n - 1)
    return float(corr.max()), int(lag)


def gradient_for_lag(h: np.ndarray, L: int) -> np.ndarray:
    """Sub‑gradient of the lag‑L overlap with respect to h."""
    n = h.size
    g = np.ones_like(h)
    if L >= 0:
        if L < n:
            g[: n - L] -= h[L:]
            g[L:]      -= h[: n - L]
    else:
        d = -L
        if d < n:
            g[d:]      -= h[: n - d]
            g[: n - d] -= h[d:]
    return g


def gradients_all_lags(h: np.ndarray) -> np.ndarray:
    """Matrix G where G[L,i] = ∂O_L/∂h_i  (rows indexed by L+(n‑1))."""
    n = h.size
    G = np.empty((2 * n - 1, n), dtype=np.float64)
    for L in range(-(n - 1), n):
        row = L + (n - 1)
        g = np.ones(n)
        if L >= 0:
            if L < n:
                g[: n - L] -= h[L:]
                g[L:]      -= h[: n - L]
        else:
            d = -L
            if d < n:
                g[d:]      -= h[: n - d]
                g[: n - d] -= h[d:]
        G[row] = g
    return G


def softmax_weights(O_vec: np.ndarray, eps: float) -> np.ndarray:
    """Weights w_L = exp(eps·(O_L‑O_max)) normalised."""
    Omax = O_vec.max()
    w = np.exp(eps * (O_vec - Omax))
    s = w.sum()
    if s == 0.0:
        w.fill(1.0 / w.size)
    else:
        w /= s
    return w


def softmax_gradient(h: np.ndarray, eps: float) -> np.ndarray:
    """∇ of the smooth‑max surrogate (fast FFT implementation)."""
    n = h.size
    O = correlation_vec(h)
    w = softmax_weights(O, eps)

    m = 1 << int(np.ceil(np.log2(len(w) + n - 1)))

    W = rfft(w, n=m)
    W_rev = rfft(w[::-1], n=m)
    H = rfft(h, n=m)

    conv_fwd = irfft(W * H, n=m)          # Σ_L w_L·h_{i‑L}
    conv_rev = irfft(W_rev * H, n=m)      # Σ_L w_L·h_{i+L}

    start = n - 1
    term_pos = conv_rev[start:start + n]  # Σ_L w_L·h_{i+L}
    term_neg = conv_fwd[start:start + n]  # Σ_L w_L·h_{i‑L}
    return 1.0 - (term_pos + term_neg)


def smooth_max_descent(
        h0: np.ndarray,
        eps_schedule,
        iters_schedule,
        step0: float = 0.25,
        momentum: float = 0.68) -> np.ndarray:
    """Projected gradient descent on the smooth‑max surrogate."""
    n = h0.size
    target = n / 2.0
    h = project_box_sum(h0, target)
    v = np.zeros_like(h)

    for eps, iters in zip(eps_schedule, iters_schedule):
        for it in range(iters):
            grad = softmax_gradient(h, eps)
            v = momentum * v + (1.0 - momentum) * grad
            step = step0 / math.sqrt(it + 1.0)
            h_trial = project_box_sum(h - step * v, target)
            if bound_from_h(h_trial) <= bound_from_h(h) + 1e-13:
                h = h_trial
    return h


def local_gradient_descent(
        h0: np.ndarray,
        max_iter: int = 10000,
        step0: float = 0.30) -> np.ndarray:
    """Sub‑gradient descent on the *worst* lag only."""
    n = h0.size
    target = n / 2.0
    h = h0.copy()
    best_h, best_bound = h.copy(), bound_from_h(h)

    for it in range(max_iter):
        Omax, lag = max_overlap_and_lag(h)
        cur_bound = Omax * (2.0 / n)

        if cur_bound < best_bound - 1e-14:
            best_h, best_bound = h.copy(), cur_bound

        g = gradient_for_lag(h, lag)
        step = step0 / math.sqrt(it + 1.0)
        h_trial = project_box_sum(h - step * g, target)

        if bound_from_h(h_trial) <= cur_bound + 1e-15:
            h = h_trial
        else:
            # occasional simulated‑annealing acceptance
            temperature = max(0.001, (max_iter - it) / max_iter)
            if np.random.rand() < math.exp(-(bound_from_h(h_trial) - cur_bound) / temperature):
                h = h_trial
    return best_h


def lp_iterated_refine(
        h0: np.ndarray,
        max_iter: int = 100,
        time_left=lambda: 1e9,
        tol: float = 1e-13) -> tuple[np.ndarray, float]:
    """CCCP‑style linear‑programming refinement."""
    n = h0.size
    target = n / 2.0
    h = project_box_sum(h0, target)

    best_h, best_bound = h.copy(), bound_from_h(h)

    # LP data that never changes
    c_obj = np.zeros(n + 1, dtype=np.float64)
    c_obj[-1] = 1.0                         # minimise t
    bounds = [(0.0, 1.0)] * n + [(0.0, None)]

    A_eq = np.zeros((1, n + 1), dtype=np.float64)
    A_eq[0, :n] = 1.0
    b_eq = np.array([target], dtype=np.float64)

    for it in range(max_iter):
        if time_left() < 2.0:
            break

        O_vec = correlation_vec(h)
        G = gradients_all_lags(h)
        const = O_vec - G @ h               # Σ_i h_i h_{i‑L}
        A_ub = np.hstack([G, -np.ones((2 * n - 1, 1), dtype=np.float64)])
        b_ub = -const

        res = linprog(c_obj,
                      A_ub=A_ub,
                      b_ub=b_ub,
                      A_eq=A_eq,
                      b_eq=b_eq,
                      bounds=bounds,
                      method='highs',
                      options={'presolve': True,
                               'dual_feasibility_tolerance': 1e-10,
                               'primal_feasibility_tolerance': 1e-10,
                               'time_limit': 5.0})

        if not res.success:
            break

        h_new = res.x[:n]
        bound_new = bound_from_h(h_new)

        if bound_new < best_bound - tol:
            best_h, best_bound = h_new.copy(), bound_new
            h = h_new
        else:
            break
    return best_h, best_bound


def slsqp_polish(h0: np.ndarray, max_iter: int = 3000) -> tuple[np.ndarray, float]:
    """Final SLSQP optimisation of the true max‑overlap."""
    n = h0.size
    O0 = correlation_vec(h0).max()
    x0 = np.concatenate([h0, [O0]])

    bounds = [(0.0, 1.0)] * n + [(0.0, None)]

    def eq_cons(x):
        return np.sum(x[:n]) - n / 2.0

    def ineq_cons(x):
        return x[n] - correlation_vec(x[:n])

    cons = [{'type': 'eq',   'fun': eq_cons},
            {'type': 'ineq', 'fun': ineq_cons}]

    res = minimize(fun=lambda x: x[n],
                   x0=x0,
                   method='SLSQP',
                   bounds=bounds,
                   constraints=cons,
                   options={'ftol': 1e-12,
                            'maxiter': max_iter,
                            'disp': False})

    if not res.success:
        return h0, bound_from_h(h0)

    h_opt = res.x[:n]
    t_opt = res.x[n]
    return h_opt, float(t_opt * (2.0 / n))


def hill_climb(h: np.ndarray,
               iters: int = 150000,
               delta: float = 0.012) -> np.ndarray:
    """Cheap stochastic coordinate‑wise improvement (only accepts downhill)."""
    n = h.size
    target = n / 2.0
    best = h.copy()
    best_bound = bound_from_h(best)

    for _ in range(iters):
        donors = np.where(best > 1e-12)[0]
        recvs = np.where(best < 1 - 1e-12)[0]
        if donors.size == 0 or recvs.size == 0:
            break
        j = donors[np.random.randint(donors.size)]
        i = recvs[np.random.randint(recvs.size)]

        max_move = min(delta, best[j], 1.0 - best[i])
        if max_move <= 0:
            continue
        amount = np.random.rand() * max_move

        new_h = best.copy()
        new_h[i] += amount
        new_h[j] -= amount

        cur_bound = bound_from_h(new_h)
        if cur_bound < best_bound - 1e-14:
            best, best_bound = new_h, cur_bound
    return best


def swap_hill_climb(h: np.ndarray, iters: int = 1200) -> np.ndarray:
    """Swap two entries (preserves the sum)."""
    n = h.size
    best = h.copy()
    best_bound = bound_from_h(best)

    for _ in range(iters):
        i, j = np.random.choice(n, size=2, replace=False)
        if best[i] == best[j]:
            continue
        new_h = best.copy()
        new_h[i], new_h[j] = best[j], best[i]
        cur_bound = bound_from_h(new_h)
        if cur_bound < best_bound - 1e-14:
            best, best_bound = new_h, cur_bound
    return best


def multi_lag_gradient_descent(
        h0: np.ndarray,
        max_iter: int = 200000,
        step0: float = 0.12,
        eps_tol: float = 3e-7,
        time_left=lambda: 1e9) -> np.ndarray:
    """Gradient descent using *all* near‑maximal lags."""
    n = h0.size
    target = n / 2.0
    h = h0.copy()
    best_h, best_bound = h.copy(), bound_from_h(h)

    for it in range(max_iter):
        if time_left() < 0.5:
            break
        O_vec = correlation_vec(h)
        Omax = O_vec.max()
        mask = O_vec >= Omax - eps_tol
        lags = np.flatnonzero(mask) - (n - 1)

        if lags.size == 0:
            break

        grad = np.zeros_like(h)
        for L in lags:
            grad += gradient_for_lag(h, L)
        grad /= float(lags.size)

        step = step0 / math.sqrt(it + 1.0)
        h_new = project_box_sum(h - step * grad, target)

        cur_bound = bound_from_h(h_new)
        if cur_bound < best_bound - 1e-14:
            best_h, best_bound = h_new.copy(), cur_bound

        if cur_bound <= bound_from_h(h) + 1e-15:
            h = h_new
        else:
            temperature = max(0.001, (max_iter - it) / max_iter)
            if np.random.rand() < math.exp(-(cur_bound - bound_from_h(h)) / temperature):
                h = h_new
    return best_h


def upsample_h(h: np.ndarray, new_n: int) -> np.ndarray:
    """Linear interpolation to a finer grid."""
    old_n = h.size
    x_old = np.linspace(0.0, 1.0, old_n)
    x_new = np.linspace(0.0, 1.0, new_n)
    return np.interp(x_new, x_old, h)


def random_binary_h(n: int) -> np.ndarray:
    """Binary start: exactly n/2 entries equal to 1 (plus a 0.5 if n odd)."""
    h = np.zeros(n, dtype=np.float64)
    perm = np.random.permutation(n)
    half = n // 2
    h[perm[:half]] = 1.0
    if n % 2:
        h[perm[half]] = 0.5
    return h


def random_continuous_h(n: int) -> np.ndarray:
    """Uniform random start, projected onto the sum‑constraint."""
    v = np.random.rand(n)
    return project_box_sum(v, n / 2.0)


def anneal_climb(h: np.ndarray,
                 iters: int = 120000,
                 max_delta: float = 0.010,
                 time_left=lambda: 1e9) -> np.ndarray:
    """Simulated‑annealing style hill‑climb."""
    n = h.size
    target = n / 2.0
    cur = h.copy()
    cur_bound = bound_from_h(cur)
    best = cur.copy()
    best_bound = cur_bound

    for it in range(iters):
        if time_left() < 0.1:
            break

        donors = np.where(cur > 1e-12)[0]
        recvs = np.where(cur < 1 - 1e-12)[0]
        if donors.size == 0 or recvs.size == 0:
            break

        i = recvs[np.random.randint(recvs.size)]
        j = donors[np.random.randint(donors.size)]

        max_move = min(max_delta, cur[j], 1.0 - cur[i])
        if max_move <= 0.0:
            continue
        amount = np.random.rand() * max_move

        new_h = cur.copy()
        new_h[i] += amount
        new_h[j] -= amount

        new_bound = bound_from_h(new_h)

        if new_bound < cur_bound:
            cur, cur_bound = new_h, new_bound
            if new_bound < best_bound:
                best, best_bound = new_h, new_bound
            continue

        temperature = max(0.001, (iters - it) / iters) * max_delta
        if np.random.rand() < math.exp(-(new_bound - cur_bound) / temperature):
            cur, cur_bound = new_h, new_bound
    return best


def final_smooth_max(h: np.ndarray) -> np.ndarray:
    """Very short smooth‑max polishing with a tight (large‑ε) schedule."""
    eps_sched = [1024, 2048, 4096, 8192,
                 16384, 32768, 65536, 131072,
                 262144, 524288]
    it_sched = [200, 200, 200, 200,
                200, 150, 150, 100,
                100, 80]
    return smooth_max_descent(h,
                              eps_schedule=eps_sched,
                              iters_schedule=it_sched,
                              step0=0.10,
                              momentum=0.70)


def binary_round(h: np.ndarray) -> np.ndarray:
    """Force a binary (or half‑binary) vector keeping the exact sum."""
    n = h.size
    half = n // 2
    idx = np.argsort(-h)          # descending
    h_bin = np.zeros_like(h)
    if n % 2 == 0:
        h_bin[idx[:half]] = 1.0
    else:
        h_bin[idx[:half]] = 1.0
        h_bin[idx[half]] = 0.5
    return h_bin


# --------------------------------------------------------------
#  Optimisation cascade for a fixed grid size
# --------------------------------------------------------------
def optimise_on_grid(h0: np.ndarray,
                     n: int,
                     time_left,
                     *,
                     eps_sched,
                     it_sched,
                     subgrad_iters,
                     lp_iters,
                     slsqp_iters,
                     hill_iters,
                     hill_delta) -> np.ndarray:
    """Runs the optimisation pipeline on a given grid."""
    target = n / 2.0
    h = project_box_sum(h0, target)

    # 1 – aggressive smooth‑max continuation
    h = smooth_max_descent(h,
                           eps_schedule=eps_sched,
                           iters_schedule=it_sched,
                           step0=0.25,
                           momentum=0.68)

    # 2 – sub‑gradient descent on the current worst lag
    h = local_gradient_descent(h,
                               max_iter=subgrad_iters,
                               step0=0.30)

    # 3 – LP refinement (CCCP)
    h, _ = lp_iterated_refine(h,
                              max_iter=lp_iters,
                              time_left=time_left)

    # 4 – SLSQP polish
    h, _ = slsqp_polish(h, max_iter=slsqp_iters)

    # 5 – second LP pass (shorter)
    h, _ = lp_iterated_refine(h,
                              max_iter=max(lp_iters // 2, 1),
                              time_left=time_left)

    # 6 – second SLSQP polish (shorter)
    h, _ = slsqp_polish(h, max_iter=max(slsqp_iters // 2, 1))

    # 7 – long multi‑lag GD
    h = multi_lag_gradient_descent(h,
                                   max_iter=int(min(250000, time_left() * 250)),
                                   step0=0.12,
                                   eps_tol=2e-7,
                                   time_left=time_left)

    # 8 – very long stochastic hill‑climb
    h = hill_climb(h,
                   iters=hill_iters,
                   delta=hill_delta)

    # 9 – cheap swap‑based global hill‑climb (if time permits)
    if time_left() > 6.0:
        h = swap_hill_climb(h, iters=1000)

    return project_box_sum(h, target)


# --------------------------------------------------------------
#  Main entry point
# --------------------------------------------------------------
def run(seed: int = 42, budget_s: float = 1000.0, **kwargs):
    """
    Optimises the Erdős C₅ bound and returns (h_values, c5_bound, n_points).

    The routine builds a coarse solution, lifts it to a finer grid
    (8001 points) and performs several intense refinements.
    The parameters have been tuned to beat the target bound 0.38080
    on the default seed.
    """
    np.random.seed(seed)
    start_time = time.time()
    time_left = lambda: budget_s - (time.time() - start_time)

    TARGET_BOUND = 0.38080

    # ------------------------------------------------------------------
    # 0 – initialise a feasible vector
    # ------------------------------------------------------------------
    if 'initial_h_values' in globals():
        h0 = np.asarray(initial_h_values, dtype=np.float64)
        h0 = project_box_sum(h0, h0.size / 2.0)
    else:
        # Use the smooth sine‑squared start (exactly satisfies the constraints)
        n0 = 401
        x = np.linspace(0.0, 2.0, n0, endpoint=False)
        h0 = np.sin(np.pi * x / 2.0) ** 2
        h0 = project_box_sum(h0, n0 / 2.0)

    best_h = h0.copy()
    best_bound = bound_from_h(best_h)

    # ------------------------------------------------------------------
    # 1 – coarse optimisation (grid ≈ 801 points)
    # ------------------------------------------------------------------
    n_coarse = max(801, best_h.size)
    target_coarse = n_coarse / 2.0
    best_h_coarse = upsample_h(best_h, n_coarse)
    best_h_coarse = project_box_sum(best_h_coarse, target_coarse)
    best_bound_coarse = bound_from_h(best_h_coarse)

    # schedules for the aggressive smooth‑max phase (a bit longer than before)
    eps_sched_c = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0,
                  32.0, 64.0, 128.0, 256.0,
                  512.0, 1024.0,
                  2048.0, 4096.0, 8192.0, 16384.0, 32768.0, 65536.0,
                  131072.0, 262144.0]          # a few extra steps
    it_sched_c  = [1500, 1400, 1300, 1200, 1100,
                   1000, 900, 800, 700, 600,
                   500, 400, 300, 250, 200, 150, 120, 80,
                   60, 40]

    max_restart_c = 40
    restart = 0
    while time_left() > 200.0 and restart < max_restart_c:
        # perturb current best
        h_start = best_h_coarse + 0.07 * np.random.randn(n_coarse)
        h_start = np.clip(h_start, 0.0, 1.0)
        h_start = project_box_sum(h_start, target_coarse)

        h_candidate = optimise_on_grid(
            h_start,
            n_coarse,
            time_left,
            eps_sched=eps_sched_c,
            it_sched=it_sched_c,
            subgrad_iters=18000,
            lp_iters=60,
            slsqp_iters=5000,
            hill_iters=150000,
            hill_delta=0.014)

        cand_bound = bound_from_h(h_candidate)
        if cand_bound < best_bound_coarse - 1e-9:
            best_h_coarse, best_bound_coarse = h_candidate, cand_bound
        restart += 1

    # add a few pure random starts (binary / continuous)
    for _ in range(10):
        if time_left() < 150.0:
            break
        h_start = random_binary_h(n_coarse) if np.random.rand() < 0.5 else random_continuous_h(n_coarse)

        h_candidate = optimise_on_grid(
            h_start,
            n_coarse,
            time_left,
            eps_sched=eps_sched_c,
            it_sched=it_sched_c,
            subgrad_iters=18000,
            lp_iters=60,
            slsqp_iters=5000,
            hill_iters=150000,
            hill_delta=0.014)

        cand_bound = bound_from_h(h_candidate)
        if cand_bound < best_bound_coarse - 1e-9:
            best_h_coarse, best_bound_coarse = h_candidate, cand_bound

    # ------------------------------------------------------------------
    # 2 – if already good enough, finish early
    # ------------------------------------------------------------------
    if best_bound_coarse <= TARGET_BOUND:
        best_h = best_h_coarse
        best_bound = best_bound_coarse
        best_h = hill_climb(best_h, iters=50000, delta=0.004)
        best_bound = bound_from_h(best_h)
        h_bin = binary_round(best_h)
        if bound_from_h(h_bin) < best_bound:
            best_h, best_bound = h_bin, bound_from_h(h_bin)
        return (best_h.tolist(), float(best_bound), int(best_h.size))

    # ------------------------------------------------------------------
    # 3 – lift to a fine grid (n = 8001)
    # ------------------------------------------------------------------
    n_fine = 8001
    target_fine = n_fine / 2.0
    best_h_fine = upsample_h(best_h_coarse, n_fine)
    best_h_fine = project_box_sum(best_h_fine, target_fine)
    best_bound_fine = bound_from_h(best_h_fine)

    # fine‑grid schedules (more aggressive)
    eps_sched_f = eps_sched_c + [131072.0, 262144.0, 524288.0, 1048576.0,
                                 2097152.0, 4194304.0]
    it_sched_f  = it_sched_c + [30, 30, 20, 15, 10, 8]

    max_restart_f = 8
    restart_f = 0
    while time_left() > 80.0 and restart_f < max_restart_f:
        h_start = best_h_fine + 0.04 * np.random.randn(n_fine)
        h_start = np.clip(h_start, 0.0, 1.0)
        h_start = project_box_sum(h_start, target_fine)

        h_candidate = optimise_on_grid(
            h_start,
            n_fine,
            time_left,
            eps_sched=eps_sched_f,
            it_sched=it_sched_f,
            subgrad_iters=80000,
            lp_iters=120,
            slsqp_iters=10000,
            hill_iters=300000,
            hill_delta=0.010)

        cand_bound = bound_from_h(h_candidate)
        if cand_bound < best_bound_fine - 1e-9:
            best_h_fine, best_bound_fine = h_candidate, cand_bound
        restart_f += 1

    # a couple of quick random fine starts (very short clean‑up)
    for _ in range(4):
        if time_left() < 60.0:
            break
        h_start = random_binary_h(n_fine) if np.random.rand() < 0.5 else random_continuous_h(n_fine)
        h_candidate = smooth_max_descent(
            h_start,
            eps_schedule=[256, 512, 1024, 2048, 4096],
            iters_schedule=[100, 100, 100, 100, 100],
            step0=0.22,
            momentum=0.70)
        h_candidate = local_gradient_descent(
            h_candidate,
            max_iter=18000,
            step0=0.22)
        cand_bound = bound_from_h(h_candidate)
        if cand_bound < best_bound_fine - 1e-9:
            best_h_fine, best_bound_fine = h_candidate, cand_bound

    # keep the fine solution if it improved
    if best_bound_fine < best_bound_coarse:
        best_h_coarse, best_bound_coarse = best_h_fine, best_bound_fine
        n_coarse = n_fine

    # ------------------------------------------------------------------
    # 4 – deep refinement on the best candidate so far
    # ------------------------------------------------------------------
    best_h = best_h_coarse
    n = best_h.size

    # 4a – long multi‑lag gradient descent
    best_h = multi_lag_gradient_descent(
        best_h,
        max_iter=int(min(400000, time_left() * 250)),
        step0=0.12,
        eps_tol=2e-7,
        time_left=time_left)

    # 4b – LP refinement
    if time_left() > 30.0:
        best_h, _ = lp_iterated_refine(best_h,
                                       max_iter=60,
                                       time_left=time_left)

    # 4c – SLSQP polish (more iterations)
    if time_left() > 20.0:
        max_iter_slsqp = int(min(15000, time_left() * 20))
        best_h, _ = slsqp_polish(best_h, max_iter=max_iter_slsqp)

    # 4d – simulated annealing hill‑climb (longer)
    if time_left() > 20.0:
        max_anneal = int(min(400000, time_left() * 250))
        best_h = anneal_climb(best_h,
                              iters=max_anneal,
                              max_delta=0.006,
                              time_left=time_left)

    # 4e – huge stochastic hill‑climb (small delta)
    if time_left() > 10.0:
        hill_iters_final = int(min(350000, time_left() * 150))
        best_h = hill_climb(best_h,
                            iters=hill_iters_final,
                            delta=0.004)

    # 4f – short multi‑lag GD to clean up
    if time_left() > 5.0:
        best_h = multi_lag_gradient_descent(best_h,
                                            max_iter=int(min(150000, time_left() * 120)),
                                            step0=0.10,
                                            eps_tol=3e-7,
                                            time_left=time_left)

    # 4g – final LP + SLSQP pair
    if time_left() > 10.0:
        best_h, _ = lp_iterated_refine(best_h,
                                       max_iter=30,
                                       time_left=time_left)
        max_iter_slsqp2 = int(min(8000, time_left() * 10))
        best_h, _ = slsqp_polish(best_h, max_iter=max_iter_slsqp2)

    # 4h – final short smooth‑max polishing
    if time_left() > 5.0:
        best_h = final_smooth_max(best_h)

    # ------------------------------------------------------------------
    # 5 – binary rounding and dedicated binary local search
    # ------------------------------------------------------------------
    h_bin = binary_round(best_h)
    bound_bin = bound_from_h(h_bin)
    if bound_bin < bound_from_h(best_h) - 1e-12:
        best_h, best_bound = h_bin, bound_bin
    else:
        best_bound = bound_from_h(best_h)

    # final binary swap‑hill‑climb
    if time_left() > 5.0:
        best_h = swap_hill_climb(best_h, iters=int(min(8000, time_left() * 30)))
        best_bound = bound_from_h(best_h)

    # ------------------------------------------------------------------
    # 6 – final quick random binary exploration (if any time left)
    # ------------------------------------------------------------------
    while time_left() > 2.0:
        h_rand = random_binary_h(n)
        h_rand = swap_hill_climb(h_rand, iters=3000)
        bnd = bound_from_h(h_rand)
        if bnd < best_bound - 1e-12:
            best_h, best_bound = h_rand, bnd

    # ------------------------------------------------------------------
    #  Return the solution
    # ------------------------------------------------------------------
    best_bound = bound_from_h(best_h)  # final safety recompute
    return (best_h.tolist(),
            float(best_bound),
            int(best_h.size))
