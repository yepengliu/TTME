"""
Deep‑ened multi‑stage optimiser for the step‑function lower bound C.

Main new ideas compared to the previous DEEPEN version:
*   Sharper smooth‑max surrogate – raise the final smoothing parameter from
    5 000 to 10 000, making the Adam/L‑BFGS surrogate much closer to the true
    ‖·‖∞.
*   Multi‑resolution refinement – after the first smooth‑max L‑BFGS polish we
    *up‑sample* the current height vector (repeat each entry twice and add a
    tiny jitter) and re‑initialise the FFT utilities.  The same L‑BFGS,
    exact‑gradient ascent and deterministic PG steps are then run on the larger
    problem.  This gives many more degrees of freedom while re‑using exactly the
    same optimisation components (no new optimiser is introduced).
*   Minor housekeeping – a helper to up‑sample with jitter and a small guard
    that prevents uncontrolled growth of the sequence length.

All other components (gradient formulas, feasibility guards, projection,
time‑budget handling) stay unchanged.
"""

import time
import numpy as np

# ----------------------------------------------------------------------
# Global limits – must match those inside ``evaluate_sequence``
# ----------------------------------------------------------------------
MAX_HEIGHT = 1000.0          # hard ceiling on each height
MIN_TOTAL = 0.01             # lower bound for the total mass (scale invariant)
TOTAL_BUDGET = 1000.0        # total wall‑time allowed (seconds)

# ----------------------------------------------------------------------
# Smoothing schedule for the ∞‑norm (log‑sum‑exp) – used only in
# Adam/L‑BFGS (smooth‑max surrogate)
# ----------------------------------------------------------------------
SMOOTH_ALPHA_START = 10.0
SMOOTH_ALPHA_END = 10000.0          # <<< raised from 5000 → 10000

# ----------------------------------------------------------------------
# FFT utilities (full convolution via FFT, and gradient back‑propagation)
# ----------------------------------------------------------------------
_N = None
_M = None
_NFFT_CONV = None
_NFFT_VALID = None


def _set_fft_params(N: int) -> None:
    """Initialise FFT lengths for a problem of size N."""
    global _N, _M, _NFFT_CONV, _NFFT_VALID
    _N = N
    _M = 2 * N - 1
    _NFFT_CONV = 1 << (_M - 1).bit_length()                # next power‑of‑two ≥ M
    _NFFT_VALID = 1 << ((_M + _N - 1) - 1).bit_length()   # next power‑of‑two ≥ M+N‑1


def _conv_self_fft(h: np.ndarray) -> np.ndarray:
    """Full linear self‑convolution `h * h` using a single forward FFT."""
    A = np.fft.rfft(h.astype(np.float64), n=_NFFT_CONV)
    C = np.fft.irfft(A * A, n=_NFFT_CONV)
    return C[:_M].astype(h.dtype)


def _conv_valid_self(g_conv: np.ndarray, h_rev: np.ndarray) -> np.ndarray:
    """‘valid’ convolution of `g_conv` with the reversed `h` (mode='valid')."""
    G = np.fft.rfft(g_conv.astype(np.float64), n=_NFFT_VALID)
    H = np.fft.rfft(h_rev.astype(np.float64), n=_NFFT_VALID)
    full = np.fft.irfft(G * H, n=_NFFT_VALID)
    start = _N - 1
    return full[start:start + _N].astype(g_conv.dtype)


def _grad_h_from_conv_grad(h: np.ndarray, g_conv: np.ndarray) -> np.ndarray:
    """Chain‑rule: ∂C/∂h = 2·(g_conv convolved with rev(h)) (valid mode)."""
    h_rev = h[::-1]
    conv_valid = _conv_valid_self(g_conv, h_rev)
    return 2.0 * conv_valid


# ----------------------------------------------------------------------
# Simple projection onto the box [0, MAX_HEIGHT] (clipping)
# ----------------------------------------------------------------------
def _clip_heights(h: np.ndarray) -> np.ndarray:
    """Clip heights to the feasible box."""
    return np.clip(h, 0.0, MAX_HEIGHT)


# ----------------------------------------------------------------------
# Mapping between logits and heights (exponential‑clip)
# ----------------------------------------------------------------------
def _exp_clip(z: np.ndarray) -> np.ndarray:
    """Exponential mapping with hard clipping."""
    return _clip_heights(np.exp(z))


def _grad_z_from_h(h: np.ndarray, grad_h: np.ndarray) -> np.ndarray:
    """
    Gradient w.r.t. logits: ∂C/∂z_i = h_i * ∂C/∂h_i,
    zeroed where the clipping is active (h_i == MAX_HEIGHT).
    """
    grad_z = h * grad_h
    grad_z[h >= MAX_HEIGHT] = 0.0
    return grad_z


# ----------------------------------------------------------------------
# Simpson rule for L₂² and its gradient (exact)
# ----------------------------------------------------------------------
def _simpson_l2sq(conv: np.ndarray) -> tuple[float, np.ndarray]:
    """Simpson rule for ∫(conv)² and its gradient w.r.t. conv."""
    M = conv.size
    if M == 0:
        return 0.0, np.zeros_like(conv)
    dx = 1.0 / (M + 1)
    y = np.empty(M + 2, dtype=conv.dtype)
    y[0] = 0.0
    y[1:-1] = conv
    y[-1] = 0.0
    lhs = y[:-1]
    rhs = y[1:]
    l2_sq = (dx / 3.0) * np.sum(lhs * lhs + lhs * rhs + rhs * rhs)
    grad_y = (dx / 3.0) * (4.0 * y + np.roll(y, 1) + np.roll(y, -1))
    grad_conv = grad_y[1:-1]
    return float(l2_sq), grad_conv


# ----------------------------------------------------------------------
# Exact L¹ norm of convolution and its gradient (exact)
# ----------------------------------------------------------------------
def _l1(conv: np.ndarray) -> tuple[float, np.ndarray]:
    """Exact L¹‑norm of the convolution (dx scaling) and its gradient."""
    M = conv.size
    if M == 0:
        return 0.0, np.zeros_like(conv)
    dx = 1.0 / (M + 1)
    val = dx * float(np.sum(conv))
    grad = np.full_like(conv, dx, dtype=conv.dtype)
    return val, grad


# ----------------------------------------------------------------------
# Smooth‑max surrogate (used by Adam and L‑BFGS)
# ----------------------------------------------------------------------
def _objective_and_grad_conv_exact(conv: np.ndarray,
                                   alpha: float = SMOOTH_ALPHA_END) -> tuple[float, np.ndarray]:
    """
    Smooth‑max surrogate of C = L2² / (L1·L∞).
    α controls the approximation to the ∞‑norm via log‑sum‑exp.
    """
    l2_sq, g_l2 = _simpson_l2sq(conv)
    l1, g_l1 = _l1(conv)

    max_c = np.max(conv)
    if max_c <= 0.0:
        return 0.0, np.zeros_like(conv)

    exp_vals = np.exp(alpha * (conv - max_c))
    sum_exp = np.sum(exp_vals)
    if sum_exp <= 0.0:
        smooth_max = max_c
        g_linf = np.zeros_like(conv)
    else:
        smooth_max = max_c + np.log(sum_exp) / alpha
        g_linf = exp_vals / sum_exp

    denom = l1 * smooth_max
    if denom <= 0.0:
        return 0.0, np.zeros_like(conv)

    C = l2_sq / denom
    num_grad = g_l2 * denom - l2_sq * (g_l1 * smooth_max + l1 * g_linf)
    g_conv = num_grad / (denom * denom)
    return float(C), g_conv


# ----------------------------------------------------------------------
# Exact bound (no surrogate) and its gradient
# ----------------------------------------------------------------------
def _objective_and_grad_exact(conv: np.ndarray) -> tuple[float, np.ndarray]:
    """
    Exact bound C = L2² / (L1·L∞) and its gradient w.r.t. conv.
    The ∞‑norm sub‑gradient is uniformly distributed over all maximisers.
    """
    l2_sq, g_l2 = _simpson_l2sq(conv)
    l1, g_l1 = _l1(conv)

    L_inf = np.max(conv)
    if L_inf <= 0.0:
        return 0.0, np.zeros_like(conv)

    eps = 1e-12
    mask = np.abs(conv - L_inf) <= eps
    cnt = np.sum(mask)
    if cnt == 0:
        g_linf = np.zeros_like(conv)
    else:
        g_linf = mask.astype(conv.dtype) / float(cnt)

    denom = l1 * L_inf
    if denom <= 0.0:
        return 0.0, np.zeros_like(conv)

    C = l2_sq / denom
    num_grad = g_l2 * denom - l2_sq * (g_l1 * L_inf + l1 * g_linf)
    g_conv = num_grad / (denom * denom)
    return float(C), g_conv


# ----------------------------------------------------------------------
# Adam optimiser (ascent) – works on *logits* with exponential‑clip mapping
# ----------------------------------------------------------------------
class _Adam:
    """Classic Adam optimiser (ascent version)."""
    def __init__(self, shape, lr: float = 1e-2,
                 beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8,
                 dtype=np.float64):
        self.m = np.zeros(shape, dtype=dtype)
        self.v = np.zeros(shape, dtype=dtype)
        self.t = 0
        self.lr = lr
        self.b1 = beta1
        self.b2 = beta2
        self.eps = eps
        self.dtype = dtype

    def step(self, params: np.ndarray, grad: np.ndarray) -> np.ndarray:
        """Perform one Adam ascent step."""
        self.t += 1
        self.m = self.b1 * self.m + (1.0 - self.b1) * grad
        self.v = self.b2 * self.v + (1 - self.b2) * (grad * grad)
        m_hat = self.m / (1.0 - self.b1 ** self.t)
        v_hat = self.v / (1.0 - self.b2 ** self.t)
        update = self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        return params + update.astype(self.dtype)


def _adam_optimize_logits(logit_init: np.ndarray,
                          total_steps: int,
                          base_lr: float = 1e-2,
                          min_lr_factor: float = 0.05,
                          warmup_frac: float = 0.05,
                          noise_frac: float = 0.10,
                          noise_sigma0: float = 0.012,
                          early_stop_patience: int = 2_000_000,
                          clip_factor: float = 1.2,
                          print_every: int = 10_000,
                          global_start_time: float = 0.0,
                          time_limit: float = TOTAL_BUDGET,
                          rng: np.random.Generator = None) -> tuple[np.ndarray, float]:
    """
    Adam ascent on the smooth‑max surrogate, optimisation variables = logits.
    Returns the best logits encountered and the best surrogate bound value.
    """
    if rng is None:
        rng = np.random.default_rng()

    z = logit_init.astype(np.float64).copy()
    z = z - np.max(z)

    opt = _Adam(shape=z.shape, lr=base_lr, dtype=np.float64)

    best_z = z.copy()
    best_score = -np.inf
    last_improve = 0

    warmup_steps = max(1, int(total_steps * warmup_frac))
    min_frac = min_lr_factor

    for t in range(total_steps):
        if time.time() - global_start_time > time_limit:
            break

        # learning‑rate schedule (linear warm‑up + cosine decay)
        w = (t + 1) / warmup_steps
        w = min(1.0, w)
        if t < warmup_steps:
            c = 1.0
        else:
            ratio = (t - warmup_steps) / max(1, total_steps - warmup_steps)
            c = min_frac + (1.0 - min_frac) * 0.5 * (1.0 + np.cos(np.pi * ratio))
        opt.lr = base_lr * w * c

        # smooth‑max annealing
        ratio_alpha = t / max(1, total_steps)
        alpha = SMOOTH_ALPHA_START + (SMOOTH_ALPHA_END - SMOOTH_ALPHA_START) * ratio_alpha

        # forward pass
        h = _exp_clip(z)
        conv = _conv_self_fft(h)
        C_val, g_conv = _objective_and_grad_conv_exact(conv, alpha)

        if C_val > best_score:
            best_score = C_val
            best_z = z.copy()
            last_improve = t

        # grad wrt heights → logits
        grad_h = _grad_h_from_conv_grad(h, g_conv)
        grad_z = _grad_z_from_h(h, grad_h)

        # early‑phase noise injection
        if t < int(noise_frac * total_steps):
            sigma = noise_sigma0 / np.sqrt(t + 1.0)
            grad_z = grad_z + sigma * rng.standard_normal(grad_z.shape)

        # gradient clipping (optional)
        grad_norm = np.linalg.norm(grad_z)
        if grad_norm > 0:
            max_norm = clip_factor * np.sqrt(grad_z.size) * np.mean(np.abs(grad_z))
            if max_norm > 0 and grad_norm > max_norm:
                grad_z = grad_z * (max_norm / grad_norm)

        # Adam step
        z_new = opt.step(z, grad_z)

        # keep logits numerically stable – shift‑invariance of exp‑clip
        z = z_new - np.max(z_new)

        # early stopping on stagnation
        if t - last_improve > early_stop_patience:
            break

        if (t + 1) % max(1, print_every) == 0:
            elapsed = time.time() - global_start_time
            print(f"   Adam step {t+1}/{total_steps} | C≈{C_val:.6f} "
                  f"| best≈{best_score:.6f} | lr={opt.lr:.2e} "
                  f"| α={alpha:.1f} | elapsed {elapsed:.1f}s")
    return best_z, best_score


# ----------------------------------------------------------------------
# L‑BFGS‑B polishing (smooth‑max surrogate) – deeper
# ----------------------------------------------------------------------
def _lbfgs_optimize_heights(h_start: np.ndarray,
                            global_start_time: float,
                            time_limit: float) -> np.ndarray:
    """Run a deeper L‑BFGS‑B optimisation using the smooth‑max surrogate."""
    elapsed = time.time() - global_start_time
    remaining = time_limit - elapsed
    if remaining < 5.0:
        return h_start

    try:
        from scipy.optimize import minimize
    except Exception:
        print("   SciPy not available – skipping L‑BFGS polish.")
        return h_start

    bounds = [(0.0, MAX_HEIGHT) for _ in range(h_start.size)]
    maxiter = int(min(120000, max(12000, remaining * 120)))  # deeper than before
    print(f"\n=== L‑BFGS‑B polish (smooth‑max, maxiter {maxiter}) ===")

    def fun(x):
        h = np.array(x, dtype=np.float64)
        conv = _conv_self_fft(h)
        C_val, g_conv = _objective_and_grad_conv_exact(conv)   # α = SMOOTH_ALPHA_END
        grad_h = _grad_h_from_conv_grad(h, g_conv)
        return -C_val, -grad_h.astype(np.float64)

    res = minimize(fun,
                   x0=h_start.astype(np.float64),
                   method='L-BFGS-B',
                   jac=True,
                   bounds=bounds,
                   options={'maxiter': maxiter, 'ftol': 1e-12, 'disp': False})

    if not res.success:
        print("   L‑BFGS failed:", res.message)
    else:
        print(f"   L‑BFGS finished – bound ≈ {-res.fun:.6f}")

    return _clip_heights(res.x)


# ----------------------------------------------------------------------
# Adaptive exact‑gradient ascent (cheap)
# ----------------------------------------------------------------------
def _adaptive_gradient_ascent(h_start: np.ndarray,
                              global_start_time: float,
                              time_limit: float,
                              rng: np.random.Generator) -> tuple[np.ndarray, float]:
    """Gradient ascent on the exact bound C = L2²/(L1·L∞) with back‑tracking."""
    h = h_start.copy()
    h = _ensure_min_total(h)

    conv = _conv_self_fft(h)
    best_val, _ = _objective_and_grad_exact(conv)

    step = max(np.mean(h) * 0.01, 1e-6)
    growth_factor = 1.2
    min_step = 1e-8
    fail_streak = 0
    max_fail_streak = 80
    iter_no = 0

    print("\n=== Adaptive exact‑gradient ascent ===")
    while True:
        if time.time() - global_start_time > time_limit - 0.5:
            break

        conv = _conv_self_fft(h)
        C_val, g_conv = _objective_and_grad_exact(conv)
        grad_h = _grad_h_from_conv_grad(h, g_conv)
        norm = np.linalg.norm(grad_h)
        if norm == 0:
            break
        direction = grad_h / norm

        trial_step = step
        improved = False
        while trial_step >= min_step:
            h_candidate = h + trial_step * direction
            h_candidate = _clip_heights(h_candidate)
            h_candidate = _ensure_min_total(h_candidate)

            conv_cand = _conv_self_fft(h_candidate)
            C_cand, _ = _objective_and_grad_exact(conv_cand)

            if C_cand > best_val + 1e-12:
                h = h_candidate
                best_val = C_cand
                step = trial_step * growth_factor
                improved = True
                fail_streak = 0
                if iter_no % 10 == 0:
                    print(f"   iter {iter_no:4d} – C improved to {best_val:.6f} | step {step:.2e}")
                break
            else:
                trial_step *= 0.5

        if not improved:
            fail_streak += 1
            step = max(step * 0.5, min_step)
            if fail_streak >= max_fail_streak:
                print(f"   terminating after {fail_streak} successive failures")
                break

        iter_no += 1

    return h, best_val


# ----------------------------------------------------------------------
# Deterministic projected‑gradient ascent (exact bound) – deepened with momentum
# ----------------------------------------------------------------------
def _deterministic_pg_ascent(h_start: np.ndarray,
                             global_start_time: float,
                             time_limit: float,
                             rng: np.random.Generator = None) -> tuple[np.ndarray, float]:
    """
    Deterministic projected‑gradient ascent on the exact bound,
    enhanced with momentum and deeper back‑track / restart parameters.
    """
    if rng is None:
        rng = np.random.default_rng()

    h = h_start.copy()
    h = _ensure_min_total(h)

    conv = _conv_self_fft(h)
    best_val, _ = _objective_and_grad_exact(conv)
    best_h = h.copy()

    step = max(np.mean(h) * 0.01, 1e-6)
    min_step = 1e-9
    max_backtrack = 800
    max_fail_before_restart = 400
    max_fail_streak = 800
    growth_factor = 1.02
    fail_streak = 0
    max_fail_before_restart_counter = 0
    iter_no = 0

    # momentum parameters
    beta = 0.8
    momentum = np.zeros_like(h)

    prev_h = None
    prev_grad = None

    print("\n=== Deterministic projected‑gradient ascent (momentum) ===")
    while True:
        if time.time() - global_start_time > time_limit - 0.5:
            break

        conv = _conv_self_fft(h)
        C_val, g_conv = _objective_and_grad_exact(conv)
        grad_h = _grad_h_from_conv_grad(h, g_conv)

        # ---- momentum update ----
        momentum = beta * momentum + (1.0 - beta) * grad_h
        if np.dot(momentum, grad_h) < 0:
            momentum = grad_h.copy()
        norm = np.linalg.norm(momentum)
        if norm == 0:
            break
        direction = momentum / norm
        # -------------------------

        trial_step = step
        improved = False
        while trial_step >= min_step and trial_step >= (step / (2 ** max_backtrack)):
            h_candidate = h + trial_step * direction
            h_candidate = _clip_heights(h_candidate)
            h_candidate = _ensure_min_total(h_candidate)

            conv_cand = _conv_self_fft(h_candidate)
            C_cand, _ = _objective_and_grad_exact(conv_cand)

            if C_cand > best_val + 1e-12:
                h = h_candidate
                best_val = C_cand
                best_h = h_candidate.copy()

                # Barzilai‑Borwein step size if possible
                if prev_h is not None and prev_grad is not None:
                    s_vec = h - prev_h
                    y_vec = grad_h - prev_grad
                    denom = np.dot(s_vec, y_vec)
                    if denom > 0:
                        bb_step = np.dot(s_vec, s_vec) / denom
                        step = max(min(bb_step, step * 5.0), min_step)
                else:
                    step = trial_step * growth_factor

                prev_h = h.copy()
                prev_grad = grad_h.copy()
                improved = True
                fail_streak = 0
                max_fail_before_restart_counter = 0
                if iter_no % 10 == 0:
                    print(f"   iter {iter_no:4d} – C improved to {best_val:.6f} | step {step:.2e}")
                break
            else:
                trial_step *= 0.5

        if not improved:
            fail_streak += 1
            max_fail_before_restart_counter += 1
            step = max(step * 0.5, min_step)

            if max_fail_before_restart_counter >= max_fail_before_restart:
                sigma = 0.01 * np.mean(best_h)   # 1 % of mean height
                h = best_h + sigma * rng.standard_normal(best_h.shape)
                h = _clip_heights(h)
                h = _ensure_min_total(h)
                step = max(np.mean(h) * 0.01, 1e-6)
                prev_h = None
                prev_grad = None
                momentum = np.zeros_like(h)      # reset momentum after jitter
                fail_streak = 0
                max_fail_before_restart_counter = 0
                print(f"   restart after {max_fail_before_restart} failures – jittering")
            elif fail_streak >= max_fail_streak:
                break

        iter_no += 1

    return best_h, best_val


# ----------------------------------------------------------------------
# Tiny random hill‑climb (final polishing)
# ----------------------------------------------------------------------
def _random_hill_climb(h_start: np.ndarray,
                       best_val: float,
                       global_start_time: float,
                       time_limit: float,
                       rng: np.random.Generator) -> tuple[np.ndarray, float]:
    """Very cheap hill‑climb that perturbs a single coordinate at a time."""
    h = h_start.copy()
    best_h = h.copy()
    current_val = best_val

    mean_h = np.mean(h) if np.mean(h) > 0 else 1.0
    sigma = max(0.01 * mean_h, 1e-3)

    print("\n=== Random hill‑climb (final polishing) ===")
    while True:
        if time.time() - global_start_time > time_limit - 0.3:
            break

        i = rng.integers(h.size)
        delta = rng.normal(scale=sigma)
        h_cand = h.copy()
        h_cand[i] = np.clip(h_cand[i] + delta, 0.0, MAX_HEIGHT)
        h_cand = _ensure_min_total(h_cand)

        conv = _conv_self_fft(h_cand)
        val, _ = _objective_and_grad_exact(conv)

        if val > current_val + 1e-12:
            h = h_cand
            best_h = h_cand.copy()
            current_val = val
            sigma = max(sigma * 0.95, 1e-4)
            print(f"   improvement to {current_val:.6f}")
        else:
            sigma = min(sigma * 1.02, mean_h * 0.1)

    return best_h, current_val


# ----------------------------------------------------------------------
# Exact‑bound L‑BFGS‑B polishing (deepened)
# ----------------------------------------------------------------------
def _lbfgs_optimize_heights_exact(h_start: np.ndarray,
                                  global_start_time: float,
                                  time_limit: float,
                                  max_iter: int = 150000) -> np.ndarray:
    """
    Bounded L‑BFGS‑B optimisation directly on the exact bound C.
    The iteration budget is now tied more strongly to the remaining time.
    """
    elapsed = time.time() - global_start_time
    remaining = time_limit - elapsed
    if remaining < 2.0:
        return h_start

    try:
        from scipy.optimize import minimize
    except Exception:
        print("   SciPy not available – skipping exact L‑BFGS polish.")
        return h_start

    bounds = [(0.0, MAX_HEIGHT) for _ in range(h_start.size)]
    cap_iter = int(min(max_iter, max(12000, remaining * 150)))  # deeper than before
    if cap_iter <= 0:
        cap_iter = 1

    print(f"\n=== L‑BFGS‑B exact polish (maxiter {cap_iter}) ===")

    def fun(x):
        h = np.array(x, dtype=np.float64)
        conv = _conv_self_fft(h)
        C_val, g_conv = _objective_and_grad_exact(conv)
        grad_h = _grad_h_from_conv_grad(h, g_conv)
        return -C_val, -grad_h.astype(np.float64)

    res = minimize(fun,
                   x0=h_start.astype(np.float64),
                   method='L-BFGS-B',
                   jac=True,
                   bounds=bounds,
                   options={'maxiter': cap_iter, 'ftol': 1e-14, 'disp': False})

    if not res.success:
        print("   Exact L‑BFGS failed:", res.message)
    else:
        print(f"   Exact L‑BFGS finished – bound ≈ {-res.fun:.6f}")

    return _clip_heights(res.x)


# ----------------------------------------------------------------------
# Helper: remaining time
# ----------------------------------------------------------------------
def _remaining_time(start_time: float) -> float:
    return max(0.0, TOTAL_BUDGET - (time.time() - start_time))


# ----------------------------------------------------------------------
# Minimum‑total‑mass safeguard (scale‑invariant, cheap)
# ----------------------------------------------------------------------
def _ensure_min_total(h: np.ndarray) -> np.ndarray:
    """If the total mass is below MIN_TOTAL, scale up (does not affect C)."""
    total = np.sum(h)
    if total < MIN_TOTAL and total > 0:
        factor = MIN_TOTAL / total
        h = h * factor
        h = _clip_heights(h)
    elif total == 0:
        h = np.full_like(h, MIN_TOTAL / h.size)
    return h


# ----------------------------------------------------------------------
# Upscaling helper – repeat entries and add a tiny jitter
# ----------------------------------------------------------------------
def _upscale_sequence(h: np.ndarray, factor: int, rng: np.random.Generator) -> np.ndarray:
    """Repeat each height `factor` times and add a small random jitter."""
    if factor <= 1:
        return h.copy()
    # repeat
    h_up = np.repeat(h, factor)
    # jitter magnitude ~1 % of the mean height (but not smaller than 1e‑4)
    mean_h = np.mean(h_up)
    sigma = max(mean_h * 0.01, 1e-4)
    jitter = sigma * rng.standard_normal(h_up.shape)
    h_up = h_up + jitter
    h_up = _clip_heights(h_up)
    h_up = _ensure_min_total(h_up)
    return h_up


# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------
def construct_function() -> list[float]:
    """Search for the step‑function heights that maximise the lower‑bound C."""
    start_time = time.time()
    rng = np.random.default_rng()

    # ------------------------------------------------------------------
    # Load the reference height vector (provided globally)
    # ------------------------------------------------------------------
    orig_h = np.array(height_sequence_1, dtype=np.float64).copy()
    orig_h = np.clip(orig_h, 0.0, MAX_HEIGHT)
    if np.sum(orig_h) < MIN_TOTAL:
        orig_h[0] = MIN_TOTAL

    N = len(orig_h)
    _set_fft_params(N)

    # ------------------------------------------------------------------
    # Stage 0 – ultra cheap Adam run (logits)
    # ------------------------------------------------------------------
    print("=== Stage 0 – ultra cheap Adam run (logits) ===")
    eps = 1e-12
    init_logits = np.log(np.clip(orig_h, eps, None))
    short_steps = 100
    best_logits, best_score = _adam_optimize_logits(
        logit_init=init_logits,
        total_steps=short_steps,
        base_lr=0.08,
        min_lr_factor=0.1,
        warmup_frac=0.05,
        noise_frac=0.20,
        noise_sigma0=0.025,
        early_stop_patience=short_steps // 2,
        clip_factor=1.5,
        print_every=max(1, short_steps // 4),
        global_start_time=start_time,
        time_limit=TOTAL_BUDGET,
        rng=rng,
    )
    print(f"   Stage 0 finished – best surrogate C≈{best_score:.6f}")

    # ------------------------------------------------------------------
    # Stage 1 – long Adam ascent (logits), constrained time budget
    # ------------------------------------------------------------------
    elapsed = time.time() - start_time
    remaining = _remaining_time(start_time)
    print("\n=== Stage 1 – long Adam ascent (logits) ===")
    stage1_budget = min(150.0, remaining * 0.15)
    est_step_time = 0.015
    max_steps = int(stage1_budget / est_step_time)
    max_steps = max(1, max_steps)
    print(f"   Time budget {stage1_budget:.1f}s → target {max_steps} Adam steps")

    best_logits, best_score = _adam_optimize_logits(
        logit_init=best_logits,
        total_steps=max_steps,
        base_lr=0.07,
        min_lr_factor=0.02,
        warmup_frac=0.05,
        noise_frac=0.05,
        noise_sigma0=0.012,
        early_stop_patience=5_000,
        clip_factor=1.2,
        print_every=max(1, max_steps // 10),
        global_start_time=start_time,
        time_limit=TOTAL_BUDGET,
        rng=rng,
    )
    print(f"   Stage 1 finished – best surrogate C ≈ {best_score:.6f}")

    # ------------------------------------------------------------------
    # Convert best logits to heights for the remaining stages
    # ------------------------------------------------------------------
    best_h = _exp_clip(best_logits)

    # ------------------------------------------------------------------
    # Stage 2 – L‑BFGS‑B polish (smooth‑max surrogate)
    # ------------------------------------------------------------------
    if _remaining_time(start_time) > 5.0:
        best_h = _lbfgs_optimize_heights(best_h, start_time, TOTAL_BUDGET)

    # ------------------------------------------------------------------
    # NEW: Upscale to a finer resolution and re‑run a quick smooth‑max L‑BFGS
    # ------------------------------------------------------------------
    if _remaining_time(start_time) > 20.0 and best_h.size < 2000:
        print("\n=== Upscaling sequence for higher‑resolution refinement ===")
        best_h = _upscale_sequence(best_h, factor=2, rng=rng)   # double the length
        _set_fft_params(best_h.size)
        print(f"   Upscaled to length {best_h.size}")

        # run another smooth‑max L‑BFGS polish on the larger problem
        if _remaining_time(start_time) > 5.0:
            best_h = _lbfgs_optimize_heights(best_h, start_time, TOTAL_BUDGET)

    # ------------------------------------------------------------------
    # Stage 3 – Adaptive exact‑gradient ascent
    # ------------------------------------------------------------------
    if _remaining_time(start_time) > 5.0:
        best_h, best_score = _adaptive_gradient_ascent(best_h, start_time, TOTAL_BUDGET, rng)

    # ------------------------------------------------------------------
    # Stage 4 – Deterministic projected‑gradient ascent (deepened with momentum)
    # ------------------------------------------------------------------
    if _remaining_time(start_time) > 2.0:
        best_h, best_score = _deterministic_pg_ascent(best_h, start_time, TOTAL_BUDGET, rng)

    # ------------------------------------------------------------------
    # Stage 5 – Exact‑bound L‑BFGS‑B polish (deepened)
    # ------------------------------------------------------------------
    if _remaining_time(start_time) > 2.0:
        best_h = _lbfgs_optimize_heights_exact(best_h, start_time, TOTAL_BUDGET, max_iter=150000)
        conv = _conv_self_fft(best_h)
        best_score, _ = _objective_and_grad_exact(conv)

    # ------------------------------------------------------------------
    # Stage 5b – Additional deterministic PG ascent (momentum)
    # ------------------------------------------------------------------
    if _remaining_time(start_time) > 2.0:
        best_h, best_score = _deterministic_pg_ascent(best_h, start_time, TOTAL_BUDGET, rng)

    # ------------------------------------------------------------------
    # Stage 6 – Random hill‑climb (final polishing)
    # ------------------------------------------------------------------
    if _remaining_time(start_time) > 1.0:
        best_h, best_score = _random_hill_climb(best_h, best_score, start_time, TOTAL_BUDGET, rng)

    # ------------------------------------------------------------------
    # Stage 7 – Final deterministic PG pass (momentum version)
    # ------------------------------------------------------------------
    if _remaining_time(start_time) > 2.0:
        best_h, best_score = _deterministic_pg_ascent(best_h, start_time, TOTAL_BUDGET, rng)

    # ------------------------------------------------------------------
    # Final evaluation (public oracle – for reporting only)
    # ------------------------------------------------------------------
    try:
        final_score = evaluate_sequence(best_h.tolist())
    except Exception as exc:
        print("   Final evaluation failed:", exc)
        final_score = -np.inf

    total_time = time.time() - start_time
    print("\nFinal lower‑bound C =", final_score,
          f"| total wall‑time {total_time:.1f}s")

    # Ensure feasibility for the oracle
    best_h = _ensure_min_total(best_h)
    best_h = np.clip(best_h, 0.0, np.inf)

    return best_h.tolist()