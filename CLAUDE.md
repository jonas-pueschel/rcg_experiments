# CLAUDE.md

Guidance for working in this repository.

## Purpose

Numerical experiments for **Riemannian optimization** methods (via
[Manopt.jl](https://github.com/JuliaManifolds/Manopt.jl)) on a Brockett-type
objective over the Stiefel manifold. The focus is Riemannian conjugate gradient
(RCG) with different conjugacy update rules, hence the repo name
`rcg_experiments`.

## The objective

For a tall–skinny `X ∈ St(n, p)` (so `Xᵀ X = Iₚ`, `n ≫ p`) and **symmetric**
matrices `A = Aᵀ ∈ ℝⁿˣⁿ`, `B = Bᵀ ∈ ℝᵖˣᵖ`:

```
f(X) = 0.5 · tr( Xᵀ A X B )
```

| quantity            | formula                                        | size |
|---------------------|------------------------------------------------|------|
| Euclidean gradient  | `∇f(X) = A X B`                                 | n×p  |
| Riemannian gradient  | `grad f(X) = ∇f(X) − X · sym(Xᵀ ∇f(X))`        | n×p  |
| differential         | `Df(X)[V] = ⟨∇f(X), V⟩`                        | scalar |
| Riemannian Hessian   | `Hess f(X)[ξ] = Pₓ( A ξ B − ξ · sym(Xᵀ A X B) )` | n×p |

`grad f` is the Euclidean-metric projection `Pₓ Z = Z − X sym(Xᵀ Z)` of `∇f`
onto `Tₓ St(n, p)`; the Hessian is the projected derivative of that field. Each
Hessian-vector product costs one fresh `A · ξ` (`sym(Xᵀ A X B)` reuses the cached
`Xᵀ A X`).

### Closed-form global minimum (`analytic_minimizer`)

For **arbitrary symmetric** `A`, `B` (neither needs to be definite):

- eigendecompose `A = Q_A diag(a) Q_Aᵀ` (`a₁ ≤ … ≤ aₙ`), `B = Q_B diag(β) Q_Bᵀ`
  (`β₁ ≤ … ≤ βₚ`);
- `m = #{k : βₖ < 0}`; pick `A`-eigenvectors for indices
  `{1,…,p−m} ∪ {n−m+1,…,n}` (the `p−m` smallest + `m` largest) as `E_A`, with
  `â = a` on those indices;
- `X⋆ = E_A · J · Q_Bᵀ` with `J` the `p×p` reversal permutation;
- `f⋆ = ½ Σₖ â_{p−k+1} βₖ = ½ Σₖ (â↓)ₖ (β↑)ₖ`.

Why: with `W = Q_Aᵀ X Q_B`, `tr(Xᵀ A X B) = Σ aᵢ βₖ Wᵢₖ²` and `P = (Wᵢₖ²)` lives
in a Birkhoff-type polytope (column sums 1, row sums ≤ 1); a linear objective is
minimised at a vertex = partial permutation, and an exchange argument gives the
pairing above. For `B ⪰ 0` (`m = 0`) this is the familiar "`X` spans the `p`
smallest eigenvectors of `A`"; indefinite `B` sends its `m` most-negative modes
to `A`'s *largest* eigenvalues instead. `reference_optimum` returns this `X⋆`
and cross-checks it with a short LBFGS polish (warns if LBFGS beats it). The
polish's `quasi_Newton` line search is `greedy_secant.jl`'s raw
`GreedySecantLinesearch` (`initial_stepsize = 1.0`), not Manopt's default
`WolfePowellLinesearch` — the default stalls for a very long time on some
seeds (e.g. `76`) polishing an already near-optimal `Xa`; a short polish this
close to the optimum doesn't need Wolfe-condition-guaranteed LBFGS
convergence, so the safeguard-free secant step is fine here.

**Hessian spectrum at `X⋆`** — `riemannian_hessian_eigenvalues(A, B)` gives the
closed-form spectrum of the Riemannian Hessian at the minimiser (verified against
the dense Hessian to machine precision). In the `A`/`B` eigenbases it is
diagonal, with `ǎ = reverse(a[idx])` (descending):

* `βₖ (aᵢ − ǎₖ)` for `k = 1..p`, `i ∉ idx` — `p(n−p)` "off-subspace" modes
* `½ (ǎ_l − ǎₖ)(βₖ − β_l)` for `1 ≤ l < k ≤ p` — `p(p−1)/2` "in-subspace" modes

All `≥ 0` at the minimiser; a `0` (⇒ singular Hessian, `cond = ∞`) only when `B`
is singular or `A`/`B` has repeated eigenvalues. `analytic_minimizer(A, B)`
(hence every `reference_optimum` call) prints `λmin`, `λmax` and `cond` to
stderr; pass `verbose = false` to silence it.
There is **no weighted metric** in this repo (unlike the reference
[`MetricFreeTest/weighted_objective.jl`](https://github.com/jonas-pueschel/MetricFreeTest/blob/main/weighted_objective.jl)
that inspired the storage design).

The expensive kernel is the dense product `A * X` (`n×n · n×p`). Everything else
(`Xᵀ(AX)`, `(AX)B`, the projection) is `O(n p²)` or smaller.

## Files

| file           | contents |
|----------------|----------|
| `objective.jl` | `ObjectiveStorage`, `get_objective`, `get_hessian`, `hessian_solve` (CG / MINRES), `analytic_minimizer`, `riemannian_hessian_eigenvalues`, math of `f` |
| `main.jl`      | exploratory driver: `run_newton`, solver zoo, results table, PNG plots; also `FloorStepsize`, `RecordApplications` / `RecordIterateDistance` / `RecordStepsizeValue`, `build_problem` (objective switch), `iterate_distance` / `grassmann_distance`; `include`s `greedy_secant.jl` (used by `reference_optimum`'s LBFGS polish) |
| `experiments.jl` | the paper experiments — each writes pgfplots `.dat` files into `data/` |
| `spectral_stepsizes.jl` | `SpectralLength` — raw BB / norm-ratio step sizes, no line search or safeguard |
| `greedy_secant.jl` | `GreedySecantLinesearch` — raw greedy secant step (one secant step on `ϕ'` per iteration), no bracketing/backtracking/safeguard; `include`d from `main.jl` |
| `data/`        | generated `.dat` / preview PNGs (git-ignored PNGs only) |
| `old/`         | retired/superseded scratch files, kept for reference — git-ignored, not loaded by anything |
| `Project.toml` / `Manifest.toml` | pinned local environment |

`old/` holds files no longer `include`d anywhere: `secant_linesearch.jl` (the
*heavier* derivative-only line search — bracket-then-secant-loop with a
strong-Wolfe accept; superseded in experiment 2 by `greedy_secant.jl`'s
lighter one-secant-step-per-iteration version, and by Hager–Zhang for the
"stays accurate on the plateau" role), `cubic_bracketing_atol.jl`
(noise-tolerant cubic bracketing — the tolerance alone didn't fix the plateau,
see its header), `beale_restart.jl` (`BealeRestartCondition` — tried against
the HS/DY stagnation in experiment 1, didn't help), `scratch_test.jl` (ad hoc
dev harness). None of these are referenced from `main.jl` / `experiments.jl` /
`spectral_stepsizes.jl` / `greedy_secant.jl`; move a file back to the repo
root and add its `include` to `experiments.jl` to resurrect it.

### `experiments.jl`

`include("main.jl")` (which itself pulls in `greedy_secant.jl`) and
`include("spectral_stepsizes.jl")` for all the machinery, then one function per
experiment, each writing whitespace-separated pgfplots `.dat` files (header row
`<x> <y>`, column 1 = the integer x-series `iter`/`apps`, column 2 =
`err = ‖Xₖ − X⋆‖` or `step = α_k`) plus a `*_preview.png`. Run the file to
execute experiments 1–4, or `include` it and call one by name. Every solver in
an experiment starts from the same
`perturbed_start`, and each series is prepended with the shared iteration-0
point.

**Objective switch** — `const OBJECTIVE_KIND` at the top of the file, `:brockett`
(default) or `:eigenvalue`, is the *one* place to change which objective every
experiment below runs:

- `:brockett`: `f(X) = 0.5 tr(Xᵀ A X B)`, `A, B` random symmetric — the general
  objective this repo is built around (all figures/numbers in this doc use it
  unless noted). `X⋆` is unique up to a per-column sign flip;
  `iterate_distance` (sign-aligned Frobenius norm, `main.jl`) measures it.
- `:eigenvalue`: `f(X) = 0.5 tr(Xᵀ A X)` (`B = I(p)`, via
  `build_problem(...; objective = :eigenvalue)`) — the plain block eigenvalue /
  trace-minimisation problem. `X⋆` spans the `p` smallest eigenvectors of `A`
  but is only unique as a **subspace**: the objective is invariant under
  `X ↦ X U` for *any* orthogonal `U` (`B`'s spectrum is fully repeated, β = 1
  with multiplicity `p`), not just a sign flip, so `iterate_distance` is the
  wrong metric. Use `grassmann_distance(X, X_star)` (`main.jl`) instead — the
  Procrustes/chordal distance `min_{U∈O(p)} ‖XU − X⋆‖ = √(2p − 2Σσᵢ(Xᵀ X⋆))`
  (`σ` = singular values, achieved at the orthogonal polar factor of `Xᵀ X⋆`).
  A second consequence: the Riemannian Hessian at `X⋆` is singular along the
  `p(p-1)/2` in-subspace "gauge" directions (β all equal ⇒ the "in-subspace
  modes" term in `riemannian_hessian_eigenvalues` is exactly zero there) — this
  is already documented/expected (`analytic_minimizer` prints `cond = Inf`);
  `hessian_spectrum_bounds(λ)` (`experiments.jl`) computes `L, μ, κ` excluding
  those exactly-zero eigenvalues so the rate lines / `2/(L+μ)` constant step
  stay well-defined (falls back to plain `λ[end]/λ[1]` — bit-identical to the
  old code — whenever nothing is exactly zero, i.e. always for `:brockett`).

Every experiment routes through two small dispatch points so `OBJECTIVE_KIND`
only needs to be set once: `problem_distance(X, X_star)` (picks
`grassmann_distance` vs. `iterate_distance`) feeds `d0`,
`RecordIterateDistance(X_star; dist = problem_distance)`,
`StopWhenIterateErrorLess(X_star, err_tol; dist = problem_distance)`, and
`run_newton(...; dist = problem_distance)`; `ERR_YLABEL` swaps the preview
y-axis label accordingly. `:brockett` keeps writing to the flat `data/`
directory used throughout this file's history; `:eigenvalue` writes to
`data/eigenvalue/` so it never overwrites the `:brockett` baseline.

**Result** (`experiment_retractions`, `perturb = 1.0`, only `rcg`/`rgd-constant`
active): under `:eigenvalue` the polar/QR curves are *even closer* than under
`:brockett` — `rcg` needs exactly `89` A·X under **both** retractions (`44`
it), `rgd-constant` `1001` A·X under both (`‖X−X⋆‖ ≈ 2.63e-6` vs `2.629e-6`).
Two compounding reasons: (1) `κ ≈ 212` here (vs. `2246` for `:brockett` on the
same `A`) — far better conditioned, so convergence is fast and stays deep in
the asymptotic regime where retraction curvature is a vanishing higher-order
effect (see the retraction-comparison discussion earlier in this file's
history); (2) `grassmann_distance` measures the **subspace**, so any
retraction-dependent difference in exactly *which point* of the equivalence
class each run lands on is invisible to it by construction — a coarser metric
that structurally suppresses retraction sensitivity, on top of the dynamics
already suppressing it. (`rgd-bb` and `newton` are currently commented out of
`methods` in `experiment_retractions`.)

**Why Newton specifically is *not* retraction-sensitive under `:eigenvalue`
(unlike `:brockett`)** — this looks like it should be the one place a real
polar-vs-QR gap survives (Newton has no line-search-driven re-adaptation to
paper over a retraction's higher-order defects, and `:brockett` Newton
genuinely does show a gap — see below). It doesn't, and not just empirically:
it's forced by the objective's symmetry. `f`, `grad f`, `Hess f` are all
*exactly* equivariant under the **full** `O(p)` action `X ↦ XU` for
`:eigenvalue` (`grad f(XU) = grad f(X)U`, `Hess f(XU)[ξU] = Hess f(X)[ξ]U` —
direct from `f(X) = ½tr(XᵀAX)`), and the Frobenius inner product (hence Armijo
slope/cost) is `O(p)`-invariant too. Separately, *any* two retractions of the
same tangent candidate land on the same subspace — `polar_factor(Y)` and
`qr_factor(Y)` are both just orthonormal bases of `range(Y)`, related by some
orthogonal `V`, for *any* `Y`, independent of the objective. Chaining these: if
the polar- and QR-driven Newton iterates at step `k` are `X_k` and `X_k·V` for
some orthogonal `V` (true at `k=0` — `perturbed_start` retracts the *same*
tangent from the *same* `X⋆` two ways), then the Newton direction, cost, and
Armijo slope at `X_k·V` are *exactly* `V`-transports of those at `X_k` — so
**both branches accept the identical sequence of trial step sizes**, and the
next candidate matrices differ by exactly `V` again, which doesn't change its
column space. The whole trajectory stays co-subspace by induction, with the
*only* leak being the Newton system's finite Krylov solve tolerance (equivariance
holds for the exact linear system, not bit-for-bit for an inexact solve).
Verified directly: tracking `grassmann_distance` **between** the polar- and
QR-driven iterates of the same run (not each to `X⋆`) at `inner_rtol = 1e-4`
gives a small but clearly nonzero gap starting at iteration 1 (`3.3e-4`,
`4.6e-4`, `4.2e-4`, `2.0e-3`, …) that later explodes (`0.15` by iteration 5,
`O(1)` by iteration 7) as Newton's known pre-asymptotic chaos ("attracted to
the nearest stationary point of *any* type", above) amplifies it; tightening to
`inner_rtol = 1e-10` collapses the gap to **exactly `0`** for iterations 1–4
and it first appears at roundoff scale (`5.3e-7`) at iteration 5, before the
same chaotic blow-up — i.e. the divergence is 100% attributable to Krylov
solve precision, not to retraction choice, exactly as the equivariance
argument requires. (Both trajectories still land on `X⋆` — `44` vs `89`
iterations respectively for the two branches shown — `grassmann_distance = 0`
to machine precision at the end either way; retraction choice affects how fast
Newton's chaotic pre-asymptotic phase resolves, never whether it resolves to
the right point.)

**`:brockett` has no such protection, and the gap there is structural, not a
solver-tolerance artifact.** Generic `B` is only invariant under the *discrete*
sign-flip subgroup of `O(p)`, not the continuous group — but the `V` relating
`polar_factor(Y)` and `qr_factor(Y)` of a generic candidate `Y` is a generic
orthogonal matrix, essentially never a signed permutation. So
`f(polar_factor(Y)) ≠ f(qr_factor(Y))` in general: the two branches disagree on
cost from the very first candidate, Armijo accepts different step sizes, and
the induction above never gets started. Confirmed with the same
`inner_rtol = 1e-10` (ruling out solver precision as the cause): tracking the
gap between the two branches' iterates shows a real, `O(1e-2)`-scale departure
already by iteration 2 (vs. exactly `0` through iteration 4 for `:eigenvalue`
at the same tolerance), and at `perturb = 0.2` the two branches don't even
converge to the same point — `newton_polar` reaches `X⋆` to machine precision
in `8` iterations / `3260` A·X, while `newton_qr` gets chaotically diverted to
a **different** critical point (`iterate_distance` to `X⋆` stalls at `2.0`) in
`12` iterations / `4250` A·X. So under `:brockett`, polar's exact
retraction-quality and equivariance advantages are real and can cost QR
correctness, not just speed; under `:eigenvalue` they're provably moot up to
floating point.

Shared helpers factor out what all four experiments need: `series_from_record`
(pulls `(iters, work, errs)` out of a Manopt record or `run_newton`'s record —
both are tuples with iteration/A·X-count/`‖Xₖ−X⋆‖` in slots 1/4/5),
`costs_from_record` / `gradnorms_from_record` (`f(Xₖ)` and `‖grad f(Xₖ)‖`,
slots 2/3 of the same tuples — already being recorded via `:Cost` /
`:GradientNorm` everywhere, just unused before), `prepend_start!` /
`prepend_cost!` / `prepend_gradnorm!` (fill in the shared iteration-0 point,
for the error series and the cost/gradnorm series respectively — the latter
two run *after* `prepend_start!` and just check the length mismatch to know
whether a point is needed), `write_series_dat` (writes the `_iter.dat`/
`_apps.dat` pair, `y = err`) / `write_fgap_dat` (`y = fgap = f(Xₖ) − f*`) /
`write_gradnorm_dat` (`y = gradnorm = ‖grad f(Xₖ)‖`), `rate_series` (the
`√κ`/`κ` linear-rate curves, computed on the fly for the plots — **not**
exported to `.dat` any more; `κ` in the LaTeX side is enough to redraw them),
and `error_preview` / `fgap_preview` / `gradnorm_preview` (log-y panels —
`error_preview` returns two panels, err vs iteration *with* dashed `√κ`/`κ`
rate lines and err vs A·X *without* them, used by experiments 2–3;
`fgap_preview` / `gradnorm_preview` each return one panel, `f(Xₖ)−f*` /
`‖grad f(Xₖ)‖` vs iteration, no rate lines since `√κ`/`κ` are calibrated to
`‖Xₖ−X⋆‖`, not these — still used only by experiment 3 now; experiments 1 and
4 build their own pared-down err-vs-iteration two-panel previews — experiment
1's two panels each carry the `√κ`/`κ` rate lines, experiment 4's left
(RCG/RGD) panel does and its right (Newton) panel does not). Only experiment 3
still records/exports cost and gradient-norm series (and its `prepend_cost!` /
`prepend_gradnorm!` needs `f(X0)` / `‖grad f(X0)‖`, evaluated once via a
throwaway `ObjectiveStorage` so they never inflate a solver's reported `A·X`
count); experiments 1, 2, 4 now export only `‖Xₖ − X⋆‖` series.

**Rate lines** are `d0·((√κ−1)/(√κ+1))ᵏ` (optimal / linear-CG) and
`d0·((κ−1)/(κ+1))ᵏ` (steepest descent), dashed / dash-dot, on every
err-vs-**iteration** panel *except* experiment 4's Newton panel, anchored at
`d0 = ‖X0 − X⋆‖` with `κ = cond(Hess f @ X⋆)` — drawn straight from `κ` at
plot time, no `.dat`. The error-panel y-axis is clipped to `[err_tol, ymax]`
(`ymax` a bit above the largest error actually reached) so the plot isn't
dominated by empty space below where every run stops.

- **`experiment_cg_params`** `(; n, p, seed, perturb, max_iters,
  err_tol = 1e-6, wall_secs, min_stepsize = 1e-2, init_stepsize = nothing,
  const_stepsize = nothing, outdir = data/, preview)` — RCG on the polar
  retraction, a **full** comparison of **FR, PRP, HS, DY, HS–DY (Hybrid1),
  FR–PRP (Hybrid2)** to justify FR–PRP against the whole field. Convergence
  criterion is `StopWhenIterateErrorLess(X⋆, err_tol)` (‖Xₖ − X⋆‖, not the
  gradient). HS/DY/HS–DY are *expected* not to converge here (known RCG
  behaviour with a non-isometric transport — the fragile `⟨𝒯d, y⟩`
  denominators; `c₂`-tightening, the Beale restart (`old/`) and DY⁺ were all
  tried and don't help). The whole 6-coefficient comparison is run **twice**,
  once per step size — **`hz`** (`rcg_hagerzhang_stepsize`, approximate Wolfe,
  `c₂ = 0.2`, pinned initial guess `α₀ = αc = 2/(L+μ)`, floored at
  `α ≥ min_stepsize`) and **`armijo`** (Armijo backtracking `c₁ = 0.2` /
  contraction 0.5, pinned `α₀ = αc`, floored at `min_stepsize`; `αc` is
  `init_stepsize`'s default, resolved the same `something(init_stepsize, αc)`
  way as experiment 2). **HZ replaced the raw greedy secant step here**: secant
  has no safeguard, so near a near-zero conjugacy denominator (the DY
  `⟨𝒯d, y⟩` blow-up) it takes wildly oversized steps that visibly wreck the DY
  curve; HZ's approximate-Wolfe accept keeps every coefficient rule on a fair
  footing. Writes **only** the two panels' data — `cgparam_<pn>_hz_iter.dat`
  and `cgparam_<pn>_armijo_iter.dat` (`y = err = ‖Xₖ − X⋆‖`) for
  `pn ∈ {fr, prp, hs, dy, hsdy, frprp}` — and a two-panel `cgparam_preview.png`
  (left: Hager–Zhang step, right: armijo step; both `‖Xₖ − X⋆‖` vs iteration
  with dashed `√κ`/`κ` rate lines drawn from `κ`, one line per coefficient
  rule, shared log-y scale). Returns a `Dict("hz" => series, "armijo" => series)`.
- **`experiment_cg_stepsize`** `(; …, min_stepsize = 1e-2, init_stepsize = nothing,
  const_stepsize = nothing)` — RCG with the **fixed FR–PRP** coefficient and the
  polar retraction, comparing **five** step-size rules by their own names:
  **`constant`** (`ConstantLength`, `α ≡ 2/(L+μ)` — the GD-optimal fixed step
  for a quadratic with Hessian spectrum `[μ, L]`; `const_stepsize` overrides).
  A **baseline that does not converge** here: `κ ≈ 2246` ⇒ linear rate
  `≈ 1 − 2/(κ+1)`. **`armijo`** (`ArmijoLinesearch`, backtracking, `c₁ = 0.2`,
  contraction 0.5). **`hagerzhang`** (`rcg_hagerzhang_stepsize` — approximate
  Wolfe + secant², curvature target `c₂ = 0.2` — the two-sided
  approximate-Wolfe test stays accurate on the machine-precision plateau,
  unlike cubic bracketing, tried earlier and dropped to `old/`; the `c₂ =
  0.05` / `0.4` variants performed almost identically and were dropped).
  **`wolfe-powell`** (Manopt's `WolfePowellLinesearch` — strong-Wolfe, bracket
  + bisection, `(c₁,c₂) = (0.1, 0.2)`; **temporary** — this is the search whose
  final bisection stalls badly on some seeds, see `reference_optimum`, kept
  only to see how it fares as an RCG step size). **`secant`**
  (`greedy_secant.jl`'s `GreedySecantLinesearch` — *no* line search,
  backtracking, or safeguard: every iteration takes exactly one secant step on
  `ϕ'` between `0` and the previous accepted step
  (`α = −ϕ'(0)·b / (ϕ'(b) − ϕ'(0))`), warm-started on the first call at
  `b = 2/(L+μ)` — the same value as `constant`'s α; one `A·X`/iteration). A
  variant that accepts `b` outright whenever it already satisfies a
  strong-curvature test (skipping the secant step) was tried at `c₂ = 0.2` and
  `0.1` and measured **worse both times** (more total iterations/A·X, not
  fewer — reusing a stale `b` under-refines the step for FR-PRP's
  fast-changing conjugate direction), so `greedy_secant.jl` always takes the
  secant step. `armijo`, `hagerzhang` and `wolfe-powell` are wrapped in
  `FloorStepsize` (`main.jl`) which clamps the accepted step to
  `α ≥ min_stepsize` and pins the **initial** trial step to
  `α₀ = init_stepsize` (default `αc = 2/(L+μ)`, resolved
  `something(init_stepsize, αc)`); `constant` and `secant` are wrapped only for
  the `α ≥ min_stepsize` floor — `secant` is *not* pinned to `init_stepsize`
  every call (`pin_init = false`), since its formula needs the genuine
  previous step. `secant` has been the cheapest strategy in past runs; it's
  the shared RCG stepsize used everywhere else in this file
  (`rcg_secant_stepsize`, experiments 1 and 4). Same convergence criterion as
  experiment 1. Writes **only the two panels' data** —
  `cgstepsize_<name>_iter.dat` and `_apps.dat` (`y = err`) for
  `name ∈ {constant, armijo, hagerzhang, wolfe-powell, secant}` — and a
  two-panel `cgstepsize_preview.png` (err vs iteration with dashed `√κ`/`κ`
  rate lines, err vs A·X without them).
- **`experiment_gd_stepsize`** `(; …, max_iters = 1000, min_stepsize = 1e-2,
  init_stepsize = 5e-2, spectral_max_stepsize = 1e2, const_stepsize = nothing)`
  — **temporarily disabled** (its call is commented out in the run-all block;
  the function is intact). **Riemannian gradient descent** (`gradient_descent`,
  polar retraction),
  comparing **four** step-size rules, all under the same `max_iters = 1000`
  cap: **`constant`** (same `2/(L+μ)` baseline, does not converge),
  **`bb-alternating`** (raw Barzilai–Borwein, `spectral_stepsizes.jl`'s
  `SpectralLength` — *no* line search / backtracking / safeguard, just
  `α = ⟨s,s⟩/⟨s,y⟩` on odd iterations and `α = ⟨s,y⟩/⟨y,y⟩` on even, where
  `s_k = X_k − X_{k-1}`, `y_k = grad f(X_k) − grad f(X_{k-1})` are ambient
  Frobenius differences), **`norm-ratio`** (raw spectral step
  `α = ‖s‖/‖y‖ = √(⟨s,s⟩/⟨y,y⟩)`, the geometric mean of the long/short BB
  steps), **`hagerzhang`** (same line search as experiment 2, here driving
  plain gradient descent). `hagerzhang` gets the same `FloorStepsize` /
  `init_stepsize` treatment as experiment 2; the raw spectral steps and
  `constant` are wrapped only for uniform step recording (their own internal
  clamps already keep them away from `min_stepsize` in practice). See below for
  why the *raw* (unsafeguarded) spectral steps were chosen over Manopt's
  `NonmonotoneLinesearch`. Writes `gdstepsize_<name>_{iter,apps,step}.dat`,
  `gdstepsize_<name>_fgap_{iter,apps}.dat` (`y = fgap = f(Xₖ)−f*`), and
  `gdstepsize_<name>_gradnorm_{iter,apps}.dat` (`y = gradnorm = ‖grad f(Xₖ)‖`)
  for `name ∈ {constant, bb-alternating, norm-ratio, hagerzhang}`, and a
  five-panel `gdstepsize_preview.png` (err vs iteration with `√κ`/`κ` rate
  lines, err vs A·X, `f(Xₖ)−f*` vs iteration, `‖grad f(Xₖ)‖` vs iteration,
  `α_k` vs iteration). (Since it's disabled, its `write_*` set hasn't been
  trimmed like experiments 1/2/4 — that'll happen when it's re-enabled.)
- **`experiment_retractions`** `(; …, min_stepsize = 1e-2, newton_max_iters = 30,
  newton_grad_tol = 1e-10, inner_rtol = 1e-4, inner_maxiter = 600,
  const_stepsize = nothing, spectral_max_stepsize = 1e2)` — compares the
  **polar** vs. **QR** retraction (transport = `DifferentiatedRetractionVectorTransport`
  of that same retraction in both cases) across four solver families:
  **`newton`** (globalised Riemannian Newton, MINRES inner solve —
  `run_newton`, `inner_method = :minres`; needs
  `inner_maxiter ≳ dim Tₓ St(n,p) = np − p(p+1)/2` = 485 here, hence the raised
  default), **`rcg`** (FR-PRP + `rcg_secant_stepsize`, the raw greedy secant
  step — experiment 1 switched *its* RCG runs to Hager–Zhang, but this one
  still uses secant), **`rgd-constant`** (RGD, `α ≡ 2/(L+μ)`), **`rgd-bb`** (RGD,
  raw Barzilai–Borwein alternating step, same as experiment 3's
  `bb-alternating`). Each (solver, retraction) pair gets its **own**
  `reference_optimum` / `perturbed_start` call for that retraction (same seed
  ⇒ the same underlying random tangent direction, retracted differently — not
  bit-identical starting points across polar/QR, by construction of
  `perturbed_start`) and its own convergence check; Newton has no
  iterate-error stop (its own `newton_max_iters` / `newton_grad_tol` loop) but
  converges quadratically once inside the basin. Writes **only the two panels'
  data** — `retractions_<name>_<retraction>_iter.dat` (`y = err`) for
  `name ∈ {newton, rcg, rgd-constant, rgd-bb}` × `retraction ∈ {polar, qr}` —
  and a two-panel `retractions_preview.png`, both `‖Xₖ − X⋆‖` vs iteration
  (solid = polar, dashed = QR, one colour per solver): **left** panel has
  `rcg` + `rgd-constant` + `rgd-bb` *with* the dashed `√κ`/`κ` rate lines,
  **right** panel has `newton` alone *without* rate lines (Newton converges —
  or, on a bad `perturb`, stalls at a saddle — in ~15–30 iterations, so
  sharing the ~1000-it first-order x-axis, or overlaying an asymptotic linear
  rate, would be meaningless).

**Why raw spectral steps, not Manopt's `NonmonotoneLinesearch`, for
experiment 3 / `rgd-bb`**: Manopt's own safeguarded Barzilai–Borwein
(`NonmonotoneLinesearch`, nonmonotone Armijo backtracking on top of the BB
formula) was tried first and needs `≈ 3900` it / `≈ 4.5` A·X/it to converge on
`strategy = :direct`, and never converges on `:inverse` (stalls `≈ 5e-3`) —
plausibly because it builds `s`/`y` from a *transported* tangent step
interacting with its own backtracking history, vs. the raw steps' ambient
differences here. The raw (unsafeguarded) versions are both far cheaper
(**1 A·X/iteration**, no backtracking) and converge markedly faster. Also:
confirmed upstream bug — in the registered **Manopt v0.6.6** (pinned here),
`NonmonotoneLinesearchStepsize`'s `:alternating` strategy's odd-iteration
branch uses the standard long step `⟨s,s⟩/⟨s,y⟩`, but its non-alternating
`:direct` strategy computes `⟨y,y⟩/⟨s,y⟩` instead (a different, not-standard
formula; `stepsizes.jl:1663-1673` vs. `:1678-1685`). Fixed on Manopt's
unreleased `master` (retargeted as `BarzilaiBorweinStepsize`, `Project.toml`
version 0.6.7 — not yet tagged/registered; `:direct` there matches the
standard `s3/s1` formula). No action needed in this repo: nothing here calls
`NonmonotoneLinesearch`, and `spectral_stepsizes.jl`'s `:bb_direct` already
uses the correct (master) formula.

Two shared *RCG* stepsize builders live in `experiments.jl`:

- `rcg_hagerzhang_stepsize(M, rm, vtm; min_stepsize = 1e-2, init_stepsize = 5e-2,
  sufficient_curvature = 0.2)`: `HagerZhangLinesearch(; wolfe_condition_mode =
  :approximate, δ = min(0.1, sufficient_curvature/2), σ = sufficient_curvature,
  stepsize_limit = 1e3, initial_guess =
  Manopt.ConstantInitialGuess(init_stepsize))`, wrapped in
  `FloorStepsize(…, min_stepsize; init = init_stepsize)`. Used for experiment
  1's `hz` step and experiment 2's `hagerzhang` (the `sufficient_curvature`
  kwarg — with the `δ ≤ σ` scaling — is left over from the dropped
  `hagerzhang-tight` `σ = 0.05` / `hagerzhang-loose` `σ = 0.4` variants; still
  handy for probing). **Never pair it with a `:Stepsize` record entry** —
  `RecordStepsize`'s `k = 0` `get_last_stepsize` probe re-invokes the line
  search while its `last_stepsize` is still `NaN`, feeding a `NaN` step into
  the polar retraction and throwing `invalid argument #4 to LAPACK call`
  (experiments 1 and 2 both keep it out of their record lists).
- `rcg_secant_stepsize(M, rm, vtm; min_stepsize = 1e-2, initial_stepsize = 5e-2)`:
  `GreedySecantLinesearch(; initial_stepsize)` (`greedy_secant.jl`), wrapped in
  `FloorStepsize(…, min_stepsize)` — **not** pinned (`init` left `NaN`), since
  the secant step needs its own previous accepted step. Used by experiment 4's
  `rcg` (experiment 1 used to share it too, but switched its RCG runs to
  Hager–Zhang — see above — because the unsafeguarded secant step wrecks the
  DY curve near a near-zero conjugacy denominator). Experiment 3's RGD
  `hagerzhang` intentionally builds its own `HagerZhangLinesearch` (not RCG),
  and experiment 2 keeps its own `hagerzhang` / `wolfe-powell` / `secant`
  variants for the comparison itself.

`_iter.dat` has `x = iteration`, `_apps.dat` has `x = cumulative A·X products`;
both have `y = err`. Plot with `\addplot table[x=iter, y=err] {…_iter.dat};`.
Every `WolfePowellLinesearch`/`ArmijoLinesearch` is built with
`stop_when_stepsize_less = 1e-12` (Manopt's `0.0`
default can spin forever — see below).

### `ObjectiveStorage`

A mutable ring buffer that keeps the intermediate results of the last `n_size`
**distinct** iterates (`X`, `A*X`, `Xᵀ A X`, Euclidean grad, Riemannian grad,
cost). Manopt solvers query cost / gradient / differential repeatedly at the
same point (line-search trial points, stopping criterion, conjugacy update); the
cache serves those from a single `A * X`.

- newest iterate at index `1`; `_update!` shifts the buffer down on a miss.
- lookup (`_find`) matches by `===` or `==` on the iterate matrix.
- counters: `n_applications` (real `A*X` products) and `n_hits` (cache hits).
  `naive = n_applications + n_hits` is the count without a cache.
- `reset!(os)` clears history + counters for reuse across runs.

`get_objective(A, B, os; use_differential = true)` returns a
`ManifoldFirstOrderObjective`. With `use_differential = true` the differential is
exposed so Wolfe-type line searches reuse the cached `A * X` instead of forcing a
new one.

Extra counter `n_hvp` tracks Hessian-vector products; each also bumps
`n_applications` (same `A · (n×p)` kernel). In the results table, `A*X` is the
total, `hvp` the Newton-only subset.

### Hessian + Newton

- `get_hessian(A, B, os)` → `hess(M, X, ξ)`, the Riemannian Hessian above.
- `hessian_solve(applyH, b; method, rtol = 1e-4, maxiter)` — matrix-free Krylov
  solve of the Newton system `Hess f(X)[ξ] = −grad f(X)` on the tangent space,
  Frobenius inner product, **relative-residual** stop `‖r‖ ≤ rtol·‖b‖`:
  - `:cg` — Newton–CG with a non-positive-curvature guard (returns `b` on the
    first step, the current iterate later).
  - `:minres` — Paige–Saunders MINRES for indefinite / singular systems.
- `run_newton(M, obj, hess, X0; …)` (in `main.jl`) — globalised Riemannian
  Newton: inner Krylov solve → descent-direction check (fall back to `−grad` if
  the direction is uphill or the inner solve stalled) → Armijo backtracking on
  the retraction. Registered as `Newton-CG` and `Newton-MINRES` in
  `solver_list()`.

Newton-CG shows the expected quadratic tail and reaches the global minimum: its
non-positive-curvature guard turns the step toward `−grad` whenever the Hessian
is indefinite, so it cannot be pulled into a saddle.

`Newton-MINRES` is included because it was asked for, but it is instructive
rather than competitive: MINRES solves the raw (indefinite) Newton system
accurately, and the resulting step is attracted to the nearest **stationary
point of any type** — on a random start it typically converges (`‖grad‖ → 0`) to
a saddle with `f ≫ f*`. A trust region / cubic regularisation (`trust_regions`,
`adaptive_regularization_with_cubics` in Manopt) is the proper globalisation and
is out of scope here. Needs `inner_maxiter ≳ dim Tₓ St(n, p) = np − p(p+1)/2`.

## Environment

Local project environment; standard deps: `Manopt`, `Manifolds`, `Plots`
(+ stdlib `LinearAlgebra`, `Random`, `Printf`).

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'   # first time
julia --project=. main.jl                             # run experiments
```

`main.jl` prints a results table and writes four PNGs next to the script:
`convergence.png` (`f − f*`) and `gradient_norm.png` (`‖grad‖`) against
**cumulative `A * X` applications** — not iteration count — so a Newton step
(~100 Hessian-vector products) is comparable to ~100 CG steps;
`iterate_error.png` (`‖Xₖ − X⋆‖`, sign-aligned) against **iteration**; and
`matrix_products.png` (cache vs. naive bar chart). The per-iteration work count
and iterate error come from the `RecordApplications` / `RecordIterateDistance`
Manopt record actions (and, for `run_newton`, the 4th and 5th record-tuple
slots); record tuples are `(iteration, cost, ‖grad‖, A*X count, ‖Xₖ − X⋆‖)`.

`run_experiments(; n, p, n_size, seed, max_iters, grad_tol, wall_secs = 30,
newton_max_iters, inner_rtol = 1e-4, inner_maxiter = 200, perturb = 1e-1,
retraction = :polar, use_differential, verbose)` is the entry point; call it
directly from the REPL to tweak sizes. It returns `(results, f_star)`.

### Retraction / vector transport

`retraction = :polar` (polar = metric-projection retraction, the Stiefel
default) or `:qr` (QR retraction). `retraction_pair(kind)` maps it to
`(retraction_method, DifferentiatedRetractionVectorTransport(retraction))` — the
matching *differentiated-retraction* transport — and `run_experiments` threads
that pair into every solver (`retraction_method` + `vector_transport_method`),
the shared `WolfePowellLinesearch`, the CG coefficient rules that transport
(`PolakRibiere`, `HestenesStiefel`, `DaiYuan`, and the hybrids built from them —
`FletcherReeves` / `ConjugateDescent` use no transport), `run_newton`'s
`retract`, `perturbed_start`, and the LBFGS polish in `reference_optimum`.

**All solvers start from the same point** `perturbed_start(M, X_star; radius =
perturb)` — the closed-form optimum nudged a geodesic distance `perturb` (≈1e-1)
away — so the runs probe the local basin rather than global search. Raise
`perturb` toward `O(1)` for a harder, more global test.

### Line-search / stopping guards

`solver_list`'s shared `WolfePowellLinesearch` is created with
`stop_when_stepsize_less = 1e-12` (Manopt's default is `0.0`, which lets its
final bisection spin forever when the curvature test can't be met — e.g. a bad
Polak–Ribière direction) plus bounded `stop_increasing/decreasing_at_step`. The
first-order `sc` also carries `StopWhenChangeLess(M, 1e-13)` and
`StopAfter(Second(wall_secs))` as belt-and-suspenders against a stalled run.

## Progress logging / "it looks stuck"

The first call to any Manopt solver spends ~10 s JIT-compiling the
`quasi_Newton` + Wolfe–Powell + debug stack — that is compilation, not a hang.
`_log(...)` writes progress markers to **stderr** (never block-buffered), so
`[ref k/starts] ...`, `[i/N] <method> running ...` and the per-Newton-iteration
lines always appear even when stdout is redirected. With `verbose = true`
(default) the reference solve also streams Manopt's own `debug=` iteration lines.
If a run genuinely stalls, the last `_log` line tells you which solver.

## Conventions

- Everything runs against the local `--project=.` env; do not add global deps.
- Keep the heavy linear algebra inside `ObjectiveStorage`; solver code in
  `main.jl` should only touch Manopt.
- New experiments: add a `(name, run)` entry to `solver_list()` in `main.jl`;
  every `run(M, obj, hess, X0, os, ctx)` must return `(X_final, record)` with
  `record` a vector of `(iteration, cost, ‖grad‖, A*X count, ‖Xₖ − X⋆‖)` tuples
  (helper `fo(...)` wraps a Manopt first-order solver into that shape; `ctx`
  carries `sc`, `X_star`, `grad_tol` and the Newton knobs).
- The reference optimum comes from the closed form (`analytic_minimizer`),
  LBFGS-verified; `reference_optimum` returns `(f_star, X_star)`.
