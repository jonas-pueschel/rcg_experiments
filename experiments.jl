# =============================================================================
#  experiments.jl
#
#  Paper experiments.  Each one runs a family of solvers on the Stiefel Brockett
#  objective and exports pgfplots-ready `.dat` files into `data/` for LaTeX
#  plotting: whitespace-separated, header row `<x> <y>`, column 1 = the integer
#  x-series (`iter` or `apps`), column 2 = the y-series (`err = ‖Xₖ − X⋆‖`,
#  `fgap = f(Xₖ) − f*`, `gradnorm = ‖grad f(Xₖ)‖`, or `step = α_k` for the
#  step-size panels).  Plus a `*_preview.png`.
#
#    1. experiment_cg_params    — RCG conjugacy-coefficient comparison
#    2. experiment_cg_stepsize  — RCG step-size / line-search comparison (FR-PRP)
#    3. experiment_gd_stepsize  — RGD step-size comparison (constant / BB / HZ)
#    4. experiment_retractions  — polar vs. QR retraction, across solver families
#
#  Run:  julia --project=. experiments.jl     # runs experiments 1–4
#   or:  julia --project=. -e 'include("experiments.jl"); experiment_retractions()'
# =============================================================================

include("main.jl")                # also pulls in greedy_secant.jl (GreedySecantLinesearch)
include("spectral_stepsizes.jl")  # SpectralLength (raw BB / norm-ratio step, no safeguard)

# -----------------------------------------------------------------------------
#  Objective switch — fix once here; every experiment below picks it up via
#  `build_problem(...; objective = OBJECTIVE_KIND)` and `problem_distance`.
# -----------------------------------------------------------------------------
#   :brockett   f(X) = 0.5 tr(Xᵀ A X B),  A, B ~ random symmetric — the general
#               Brockett-type objective this repo is built around. X⋆ unique up
#               to a per-column sign flip -> `iterate_distance`.
#   :eigenvalue f(X) = 0.5 tr(Xᵀ A X)  (B = I(p)) — the plain block eigenvalue /
#               trace-minimisation problem: X⋆ spans the p smallest eigenvectors
#               of A, but is only unique as a *subspace* (invariant under
#               X ↦ X U for any orthogonal U, not just a sign flip) ->
#               `grassmann_distance`. The Riemannian Hessian at X⋆ is then
#               singular along the p(p-1)/2 in-subspace "gauge" directions
#               (B's spectrum is fully repeated); `hessian_spectrum_bounds`
#               below excludes those exactly-zero eigenvalues from L, μ, κ so
#               the rate lines / constant step stay well-defined.
const OBJECTIVE_KIND = :brockett   # :brockett or :eigenvalue

problem_distance(X, X_star) =
    OBJECTIVE_KIND === :eigenvalue ? grassmann_distance(X, X_star) : iterate_distance(X, X_star)
const ERR_YLABEL = OBJECTIVE_KIND === :eigenvalue ? "dist_Gr(Xₖ, X*)" : "‖Xₖ - X*‖"

# :brockett keeps writing to the flat data/ directory used throughout this
# file's history; :eigenvalue gets its own subfolder so it never overwrites
# the :brockett baseline.
const DATA_DIR = OBJECTIVE_KIND === :brockett ?
    joinpath(@__DIR__, "data") : joinpath(@__DIR__, "data", string(OBJECTIVE_KIND))

# write_dat(path, x, y; xname, yname)
#   Two columns — the integer x-series (iteration or A·X count) and a y-series
#   (err = ‖Xₖ − X⋆‖ by default, or the step size) — into a pgfplots `.dat` file
#   with header  `<xname> <yname>`.
function write_dat(path, x::AbstractVector{<:Integer}, y::AbstractVector;
                   xname::AbstractString, yname::AbstractString = "err")
    open(path, "w") do io
        println(io, xname, " ", yname)
        for (k, e) in zip(x, y)
            @printf(io, "%d %.12e\n", k, e)
        end
    end
    _log(@sprintf("    wrote %s  (%d rows)", relpath(path, @__DIR__), length(y)))
end

# write_series_dat(outdir, prefix, name, iters, work, errs)
#   `<prefix>_<name>_iter.dat` (x = iter) and `<prefix>_<name>_apps.dat`
#   (x = cumulative A·X); both y = err = ‖Xₖ − X⋆‖.
function write_series_dat(outdir, prefix, name, iters, work, errs)
    write_dat(joinpath(outdir, "$(prefix)_$(name)_iter.dat"), iters, errs; xname = "iter")
    write_dat(joinpath(outdir, "$(prefix)_$(name)_apps.dat"), work,  errs; xname = "apps")
    return nothing
end

# write_fgap_dat(outdir, prefix, name, iters, work, fgaps)
#   `<prefix>_<name>_fgap_iter.dat` (x = iter) and `<prefix>_<name>_fgap_apps.dat`
#   (x = cumulative A·X); both y = fgap = f(Xₖ) - f*.
function write_fgap_dat(outdir, prefix, name, iters, work, fgaps)
    write_dat(joinpath(outdir, "$(prefix)_$(name)_fgap_iter.dat"), iters, fgaps; xname = "iter", yname = "fgap")
    write_dat(joinpath(outdir, "$(prefix)_$(name)_fgap_apps.dat"), work,  fgaps; xname = "apps", yname = "fgap")
    return nothing
end

# write_gradnorm_dat(outdir, prefix, name, iters, work, gradnorms)
#   `<prefix>_<name>_gradnorm_iter.dat` (x = iter) and
#   `<prefix>_<name>_gradnorm_apps.dat` (x = cumulative A·X); both
#   y = gradnorm = ‖grad f(Xₖ)‖.
function write_gradnorm_dat(outdir, prefix, name, iters, work, gradnorms)
    write_dat(joinpath(outdir, "$(prefix)_$(name)_gradnorm_iter.dat"), iters, gradnorms; xname = "iter", yname = "gradnorm")
    write_dat(joinpath(outdir, "$(prefix)_$(name)_gradnorm_apps.dat"), work,  gradnorms; xname = "apps", yname = "gradnorm")
    return nothing
end

# The Hager–Zhang line search ([HagerZhang:2006:2]) — approximate-Wolfe
# termination (the derivative test  ϕ'(α) ≤ (2δ-1)ϕ'(0)  guarded by the slack
# check  ϕ(α) ≤ ϕ(0) + εₖ)  plus a secant² bracket update.  The curvature target
# σ = 0.2 matches the strong-Wolfe c2 used elsewhere.  Unlike the cubic
# bracketing line search it stays accurate on the machine-precision plateau,
# where `ϕ(a)`-vs-`ϕ(b)` comparisons collapse into rounding noise and the step
# otherwise decays to the `min_bracket_width` floor.
#
# NB: never pair this with a `:Stepsize` record — `RecordStepsize`'s k = 0
# `get_last_stepsize` probe re-invokes the search while its `last_stepsize` is
# still NaN, which sends a NaN step into the retraction and throws in LAPACK.
#
# This exact construction — pinned to start every line search at
# α₀ = init_stepsize (`Manopt.ConstantInitialGuess`) and floored at
# α ≥ min_stepsize (`FloorStepsize`, `main.jl`) — is the *RCG* Hager–Zhang
# stepsize used by experiment 1 (the `hz` step, one of its two panels) and
# experiment 2 (the `hagerzhang` comparison point).  It is RCG-specific:
# `experiment_gd_stepsize`'s RGD `hagerzhang` builds its own (same
# approximate-Wolfe/guard settings, but not derived from this function, since
# it isn't RCG). Experiment 4's `rcg` uses `rcg_secant_stepsize` instead — see
# below.
#
# `sufficient_curvature` (σ = c₂, default 0.2) is a knob left from experiment
# 2's dropped `hagerzhang-tight` / `hagerzhang-loose` (σ = 0.05 / 0.4)
# variants. δ = min(0.1, σ/2): Manopt's approximate-Wolfe test requires δ ≤ σ,
# so a fixed δ = 0.1 breaks for σ = 0.05; scaling it keeps δ = 0.1 for σ ≥ 0.2.
function rcg_hagerzhang_stepsize(M, rm, vtm; min_stepsize = 0, init_stepsize = 5.0e-2,
                                 sufficient_curvature = 0.2)
    hz = HagerZhangLinesearch(; retraction_method = rm, vector_transport_method = vtm,
                                wolfe_condition_mode = :approximate,
                                δ = min(0.1, sufficient_curvature / 2),
                                σ = sufficient_curvature, stepsize_limit = 1.0e3,
                                initial_guess = Manopt.ConstantInitialGuess(init_stepsize))
    return FloorStepsize(Manopt._produce_type(hz, M), min_stepsize; init = init_stepsize)
end

# The raw, greedy secant step (`greedy_secant.jl`) — found in experiment 2 to
# outperform the guarded Hager–Zhang stepsize above (fewer total A·X to reach
# the same ‖Xₖ − X⋆‖ < err_tol), so it is used for every *other* RCG run in
# this file: `experiment_cg_params` and `experiment_retractions`' `rcg`.
# Floored at α ≥ min_stepsize like the Hager–Zhang stepsize, but *not* pinned
# to a fixed initial guess every call (`init = NaN`) — the secant step needs
# its own previous accepted step, not a reset one; `initial_stepsize` only
# seeds the very first call.
#
# From iteration `corridor_start_iter` onward, the accepted step is
# additionally ceiled at `max_stepsize` (via `FloorStepsize`'s `ceil` /
# `start_iter`). This corridor was read off `cgstepsize_secant_step.dat`:
# between iterations ~50 and ~200 (well past the initial transient, well
# before the near-machine-precision tail) essentially all accepted steps sit
# in `[0.019, 0.074]` — 149/151 samples, the 2 exceptions already at the
# `min_stepsize` floor. Past iteration ~200 the secant formula's denominator
# `ϕ'(b) − ϕ'(0)` starts cancelling as the run nears the noise floor, and the
# step swings wildly (observed up to `1.2` in one run) instead of staying in
# that band; ceiling it at `max_stepsize` keeps it in the well-behaved
# corridor instead. Before `corridor_start_iter` the step never approached
# either bound in practice, so gating the ceiling to start there (rather than
# from iteration 0) costs nothing in the early phase.
function rcg_secant_stepsize(M, rm, vtm; min_stepsize = 0, initial_stepsize = 5.0e-2,
                              max_stepsize = Inf, corridor_start_iter = 50)
    sec = GreedySecantLinesearch(; retraction_method = rm, vector_transport_method = vtm,
                                   initial_stepsize = initial_stepsize)
    return FloorStepsize(Manopt._produce_type(sec, M), min_stepsize;
                          ceil = max_stepsize, start_iter = corridor_start_iter)
end

# -----------------------------------------------------------------------------
#  Convergence criterion on the iterate error  ‖Xₖ − X⋆‖  (not the gradient)
# -----------------------------------------------------------------------------
mutable struct StopWhenIterateErrorLess{P, F <: Real} <: StoppingCriterion
    X_star::P
    threshold::F
    last_error::F
    at_iteration::Int
    dist::Function
end
StopWhenIterateErrorLess(X_star, tol::Real; dist::Function = iterate_distance) =
    StopWhenIterateErrorLess(X_star, float(tol), float(tol), -1, dist)

function (c::StopWhenIterateErrorLess)(::AbstractManoptProblem,
                                       s::AbstractManoptSolverState, k::Int)
    k == 0 && (c.at_iteration = -1)
    if k > 0
        c.last_error = c.dist(get_iterate(s), c.X_star)
        if c.last_error < c.threshold
            c.at_iteration = k
            return true
        end
    end
    return false
end
function Manopt.get_reason(c::StopWhenIterateErrorLess)
    (c.at_iteration >= 0) || return ""
    return "Iterate error ‖Xₖ − X⋆‖ = $(c.last_error) < $(c.threshold) after $(c.at_iteration) iterations.\n"
end
Manopt.indicates_convergence(::StopWhenIterateErrorLess) = true
Manopt.status_summary(c::StopWhenIterateErrorLess; context::Symbol = :default) =
    "‖Xₖ − X⋆‖ < $(c.threshold): " * (c.at_iteration >= 0 ? "reached" : "not reached")
Base.show(io::IO, c::StopWhenIterateErrorLess) =
    print(io, "StopWhenIterateErrorLess(", c.threshold, ")")

# -----------------------------------------------------------------------------
#  Small shared helpers: record extraction, rate curves, preview plots
# -----------------------------------------------------------------------------
# Pull (iters, work, errs) out of a Manopt record or `run_newton`'s record —
# both are vectors of tuples with iteration in slot 1, A·X count in slot 4,
# ‖Xₖ − X⋆‖ in slot 5 (see CLAUDE.md's record-tuple convention).
series_from_record(rc) =
    (Int[x[1] for x in rc], Int[x[4] for x in rc], Float64[x[5] for x in rc])

# f(Xₖ) is slot 2 of the same record tuples; ‖grad f(Xₖ)‖ is slot 3.
costs_from_record(rc)     = Float64[x[2] for x in rc]
gradnorms_from_record(rc) = Float64[x[3] for x in rc]

# Prepend the shared iteration-0 point (every solver starts at the same X0).
function prepend_start!(iters, work, errs, d0)
    first(iters) == 0 && return (iters, work, errs)
    pushfirst!(iters, 0); pushfirst!(work, 0); pushfirst!(errs, d0)
    return iters, work, errs
end

# Prepend f(X0) to a cost series -- call *after* `prepend_start!` has already
# (maybe) grown `iters` by one, and this fills in the matching cost so the two
# stay the same length.
function prepend_cost!(iters, costs, f0)
    length(costs) == length(iters) && return costs
    pushfirst!(costs, f0)
    return costs
end

# Same idea, for a gradient-norm series and ‖grad f(X0)‖.
function prepend_gradnorm!(iters, gradnorms, g0)
    length(gradnorms) == length(iters) && return gradnorms
    pushfirst!(gradnorms, g0)
    return gradnorms
end

# L, μ, κ = largest / smallest-*nonzero* / condition number of a Riemannian-
# Hessian spectrum from `riemannian_hessian_eigenvalues` (sorted, all ≥ 0).
# Excluding exactly-zero eigenvalues matters for `:eigenvalue`-objective runs,
# where B = I(p) has a fully repeated spectrum and the p(p-1)/2 in-subspace
# "gauge" directions (X ↦ XU invariance) carry zero curvature — using the raw
# λ[1] there would give μ = 0, κ = Inf and break the rate lines / constant step.
function hessian_spectrum_bounds(λ; tol = 1.0e-10 * maximum(abs, λ))
    nz = filter(x -> abs(x) > tol, λ)
    L  = maximum(λ)
    μ  = isempty(nz) ? 0.0 : minimum(nz)
    κ  = μ > 0 ? L / μ : Inf
    return L, μ, κ
end

# Expected linear rates for a quadratic with Hessian condition number κ,
# anchored at ‖X0 − X⋆‖ = d0:  κ rate (steepest descent) and √κ rate (optimal
# / linear-CG), both ρᵏ with ρ = (c-1)/(c+1) for c ∈ {κ, √κ}.
function rate_series(κ, d0, kmax)
    kk  = collect(0:kmax)
    ρκ  = (κ - 1) / (κ + 1)
    ρκ½ = (sqrt(κ) - 1) / (sqrt(κ) + 1)
    return kk, d0 .* ρκ½ .^ kk, d0 .* ρκ .^ kk
end

# y-axis limits shared by the two error panels: clipped at `ymin` (below the
# convergence threshold nothing in the plot is meaningful) up to a bit above
# the largest value actually reached (so a divergent run doesn't blow up the
# scale, but nothing is clipped off the top either).
error_ylims(series, d0; ymin = 1e-6) =
    (ymin, maximum(max(maximum(s.errs), d0) for s in series) * 1.3)

# Two log-y error panels (err vs iteration, err vs A·X applications), each
# series as one line, plus √κ / κ dashed rate lines on the iteration panel.
function error_preview(series, κ, d0; title, ymin = 1e-6)
    kmax = maximum(last(s.iters) for s in series)
    kk, rate_sqrt, rate_cond = rate_series(κ, d0, kmax)
    ylims = error_ylims(series, d0; ymin)
    pit = plot(; yscale = :log10, ylims = ylims, legend = :topright,
               xlabel = "iteration", ylabel = ERR_YLABEL, title = title)
    pap = plot(; yscale = :log10, ylims = ylims, legend = :topright,
               xlabel = "A*X applications", ylabel = ERR_YLABEL, title = title)
    for s in series
        plot!(pit, s.iters, max.(s.errs, ymin); label = s.name, lw = 2)
        plot!(pap, s.work,  max.(s.errs, ymin); label = s.name, lw = 2)
    end
    plot!(pit, kk, max.(rate_sqrt, ymin);
          label = @sprintf("√κ rate  (κ=%.0f)", κ), lc = :black, ls = :dash, lw = 1.5)
    plot!(pit, kk, max.(rate_cond, ymin);
          label = "κ rate", lc = :gray, ls = :dashdot, lw = 1.5)
    return pit, pap
end

# y-axis floor for the f(Xₖ) - f* panel: near a nondegenerate minimum
# f - f* ~ ‖X - X⋆‖², so the natural floor sits far below the `err` panels'
# 1e-6 (a good chunk of that is genuine signal down to machine precision on
# the cost, not just noise the way ‖X-X⋆‖ itself bottoms out).
fgap_ylims(series, fgap0; ymin = 1e-15) =
    (ymin, maximum(max(maximum(s.fgaps), fgap0) for s in series) * 1.3)

# Single log-y panel: f(Xₖ) - f* vs iteration, one line per series. No rate
# lines here -- the √κ / κ reference rates are calibrated to ‖Xₖ - X⋆‖, not
# to the (roughly squared) cost gap.
function fgap_preview(series; title, ymin = 1e-14)
    fgap0 = maximum(first(s.fgaps) for s in series)
    ylims = fgap_ylims(series, fgap0; ymin)
    pfg = plot(; yscale = :log10, ylims = ylims, legend = :topright,
               xlabel = "iteration", ylabel = "f(Xₖ) - f*", title = title)
    for s in series
        plot!(pfg, s.iters, max.(s.fgaps, ymin); label = s.name, lw = 2)
    end
    return pfg
end

# y-axis floor for the ‖grad f(Xₖ)‖ panel -- gradient norm shrinks roughly like
# ‖Xₖ - X⋆‖ (both hit zero together), so a floor between the error panels'
# 1e-6 and the fgap panel's 1e-14 is the natural middle ground.
gradnorm_ylims(series, g0; ymin = 1e-8) =
    (ymin, maximum(max(maximum(s.gradnorms), g0) for s in series) * 1.3)

# Single log-y panel: ‖grad f(Xₖ)‖ vs iteration, one line per series. No rate
# lines -- same reasoning as `fgap_preview`.
function gradnorm_preview(series; title, ymin = 1e-8)
    g0 = maximum(first(s.gradnorms) for s in series)
    ylims = gradnorm_ylims(series, g0; ymin)
    pgn = plot(; yscale = :log10, ylims = ylims, legend = :topright,
               xlabel = "iteration", ylabel = "‖grad f(Xₖ)‖", title = title)
    for s in series
        plot!(pgn, s.iters, max.(s.gradnorms, ymin); label = s.name, lw = 2)
    end
    return pgn
end

# -----------------------------------------------------------------------------
#  Experiment 1 — comparison of RCG conjugacy coefficients
# -----------------------------------------------------------------------------
# Riemannian CG (polar retraction / its differentiated vector transport), full
# comparison of the conjugacy-coefficient rules
#
#     FR, PRP, HS, DY, HS–DY (Hybrid1), FR–PRP (Hybrid2)
#
# — some (HS/DY/HS–DY) are expected not to converge here, that is a known
# behaviour of RCG with a non-isometric transport; the point of the plot is to
# justify FR–PRP against the full field.  Convergence criterion:
# ‖Xₖ − X⋆‖ < err_tol.  The whole comparison is run twice, once per step size:
#
#     hz     : Hager–Zhang (`rcg_hagerzhang_stepsize`, approximate Wolfe,
#              curvature target c₂ = 0.2), pinned initial guess α₀ = αc = 2/(L+μ),
#              floored at α ≥ min_stepsize.  Replaced the raw greedy secant step
#              here: secant has *no* safeguard, so near a near-zero conjugacy
#              denominator (the DY `⟨𝒯d, y⟩` blow-up) it takes wildly oversized
#              steps that visibly wreck the DY curve; HZ's approximate-Wolfe
#              accept keeps every coefficient rule on a fair footing.
#     armijo : Armijo backtracking (c₁ = 0.2, contraction 0.5), pinned initial
#              guess α₀ = αc, floored / stopped at α ≥ min_stepsize
#
# All started from the same point `perturb` away from the closed-form optimum.
# Writes *only* the err-vs-iteration series the two panels plot:
#
#     cgparam_<pn>_hz_iter.dat       columns:  iter  err   (err = ‖Xₖ − X⋆‖)
#     cgparam_<pn>_armijo_iter.dat   columns:  iter  err
#
# with pn ∈ {fr, prp, hs, dy, hsdy, frprp}, and a two-panel
# `cgparam_preview.png` (left: Hager–Zhang step, right: armijo step — both
# ‖Xₖ − X⋆‖ vs iteration with dashed √κ / κ rate lines drawn from κ, one line
# per coefficient rule, shared log-y scale).
function experiment_cg_params(; n = 100, p = 5, seed = 42, perturb = 1e-1,
                                max_iters = 1000, err_tol = 1e-6, wall_secs = 60,
                                min_stepsize = 1e-2, init_stepsize = nothing,
                                const_stepsize = nothing,
                                outdir = DATA_DIR, preview = true, verbose = true)
    mkpath(outdir)
    rm, vtm = retraction_pair(:polar)

    M, A, B = build_problem(; n, p, seed, objective = OBJECTIVE_KIND)
    f_star, X_star = reference_optimum(M, A, B; verbose = verbose,
                                       retraction_method = rm, vector_transport_method = vtm)
    X0 = perturbed_start(M, X_star; radius = perturb, seed = seed + 1, retraction_method = rm)
    d0 = problem_distance(X0, X_star)
    λ  = riemannian_hessian_eigenvalues(A, B)
    L, μ, κ = hessian_spectrum_bounds(λ)                            # cond(Hess f @ X⋆)
    αc = something(const_stepsize, 2 / (L + μ))
    init_stepsize = something(init_stepsize, αc)                    # pinned α₀ for both step sizes
    _log(@sprintf("[cg-params]  St(%d,%d)  seed=%d  perturb=%.1e  polar retraction  min α = %.1e  init α = %.1e  objective = %s",
                  n, p, seed, perturb, min_stepsize, init_stepsize, OBJECTIVE_KIND))
    _log(@sprintf("  f* = %.12g   ‖X0 - X*‖ = %.3e   cond(Hess) = %.4g", f_star, d0, κ))

    sc = StopAfterIteration(max_iters) | StopWhenIterateErrorLess(X_star, err_tol; dist = problem_distance) |
         StopAfter(Second(wall_secs))

    prp = PolakRibiereCoefficient(;    vector_transport_method = vtm)
    hs  = HestenesStiefelCoefficient(; vector_transport_method = vtm)
    dy  = DaiYuanCoefficient(;         vector_transport_method = vtm)
    fr  = FletcherReevesCoefficient()
    variants = [
        ("FR",     "fr",    fr),
        ("PRP",    "prp",   prp),
        ("HS",     "hs",    hs),
        ("DY",     "dy",    dy),
        ("HS-DY",  "hsdy",  HybridCoefficient(hs, dy)),
        ("FR-PRP", "frprp", HybridCoefficient(fr, prp)),
    ]

    cig = Manopt.ConstantInitialGuess(init_stepsize)               # fixed α₀ for Armijo
    mkhz() = rcg_hagerzhang_stepsize(M, rm, vtm; min_stepsize, init_stepsize)
    mkarmijo() = FloorStepsize(
        Manopt._produce_type(ArmijoLinesearch(; retraction_method = rm, initial_guess = cig,
                                                sufficient_decrease = 0.2, contraction_factor = 0.5,
                                                stop_when_stepsize_less = min_stepsize), M),
        min_stepsize; init = init_stepsize)
    linesearches = [("hz", mkhz), ("armijo", mkarmijo)]

    panels = Dict{String, Vector{NamedTuple}}()
    ymax   = d0
    for (ls_name, mkls) in linesearches
        series = NamedTuple[]
        for (name, pn, coeff) in variants
            os  = ObjectiveStorage(5)
            obj = get_objective(A, B, os)
            _log(@sprintf("  %-6s / %-7s running ...", ls_name, name))
            st = conjugate_gradient_descent(M, obj, X0;
                coefficient = coeff, stepsize = mkls(),
                retraction_method = rm, vector_transport_method = vtm,
                stopping_criterion = sc, return_state = true,
                record = [:Iteration, RecordIterateDistance(X_star; dist = problem_distance)])
            rc    = get_record(st)
            iters = Int[x[1] for x in rc]
            errs  = Float64[x[2] for x in rc]
            first(iters) == 0 || (pushfirst!(iters, 0); pushfirst!(errs, d0))
            write_dat(joinpath(outdir, "cgparam_$(pn)_$(ls_name)_iter.dat"), iters, errs; xname = "iter")
            push!(series, (; name, iters, errs))
            _log(@sprintf("  %-6s / %-7s  %4d it   ‖X-X*‖ = %.3e", ls_name, name, last(iters), last(errs)))
        end
        ymax = max(ymax, maximum(maximum(s.errs) for s in series))
        panels[ls_name] = series
    end

    if preview
        ylims = (err_tol, ymax * 1.3)
        kmax  = maximum(last(s.iters) for pnl in values(panels) for s in pnl)
        kk, rate_sqrt, rate_cond = rate_series(κ, d0, kmax)
        function mkpanel(series, title)
            pl = plot(; yscale = :log10, ylims = ylims, legend = :topright,
                      xlabel = "iteration", ylabel = ERR_YLABEL, title = title)
            for s in series
                plot!(pl, s.iters, max.(s.errs, err_tol); label = s.name, lw = 2)
            end
            plot!(pl, kk, max.(rate_sqrt, err_tol);
                  label = @sprintf("√κ rate  (κ=%.0f)", κ), lc = :black, ls = :dash, lw = 1.5)
            plot!(pl, kk, max.(rate_cond, err_tol); label = "κ rate", lc = :gray, ls = :dashdot, lw = 1.5)
            return pl
        end
        pv = joinpath(outdir, "cgparam_preview.png")
        savefig(plot(mkpanel(panels["hz"], "CG coefficients — Hager–Zhang step"),
                     mkpanel(panels["armijo"], "CG coefficients — armijo step");
                     size = (1400, 480), layout = (1, 2)), pv)
        _log("  preview: " * relpath(pv, @__DIR__))
    end

    _log("[cg-params] done -> " * relpath(outdir, @__DIR__))
    return panels
end

# -----------------------------------------------------------------------------
#  Experiment 2 — RCG step-size / line-search comparison (fixed FR-PRP)
# -----------------------------------------------------------------------------
# Riemannian CG with the fixed FR-PRP (Hybrid2) coefficient and the polar
# retraction / its differentiated vector transport.  Compares five step-size
# rules:
#
#     constant     : fixed step  α ≡ 2/(L+μ)  (GD-optimal for a quadratic with
#                    Hessian spectrum [μ, L]; L, μ = largest/smallest
#                    Riemannian-Hessian eigenvalue at X⋆).  A baseline — it
#                    does *not* converge here (κ ≈ 2246 ⇒ linear rate
#                    ≈ 1 − 2/(κ+1)).  Override with `const_stepsize`.
#     armijo       : Armijo backtracking, sufficient decrease  c1 = 0.2
#     hagerzhang   : Hager–Zhang (approximate Wolfe, secant²), curvature target
#                    c2 = 0.2 — the two-sided approximate-Wolfe test brackets
#                    |ϕ'(α)| ≤ c2|ϕ'(0)| and stays accurate on the
#                    machine-precision plateau (unlike cubic bracketing, tried
#                    earlier and dropped — see CLAUDE.md).  The c2 = 0.05 /
#                    0.4 variants tried earlier performed almost identically
#                    and were dropped.
#     wolfe-powell : Manopt's `WolfePowellLinesearch` (strong-Wolfe, bracket
#                    then bisection; (c1, c2) = (0.1, 0.2)).  *Temporary* — its
#                    final bisection is the search that stalls badly on some
#                    seeds (see `reference_optimum` in `main.jl`); kept here
#                    only to see how it fares as an RCG step size.
#     secant       : `greedy_secant.jl`'s raw, greedy secant step — *no*
#                    line search / backtracking / safeguard: every iteration
#                    takes exactly one secant step on ϕ' between 0 and the
#                    previous accepted step (`α = −ϕ'(0)·b / (ϕ'(b) − ϕ'(0))`),
#                    seeded on the first call with b = 2/(L+μ) (the same value
#                    as `constant`'s α); one A·X per iteration.  A curvature-
#                    accept shortcut (skip the secant step if b already
#                    satisfies |ϕ'(b)| ≤ c2|ϕ'(0)|) was tried and measured
#                    *worse* at both c2 = 0.2 and 0.1 — more total
#                    iterations/A·X, not fewer, because reusing a stale b
#                    under-refines the step for FR-PRP's fast-changing
#                    conjugate direction — so it always takes the secant step;
#                    see CLAUDE.md
#
# `armijo`, `hagerzhang` and `wolfe-powell` are wrapped in a `FloorStepsize`
# that clamps the accepted step to `α ≥ min_stepsize` (default 1e-2) and pins
# the *initial* trial step to `α₀ = init_stepsize`.  `init_stepsize` defaults
# to `αc = 2/(L+μ)` — the same problem-calibrated value `constant` and
# `secant` already use — rather than an arbitrary literal, so every step-size
# rule here starts from the same guess (pass `init_stepsize` explicitly to
# override just this pinned guess without changing `const_stepsize`/`αc`
# itself). `constant` and `secant` are wrapped only for the `α ≥ min_stepsize`
# floor — `secant` is *not* pinned to `init_stepsize` every call (that would
# discard the previous step its formula needs), so it warm-starts from `αc`
# and then keeps its own.
#
# all started from the same point `perturb` away from the closed-form optimum,
# same convergence criterion as experiment 1 (‖Xₖ − X⋆‖ < err_tol), and for
# each writes only the two panels' data:
#
#     cgstepsize_<name>_iter.dat   columns:  iter  err
#     cgstepsize_<name>_apps.dat   columns:  apps  err
#
# with  name ∈ {constant, armijo, hagerzhang, wolfe-powell, secant}, plus
# `cgstepsize_rate_{sqrtcond,cond}.dat` and a two-panel `cgstepsize_preview.png`
# (err vs iteration with dashed √κ / κ rate lines, err vs A·X).
function experiment_cg_stepsize(; n = 100, p = 5, seed = 42, perturb = 1e-1,
                                  max_iters = 1000, err_tol = 1e-6, wall_secs = 60,
                                  min_stepsize = 1e-2, init_stepsize = nothing,
                                  const_stepsize = nothing,
                                  outdir = DATA_DIR, preview = true, verbose = true)
    mkpath(outdir)
    rm, vtm = retraction_pair(:polar)

    M, A, B = build_problem(; n, p, seed, objective = OBJECTIVE_KIND)
    f_star, X_star = reference_optimum(M, A, B; verbose = verbose,
                                       retraction_method = rm, vector_transport_method = vtm)
    X0 = perturbed_start(M, X_star; radius = perturb, seed = seed + 1, retraction_method = rm)
    d0 = problem_distance(X0, X_star)
    λ  = riemannian_hessian_eigenvalues(A, B)
    L, μ, κ = hessian_spectrum_bounds(λ)
    αc = something(const_stepsize, 2 / (L + μ))
    # default the pinned initial-guess step to αc (2/(L+μ)) -- same value
    # `constant`/`secant` already use -- so `armijo`/`hagerzhang`/`wolfe-powell`
    # start from a problem-calibrated guess instead of an arbitrary literal;
    # still overridable via `init_stepsize` to probe a different guess.
    init_stepsize = something(init_stepsize, αc)

    _log(@sprintf("[cg-stepsize]  St(%d,%d)  seed=%d  perturb=%.1e  polar retraction  FR-PRP coefficient  min α = %.1e  init α = %.1e  const α = %.3e (2/(L+μ))",
                  n, p, seed, perturb, min_stepsize, init_stepsize, αc))
    _log(@sprintf("  f* = %.12g   ‖X0 - X*‖ = %.3e   cond(Hess) = %.4g", f_star, d0, κ))

    sc = StopAfterIteration(max_iters) | StopWhenIterateErrorLess(X_star, err_tol; dist = problem_distance) |
         StopAfter(Second(wall_secs))
    frprp = HybridCoefficient(FletcherReevesCoefficient(),
                              PolakRibiereCoefficient(; vector_transport_method = vtm))

    cig = Manopt.ConstantInitialGuess(init_stepsize)   # fixed α₀ for Armijo
    armijo() = ArmijoLinesearch(; retraction_method = rm, initial_guess = cig,
                                  sufficient_decrease = 0.2, contraction_factor = 0.5,
                                  stop_when_stepsize_less = min_stepsize)
    # WolfePowell (strong-Wolfe, bracket + bisection) — TEMPORARY, see header.
    wolfepowell() = WolfePowellLinesearch(; retraction_method = rm, vector_transport_method = vtm,
                                            sufficient_decrease = 0.1, sufficient_curvature = 0.2,
                                            stop_when_stepsize_less = min_stepsize,
                                            stop_increasing_at_step = 25, stop_decreasing_at_step = 50)
    # materialise each factory, clamp the step to α ≥ min_stepsize, and (unless
    # pin_init = false) restart every line search from α₀ = init_stepsize
    guard(f; pin_init = true) = FloorStepsize(Manopt._produce_type(f, M), min_stepsize;
                                              init = pin_init ? init_stepsize : NaN)
    # greedy secant: no bracketing/backtracking, no curvature-accept shortcut —
    # always one secant step, warm-started at 2/(L+μ) (same value as
    # `constant`) and thereafter using its own previous accepted step
    secant() = GreedySecantLinesearch(; retraction_method = rm, vector_transport_method = vtm,
                                        initial_stepsize = αc)
    steps = [
        ("constant",     guard(ConstantLength(αc; type = :relative); pin_init = false)),
        ("armijo",       guard(armijo())),
        ("hagerzhang",   rcg_hagerzhang_stepsize(M, rm, vtm; min_stepsize, init_stepsize)),
        ("wolfe-powell", guard(wolfepowell())),
        ("secant",       guard(secant(); pin_init = false)),
    ]

    series = NamedTuple[]
    for (name, ss) in steps
        os  = ObjectiveStorage(5)
        obj = get_objective(A, B, os)
        _log(@sprintf("  %-12s running ...", name))
        st = conjugate_gradient_descent(M, obj, X0;
            coefficient = frprp, stepsize = ss,
            retraction_method = rm, vector_transport_method = vtm,
            stopping_criterion = sc, return_state = true,
            record = [:Iteration, :Cost, :GradientNorm,
                      RecordApplications(os), RecordIterateDistance(X_star; dist = problem_distance)])
        iters, work, errs = series_from_record(get_record(st))
        prepend_start!(iters, work, errs, d0)

        write_series_dat(outdir, "cgstepsize", name, iters, work, errs)
        push!(series, (; name, iters, work, errs))
        _log(@sprintf("  %-12s  %4d it   ‖X-X*‖ = %.3e   A*X = %d",
                      name, last(iters), last(errs), last(work)))
    end

    if preview
        pit, pap = error_preview(series, κ, d0; title = "Step sizes (FR-PRP)", ymin = err_tol)
        pv = joinpath(outdir, "cgstepsize_preview.png")
        savefig(plot(pit, pap; size = (1400, 480), layout = (1, 2),
                     left_margin = 5Plots.mm, bottom_margin = 6Plots.mm), pv)
        _log("  preview: " * relpath(pv, @__DIR__))
    end

    _log("[cg-stepsize] done -> " * relpath(outdir, @__DIR__))
    return series
end

# -----------------------------------------------------------------------------
#  Experiment 3 — RGD step-size comparison (constant / spectral / Hager–Zhang)
# -----------------------------------------------------------------------------
# Riemannian *gradient descent* (steepest descent, δ = −grad f), polar
# retraction, comparing four step-size rules:
#
#     constant       : fixed α ≡ 2/(L+μ)  (GD-optimal, same baseline as
#                      experiment 2's `constant`) — does not converge (κ ≈ 2246)
#     bb-alternating : raw Barzilai–Borwein (`spectral_stepsizes.jl`, no line
#                      search / backtracking / safeguard), long step
#                      α = ⟨s,s⟩/⟨s,y⟩ on odd iterations, short step
#                      α = ⟨s,y⟩/⟨y,y⟩ on even — as Manopt's
#                      `NonmonotoneLinesearch(strategy = :alternating)`, minus
#                      the nonmonotone Armijo safeguard
#     norm-ratio     : raw spectral step  α = ‖s‖/‖y‖ = √(⟨s,s⟩/⟨y,y⟩)
#                      (geometric mean of the long/short BB steps)
#     hagerzhang     : same Hager–Zhang line search as experiment 2, here
#                      driving plain gradient descent instead of RCG
#
# where s_k = X_k − X_{k-1}, y_k = grad f(X_k) − grad f(X_{k-1}) (ambient
# differences).  `hagerzhang` is wrapped in the same `FloorStepsize` /
# `α₀ = init_stepsize` convention as experiment 2; the raw spectral steps and
# `constant` are wrapped only for uniform step recording (their own internal
# clamps already keep them well away from `min_stepsize` in practice).
#
# started from the same `perturbed_start`, same convergence criterion
# ‖Xₖ − X⋆‖ < err_tol, `max_iters = 1000` for every strategy (large enough for
# the spectral steps to converge, small enough that `constant` visibly does
# not).  For each name ∈ {constant, bb-alternating, norm-ratio, hagerzhang}
# writes
#
#     gdstepsize_<name>_iter.dat            columns:  iter  err
#     gdstepsize_<name>_apps.dat            columns:  apps  err
#     gdstepsize_<name>_fgap_iter.dat       columns:  iter  fgap       (fgap = f(Xₖ) − f*)
#     gdstepsize_<name>_fgap_apps.dat       columns:  apps  fgap
#     gdstepsize_<name>_gradnorm_iter.dat   columns:  iter  gradnorm   (gradnorm = ‖grad f(Xₖ)‖)
#     gdstepsize_<name>_gradnorm_apps.dat   columns:  apps  gradnorm
#     gdstepsize_<name>_step.dat            columns:  iter  step       (accepted α_k, from iter 1)
#
# plus `gdstepsize_rate_{sqrtcond,cond}.dat` and a five-panel
# `gdstepsize_preview.png` (err vs iteration with dashed √κ / κ rate lines, err
# vs A·X, f(Xₖ) − f* vs iteration, ‖grad f(Xₖ)‖ vs iteration, α_k vs iteration).
function experiment_gd_stepsize(; n = 100, p = 5, seed = 42, perturb = 1e-1,
                                  max_iters = 1000, err_tol = 1e-6, wall_secs = 120,
                                  min_stepsize = 1e-2, init_stepsize = 5e-2,
                                  spectral_max_stepsize = 1.0e2, const_stepsize = nothing,
                                  outdir = DATA_DIR, preview = true, verbose = true)
    mkpath(outdir)
    rm, vtm = retraction_pair(:polar)

    M, A, B = build_problem(; n, p, seed, objective = OBJECTIVE_KIND)
    f_star, X_star = reference_optimum(M, A, B; verbose = verbose,
                                       retraction_method = rm, vector_transport_method = vtm)
    X0 = perturbed_start(M, X_star; radius = perturb, seed = seed + 1, retraction_method = rm)
    d0 = problem_distance(X0, X_star)
    os0 = ObjectiveStorage(2); obj0 = get_objective(A, B, os0)
    f0 = get_cost(M, obj0, X0)
    g0 = norm(get_gradient(M, obj0, X0))
    λ  = riemannian_hessian_eigenvalues(A, B)
    L, μ, κ = hessian_spectrum_bounds(λ)
    αc = something(const_stepsize, 2 / (L + μ))

    _log(@sprintf("[gd-stepsize]  St(%d,%d)  seed=%d  perturb=%.1e  polar retraction  RGD  min α = %.1e  init α = %.1e  const α = %.3e (2/(L+μ))",
                  n, p, seed, perturb, min_stepsize, init_stepsize, αc))
    _log(@sprintf("  f* = %.12g   ‖X0 - X*‖ = %.3e   cond(Hess) = %.4g", f_star, d0, κ))

    sc = StopAfterIteration(max_iters) | StopWhenIterateErrorLess(X_star, err_tol; dist = problem_distance) |
         StopAfter(Second(wall_secs))

    cig = Manopt.ConstantInitialGuess(init_stepsize)
    hagerzhang() = HagerZhangLinesearch(; retraction_method = rm, vector_transport_method = vtm,
                                          wolfe_condition_mode = :approximate, δ = 0.1, σ = 0.2,
                                          stepsize_limit = 1.0e3, initial_guess = cig)
    spectral(rule) = SpectralLength(; rule = rule, initial_stepsize = αc,
                                      max_stepsize = spectral_max_stepsize)
    guard(f; pin_init = true) = FloorStepsize(Manopt._produce_type(f, M), min_stepsize;
                                              init = pin_init ? init_stepsize : NaN)
    steps = [
        ("constant",       guard(ConstantLength(αc; type = :relative); pin_init = false)),
        ("bb-alternating", guard(spectral(:alternating); pin_init = false)),
        ("norm-ratio",     guard(spectral(:norm_ratio);  pin_init = false)),
        ("hagerzhang",     guard(hagerzhang())),
    ]

    series = NamedTuple[]
    for (name, ss) in steps
        os  = ObjectiveStorage(5)
        obj = get_objective(A, B, os)
        _log(@sprintf("  %-15s running ...", name))
        st = gradient_descent(M, obj, X0;
            stepsize = ss, retraction_method = rm,
            stopping_criterion = sc, return_state = true,
            record = [:Iteration, :Cost, :GradientNorm,
                      RecordApplications(os), RecordIterateDistance(X_star; dist = problem_distance),
                      RecordStepsizeValue()])
        rc     = get_record(st)
        aiters = Int[x[1] for x in rc]
        alphas = Float64[x[6] for x in rc]
        iters, work, errs = series_from_record(rc)
        costs = costs_from_record(rc)
        gradnorms = gradnorms_from_record(rc)
        prepend_start!(iters, work, errs, d0)
        prepend_cost!(iters, costs, f0)
        prepend_gradnorm!(iters, gradnorms, g0)
        fgaps = costs .- f_star

        write_series_dat(outdir, "gdstepsize", name, iters, work, errs)
        write_fgap_dat(outdir, "gdstepsize", name, iters, work, fgaps)
        write_gradnorm_dat(outdir, "gdstepsize", name, iters, work, gradnorms)
        write_dat(joinpath(outdir, "gdstepsize_$(name)_step.dat"), aiters, alphas; xname = "iter", yname = "step")
        push!(series, (; name, iters, work, errs, fgaps, gradnorms, aiters, alphas))
        _log(@sprintf("  %-15s  %4d it   ‖X-X*‖ = %.3e   A*X = %d   α: %.2e … %.2e",
                      name, last(iters), last(errs), last(work),
                      minimum(alphas), maximum(alphas)))
    end

    if preview
        pit, pap = error_preview(series, κ, d0; title = "Step sizes (RGD)", ymin = err_tol)
        pfg = fgap_preview(series; title = "Step sizes (RGD)")
        pgn = gradnorm_preview(series; title = "Step sizes (RGD)")
        pst = plot(; yscale = :log10, legend = :topright,
                   xlabel = "iteration", ylabel = "step size αₖ", title = "Step sizes (RGD)")
        for s in series
            plot!(pst, s.aiters, max.(s.alphas, eps()); label = s.name, lw = 2)
        end
        pv = joinpath(outdir, "gdstepsize_preview.png")
        savefig(plot(pit, pap, pfg, pgn, pst; size = (2750, 480), layout = (1, 5),
                     left_margin = 5Plots.mm, bottom_margin = 6Plots.mm), pv)
        _log("  preview: " * relpath(pv, @__DIR__))
    end

    _log("[gd-stepsize] done -> " * relpath(outdir, @__DIR__))
    return series
end

# -----------------------------------------------------------------------------
#  Experiment 4 — retraction comparison (polar vs. QR)
# -----------------------------------------------------------------------------
# Four solvers, each run once per retraction (vector transport =
# `DifferentiatedRetractionVectorTransport` of that same retraction, both cases):
#
#     newton       : globalised Riemannian Newton, MINRES inner solve
#                    (`run_newton`, `inner_method = :minres`)
#     rcg          : RCG, FR-PRP (Hybrid2) coefficient, raw greedy secant
#                    stepsize (`rcg_secant_stepsize`, same as experiment 1 —
#                    found in experiment 2 to outperform Hager–Zhang)
#     rgd-constant : RGD, fixed step α ≡ 2/(L+μ)
#     rgd-bb       : RGD, raw Barzilai–Borwein alternating step
#                    (`spectral_stepsizes.jl`, same as experiment 3's
#                    `bb-alternating`)
#
# Each (solver, retraction) pair starts from its own `perturbed_start` (same
# seed ⇒ the same underlying random tangent direction, retracted differently)
# and stops at ‖Xₖ − X⋆‖ < err_tol (`max_iters` cap; Newton uses its own
# `newton_max_iters` / `newton_grad_tol` — it has no iterate-error stop, but
# converges quadratically once inside the basin so this rarely matters).
# Writes, for name ∈ {newton, rcg, rgd-constant, rgd-bb} × {polar, qr}, only
# the two panels' data:
#
#     retractions_<name>_<retraction>_iter.dat   columns:  iter  err
#
# and a two-panel `retractions_preview.png`, both ‖Xₖ − X⋆‖ vs iteration
# (solid = polar, dashed = QR, one colour per solver): the left panel has
# `rcg` + `rgd-constant` + `rgd-bb`, the right panel has `newton` alone
# (Newton converges in ~15 iterations, so sharing an x-axis with the ~1000-it
# first-order runs would squash it into the y-axis).
function experiment_retractions(; n = 100, p = 5, seed = 42, perturb = 1e-1,
                                   max_iters = 1000, err_tol = 1e-6, wall_secs = 180,
                                   min_stepsize = 1e-2,
                                   newton_max_iters = 30, newton_grad_tol = 1e-10,
                                   inner_rtol = 1e-4, inner_maxiter = 600,
                                   const_stepsize = nothing, spectral_max_stepsize = 1.0e2,
                                   outdir = DATA_DIR, preview = true, verbose = true)
    mkpath(outdir)
    M, A, B = build_problem(; n, p, seed, objective = OBJECTIVE_KIND)
    λ  = riemannian_hessian_eigenvalues(A, B)
    L, μ, κ = hessian_spectrum_bounds(λ)
    αc = something(const_stepsize, 2 / (L + μ))
    _log(@sprintf("[retractions]  St(%d,%d)  seed=%d  perturb=%.1e  cond(Hess) = %.4g  const α = %.3e  objective = %s",
                  n, p, seed, perturb, κ, αc, OBJECTIVE_KIND))

    method_label = (newton = "newton", rcg = "rcg",
                    rgd_constant = "rgd-constant", rgd_bb = "rgd-bb")
    method_color = (newton = :steelblue, rcg = :darkorange,
                    rgd_constant = :seagreen, rgd_bb = :purple)
    methods = (:rcg, :rgd_constant, :rgd_bb, :newton)   # left panel: first three; right panel: newton

    f_star, X_star = reference_optimum(M, A, B; verbose = verbose)

    series = NamedTuple[]
    for retr in (:polar, :qr)
        rm, vtm = retraction_pair(retr)

        X0 = perturbed_start(M, X_star; radius = perturb, seed = seed + 1, retraction_method = rm)
        d0 = problem_distance(X0, X_star)
        sc = StopAfterIteration(max_iters) | StopWhenIterateErrorLess(X_star, err_tol; dist = problem_distance) |
             StopAfter(Second(wall_secs))
        frprp = HybridCoefficient(FletcherReevesCoefficient(),
                                  PolakRibiereCoefficient(; vector_transport_method = vtm))
        _log(@sprintf("  [%s]  f* = %.12g   ‖X0 - X*‖ = %.3e", retr, f_star, d0))

        for m in methods
            os   = ObjectiveStorage(5)
            obj  = get_objective(A, B, os)
            name = "$(method_label[m])_$(retr)"
            _log(@sprintf("    %-20s running ...", name))
            local rc
            if m === :newton
                hess = get_hessian(A, B, os)
                _, rc = run_newton(M, obj, hess, X0, os, X_star;
                    max_iters = newton_max_iters, grad_tol = newton_grad_tol,
                    inner_method = :minres, inner_rtol = inner_rtol,
                    inner_maxiter = inner_maxiter, verbose = false,
                    retraction_method = rm, dist = problem_distance)
            elseif m === :rcg
                rcgstep = rcg_secant_stepsize(M, rm, vtm; min_stepsize, initial_stepsize = αc)
                st = conjugate_gradient_descent(M, obj, X0;
                    coefficient = frprp, stepsize = rcgstep,
                    retraction_method = rm, vector_transport_method = vtm,
                    stopping_criterion = sc, return_state = true,
                    record = [:Iteration, :Cost, :GradientNorm,
                              RecordApplications(os), RecordIterateDistance(X_star; dist = problem_distance)])
                rc = get_record(st)
            elseif m === :rgd_constant
                st = gradient_descent(M, obj, X0;
                    stepsize = ConstantLength(αc; type = :relative), retraction_method = rm,
                    stopping_criterion = sc, return_state = true,
                    record = [:Iteration, :Cost, :GradientNorm,
                              RecordApplications(os), RecordIterateDistance(X_star; dist = problem_distance)])
                rc = get_record(st)
            else # :rgd_bb
                st = gradient_descent(M, obj, X0;
                    stepsize = SpectralLength(; rule = :alternating, initial_stepsize = αc,
                                              max_stepsize = spectral_max_stepsize),
                    retraction_method = rm,
                    stopping_criterion = sc, return_state = true,
                    record = [:Iteration, :Cost, :GradientNorm,
                              RecordApplications(os), RecordIterateDistance(X_star; dist = problem_distance)])
                rc = get_record(st)
            end

            iters, work, errs = series_from_record(rc)
            prepend_start!(iters, work, errs, d0)
            write_dat(joinpath(outdir, "retractions_$(name)_iter.dat"), iters, errs; xname = "iter")
            push!(series, (; name, method = m, retraction = retr, color = method_color[m], iters, errs))
            _log(@sprintf("    %-20s  %4d it   ‖X-X*‖ = %.3e   A*X = %d",
                          name, last(iters), last(errs), last(work)))
        end
    end

    if preview
        # two panels, both ‖Xₖ - X⋆‖ vs iteration: first-order runs on the left
        # (with √κ / κ rate lines), Newton (converges in a handful of iterations,
        # rate lines meaningless there) alone on the right.
        function errpanel(subseries, title; rates = false)
            emin = min(err_tol, minimum(minimum(s.errs) for s in subseries) * 0.5)
            emax = maximum(maximum(s.errs) for s in subseries) * 1.3
            pl = plot(; yscale = :log10, ylims = (emin, emax), legend = :outertopright,
                      legendfontsize = 7, xlabel = "iteration", ylabel = ERR_YLABEL, title = title)
            for s in subseries
                ls = s.retraction === :polar ? :solid : :dash
                plot!(pl, s.iters, max.(s.errs, emin); label = s.name, lw = 2, ls = ls, lc = s.color)
            end
            if rates
                d0   = maximum(first(s.errs) for s in subseries)
                kmax = maximum(last(s.iters) for s in subseries)
                kk, rate_sqrt, rate_cond = rate_series(κ, d0, kmax)
                plot!(pl, kk, max.(rate_sqrt, emin);
                      label = @sprintf("√κ rate  (κ=%.0f)", κ), lc = :black, ls = :dash, lw = 1.5)
                plot!(pl, kk, max.(rate_cond, emin); label = "κ rate", lc = :gray, ls = :dashdot, lw = 1.5)
            end
            return pl
        end
        first_order = filter(s -> s.method !== :newton, series)
        newton      = filter(s -> s.method === :newton, series)
        pv = joinpath(outdir, "retractions_preview.png")
        savefig(plot(errpanel(first_order, "Retractions — RCG / RGD"; rates = true),
                     errpanel(newton, "Retractions — Newton");
                     size = (1600, 520), layout = (1, 2)), pv)
        _log("  preview: " * relpath(pv, @__DIR__))
    end

    _log("[retractions] done -> " * relpath(outdir, @__DIR__))
    return series
end

# -----------------------------------------------------------------------------
if abspath(PROGRAM_FILE) == @__FILE__

    perturb = 0.5
    seed = 2 #rand(1:300) #2
    experiment_cg_params(seed = seed, perturb = perturb)
    experiment_cg_stepsize(seed = seed, perturb = perturb)
    # experiment_gd_stepsize(seed = seed, perturb = perturb)   # temporarily disabled
    experiment_retractions(seed = seed, perturb = perturb)
end
