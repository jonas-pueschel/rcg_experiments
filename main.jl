# =============================================================================
#  main.jl
#
#  Numerical experiments for Riemannian optimization methods (Manopt.jl) on the
#  Stiefel Brockett-type objective
#
#        f(X) = 0.5 * tr( Xᵀ A X B ),   X ∈ St(n, p),  A = Aᵀ,  B = Bᵀ.
#
#  We compare Riemannian conjugate gradient (RCG, several conjugacy rules),
#  steepest descent and Riemannian Newton (Krylov inner solve, :cg / :minres),
#  all started from the closed-form optimum perturbed by `perturb`, and measure
#  how much repeated `A * X` work the `ObjectiveStorage` cache removes.
#
#  Run:   julia --project=. main.jl
# =============================================================================

using LinearAlgebra
using Random
using Printf
using Dates
using Manifolds
using Manopt
using Plots

include("objective.jl")
include("greedy_secant.jl")   # GreedySecantLinesearch -- used by `reference_optimum`'s LBFGS polish

# stdout is block-buffered when it is not a TTY (pipes, IDE consoles, files),
# so every progress line is pushed out explicitly.
_log(args...) = (println(stderr, args...); flush(stderr))

# Manopt record action: log the running `A * X` count from an `ObjectiveStorage`
# at every iteration, so convergence can be plotted against work instead of steps.
mutable struct RecordApplications <: Manopt.RecordAction
    os::ObjectiveStorage
    recorded_values::Vector{Int}
    RecordApplications(os::ObjectiveStorage) = new(os, Int[])
end
function (r::RecordApplications)(::AbstractManoptProblem, ::AbstractManoptSolverState, k::Int)
    return Manopt.record_or_reset!(r, r.os.n_applications, k)
end

"""
    iterate_distance(X, X_star)

Frobenius distance `‖X − X⋆‖`, minimised over the per-column sign flips that
leave the objective invariant (`X⋆ = E_A J Q_Bᵀ` is unique only up to
`X⋆ diag(±1)`).  Near the optimum this equals the ordinary `‖X − X⋆‖`.
"""
function iterate_distance(X, X_star)
    s = sign.(sum(X .* X_star; dims = 1))
    s = ifelse.(s .== 0, 1.0, s)
    return norm(X .- X_star .* s)
end

"""
    grassmann_distance(X, X_star)

Procrustes / chordal distance between the `p`-dimensional subspaces spanned by
the orthonormal `n×p` bases `X` and `X_star`:
`min_{U ∈ O(p)} ‖X U − X_star‖`, achieved at the orthogonal polar factor of
`Xᵀ X_star`, giving `dist² = 2p − 2 Σᵢ σᵢ(Xᵀ X_star)` (`σ` = singular values).
Use this instead of `iterate_distance` whenever the objective is invariant
under `X ↦ X U` for *any* orthogonal `U` — e.g. the block eigenvalue problem
`f(X) = 0.5 tr(Xᵀ A X)` (`build_problem(...; objective = :eigenvalue)`), where
the minimiser is unique only as a subspace, not as a point of `St(n, p)`.
"""
function grassmann_distance(X, X_star)
    p = size(X, 2)
    σ = svdvals(X' * X_star)
    return sqrt(max(0.0, 2p - 2 * sum(σ)))
end

# Manopt record action: distance of the current iterate to the reference
# solution, via `dist` (default `iterate_distance`; pass `grassmann_distance`
# for a subspace-invariant objective).
mutable struct RecordIterateDistance <: Manopt.RecordAction
    X_star::Matrix{Float64}
    dist::Function
    recorded_values::Vector{Float64}
end
RecordIterateDistance(X_star; dist::Function = iterate_distance) =
    RecordIterateDistance(Matrix{Float64}(X_star), dist, Float64[])
function (r::RecordIterateDistance)(::AbstractManoptProblem, s::AbstractManoptSolverState, k::Int)
    return Manopt.record_or_reset!(r, r.dist(get_iterate(s), r.X_star), k)
end

# Manopt record action: the accepted line-search step size α at each iteration.
# Reads the materialised stepsize's `last_stepsize` field directly (or `length`
# for a `ConstantStepsize`).  The built-in `:Stepsize` / `RecordStepsize` calls
# `get_last_stepsize`, which for stateful searches without a stored-value accessor
# (cubic bracketing, Hager–Zhang) re-invokes the whole line search; at the k = 0
# init probe that runs with `last_stepsize = NaN` and throws in LAPACK.
# `k = 0` records `NaN` (no step yet).
mutable struct RecordStepsizeValue <: Manopt.RecordAction
    recorded_values::Vector{Float64}
    RecordStepsizeValue() = new(Float64[])
end
function (r::RecordStepsizeValue)(::AbstractManoptProblem, s::AbstractManoptSolverState, k::Int)
    ss = Manopt.get_state(s).stepsize
    α  = k <= 0                              ? NaN :
         hasfield(typeof(ss), :last_stepsize) ? float(getfield(ss, :last_stepsize)) :
         hasfield(typeof(ss), :length)        ? float(getfield(ss, :length)) : NaN
    return Manopt.record_or_reset!(r, α, k)
end

# -----------------------------------------------------------------------------
#  Step-size guard wrapper
# -----------------------------------------------------------------------------
# Wrap any materialised line search and
#   * clamp its accepted step to `floor ≤ α ≤ ceil` — a uniform bound across
#     strategies whose native knobs differ (`stop_when_stepsize_less` for Armijo /
#     Wolfe–Powell, `min_bracket_width` for cubic bracketing, nothing for
#     Hager–Zhang); `ceil = Inf` (the default) disables the upper bound;
#   * only from iteration `start_iter` onward — for `k < start_iter` the raw
#     inner step is returned unclamped, so e.g. a wide-open early phase can run
#     on the bare line search before a floor/ceiling starts constraining it
#     (`start_iter = 0`, the default, clamps from the first call, matching the
#     original always-on behaviour);
#   * optionally pin the *initial* trial step: when `init` is finite, the inner
#     search's `last_stepsize` field is reset to `init` before every call, so
#     warm-starting searches (cubic bracketing, secant) always restart the line
#     search from the same `α₀` instead of the previous accepted step.  Searches
#     with an explicit initial-guess knob (Armijo, Hager–Zhang) should still be
#     built with `Manopt.ConstantInitialGuess(init)` — the reset is a no-op there.
#     (`init` pinning is unaffected by `start_iter` — it's a separate knob.)
# Pass a *materialised* `Stepsize` (`Manopt._produce_type(factory, M)`), not the
# `ManifoldDefaultsFactory`.
mutable struct FloorStepsize{S <: Manopt.Stepsize, F <: Real} <: Manopt.Stepsize
    inner::S
    floor::F
    ceil::F        # upper bound; Inf = none
    init::F        # reset inner `last_stepsize` to this before each call; NaN = leave alone
    start_iter::Int  # clamp only applies for k ≥ start_iter; 0 = always (original behaviour)
    last_stepsize::F
end
FloorStepsize(inner::Manopt.Stepsize, floor::Real; init::Real = NaN, ceil::Real = Inf,
              start_iter::Integer = 0) =
    FloorStepsize(inner, float(floor), float(ceil), float(init), Int(start_iter), float(floor))

function (fs::FloorStepsize)(mp::AbstractManoptProblem, s::AbstractManoptSolverState,
                             k::Int, args...; kwargs...)
    if isfinite(fs.init) && hasfield(typeof(fs.inner), :last_stepsize)
        fs.inner.last_stepsize = oftype(fs.inner.last_stepsize, fs.init)
    end
    raw = fs.inner(mp, s, k, args...; kwargs...)
    fs.last_stepsize = k >= fs.start_iter ? clamp(raw, fs.floor, fs.ceil) : raw
    return fs.last_stepsize
end
Manopt.get_last_stepsize(fs::FloorStepsize, ::Any...) = fs.last_stepsize
function Manopt.get_initial_stepsize(fs::FloorStepsize)
    v = applicable(Manopt.get_initial_stepsize, fs.inner) ?
        Manopt.get_initial_stepsize(fs.inner) : fs.floor
    return fs.start_iter <= 0 ? clamp(v, fs.floor, fs.ceil) : v
end
function Manopt.initialize_stepsize!(fs::FloorStepsize)
    Manopt.initialize_stepsize!(fs.inner)
    fs.last_stepsize = fs.floor
    return fs
end
Base.show(io::IO, fs::FloorStepsize) =
    print(io, "FloorStepsize(", fs.inner, "; floor = ", fs.floor,
          isfinite(fs.ceil) ? ", ceil = $(fs.ceil)" : "",
          fs.start_iter > 0 ? ", start_iter = $(fs.start_iter)" : "",
          isfinite(fs.init) ? ", init = $(fs.init)" : "", ")")

# -----------------------------------------------------------------------------
#  Retraction / vector transport choice
# -----------------------------------------------------------------------------
"""
    retraction_pair(kind) -> (retraction_method, vector_transport_method)

`kind = :polar` gives the polar (= metric-projection) retraction on Stiefel,
`kind = :qr` the QR retraction; each is paired with *its own* differentiated
retraction vector transport `DifferentiatedRetractionVectorTransport`.
"""
function retraction_pair(kind::Symbol)
    r = kind === :polar ? PolarRetraction() :
        kind === :qr     ? QRRetraction()    :
        throw(ArgumentError("retraction must be :polar or :qr, got :$kind"))
    return r, DifferentiatedRetractionVectorTransport(r)
end

# -----------------------------------------------------------------------------
#  Problem construction
# -----------------------------------------------------------------------------
"""
    build_problem(; n, p, seed, objective = :brockett)

Return `(M, A, B)` with `M = Stiefel(n, p)` and symmetric `A ∈ ℝⁿˣⁿ`.
`objective = :brockett` (default) draws a random symmetric `B ∈ ℝᵖˣᵖ`, the
general Brockett-type objective `f(X) = 0.5 tr(Xᵀ A X B)`.
`objective = :eigenvalue` sets `B = I(p)`, the plain block eigenvalue /
trace-minimisation problem `f(X) = 0.5 tr(Xᵀ A X)` — `X⋆` then spans the `p`
smallest eigenvectors of `A` but is only unique as a *subspace* (invariant
under `X ↦ X U` for any orthogonal `U`, not just a per-column sign flip); use
`grassmann_distance`, not `iterate_distance`, to measure convergence.
"""
function build_problem(; n::Int, p::Int, seed::Int, objective::Symbol = :brockett)
    rng = MersenneTwister(seed)
    A0 = randn(rng, n, n)
    A = (A0 .+ A0') ./ 2
    B = if objective === :brockett
        B0 = randn(rng, p, p)
        (B0 .+ B0') ./ 2
    elseif objective === :eigenvalue
        Matrix{Float64}(I, p, p)
    else
        throw(ArgumentError("objective must be :brockett or :eigenvalue, got :$objective"))
    end
    return Stiefel(n, p), A, B
end

"""
    perturbed_start(M, Xstar; radius, seed)

Initial guess a geodesic distance ≈ `radius` from the reference solution
`Xstar`: draw a random tangent vector at `Xstar`, scale it to length `radius`,
and retract.
"""
function perturbed_start(M, Xstar; radius::Real, seed::Int,
                         retraction_method = default_retraction_method(M))
    rng = MersenneTwister(seed)
    ξ = project(M, Xstar, randn(rng, size(Xstar)))
    ξ .*= radius / norm(ξ)
    return retract(M, Xstar, ξ, retraction_method)
end

# -----------------------------------------------------------------------------
#  Reference optimum — closed form (see `analytic_minimizer`), LBFGS-verified
# -----------------------------------------------------------------------------
"""
    reference_optimum(M, A, B; verbose, polish, tol) -> (f_star, X_star)

`X_star` is the closed-form global minimiser from `analytic_minimizer`;
`f_star` its cost.  With `polish = true` a short LBFGS run starting from `X_star`
double-checks it (and floating-point-polishes the point); a WARNING is logged if
LBFGS beats the analytic value by more than `tol`.

The LBFGS step size is `greedy_secant.jl`'s raw `GreedySecantLinesearch`
(`initial_stepsize = 1.0`, the natural quasi-Newton trial step), not Manopt's
default `WolfePowellLinesearch` — for some seeds (e.g. `76`) that default
search stalls for a very long time polishing an already near-optimal `Xa`
(the Wolfe–Powell bracketing/bisection apparently struggling right at the
point where gradients are already tiny); the greedy secant step doesn't have
that failure mode and this is a short, already-close-to-optimal polish, not a
run that needs Wolfe-condition-guaranteed LBFGS convergence.
"""
function reference_optimum(M, A, B; verbose = true, polish = true, tol = 1e-6,
                           retraction_method = default_retraction_method(M),
                           vector_transport_method = default_vector_transport_method(M))
    Xa, fa = analytic_minimizer(A, B)
    Xa = project(M, Xa)                       # scrub any O(1e-15) drift off St(n,p)
    _log(@sprintf("  analytic minimiser:  f* = %.12e   ‖XᵀX-I‖ = %.1e",
                  fa, norm(Xa' * Xa - I)))

    polish || return fa, Xa

    os  = ObjectiveStorage(4)
    obj = get_objective(A, B, os)
    dbg = verbose ? [:Iteration, (:Cost, "  f = %.10e"),
                     (:GradientNorm, "  ‖g‖ = %.3e"), 25, "\n", :Stop] : []
    ls = GreedySecantLinesearch(; retraction_method = retraction_method,
                                  vector_transport_method = vector_transport_method,
                                  initial_stepsize = 1.0)
    st = quasi_Newton(M, obj, Xa;
        stepsize = ls,
        retraction_method = retraction_method,
        vector_transport_method = vector_transport_method,
        stopping_criterion = StopAfterIteration(100) |
                             StopWhenGradientNormLess(1e-12) |
                             StopWhenChangeLess(M, 1e-16),
        debug = dbg, return_state = true)
    Xp = get_solver_result(st)
    fp = get_cost(M, obj, Xp)
    _log(@sprintf("  LBFGS-polished:      f  = %.12e   (Δ vs analytic = %+.2e)", fp, fp - fa))
    if fp < fa - tol * max(1.0, abs(fa))
        _log("  WARNING: LBFGS improved on the analytic minimiser by > tol — check analytic_minimizer assumptions")
        return fp, Xp
    end
    return min(fa, fp), (fp <= fa ? Xp : Xa)
end

# -----------------------------------------------------------------------------
#  Riemannian Newton with an inexact Krylov inner solve
# -----------------------------------------------------------------------------
"""
    run_newton(M, obj, hess, X0; max_iters, grad_tol, inner_method, inner_rtol,
               inner_maxiter, verbose)

Globalised Riemannian Newton iteration.  Each step solves the Newton system

    Hess f(Xₖ)[ξ] = -grad f(Xₖ)

with `hessian_solve` (`:cg` or `:minres`, stopping at relative residual
`inner_rtol`), forces `ξ` to be a descent direction (falling back to `-grad`),
then runs an Armijo backtracking line search along the retraction.

Returns `(X, record)` where `record` is a vector of
`(iteration, cost, ‖grad‖, A*X count, ‖X − X⋆‖)` tuples — the same shape
`get_record` produces for the Manopt solvers (with the `RecordApplications` and
`RecordIterateDistance` actions attached).
"""
function run_newton(M, obj, hess, X0, os, X_star; max_iters = 50, grad_tol = 1e-8,
                    inner_method::Symbol = :cg, inner_rtol = 1e-8,
                    inner_maxiter = 200, verbose = true,
                    retraction_method = default_retraction_method(M),
                    dist::Function = iterate_distance)
    X  = copy(X0)
    c  = get_cost(M, obj, X)
    g  = get_gradient(M, obj, X); gn = norm(g)
    record = Tuple{Int,Float64,Float64,Int,Float64}[
        (0, c, gn, os.n_applications, dist(X, X_star))]

    for k in 1:max_iters
        gn <= grad_tol && break

        ξ, it_in, relres, info = hessian_solve(v -> hess(M, X, v), -g;
            method = inner_method, rtol = inner_rtol, maxiter = inner_maxiter)

        slope = dot(ξ, g)
        stalled = info === :maxiter && relres > 0.9      # inner solve made no headway
        if !isfinite(slope) || slope >= 0 || stalled || !all(isfinite, ξ)
            ξ, slope, info = -g, -gn^2, :grad_fallback   # not a usable descent direction
        end

        α, cnew, Xnew = 1.0, c, X
        for _ in 1:30                                     # Armijo backtracking
            Xnew = retract(M, X, α .* ξ, retraction_method)
            cnew = get_cost(M, obj, Xnew)
            cnew <= c + 1e-4 * α * slope && break
            α *= 0.5
        end

        X, c = Xnew, cnew
        g = get_gradient(M, obj, X); gn = norm(g)
        push!(record, (k, c, gn, os.n_applications, dist(X, X_star)))
        verbose && _log(@sprintf(
            "      newton %2d  f = %.10e  ‖g‖ = %.3e  [%s: %d it, relres %.1e, %s, α=%.3g]",
            k, c, gn, inner_method, it_in, relres, info, α))
    end
    return X, record
end

# -----------------------------------------------------------------------------
#  Solver zoo
# -----------------------------------------------------------------------------
"""
    solver_list(rm, vtm)

`Vector` of `(name, run)` pairs.  Every `run(M, obj, hess, X0, os, ctx)` returns
`(X_final, record)` with `record` a vector of
`(iteration, cost, ‖grad‖, A*X count, ‖Xₖ − X⋆‖)` tuples.  `rm` / `vtm` are the
shared retraction and (differentiated-retraction) vector transport — threaded
into every solver, its line search, and the CG coefficient rules.  `ctx` carries
`sc`, `X_star`, `grad_tol` and the Newton inner-solver settings.
"""
function solver_list(rm, vtm)
    # shared strong-Wolfe line search, on the chosen retraction / transport.
    # `stop_when_stepsize_less` defaults to 0.0, so Manopt's final bisection can
    # spin forever on a bad Polak–Ribière direction — give it a real floor.
    ls() = WolfePowellLinesearch(; retraction_method = rm,
                                   vector_transport_method = vtm,
                                   sufficient_decrease = 0.1,
                                   sufficient_curvature = 0.2,
                                   stop_when_stepsize_less = 1e-12,
                                   stop_increasing_at_step = 25,
                                   stop_decreasing_at_step = 50)
    rec(os, X_star) = [:Iteration, :Cost, :GradientNorm,
                       RecordApplications(os), RecordIterateDistance(X_star)]

    # wrap a Manopt first-order solver into the uniform (X_final, record) form
    fo(build) = (M, obj, hess, X0, os, ctx) -> begin
        s = build(M, obj, X0, os, ctx)
        (get_solver_result(s), get_record(s))
    end
    cg(coeff) = fo((M, obj, X0, os, ctx) -> conjugate_gradient_descent(
        M, obj, X0; coefficient = coeff, stepsize = ls(),
        retraction_method = rm, vector_transport_method = vtm,
        stopping_criterion = ctx.sc, record = rec(os, ctx.X_star), return_state = true))

    newton(method) = (M, obj, hess, X0, os, ctx) -> run_newton(
        M, obj, hess, X0, os, ctx.X_star; max_iters = ctx.newton_max_iters,
        grad_tol = ctx.grad_tol, inner_method = method,
        inner_rtol = ctx.inner_rtol, inner_maxiter = ctx.inner_maxiter,
        verbose = ctx.verbose, retraction_method = rm)

    # CG coefficients that transport gradients/directions get the chosen `vtm`
    # (Fletcher–Reeves and Conjugate-Descent use no transport, so take none)
    pr = PolakRibiereCoefficient(;    vector_transport_method = vtm)
    hs = HestenesStiefelCoefficient(; vector_transport_method = vtm)
    dy = DaiYuanCoefficient(;         vector_transport_method = vtm)

    return [
        ("Steepest descent",  fo((M, obj, X0, os, ctx) -> gradient_descent(
            M, obj, X0; stepsize = ls(), retraction_method = rm,
            stopping_criterion = ctx.sc, record = rec(os, ctx.X_star), return_state = true))),
        ("RCG Fletcher-Reeves",  cg(FletcherReevesCoefficient())),
        ("RCG Polak-Ribiere",    cg(pr)),
        ("RCG Hestenes-Stiefel", cg(hs)),
        ("RCG Dai-Yuan",         cg(dy)),
        ("RCG Hybrid2", cg(HybridCoefficient(FletcherReevesCoefficient(), pr))),
        ("RCG Hybrid1", cg(HybridCoefficient(hs, dy))),
        ("Newton-CG",            newton(:cg)),
        ("Newton-MINRES",        newton(:minres)),
    ]
end

# -----------------------------------------------------------------------------
#  Experiment driver
# -----------------------------------------------------------------------------
function run_experiments(; n = 100, p = 5, n_size = 5, seed = 42,
                           max_iters = 400, grad_tol = 1e-8, wall_secs = 30,
                           newton_max_iters = 50, inner_rtol = 1e-4,
                           inner_maxiter = 200, perturb = 5e-1,
                           retraction = :polar,
                           use_differential = true, verbose = true,
                           outdir = @__DIR__)

    rm, vtm = retraction_pair(retraction)
    M, A, B = build_problem(; n, p, seed)
    _log(@sprintf("Problem:  St(%d, %d)   n_size = %d   seed = %d   retraction = :%s",
                  n, p, n_size, seed, retraction))
    _log("Computing reference optimum ...")
    f_star, X_star = reference_optimum(M, A, B; verbose = verbose,
                                       retraction_method = rm, vector_transport_method = vtm)
    _log(@sprintf("reference optimum  f* = %.12g", f_star))

    # all solvers start from the same point, a distance `perturb` off X_star
    X0 = perturbed_start(M, X_star; radius = perturb, seed = seed + 1, retraction_method = rm)
    _log(@sprintf("initial guess: dist ≈ %.2e from X_star,  f(X0) - f* = %.3e\n",
                  perturb, 0.5 * tr(X0' * A * X0 * B) - f_star))

    sc  = StopAfterIteration(max_iters) | StopWhenGradientNormLess(grad_tol) |
          StopWhenChangeLess(M, 1e-13) |        # bail out once a method plateaus
          StopAfter(Second(wall_secs))          # ... or after a wall-clock budget
    ctx = (; sc, grad_tol, verbose, newton_max_iters, inner_rtol, inner_maxiter, X_star)
    sl  = solver_list(rm, vtm)

    # common iteration-0 values (identical X0 for every solver); Manopt records
    # start at iteration 1, so this is prepended to their series for a shared left edge
    obj0 = get_objective(A, B, ObjectiveStorage(2))
    c0   = get_cost(M, obj0, X0)
    gn0  = norm(get_gradient(M, obj0, X0))
    d0   = iterate_distance(X0, X_star)

    results = NamedTuple[]
    for (i, (name, run)) in enumerate(sl)
        os   = ObjectiveStorage(n_size)
        obj  = get_objective(A, B, os; use_differential = use_differential)
        hess = get_hessian(A, B, os)
        _log(@sprintf("[%d/%d] %-22s running ...", i, length(sl), name))
        t = @elapsed ((Xs, rec) = run(M, obj, hess, X0, os, ctx))

        # `rec`: (iteration, cost, gradientnorm, A*X count, ‖X − X⋆‖) tuples
        iters    = [Int(x[1]) for x in rec]
        costs    = [Float64(x[2]) for x in rec]
        gnorms   = [Float64(x[3]) for x in rec]
        work     = [Int(x[4]) for x in rec]           # cumulative A*X applications
        errs     = [Float64(x[5]) for x in rec]       # ‖Xₖ − X⋆‖
        if first(iters) != 0                          # prepend the shared start
            pushfirst!(iters, 0); pushfirst!(costs, c0); pushfirst!(gnorms, gn0)
            pushfirst!(work, 0);  pushfirst!(errs, d0)
        end
        naive    = os.n_applications + os.n_hits

        fin_cost = get_cost(M, obj, Xs)
        push!(results, (; name, iters, costs, gnorms, work, errs,
                          fin_cost = fin_cost,
                          fin_gap  = fin_cost - f_star,
                          fin_gn   = last(gnorms),
                          n_iter   = last(iters),
                          apps     = os.n_applications,
                          hits     = os.n_hits,
                          hvp      = os.n_hvp,
                          naive    = naive,
                          time     = t))
        _log(@sprintf("      %-22s %4d it  %.2fs  f-f* = %.3e  ‖g‖ = %.3e  A*X = %d  hvp = %d  reuse = %d",
                      name, last(iters), t, fin_cost - f_star, last(gnorms),
                      os.n_applications, os.n_hvp, os.n_hits))
    end

    print_table(results)
    make_plots(results, f_star, outdir)
    return results, f_star
end

# -----------------------------------------------------------------------------
#  Reporting
# -----------------------------------------------------------------------------
function print_table(results)
    @printf("\n%-22s %6s %14s %12s %8s %6s %8s %8s %7s %8s\n",
            "method", "iters", "f - f*", "‖grad‖", "A*X", "hvp", "naive",
            "saved%", "reuse", "time[s]")
    println("-"^112)
    for r in results
        fo    = r.apps - r.hvp                 # first-order A*X products (misses)
        saved = 100 * (1 - fo / (fo + r.hits)) # first-order only; hvp is not cacheable
        reuse = (fo + r.hits) / fo
        @printf("%-22s %6d %14.3e %12.3e %8d %6d %8d %7.1f %6.2fx %8.3f\n",
                r.name, r.n_iter, r.fin_gap, r.fin_gn,
                r.apps, r.hvp, r.naive, saved, reuse, r.time)
    end
    println("-"^112)
    println("A*X    : total dense n×(n×p) products (first-order misses + Hessian-vector products)")
    println("hvp    : Hessian-vector products  Hess f(X)[ξ]  (subset of A*X, Newton only, not cacheable)")
    println("naive  : first-order queries served (cost/gradient/differential = A*X count without the cache)")
    println("saved% : fraction of first-order A*X products removed by ObjectiveStorage")
end

# -----------------------------------------------------------------------------
#  Plots
# -----------------------------------------------------------------------------
function make_plots(results, f_star, outdir)
    gap = plot(; xlabel = "A*X applications", ylabel = "f(Xₖ) - f*",
               yscale = :log10, legend = :topright, title = "Cost gap vs. work")
    gn = plot(; xlabel = "A*X applications", ylabel = "‖grad f(Xₖ)‖",
              yscale = :log10, legend = :topright, title = "Gradient norm vs. work")
    err = plot(; xlabel = "iteration", ylabel = "‖Xₖ - X*‖",
               yscale = :log10, legend = :topright, title = "Iterate error vs. iteration")
    for r in results
        g = max.(r.costs .- f_star, eps())
        plot!(gap, r.work, g; label = r.name, lw = 2)
        plot!(gn, r.work, max.(r.gnorms, eps()); label = r.name, lw = 2)
        plot!(err, r.iters, max.(r.errs, eps()); label = r.name, lw = 2)
    end

    names = [r.name for r in results]
    # two bar series side by side (plain Plots.bar, no StatsPlots dependency)
    work = bar(names, [r.naive for r in results];
        label = "naive (no cache)", xrotation = 40, legend = :topright,
        ylabel = "A*X products", title = "Matrix products: cache vs. naive",
        bar_width = 0.6, fillalpha = 0.35)
    bar!(work, names, [r.apps for r in results];
        label = "A*X performed", bar_width = 0.6)

    p1 = joinpath(outdir, "convergence.png")
    p2 = joinpath(outdir, "gradient_norm.png")
    p3 = joinpath(outdir, "matrix_products.png")
    p4 = joinpath(outdir, "iterate_error.png")
    savefig(gap, p1)
    savefig(gn, p2)
    savefig(work, p3)
    savefig(err, p4)
    println("\nsaved plots:\n  ", p1, "\n  ", p2, "\n  ", p3, "\n  ", p4)
end

# -----------------------------------------------------------------------------
if abspath(PROGRAM_FILE) == @__FILE__
    run_experiments()
end
