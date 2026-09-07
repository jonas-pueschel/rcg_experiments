# =============================================================================
#  greedy_secant.jl
#
#  A lightweight, greedy secant step size — *no* line search, no backtracking,
#  no safeguard.  Every iteration takes exactly one secant step on the
#  directional derivative  ϕ'(t) = ⟨grad f(R_p(tη)), 𝒯η⟩  between t = 0 (slope
#  ϕ'(0)) and t = b (slope ϕ'(b)), where b is the *previous accepted step* (or,
#  on the first call / after a degenerate secant, a prescribed
#  `initial_stepsize`):
#
#      α = −ϕ'(0) · b / (ϕ'(b) − ϕ'(0)).
#
#  One extra `A·X` per iteration (the probe at t = b), nothing else — the same
#  "raw spectral step" philosophy as `spectral_stepsizes.jl`'s Barzilai–Borwein
#  / norm-ratio rules (no cost evaluation, no bracketing), just anchored on ϕ'
#  instead of consecutive iterate/gradient differences.  Falls back to
#  `initial_stepsize` if the secant is degenerate (ϕ'(b) ≤ ϕ'(0), i.e.
#  non-increasing slope) or the search direction isn't a descent direction
#  (ϕ'(0) ≥ 0).  Nothing here guarantees descent — cap the outer iteration
#  count.
#
#  NB: a variant that accepts `b` outright whenever it already satisfies a
#  strong-curvature test (skipping the secant step) was tried and measured
#  *worse* — more total iterations/A·X, not fewer, because reusing a stale `b`
#  under-refines the step for RCG's fast-changing conjugate direction (see
#  CLAUDE.md). This file intentionally always takes the secant step.
#
#  `include` after `main.jl` (needs Manopt / Manifolds in scope).
# =============================================================================

using Manopt

mutable struct GreedySecantLinesearchStepsize{
        TRM <: AbstractRetractionMethod, VTM <: AbstractVectorTransportMethod, F <: Real,
    } <: Manopt.Linesearch
    retraction_method::TRM
    vector_transport_method::VTM
    initial_stepsize::F
    last_stepsize::F
    min_stepsize::F
    max_stepsize::F
    sufficient_curvature::F
end

function GreedySecantLinesearchStepsize(
        M::AbstractManifold;
        retraction_method = default_retraction_method(M),
        vector_transport_method = default_vector_transport_method(M),
        initial_stepsize::Real = 1.0,
        min_stepsize::Real = 1.0e-10,
        max_stepsize::Real = 1.0e2,
        sufficient_curvature::Real = 0.0
    )
    F = float(promote_type(typeof(initial_stepsize), typeof(min_stepsize), typeof(max_stepsize)))
    return GreedySecantLinesearchStepsize{
        typeof(retraction_method), typeof(vector_transport_method), F,
    }(
        retraction_method, vector_transport_method,
        F(initial_stepsize), F(initial_stepsize), F(min_stepsize), F(max_stepsize), F(sufficient_curvature)
    )
end
GreedySecantLinesearchStepsize(M::AbstractManifold, ::Any; kwargs...) =
    GreedySecantLinesearchStepsize(M; kwargs...)

function (ls::GreedySecantLinesearchStepsize)(
        mp::AbstractManoptProblem, s::AbstractManoptSolverState, ::Int,
        η = -get_gradient(mp, get_iterate(s)); kwargs...,
    )
    M   = get_manifold(mp)
    p   = get_iterate(s)
    dϕ0 = get_differential(mp, p, η)
    dϕ0 < 0 || return zero(ls.last_stepsize)             # not a descent direction

    b  = ls.last_stepsize > 0 ? ls.last_stepsize : ls.initial_stepsize
    b  = clamp(b, ls.min_stepsize, ls.max_stepsize)
    q  = retract(M, p, b .* η, ls.retraction_method)
    ηq = vector_transport_to(M, p, η, q, ls.vector_transport_method)
    dϕb = get_differential(mp, q, ηq)                    # one A·X, the only probe

    # greedier strategy
    abs(dϕb) ≤ ls.sufficient_curvature * abs(dϕ0) && return b

    # secant root of ϕ' through (0, dϕ0) and (b, dϕb)
    t = dϕb > dϕ0 ? -dϕ0 * b / (dϕb - dϕ0) : NaN
    (isfinite(t) && t > 0) || (t = ls.initial_stepsize)  # degenerate -> fall back
    t = clamp(t, ls.min_stepsize, ls.max_stepsize)

    ls.last_stepsize = t
    return t
end

Manopt.get_last_stepsize(ls::GreedySecantLinesearchStepsize, ::Any...) = ls.last_stepsize
Manopt.get_initial_stepsize(ls::GreedySecantLinesearchStepsize) = ls.initial_stepsize
Manopt.initialize_stepsize!(ls::GreedySecantLinesearchStepsize) =
    (ls.last_stepsize = ls.initial_stepsize; ls)
Base.show(io::IO, ls::GreedySecantLinesearchStepsize) = print(io,
    "GreedySecantLinesearchStepsize(; initial_stepsize = ", ls.initial_stepsize, ")")

"""
    GreedySecantLinesearch(; kwargs...)
    GreedySecantLinesearch(M::AbstractManifold; kwargs...)

Factory for [`GreedySecantLinesearchStepsize`](@ref) — a raw, greedy secant step
size: every call takes exactly one secant step on `ϕ'` between `0` and the
previous accepted step (or `initial_stepsize` on the first call / after a
degenerate secant), with no bracketing, backtracking or curvature safeguard.
Keywords: `initial_stepsize = 1.0`, `min_stepsize = 1e-10`,
`max_stepsize = 1e2` (numerical-hygiene clamp only), plus `retraction_method` /
`vector_transport_method`.
"""
GreedySecantLinesearch(args...; kwargs...) =
    Manopt.ManifoldDefaultsFactory(GreedySecantLinesearchStepsize, args...; kwargs...)
