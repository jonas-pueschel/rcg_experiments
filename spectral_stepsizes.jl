# =============================================================================
#  spectral_stepsizes.jl
#
#  Raw spectral / Barzilai–Borwein step sizes for Manopt — *no* line search, *no*
#  backtracking, *no* nonmonotone safeguard.  For use with `gradient_descent`.
#
#  With the previous / current iterate and gradient, and the ambient (Frobenius)
#  finite differences
#
#      s_k = X_k − X_{k-1},        y_k = grad f(X_k) − grad f(X_{k-1}),
#
#  the step is one of
#
#      :bb_direct    α = ⟨s,s⟩ / ⟨s,y⟩              (long BB)
#      :bb_inverse   α = ⟨s,y⟩ / ⟨y,y⟩              (short BB)
#      :norm_ratio   α = ‖s‖ / ‖y‖ = √(⟨s,s⟩/⟨y,y⟩)  (geometric mean of the two)
#      :alternating  long on odd iterations, short on even — as Manopt's
#                    `NonmonotoneLinesearch(strategy = :alternating)`
#
#  clamped to `[min_stepsize, max_stepsize]` for numerical hygiene only.  If
#  ⟨s,y⟩ ≤ 0 the BB ratios fall back to `max_stepsize`.  There is no cost
#  evaluation and nothing that guarantees descent — the iteration can and does
#  diverge on ill-conditioned problems, so cap the outer iteration count.
#
#  NB: `NonmonotoneLinesearchStepsize` in the registered Manopt v0.6.6 (the
#  version pinned here) is internally inconsistent — its `:alternating` branch
#  uses the standard long step `⟨s,s⟩/⟨s,y⟩` on odd iterations, but its
#  non-alternating `:direct` branch computes `⟨y,y⟩/⟨s,y⟩` instead
#  (stepsizes.jl:1678-1685 vs. :1669-1673) — not the same formula.  Confirmed
#  fixed on Manopt's unreleased `master` (retargeted as `BarzilaiBorweinStepsize`,
#  `Project.toml` version 0.6.7 — not yet tagged/registered as of writing); no
#  action needed here since `:bb_direct` below already uses the correct,
#  standard (master / alternating-branch) formula.
#
#  `include` after `main.jl` (needs Manopt / Manifolds in scope).
# =============================================================================

using Manopt
using LinearAlgebra: dot

mutable struct SpectralStepsize{F <: Real} <: Manopt.Stepsize
    rule::Symbol                              # :bb_direct | :bb_inverse | :norm_ratio | :alternating
    initial_stepsize::F
    min_stepsize::F
    max_stepsize::F
    last_stepsize::F
    p_old::Union{Nothing, Matrix{Float64}}
    g_old::Union{Nothing, Matrix{Float64}}   # gradient at p_old
end

function SpectralStepsize(
        M::AbstractManifold;
        rule::Symbol = :norm_ratio,
        initial_stepsize::Real = 1.0e-2,
        min_stepsize::Real = 1.0e-10,
        max_stepsize::Real = 1.0e2,
    )
    rule in (:bb_direct, :bb_inverse, :norm_ratio, :alternating) || throw(ArgumentError(
        "SpectralStepsize rule must be :bb_direct, :bb_inverse, :norm_ratio or :alternating, got :$rule"))
    F = float(promote_type(typeof(initial_stepsize), typeof(min_stepsize), typeof(max_stepsize)))
    return SpectralStepsize{F}(rule, F(initial_stepsize), F(min_stepsize), F(max_stepsize),
                               F(initial_stepsize), nothing, nothing)
end
SpectralStepsize(M::AbstractManifold, ::Any; kwargs...) = SpectralStepsize(M; kwargs...)

function (ss::SpectralStepsize)(
        mp::AbstractManoptProblem, s::AbstractManoptSolverState, k::Int, args...;
        gradient = nothing, kwargs...,
    )
    p = get_iterate(s)
    g = gradient === nothing ? get_gradient(mp, p) : gradient

    if ss.p_old === nothing                        # first step: no history yet
        ss.p_old = copy(p)
        ss.g_old = copy(g)
        ss.last_stepsize = ss.initial_stepsize
        return ss.last_stepsize
    end

    sk  = p .- ss.p_old
    yk  = g .- ss.g_old
    sss = dot(sk, sk)
    sy  = dot(sk, yk)
    yy  = dot(yk, yk)
    long  = sy > 0 ? sss / sy : ss.max_stepsize    # ⟨s,s⟩/⟨s,y⟩
    short = sy > 0 ? sy / yy  : ss.max_stepsize    # ⟨s,y⟩/⟨y,y⟩
    α = if ss.rule === :norm_ratio
        yy > 0 ? sqrt(sss / yy) : ss.max_stepsize
    elseif ss.rule === :bb_direct
        long
    elseif ss.rule === :bb_inverse
        short
    else # :alternating — long on odd k, short on even k
        isodd(k) ? long : short
    end
    isfinite(α) || (α = ss.initial_stepsize)
    α = clamp(α, ss.min_stepsize, ss.max_stepsize)

    ss.p_old .= p
    ss.g_old .= g
    ss.last_stepsize = α
    return α
end

Manopt.get_last_stepsize(ss::SpectralStepsize, ::Any...) = ss.last_stepsize
Manopt.get_initial_stepsize(ss::SpectralStepsize) = ss.initial_stepsize
function Manopt.initialize_stepsize!(ss::SpectralStepsize)
    ss.p_old = nothing
    ss.g_old = nothing
    ss.last_stepsize = ss.initial_stepsize
    return ss
end
Base.show(io::IO, ss::SpectralStepsize) = print(io,
    "SpectralStepsize(:", ss.rule, "; initial_stepsize = ", ss.initial_stepsize,
    ", clamp = [", ss.min_stepsize, ", ", ss.max_stepsize, "])")

"""
    SpectralLength(; rule = :norm_ratio, kwargs...)
    SpectralLength(M::AbstractManifold; kwargs...)

Factory for [`SpectralStepsize`](@ref) — a raw (no line-search) spectral / BB step
size.  Keywords: `rule ∈ {:bb_direct, :bb_inverse, :norm_ratio, :alternating}`,
`initial_stepsize = 1e-2` (used for the first step, before any history),
`min_stepsize = 1e-10`, `max_stepsize = 1e2` (numerical-hygiene clamp only).
"""
SpectralLength(args...; kwargs...) =
    Manopt.ManifoldDefaultsFactory(SpectralStepsize, args...; kwargs...)
