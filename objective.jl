# =============================================================================
#  objective.jl
#
#  Brockett-type objective on the Stiefel manifold St(n, p):
#
#        f(X) = 0.5 * tr( Xᵀ A X B ),      X ∈ St(n, p),  A = Aᵀ ∈ ℝⁿˣⁿ,  B = Bᵀ ∈ ℝᵖˣᵖ
#
#  Euclidean gradient:      ∇f(X) = A X B                        (n×p)
#  Riemannian gradient:     grad f(X) = ∇f(X) - X sym(Xᵀ ∇f(X))  (projection onto TₓSt)
#  Differential:            Df(X)[V] = ⟨∇f(X), V⟩
#  Riemannian Hessian:      Hess f(X)[ξ] = Pₓ( A ξ B - ξ sym(Xᵀ A X B) )
#  Hessian spectrum @ X⋆:   closed form, see `riemannian_hessian_eigenvalues`
#
#  The expensive operation is the dense product  A * X  (n×n · n×p).  The Manopt
#  solvers evaluate cost / gradient / differential repeatedly at the *same* point
#  (line search trial points, stopping criterion, conjugacy update, ...).  The
#  `ObjectiveStorage` below caches the intermediate results of the last `n_size`
#  distinct iterates so that these repeated queries reuse a single `A * X`.
#
#  Design mirrors
#  https://github.com/jonas-pueschel/MetricFreeTest/blob/main/weighted_objective.jl
#  but without a weighted metric: the Riemannian gradient here is the plain
#  Euclidean-metric projection onto the tangent space of the Stiefel manifold.
# =============================================================================

using LinearAlgebra
using Printf
using Manopt

_sym(M) = (M .+ M') ./ 2

"""
    ObjectiveStorage(n_size)

Ring buffer holding the intermediate results of the last `n_size` *distinct*
iterates of the Stiefel Brockett objective `f(X) = 0.5 tr(Xᵀ A X B)`.

Most recent iterate is stored at index `1`.

# Fields
- `n_size::Int`          number of iterates kept
- `X_hist`               the iterates
- `cost_hist`            `f(X)`
- `AX_hist`              `A * X`   — the expensive n×n · n×p product
- `XtAX_hist`            `Xᵀ A X`  (p×p)
- `egrad_hist`           Euclidean gradient `A X B`  (n×p)
- `rgrad_hist`           Riemannian gradient, projected onto `TₓSt(n, p)`
- `n_applications::Int`  number of dense `A * (n×p)` products performed
                         (cost/gradient misses *and* Hessian-vector products)
- `n_hits::Int`          number of first-order queries served from the cache
- `n_hvp::Int`           number of Hessian-vector products `Hess f(X)[ξ]`
                         (each costs one fresh `A * ξ`; also counted in
                         `n_applications`)
"""
mutable struct ObjectiveStorage
    n_size::Int
    X_hist::Vector{Any}
    cost_hist::Vector{Any}
    AX_hist::Vector{Any}
    XtAX_hist::Vector{Any}
    egrad_hist::Vector{Any}
    rgrad_hist::Vector{Any}
    n_applications::Int
    n_hits::Int
    n_hvp::Int
    function ObjectiveStorage(n_size::Int)
        empty() = Any[nothing for _ in 1:n_size]
        return new(n_size, empty(), empty(), empty(), empty(), empty(), empty(), 0, 0, 0)
    end
end

"""
    reset!(os::ObjectiveStorage)

Clear the history and the counters, so the same storage object can be reused for
a fresh solver run.
"""
function reset!(os::ObjectiveStorage)
    for v in (os.X_hist, os.cost_hist, os.AX_hist, os.XtAX_hist, os.egrad_hist, os.rgrad_hist)
        fill!(v, nothing)
    end
    os.n_applications = 0
    os.n_hits = 0
    os.n_hvp = 0
    return os
end

# -- internal: locate an iterate in the history -------------------------------
function _find(X, os::ObjectiveStorage)
    for i in eachindex(os.X_hist)
        Xi = os.X_hist[i]
        Xi === nothing && continue
        (Xi === X || Xi == X) && return i
    end
    return 0
end

# -- internal: make sure the results for `X` are available, return their index -
function _update!(X, A, B, os::ObjectiveStorage)
    idx = _find(X, os)
    if idx != 0
        os.n_hits += 1
        return idx
    end

    os.n_applications += 1
    AX    = A * X                       # n×n · n×p   (expensive)
    XtAX  = X' * AX                     # p×p
    egrad = AX * B                      # n×p
    rgrad = egrad .- X * _sym(X' * egrad)
    cost  = 0.5 * dot(XtAX, B)          # 0.5 * tr(Xᵀ A X B), B symmetric

    # shift history down by one, newest goes to index 1
    for i in os.n_size:-1:2
        os.X_hist[i]     = os.X_hist[i - 1]
        os.cost_hist[i]  = os.cost_hist[i - 1]
        os.AX_hist[i]    = os.AX_hist[i - 1]
        os.XtAX_hist[i]  = os.XtAX_hist[i - 1]
        os.egrad_hist[i] = os.egrad_hist[i - 1]
        os.rgrad_hist[i] = os.rgrad_hist[i - 1]
    end
    os.X_hist[1]     = copy(X)
    os.cost_hist[1]  = cost
    os.AX_hist[1]    = AX
    os.XtAX_hist[1]  = XtAX
    os.egrad_hist[1] = egrad
    os.rgrad_hist[1] = rgrad
    return 1
end

"""
    get_objective(A, B, os::ObjectiveStorage; use_differential = true)

Build a `ManifoldFirstOrderObjective` for `f(X) = 0.5 tr(Xᵀ A X B)` on the
Stiefel manifold whose cost, (Riemannian) gradient and differential all go
through the shared `ObjectiveStorage` `os`.

`A` and `B` must be symmetric.  With `use_differential = true` the objective also
exposes the differential `Df(X)[V] = ⟨A X B, V⟩`, which lets Wolfe-type line
searches reuse the cached `A * X` instead of triggering a new one.
"""
function get_objective(A::AbstractMatrix, B::AbstractMatrix, os::ObjectiveStorage;
                       use_differential::Bool = true)

    cost_fun(M, X)    = os.cost_hist[_update!(X, A, B, os)]
    grad_fun(M, X)    = copy(os.rgrad_hist[_update!(X, A, B, os)])
    diff_fun(M, X, V) = dot(V, os.egrad_hist[_update!(X, A, B, os)])

    if use_differential
        return ManifoldFirstOrderObjective(;
            cost = cost_fun, gradient = grad_fun, differential = diff_fun)
    else
        return ManifoldFirstOrderObjective(; cost = cost_fun, gradient = grad_fun)
    end
end

# =============================================================================
#  Closed-form global minimiser
# =============================================================================
"""
    analytic_minimizer(A, B) -> (X⋆, f⋆)

Closed-form **global** minimiser of `f(X) = 0.5 tr(Xᵀ A X B)` over `St(n, p)`,
valid for *arbitrary symmetric* `A` (n×n) and `B` (p×p) — `A` and `B` need not be
positive (semi)definite.

Derivation. Diagonalise `A = Q_A diag(a) Q_Aᵀ` (a₁ ≤ … ≤ aₙ) and
`B = Q_B diag(β) Q_Bᵀ` (β₁ ≤ … ≤ βₚ). With `W = Q_Aᵀ X Q_B` (still Stiefel),

    tr(Xᵀ A X B) = Σ_{i,k} aᵢ βₖ Wᵢₖ² ,   P := (Wᵢₖ²)  has  colsums = 1, rowsums ≤ 1.

`P` ranges over a Birkhoff-type polytope whose vertices are partial permutation
matrices, and the objective is linear in `P`, so the minimum is attained at a
vertex — i.e. an injective assignment `k ↦ τ(k)` minimising `Σₖ a_{τ(k)} βₖ`.
An exchange argument fixes that assignment: with `m = #{k : βₖ < 0}`,

    chosen A-eigenvalues  â = (a₁, …, a_{p−m}, a_{n−m+1}, …, aₙ)   (p−m smallest + m largest)
    optimal pairing        βₖ  ↔  â_{p−k+1}                        (full reversal)
    X⋆ = E_A · J · Q_Bᵀ ,   J = anti-identity,  E_A = Q_A[:, idx(â)]
    f⋆ = 0.5 Σₖ â_{p−k+1} βₖ  =  0.5 Σₖ (â sorted ↓)ₖ (β sorted ↑)ₖ .

For `B ⪰ 0` (`m = 0`) this is exactly "columns of `X` span the `p` smallest
eigenvectors of `A`"; for indefinite `B` the `m` most-negative modes of `B` pair
with the `m` *largest* eigenvalues of `A` instead.

With `verbose = true` the analytic spectrum / condition number of the Riemannian
Hessian at `X⋆` is printed (see [`riemannian_hessian_eigenvalues`](@ref)).
"""
function analytic_minimizer(A::AbstractMatrix, B::AbstractMatrix; verbose::Bool = true)
    n = size(A, 1)
    p = size(B, 1)
    p <= n || throw(ArgumentError("need p ≤ n, got p=$p, n=$n"))

    Ea = eigen(Symmetric(Matrix(A)))          # ascending eigenvalues / vectors
    Eb = eigen(Symmetric(Matrix(B)))
    a  = Ea.values
    β  = Eb.values

    m   = count(<(0), β)                      # negative eigenvalues of B
    idx = vcat(1:(p - m), (n - m + 1):n)      # p−m smallest + m largest of A
    EA  = Ea.vectors[:, idx]                  # n×p,  EAᵀ A EA = diag(â)
    â   = a[idx]
    J   = reverse(Matrix{Float64}(I, p, p); dims = 2)

    X    = EA * J * Eb.vectors'
    fval = 0.5 * sum(â[p - k + 1] * β[k] for k in 1:p)

    if verbose
        λ = _riemannian_hessian_eigenvalues(a, β, n, p, m, idx)
        λmin, λmax = extrema(λ)
        κ = λmin > 0 ? λmax / λmin : Inf
        @printf(stderr,
            "  [analytic_minimizer] Riemannian Hessian @ X⋆:  %d eigvals   λmin = %.6e   λmax = %.6e   cond = %.6e\n",
            length(λ), λmin, λmax, κ)
        flush(stderr)
    end
    return X, fval
end

"""
    riemannian_hessian_eigenvalues(A, B) -> Vector{Float64}   (sorted ascending)

Closed-form spectrum of the Riemannian Hessian of `f(X) = 0.5 tr(Xᵀ A X B)` on
`St(n, p)` at the global minimiser `X⋆` (embedded / Euclidean metric).

Let `a₁ ≤ … ≤ aₙ` be the eigenvalues of `A`, `β₁ ≤ … ≤ βₚ` those of `B`,
`m = #{k : βₖ < 0}`, `idx = {1,…,p−m} ∪ {n−m+1,…,n}` the `A`-modes selected by
`analytic_minimizer`, and `ǎ = reverse(a[idx])` (so `ǎ₁ ≥ … ≥ ǎₚ`).  Working in
the `A`/`B` eigenbases the Hessian is diagonal, with eigenvalues

* `βₖ (aᵢ − ǎₖ)`               for `k = 1..p`, `i ∉ idx`   — `p(n−p)` "off-subspace" modes
* `½ (ǎ_l − ǎₖ)(βₖ − β_l)`     for `1 ≤ l < k ≤ p`         — `p(p−1)/2`  "in-subspace" (skew) modes

(`p(n−p) + p(p−1)/2 = np − p(p+1)/2 = dim T_{X⋆} St(n, p)`).  All values are
`≥ 0` (as they must be at a minimiser); a `0` appears only when `B` is singular
or `A` / `B` has repeated eigenvalues, in which case the Hessian is singular.
Verified against the dense Hessian to machine precision.
"""
function riemannian_hessian_eigenvalues(A::AbstractMatrix, B::AbstractMatrix)
    n = size(A, 1)
    p = size(B, 1)
    p <= n || throw(ArgumentError("need p ≤ n, got p=$p, n=$n"))
    a = eigen(Symmetric(Matrix(A))).values
    β = eigen(Symmetric(Matrix(B))).values
    m = count(<(0), β)
    idx = vcat(1:(p - m), (n - m + 1):n)
    return _riemannian_hessian_eigenvalues(a, β, n, p, m, idx)
end

# internal: spectrum from already-computed ascending eigenvalues `a`, `β`
function _riemannian_hessian_eigenvalues(a, β, n, p, m, idx)
    ǎ    = reverse(a[idx])                    # ǎ_k = a_{idx[p−k+1]}, descending
    rest = setdiff(1:n, idx)                  # the n−p unselected A-modes
    λ = Vector{Float64}(undef, p * length(rest) + p * (p - 1) ÷ 2)
    t = 0
    for k in 1:p, i in rest
        λ[t += 1] = β[k] * (a[i] - ǎ[k])
    end
    for l in 1:p, k in (l + 1):p
        λ[t += 1] = 0.5 * (ǎ[l] - ǎ[k]) * (β[k] - β[l])
    end
    return sort!(λ)
end

# =============================================================================
#  Riemannian Hessian
# =============================================================================
"""
    get_hessian(A, B, os::ObjectiveStorage)

Return `hess(M, X, ξ)` evaluating the Riemannian Hessian of
`f(X) = 0.5 tr(Xᵀ A X B)` on the Stiefel manifold (embedded, Euclidean metric):

    Hess f(X)[ξ] = Pₓ( A ξ B - ξ · sym(Xᵀ A X B) ),   Pₓ Z = Z - X sym(Xᵀ Z).

The point-dependent factor `sym(Xᵀ A X B)` reuses the cached `Xᵀ A X` from `os`
(no extra `A * X`).  Each call performs exactly one fresh `A * ξ` product, which
is counted in both `os.n_hvp` and `os.n_applications`.
"""
function get_hessian(A::AbstractMatrix, B::AbstractMatrix, os::ObjectiveStorage)
    function hess(M, X, ξ)
        idx = _find(X, os)
        idx == 0 && (idx = _update!(X, A, B, os))
        S = _sym(os.XtAX_hist[idx] * B)          # sym(Xᵀ A X B), p×p
        os.n_applications += 1
        os.n_hvp += 1
        Y = (A * ξ) * B .- ξ * S                 # Euclidean Hessian minus curvature term
        return Y .- X * _sym(X' * Y)             # project onto TₓSt(n, p)
    end
    return hess
end

# =============================================================================
#  Krylov solvers for the Newton system   Hess f(X)[ξ] = -grad f(X)
#
#  Both operate directly on tangent vectors (n×p matrices); the inner product is
#  the Frobenius product `dot`.  `applyH` is the linear operator ξ ↦ Hess f(X)[ξ]
#  and `b = -grad f(X)`.  Iteration stops at the *relative* residual
#  `‖r_k‖ ≤ rtol · ‖b‖`.
# =============================================================================

"""
    hessian_solve(applyH, b; method = :cg, rtol = 1e-4, maxiter = 100)

Approximately solve `applyH(ξ) = b` for the tangent vector `ξ`.
Returns `(ξ, iters, relres, info)` where `info` is `:converged`, `:maxiter` or
`:negative_curvature` (`:cg` only).

- `:cg`     – conjugate gradients with a non-positive-curvature guard
              (Newton–CG); on the first step it falls back to `b` itself.
- `:minres` – minimum-residual (handles an indefinite / singular Hessian).
"""
function hessian_solve(applyH, b; method::Symbol = :cg, rtol::Real = 1e-4,
                       maxiter::Int = 100)
    method === :cg     && return _hs_cg(applyH, b; rtol, maxiter)
    method === :minres && return _hs_minres(applyH, b; rtol, maxiter)
    error("hessian_solve: unknown method $method (use :cg or :minres)")
end

function _hs_cg(applyH, b; rtol, maxiter)
    x  = zero(b)
    r  = copy(b)
    p  = copy(b)
    rs = dot(r, r)
    nb = sqrt(rs)
    nb == 0 && return x, 0, 0.0, :converged
    for k in 1:maxiter
        Hp  = applyH(p)
        pHp = dot(p, Hp)
        if pHp <= 1e-12 * dot(p, p)                 # non-positive curvature
            return (k == 1 ? copy(b) : x), k, sqrt(rs) / nb, :negative_curvature
        end
        α  = rs / pHp
        x  = x .+ α .* p
        r  = r .- α .* Hp
        rs_new = dot(r, r)
        sqrt(rs_new) <= rtol * nb &&
            return x, k, sqrt(rs_new) / nb, :converged
        p  = r .+ (rs_new / rs) .* p
        rs = rs_new
    end
    return x, maxiter, sqrt(rs) / nb, :maxiter
end

# MINRES without preconditioning / shift, matrix-free.  Faithful transcription of
# the Paige & Saunders recurrence (cf. `scipy.sparse.linalg.minres`), with the
# Frobenius product `dot` as inner product so it runs on n×p tangent vectors.
function _hs_minres(applyH, b; rtol, maxiter)
    x  = zero(b)
    y  = copy(b)
    r1 = copy(b)
    β1 = sqrt(dot(b, y))
    β1 == 0 && return x, 0, 0.0, :converged

    oldb  = 0.0
    β     = β1
    dbar  = 0.0
    epsln = 0.0
    ϕbar  = β1
    cs, sn = -1.0, 0.0
    w, w2 = zero(b), zero(b)
    r2    = copy(r1)
    rnorm = β1

    for k in 1:maxiter
        s = 1.0 / β
        v = s .* y

        y = applyH(v)
        k >= 2 && (y = y .- (β / oldb) .* r1)
        alfa = dot(v, y)
        y  = y .- (alfa / β) .* r2
        r1 = r2
        r2 = y
        oldb = β
        β    = sqrt(dot(r2, r2))

        # apply previous rotation
        oldeps = epsln
        delta  = cs * dbar + sn * alfa
        gbar   = sn * dbar - cs * alfa
        epsln  =              sn * β
        dbar   =            - cs * β

        # new plane rotation
        γ = sqrt(gbar^2 + β^2)
        γ = max(γ, 1e-300)
        cs = gbar / γ
        sn = β / γ
        ϕ    = cs * ϕbar
        ϕbar = sn * ϕbar

        # update x
        denom = 1.0 / γ
        w1 = w2
        w2 = w
        w  = (v .- oldeps .* w1 .- delta .* w2) .* denom
        x  = x .+ ϕ .* w

        rnorm = ϕbar
        rnorm <= rtol * β1 && return x, k, rnorm / β1, :converged
        β <= 1e-300 && return x, k, rnorm / β1, :converged
    end
    return x, maxiter, rnorm / β1, :maxiter
end
