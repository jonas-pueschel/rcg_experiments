# rcg_experiments

Numerical experiments for **Riemannian optimization** methods (via
[Manopt.jl](https://github.com/JuliaManifolds/Manopt.jl)) on a Brockett-type
objective over the Stiefel manifold. The focus is Riemannian conjugate gradient
(RCG) with different parameter setups.

## Running the experiments

Local project environment; standard deps: `Manopt`, `Manifolds`, `Plots`
(+ stdlib `LinearAlgebra`, `Random`, `Printf`).

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'   # first time
julia --project=. experiments.jl                      # run experiments
```

It runs a total of three experiments:
- **`experiment_cg_params`** tests different CG parameters for two stepsizes (HZ and Armijo)
- **`experiment_cg_stepsize`** tests different step sizes for FR-PRP param
- **`experiment_retractions`** tests different retractions for RCG, RGD and Newton
The results get saved under `/data/` in `*.dat` files and a `*_preview.png` preview of the plot gets generated.