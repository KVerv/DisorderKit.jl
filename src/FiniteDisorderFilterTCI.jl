# Disorder normalization filter of a FiniteDisorderMPS, using TensorCrossInterpolation.jl.
# Same target as normalization_filter in FiniteDisorderFilter.jl, f(w) ≈ Z(w)^(-1/2) from the exact conditional norms,
# but the tensor train of f is obtained with TensorCrossInterpolation.crossinterpolate2 instead of the hand-written
# nested cross. The struct FiniteDisorderFilter, F * ψ and the residual diagnostics are reused unchanged.

# Normalization filter f(w) ≈ Z(w)^(-1/2) of ψ from TCI. Returns the filter, its bond dimensions and whether the
# final (relative) TCI error met the tolerance.
function normalization_filter_tci(ψ::FiniteDisorderMPS; tolerance::Real = 1e-8, max_rank::Int = 32, maxiter::Int = 20,
                                  nsamples::Int = 32, rng::AbstractRNG = Random.default_rng(), to::TimerOutput = TimerOutput())
    @timeit to "dense_tensors" A = _dense(ψ)
    L, d = length(A), length(A[1])
    localdims = fill(d, L)
    # Common log shift σ so that exp(-½ log Z(w) - σ) stays in floating point range, restored in log_scale
    σ = maximum(-0.5 * conditional_log_norm(A, rand(rng, 1:d, L)) for _ in 1:nsamples)
    f = TensorCrossInterpolation.CachedFunction{Float64}(w -> exp(-0.5 * conditional_log_norm(A, w) - σ), localdims)
    @timeit to "crossinterpolate2" tci, _, errors = TensorCrossInterpolation.crossinterpolate2(
        Float64, f, localdims; tolerance = Float64(tolerance), maxbonddim = max_rank, maxiter = maxiter)
    opp = map(TensorCrossInterpolation.sitetensors(tci)) do t
        [TensorMap(complex(t[:, p, :]), ℂ^size(t, 1), ℂ^size(t, 3)) for p in 1:d]
    end
    return FiniteDisorderFilter(opp, σ), TensorCrossInterpolation.linkdims(tci), last(errors) <= tolerance
end
