# Check the SRC of DisorderKit (src/FiniteSRC.jl) against the exact result: the partition √P(w) Z(w) of Us ψ and the
# compressed state f ⊙ (Us ψ), the latter also with a warm start (previous_widths) as used by the ITE driver.
using TensorKit, MPSKit, DisorderKit, Random, Statistics, LinearAlgebra

# Model and problem (as in examples/FiniteNormalizedITE.jl)
J₀, δ, W = 1.0, 0.0, 0.1
hs = exp(δ) * [1., exp(-2W)]
Js = J₀ * [1., exp(-2W)]
ps = [0.25, 0.25, 0.25, 0.25]
Hs = DisorderKit.random_transverse_field_ising(Js, hs)

L = 6
Δτ = 0.02
D = 16          # rank cap of the partition and the compressed state
rtol_Z = 1e-10  # partition rtol
rtol_ψ = 1e-11  # state rtol

Random.seed!(1234)
ψ = DisorderKit.FiniteDisorderMPS(Hs, ps, L, 2, 8)
Us = DisorderKit.time_evolution_MPO(Hs, Δτ, ψ.sectors; N = 2)
F, _ = DisorderKit.normalization_filter(ψ, Us; max_rank = D)

# Exact references
Uψ = Us * ψ
ψ_exact = F * Uψ
words = DisorderKit.assessment_words(ψ.ps, 10^6)               # all 4^(L-1)·2 words (site L only carries h)

# log|√P(w) Z(w)| of a ScalarPartition: exp(log_scale) cores[1][:, w₁, :] ⋯ cores[L][:, w_L, :]
function log_partition(Z::DisorderKit.ScalarPartition, w)
    v = ones(ComplexF64, 1, 1)
    s = Z.log_scale
    for (i, u) in enumerate(w)
        v = v * convert(Array, Z.cores[i])[:, u, :]
        m = maximum(abs, v)
        m == 0 && return -Inf
        v ./= m
        s += log(m)
    end
    return log(abs(only(v))) + s
end

# log √P(w) Z(w) of Us ψ, exact
exact_log_partition(w) = 0.5*sum(log(ψ.ps[i][u]) for (i, u) in enumerate(w)) + DisorderKit.conditional_log_norm(Uψ, w)

# Relative P-weighted distance ‖a - b‖/‖b‖ with ⟨a|b⟩ = Σ_w P(w) ⟨a_w|b_w⟩
function relative_distance(a::DisorderKit.FiniteDisorderMPS, b::DisorderKit.FiniteDisorderMPS)
    aa, bb, ab = DisorderKit.overlap(a, a), DisorderKit.overlap(b, b), DisorderKit.overlap(a, b)
    return sqrt(max(real(aa + bb - 2real(ab)), 0.0)/real(bb))
end

# Run f() once as warm-up and once timed; returns (result, seconds), or (nothing, message) if it throws
function timed(f)
    try
        f()
        t = @elapsed r = f()
        return r, t
    catch e
        msg = e isa ErrorException ? e.msg : sprint(showerror, e)
        return nothing, msg
    end
end

println("="^80, "\nPartition √P(w) Z(w) of Us ψ, L = $L, max_rank = $D, rtol = $rtol_Z\n", "="^80)
exact = exact_log_partition.(words)
out, t = timed(() -> DisorderKit.src_partition(ψ, Us; max_rank = D, rtol = rtol_Z, rng = Xoshiro(1)))
if isnothing(out)
    println("failed: ", t)
else
    Z, info = out
    err = abs.(log_partition.(Ref(Z), words) .- exact)
    println("time = ", round(t; sigdigits = 3), " s",
            ", bonds = ", DisorderKit.bond_dimensions(Z),
            ", target met = ", info.target_met,
            ", |Δ log √P Z|: max = ", round(maximum(err); sigdigits = 3), ", rms = ", round(sqrt(mean(abs2, err)); sigdigits = 3))
end

println("="^80, "\nState f ⊙ (Us ψ), maxD = $D, rtol = $rtol_ψ\n", "="^80)
cold, _ = timed(() -> DisorderKit.src_compress(ψ, Us, F; maxD = D, rtol = rtol_ψ, rng = Xoshiro(2)))
runs = isnothing(cold) ? [] : [("cold start", () -> DisorderKit.src_compress(ψ, Us, F; maxD = D, rtol = rtol_ψ, rng = Xoshiro(2))),
                               ("warm start", () -> DisorderKit.src_compress(ψ, Us, F; maxD = D, rtol = rtol_ψ, rng = Xoshiro(3),
                                                                            previous_widths = cold[2].sketch_widths))]
for (name, f) in runs
    local out, t, ψc, info, res
    out, t = timed(f)
    if isnothing(out)
        println(rpad(name, 11), " failed: ", t)
        continue
    end
    ψc, info = out
    res = DisorderKit.normalization_residuals(ψc, words)
    println(rpad(name, 11), " time = ", round(t; sigdigits = 3), " s",
            ", sketch widths = ", info.sketch_widths,
            ", bonds before/after rounding = ", info.pre_round_ranks, " / ", DisorderKit.bond_dimensions(ψc),
            ", ‖ψ - ψ_exact‖/‖ψ_exact‖ = ", round(relative_distance(ψc, ψ_exact); sigdigits = 3),
            ", max |log Z(w)| = ", round(maximum(abs, res); sigdigits = 3))
end
println("exact state: max |log Z(w)| = ", maximum(abs, DisorderKit.normalization_residuals(ψ_exact, words)))
