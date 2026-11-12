using Test, TensorKit, DisorderKit, LinearAlgebra, Random, StatsBase

# ---- Dense reference: the conditional state ψ_w as a vector (site 1 fastest) and on-site operators on the full space

function dense_state(ψ, word)
    As = [ComplexF64.(convert(Array, ψ[i, p])) for (i, p) in enumerate(word)]
    v = As[1]
    for A in As[2:end]
        @tensor w[a, s, t, b] := v[a, s, c] * A[c, t, b]
        v = reshape(w, size(w, 1), size(w, 2) * size(w, 3), size(w, 4))
    end
    return vec(v[1, :, 1])
end
site_op(O, i, d, L) = kron(Matrix(1.0I, d^(L-i), d^(L-i)), O, Matrix(1.0I, d^(i-1), d^(i-1)))
labels(q, L) = [sum(q[I[i]] for i in 1:L) for I in CartesianIndices(ntuple(_ -> length(q), L))][:]

# ψ_w normalized and, with charges q, projected onto its most likely total charge
function reference_state(ψ, word, q)
    v = dense_state(ψ, word)
    L = length(word)
    if !isnothing(q)
        N = labels(q, L)
        pN = Dict(n => sum(abs2, v[N .== n]) for n in unique(N))
        v = v .* (N .== argmax(n -> pN[n], collect(keys(pN))))
    end
    return v / LinearAlgebra.norm(v)
end
expect(v, M) = dot(v, M * v)
function entropies(v, d, L)
    map(1:L-1) do k
        λ = svdvals(reshape(v, d^k, d^(L-k))) .^ 2
        λ = λ[λ .> 1e-14]
        -sum(λ .* log.(λ))
    end
end

# ---- Small test states

# Bose-Hubbard without symmetry, from a product state that mixes particle numbers: the conditional states are
# superpositions of N, so projection matters
function mixed_bh_state(; L = 4, nmax = 3)
    Hs = DisorderKit.random_bose_hubbard([0.3], [1.0], [0.3, 0.7]; nmax)
    v = normalize(ComplexF64[1, 1, 0.7, 0.2])
    W = TensorMap(reshape(v, 1, nmax + 1, 1), ℂ^1 ⊗ ℂ^(nmax + 1) ← ℂ^1)
    site_ps, sectors = DisorderKit.site_sectors(Hs, [0.5, 0.5], L)
    ψ0 = DisorderKit.FiniteDisorderMPS([[W for _ in p] for p in site_ps], site_ps, sectors)
    alg = DisorderKit.Groundstate_FiniteNormalizedITE(; maxD = 8, order = 2, maxiter = 5, convtol = 1e-12)
    ψ, _, _ = DisorderKit.groundstate(ψ0, Hs, 0.02, alg)
    n = TensorMap(ComplexF64.(diagm(0:nmax)), ℂ^(nmax + 1) ← ℂ^(nmax + 1))
    x = TensorMap(ComplexF64.(diagm(1 => sqrt.(1:nmax)) + diagm(-1 => sqrt.(1:nmax))), ℂ^(nmax + 1) ← ℂ^(nmax + 1))   # a + a†
    return ψ, Hs, n, x, collect(0:nmax)
end

# U(1)-symmetric Bose-Hubbard at filling 1 from the Mott state
function u1_bh_state(; L = 4, nmax = 3)
    Hs = DisorderKit.random_bose_hubbard([0.3], [1.0], [-0.2, 0.2]; nmax, symmetry = U1Irrep, filling = 1)
    alg = DisorderKit.Groundstate_FiniteNormalizedITE(; maxD = 8, order = 2, maxiter = 5, convtol = 1e-12)
    ψ, _, _ = DisorderKit.groundstate(DisorderKit.mott_state(Hs, [0.5, 0.5], L; filling = 1), Hs, 0.02, alg)
    P = space(Hs.Ds[1], 1)
    pP = DisorderKit.basis_order(P, [U1Irrep(k - 1) for k in 0:nmax])
    n = TensorMap(ComplexF64.(diagm(0:nmax))[pP, pP], P ← P)
    return ψ, Hs, n, collect(0:nmax)[pP]
end

all_words(ψ) = [collect(w) for w in Iterators.product(eachindex.(ψ.ps)...)]

@testset "DisorderKit" begin
    L = 4
    ψm, _, n_m, x_m, q_m = mixed_bh_state(; L)
    ψu, _, n_u, q_u = u1_bh_state(; L)
    blocks = [a:b for a in 1:L for b in a:L]

    @testset "conditional observables vs dense ED" begin
        for (ψ, ops, q, proj) in ((ψm, (n_m, x_m), q_m, nothing), (ψm, (n_m,), q_m, q_m), (ψu, (n_u,), q_u, nothing))
            d = length(q)
            for word in all_words(ψ)[1:5]
                v = reference_state(ψ, word, proj)
                for O in ops
                    M = ComplexF64.(convert(Array, O))
                    ref = [real(expect(v, site_op(M, i, d, L))) for i in 1:L]
                    @test DisorderKit.conditional_expectation_value(ψ, word, O; charges = proj) ≈ ref atol = 1e-10
                    F = DisorderKit.conditional_block_fluctuations(ψ, word, O, blocks; charges = proj)
                    Fref = map(blocks) do b
                        X = sum(site_op(M, i, d, L) for i in b)
                        real(expect(v, X * X) - expect(v, X)^2)
                    end
                    @test F ≈ Fref atol = 1e-10
                    # blocks with a common left edge share one sweep: the same as one block at a time
                    @test F ≈ [only(DisorderKit.conditional_block_fluctuations(ψ, word, O, [b]; charges = proj)) for b in blocks] atol = 1e-12
                end
                @test DisorderKit.conditional_entanglement_entropy(ψ, word; charges = proj) ≈ entropies(v, d, L) atol = 1e-9
            end
        end
        # projection needs a charge conserving operator
        @test_throws ArgumentError DisorderKit.conditional_expectation_value(ψm, all_words(ψm)[1], x_m; charges = q_m)
    end

    @testset "exact disorder averages vs enumeration" begin
        words = all_words(ψm)
        P(w) = prod(ψm.ps[i][p] for (i, p) in enumerate(w))
        vs = [dense_state(ψm, w) for w in words]
        Zw = [sum(abs2, v) for v in vs]                 # the averages weight every word by its norm Z(w)
        weight = [P(w) * Z for (w, Z) in zip(words, Zw)]
        avg(f) = sum(weight[k] * f(vs[k] / sqrt(Zw[k])) for k in eachindex(words)) / sum(weight)
        d = length(q_m)
        M = ComplexF64.(convert(Array, x_m))
        @test DisorderKit.expectation_value(ψm, x_m) ≈ [real(avg(v -> expect(v, site_op(M, i, d, L)))) for i in 1:L] atol = 1e-10
        for (i, j) in ((1, 2), (1, 4), (2, 4))
            ref = real(avg(v -> expect(v, site_op(M, i, d, L) * site_op(M, j, d, L))))
            @test DisorderKit.correlator(ψm, x_m, x_m, i, j)[end] ≈ ref atol = 1e-10
        end
    end

    @testset "disorder_average" begin
        f(w) = DisorderKit.conditional_expectation_value(ψm, w, n_m)
        words = all_words(ψm)
        P(w) = prod(ψm.ps[i][p] for (i, p) in enumerate(w))
        m, err, v = DisorderKit.disorder_average(f, ψm; n_samples = 1000)   # all words enumerated
        mref = sum(P(w) .* f(w) for w in words)
        @test m ≈ mref atol = 1e-12
        @test all(iszero, err)
        @test v ≈ sum(P(w) .* f(w) .^ 2 for w in words) .- mref .^ 2 atol = 1e-12
        # sampled: a different estimate of the same mean
        ms, errs, _ = DisorderKit.disorder_average(f, ψm; n_samples = length(words) - 1, rng = Xoshiro(7))
        @test all(errs .> 0)
        @test all(abs.(ms .- mref) .<= 5 .* errs .+ 1e-12)
        # scalar quantities work as well
        s, s_err, _ = DisorderKit.disorder_average(w -> sum(f(w)), ψm)
        @test s ≈ sum(mref) && s_err == 0
    end

    @testset "normalized ITE" begin
        # Random transverse-field Ising chain against its exact free-fermion solution
        W = 0.8
        Js, hs, ps = [1.0, exp(-2W)], [1.0, exp(-2W)], fill(0.25, 4)
        Hs = DisorderKit.random_transverse_field_ising(Js, hs)
        Lr = 4
        alg = DisorderKit.Groundstate_FiniteNormalizedITE(; maxD = 8, order = 4, maxiter = 200, convtol = 1e-10)
        ψ, _, info = DisorderKit.groundstate(DisorderKit.plus_state(Hs, ps, Lr), Hs, 0.05, alg)
        ff = DisorderKit.sample_RTFIM(Lr, Js, hs; ps, n_samples = 10^5)
        @test ff.exact
        @test DisorderKit.energy_density(ψ, Hs) ≈ ff.E atol = 1e-5
        @test info.residual_max[end] < 1e-2   # normalization kept to ~1e-3 at cap 8
        # U(1) Bose-Hubbard: the particle number stays exactly L
        @test sum(DisorderKit.expectation_value(ψu, n_u)) ≈ L atol = 1e-8
    end

    @testset "luttinger_parameter" begin
        Lb = 10
        bl = [((Lb - l) ÷ 2 + 1):((Lb - l) ÷ 2 + l) for l in 1:8]
        F = [0.484032, 0.648079, 0.710608, 0.735269, 0.741968, 0.743108, 0.686986, 0.630427]
        K, c, X, fit = DisorderKit.luttinger_parameter(F, bl, Lb)
        @test fit == 2:6
        @test K ≈ 1.5879 atol = 1e-4
        @test isnan(first(DisorderKit.luttinger_parameter(F[1:2], bl[1:2], Lb)))   # fewer than 2 blocks qualify
    end
end
