# Groundstate of a FiniteDisorderMPS through disorder-normalized imaginary time evolution.
# Every step works on the separate layers A (state) and W (time evolution MPO), Us ψ is never formed:
# (1) compress the prior-weighted partition function √P(w) Z(w) of Us ψ with SRC (rank ≤ cap D_max),
# (2) construct a normalization filter f(w) ≈ |Z(w)|^(-1/2) from a nested cross of the compressed partition,
# (3) compress f ⊙ (Us ψ) with SRC in the prior-weighted joint metric from the layers A, W, f and round to the cap.
# Every conditional state ψ_w stays normalized, the tensors are not isometric.
# The cap of the partition, the filter and the state follows a stage schedule; the state is carried across stages.

# Product state |+x⟩ on every site and for every disorder sector
function plus_state(ps::Vector{Float64}, L::Int)
    W = TensorMap(ComplexF64[1, 1]/sqrt(2), ℂ^1 ⊗ ℂ^2, ℂ^1)
    return FiniteDisorderMPS([[W for _ in ps] for _ in 1:L], ps)
end

bond_dimensions(ψ::FiniteDisorderMPS) = [dim(space(ψ[i,1], 3)) for i in 1:length(ψ)-1]

struct Groundstate_FiniteNormalizedITE <: AbstractAlgorithm
    maxD::Vector{Pair{Int, Int}}    # stage schedule (number of steps => cap D_max), the last cap is kept afterwards
    order::Int
    partition_rtol::Float64
    cross_width::Int
    cross_cycles::Int
    filter_tail_tol::Float64
    src_rtol::Float64
    src_sketch_cap::Int             # upper bound of the state sketch width per cut
    src_oversampling::Int           # state sketch width per cut ≤ 2 D_max + src_oversampling
    src_rank_threshold::Float64
    round_tol::Float64
    n_assess::Int
    post_warn::Float64
    seed::Int
    convtol::Float64
    maxiter::Int
    verbosity::Int
    timer_output::TimerOutput
    finalizer::Finalizer

    # maxD: a single cap, or a stage schedule such as [100 => 8, 200 => 16, 200 => 32]
    function Groundstate_FiniteNormalizedITE(; maxD::Union{Int, Vector{Pair{Int, Int}}} = 16, order::Int = 4,
                                             partition_rtol::Float64 = 1e-10, cross_width::Int = 32, cross_cycles::Int = 2,
                                             filter_tail_tol::Float64 = 1e-8, src_rtol::Float64 = 1e-11, src_sketch_cap::Int = 128,
                                             src_oversampling::Int = 10,
                                             src_rank_threshold::Float64 = 1e-13, round_tol::Float64 = 1e-10, n_assess::Int = 128,
                                             post_warn::Float64 = 1e-3, seed::Int = 20266001, convtol::Float64 = 1e-8, maxiter::Int = 100,
                                             verbosity::Int = 0, timer_output::TimerOutput = TimerOutput(),
                                             finalizer::Finalizer = Finalizer(Float64, energy_density))
        schedule = maxD isa Int ? [1 => maxD] : maxD
        (!isempty(schedule) && all(s -> s.first >= 1 && s.second >= 1, schedule)) || throw(ArgumentError("invalid cap schedule"))
        return new(schedule, order, partition_rtol, cross_width, cross_cycles, filter_tail_tol, src_rtol, src_sketch_cap,
                   src_oversampling, src_rank_threshold, round_tol, n_assess, post_warn, seed, convtol, maxiter, verbosity, timer_output, finalizer)
    end
end

# Cap of step ix according to the stage schedule
function cap_at(alg::Groundstate_FiniteNormalizedITE, ix::Int)
    last = 0
    for (steps, cap) in alg.maxD
        last += steps
        ix <= last && return cap
    end
    return alg.maxD[end].second
end

# Per step diagnostics of Groundstate_FiniteNormalizedITE
struct FiniteNormalizedITEInfo
    Es::Vector{Float64}                     # energy density
    ϵsconv::Vector{Float64}                 # |ΔE|/dτ
    caps::Vector{Int}                       # cap of the step
    partition_ranks::Vector{Vector{Int}}    # bond dimensions of the compressed partition function
    partition_met::Vector{Bool}             # every partition cut met partition_rtol
    cross_max_phase::Vector{Float64}        # maximal |arg Z(w)| of the queried partition entries
    cross_negative_real::Vector{Int}        # number of queried partition entries with Re Z(w) < 0
    filter_ranks::Vector{Vector{Int}}       # bond dimensions of the filter
    filter_met::Vector{Bool}                # every filter cut met filter_tail_tol
    src_ranks::Vector{Vector{Int}}          # bond dimensions after SRC, before rounding
    src_cap_unmet::Vector{Int}              # number of cuts where the sketch cap was reached without meeting src_rtol
    bond_dimensions::Vector{Vector{Int}}    # bond dimensions of ψ after rounding
    rounding_errors::Vector{Float64}        # relative rounding error
    residual_rms::Vector{Float64}           # RMS of log|Z(w)| of ψ over the assessment words
    residual_max::Vector{Float64}           # maximal |log|Z(w)|| of ψ over the assessment words
end

FiniteNormalizedITEInfo() = FiniteNormalizedITEInfo(Float64[], Float64[], Int[], Vector{Int}[], Bool[], Float64[], Int[],
                                                    Vector{Int}[], Bool[], Vector{Int}[], Int[], Vector{Int}[], Float64[], Float64[], Float64[])

# Assessment words: every word if there are at most n of them, otherwise n words drawn from the disorder distribution
function assessment_words(ps::Vector{<:Real}, L::Int, n::Int; rng::AbstractRNG = Random.default_rng())
    if float(length(ps))^L <= n
        return [collect(w) for w in vec(collect(Iterators.product(fill(eachindex(ps), L)...)))]
    end
    return [sample(rng, eachindex(ps), Weights(ps), L) for _ in 1:n]
end

function groundstate(ψ0::FiniteDisorderMPS, Hs::DisorderMPOHam, dτ::Float64, alg::Groundstate_FiniteNormalizedITE)
    data = Vector{alg.finalizer.E}()
    info = FiniteNormalizedITEInfo()
    L = length(ψ0)
    to = alg.timer_output

    @timeit to "construct_time_evolution_operator" Us = time_evolution_MPO(Hs, dτ, L; N = alg.order)

    ψ = ψ0
    widths, previous_cap = nothing, nothing
    ix = 0
    ϵ_conv = 1.0
    @timeit to "energy_density" Eprev = energy_density(ψ, Hs)
    while (ϵ_conv > alg.convtol) && (ix+1 <= alg.maxiter)
        ix += 1
        cap = cap_at(alg, ix)
        (alg.verbosity > 0) && (@info "Iteration $ix, cap = $cap")
        # Independent random streams for every role and step
        rng(role) = Xoshiro(hash((alg.seed, L, ix, role)))

        (alg.verbosity > 0) && (@info(crayon"cyan"("Partition function")))
        @timeit to "src_partition" Z, zinfo = src_partition(ψ, Us; max_rank = cap, rtol = alg.partition_rtol, rank_threshold = alg.src_rank_threshold, rng = rng(:partition), to = to)
        (alg.verbosity > 1) && (@info(crayon"cyan"("Partition bond dimensions = $(bond_dimensions(Z)), target met = $(zinfo.target_met)")))

        (alg.verbosity > 0) && (@info(crayon"cyan"("Normalization filter")))
        @timeit to "normalization_filter" F, finfo = normalization_filter(Z; width = alg.cross_width, cycles = alg.cross_cycles, max_rank = cap, tail_tol = alg.filter_tail_tol, rng = rng(:cross), to = to)
        (alg.verbosity > 1) && (@info(crayon"cyan"("Filter bond dimensions = $(finfo.filter_ranks), tail tolerance met = $(finfo.filter_met), max |arg Z| = $(finfo.cross.max_phase)")))

        (alg.verbosity > 0) && (@info(crayon"magenta"("Compress")))
        # Warm start from the previous sketch widths only within a stage
        previous_widths = (cap == previous_cap) ? widths : nothing
        @timeit to "src_compress" ψ, src = src_compress(ψ, Us, F; maxD = cap, rtol = alg.src_rtol, sketch_cap = min(alg.src_sketch_cap, 2cap + alg.src_oversampling), rank_threshold = alg.src_rank_threshold, round_tol = alg.round_tol, previous_widths = previous_widths, rng = rng(:physical), to = to)
        widths, previous_cap = src.sketch_widths, cap
        (alg.verbosity > 1) && (@info(crayon"magenta"("Bond dimensions: before rounding = $(src.pre_round_ranks), after rounding = $(bond_dimensions(ψ)), rounding error = $(src.rounding_error)")))

        @timeit to "assess_normalization" residuals = normalization_residuals(ψ, assessment_words(ψ.ps, L, alg.n_assess; rng = rng(:post)))
        rms, rmax = sqrt(mean(abs2, residuals)), maximum(abs, residuals)
        (alg.verbosity > 1) && (@info(crayon"magenta"("Normalization residuals: rms = $(rms), max = $(rmax)")))
        (rmax > alg.post_warn) && (@warn "Step $ix: maximal normalization residual $(rmax) exceeds $(alg.post_warn)")

        @timeit to "energy_density" E = energy_density(ψ, Hs)
        ϵ_conv = abs(Eprev - E)/dτ
        Eprev = E
        (alg.verbosity > 0) && (@info(crayon"light_blue"("Convergence error: (E, dE) = ($(E), $(ϵ_conv))")))

        push!(info.Es, E)
        push!(info.ϵsconv, ϵ_conv)
        push!(info.caps, cap)
        push!(info.partition_ranks, bond_dimensions(Z))
        push!(info.partition_met, zinfo.target_met)
        push!(info.cross_max_phase, finfo.cross.max_phase)
        push!(info.cross_negative_real, finfo.cross.negative_real)
        push!(info.filter_ranks, finfo.filter_ranks)
        push!(info.filter_met, finfo.filter_met)
        push!(info.src_ranks, src.pre_round_ranks)
        push!(info.src_cap_unmet, src.cap_unmet)
        push!(info.bond_dimensions, bond_dimensions(ψ))
        push!(info.rounding_errors, src.rounding_error)
        push!(info.residual_rms, rms)
        push!(info.residual_max, rmax)

        (alg.verbosity > 0) && (@info(crayon"cyan"("Finalize")))
        @timeit to "finalizer" push!(data, alg.finalizer.f!(ψ, Hs))
    end
    return ψ, data, info
end

# Dense layers of the state and the filter with the sector as last index: A[a, t, b, u], f[x, y, u]
_stack(Ts) = cat([ComplexF64.(convert(Array, T)) for T in Ts]...; dims = numind(first(Ts))+1)
_dense_layers(ψ::FiniteDisorderMPS) = [_stack(ψ[i]) for i in 1:length(ψ)]
_dense_layers(F::FiniteDisorderFilter) = [_stack(F[i]) for i in 1:length(F)]

# Raw energy density of the filtered state, Σ_w P(w) f(w)² ⟨ψ_w|H_w|ψ_w⟩ / L, and its norm Σ_w P(w) f(w)² Z(w),
# contracted with the layers A, f and H kept separate. The energy is not divided by the norm.
# The coupling of bond (i, i+1) is set by the sector of site i, as in energy.
function filter_normalized_energy_density(ψ::FiniteDisorderMPS, Hs::DisorderMPOHam, F::FiniteDisorderFilter)
    L = length(ψ)
    Hc = outgoing_coupling(Hs)
    A = _dense_layers(ψ)
    f = _dense_layers(F)
    D = [ComplexF64.(convert(Array, x)) for x in Hc.Ds]     # D[out, in]
    C = [ComplexF64.(convert(Array, x)) for x in Hc.Cs]     # C[out, in, v]
    B = ComplexF64.(convert(Array, Hc.Bs[1]))               # B[v, out, in]
    nv = size(B, 1)

    # Transfer of a layered environment l[a, x, c, z] (ket A, ket f, bra A, bra f), ket A possibly with an operator applied
    transfer(l, Ak, fu, Au) = @tensor out[b, y, d, w] := (((l[a, x, c, z] * Ak[a, s, b]) * fu[x, y]) * conj(Au[c, s, d])) * conj(fu[z, w])
    transfer_open(l, CA, fu, Au) = @tensor out[b, y, d, w, v] := (((l[a, x, c, z] * CA[a, s, b, v]) * fu[x, y]) * conj(Au[c, s, d])) * conj(fu[z, w])
    transfer_close(l, BA, fu, Au) = @tensor out[b, y, d, w] := (((l[a, x, c, z, v] * BA[v, a, s, b]) * fu[x, y]) * conj(Au[c, s, d])) * conj(fu[z, w])

    lN = ones(ComplexF64, 1, 1, 1, 1)
    lE = zeros(ComplexF64, 1, 1, 1, 1)
    lC = zeros(ComplexF64, 1, 1, 1, 1, nv)
    log_scale = 2*F.log_scale
    for i in 1:L
        nb, ny = size(A[i], 3), size(f[i], 2)
        N′, E′, C′ = zeros(ComplexF64, nb, ny, nb, ny), zeros(ComplexF64, nb, ny, nb, ny), zeros(ComplexF64, nb, ny, nb, ny, nv)
        for (u, p) in enumerate(ψ.ps)
            Au, fu = A[i][:, :, :, u], f[i][:, :, u]
            @tensor DA[a, s, b] := D[u][s, t] * Au[a, t, b]
            N′ .+= p .* transfer(lN, Au, fu, Au)
            E′ .+= p .* (transfer(lE, Au, fu, Au) .+ transfer(lN, DA, fu, Au))
            if i > 1
                @tensor BA[v, a, s, b] := B[v, s, t] * Au[a, t, b]
                E′ .+= p .* transfer_close(lC, BA, fu, Au)
            end
            if i < L
                @tensor CA[a, s, b, v] := C[u][s, t, v] * Au[a, t, b]
                C′ .+= p .* transfer_open(lN, CA, fu, Au)
            end
        end
        m = max(maximum(abs, N′), maximum(abs, E′), maximum(abs, C′))
        (isfinite(m) && m > 0) || throw(DomainError(m, "unusable filtered environment at site $i"))
        lN, lE, lC = N′ ./ m, E′ ./ m, C′ ./ m
        log_scale += log(m)
    end
    E, N = only(lE)*exp(log_scale), only(lN)*exp(log_scale)
    abs(imag(E)) <= 1e-8*max(abs(E), 1.0) || @warn("Energy has imaginary part: E = $E")
    return real(E)/L, real(N)
end

# Finite028 post-processing: an additional normalization filter of ψ (partition SRC of ψ with rank cap max_rank,
# cross, reconstruction), then the raw filter-normalized energy density. Returns (E/L, norm, filter info)
function filter_normalized_energy_density(ψ::FiniteDisorderMPS, Hs::DisorderMPOHam; max_rank::Int = max(2, maximum(bond_dimensions(ψ); init = 1)),
                                          partition_rtol::Real = 1e-10, width::Int = 32, cycles::Int = 2, tail_tol::Real = 1e-8,
                                          rng::AbstractRNG = Xoshiro(20262803), to::TimerOutput = TimerOutput())
    F, finfo = normalization_filter(ψ; max_rank = max_rank, partition_rtol = partition_rtol, width = width, cycles = cycles,
                                    tail_tol = tail_tol, rng = rng, to = to)
    E, N = filter_normalized_energy_density(ψ, Hs, F)
    return E, N, finfo
end
