# Algorithm for computing the groundstate of a disordered system through imaginary time evolution
abstract type AbstractAlgorithm end

struct GroundstateInfo
    ϵsconv::Vector{Float64}
    ϵsent::Vector{Float64}
    ϵsz::Vector{Float64}
    ρ_trunc_err::Vector{Float64}
    R_trunc_err::Vector{Float64}
    ϵsA1::Vector{Float64}
    ϵsA2::Vector{Float64}

    function GroundstateInfo(maxiter::Int)
        return new(Vector{Float64}(undef, maxiter), Vector{Float64}(undef, maxiter), Vector{Float64}(undef, maxiter), Vector{Float64}(undef, maxiter), Vector{Float64}(undef, maxiter), Vector{Float64}(undef, maxiter), Vector{Float64}(undef, maxiter))
    end
end

struct Groundstate_iDTEBD <: AbstractAlgorithm
    trunc_method_state::MatrixAlgebraKit.TruncationStrategy
    trunc_method_norm::MatrixAlgebraKit.TruncationStrategy
    convtol::Float64
    maxiter::Int
    verbosity::Int
    timer_output::TimerOutput
    finalizer::Finalizer

    function Groundstate_iDTEBD(trunc_method::MatrixAlgebraKit.TruncationStrategy, trunc_method_norm::MatrixAlgebraKit.TruncationStrategy; convtol::Float64 = 1e-8, maxiter::Int = 100, verbosity::Int = 0, timer_output::TimerOutput = TimerOutput(), finalizer::Finalizer = default_Finalizer)
        return new(trunc_method, trunc_method_norm, convtol, maxiter, verbosity, timer_output, finalizer)
    end
end

function groundstate(ρ0::InfiniteDisorderDensityMatrix, Hs::DisorderMPOHam, dτ::Float64, alg::Groundstate_iDTEBD)
    data = Vector{alg.finalizer.E}()
    info = GroundstateInfo(alg.maxiter)

    ϵ_conv = 1.0
    
    @timeit alg.timer_output "Copy" ρs = deepcopy(ρ0)

    ix = 0
    ϵ_conv = 1.0
    @timeit alg.timer_output "Copy" ρprev = deepcopy(ρs)
    @timeit alg.timer_output "energy_density" Eprev = DisorderKit.energy_density(ρs, Hs)
    while (ϵ_conv > alg.convtol) && (ix+1 <= alg.maxiter)
        ρs = gauge(ρs)
        ix += 1
        (alg.verbosity > 0) && (@info "Iteration $ix")
        (alg.verbosity > 0) && (@info(crayon"cyan"("Constructing time evolution operator")))
        @timeit alg.timer_output "construct_time_evolution_operator" begin
            Us = time_evolution_MPO(Hs, dτ/2; N = 1)
            R, ϵR, ϵA1, ϵA2 = construct_renormalisation(ρs, Hs, dτ/2, alg.trunc_method_norm)
            info.R_trunc_err[ix] = ϵR
            info.ϵsA1[ix] = ϵA1
            info.ϵsA2[ix] = ϵA2
        end

        (alg.verbosity > 0) && (@info(crayon"cyan"("Evolve")))
        @timeit alg.timer_output "evolve_one_time_step" ρs_normalized = ρs * Us * R

        (alg.verbosity > 0) && (@info(crayon"magenta"("Truncating ρ")))
        (alg.verbosity > 1) && (@info(crayon"magenta"("Before truncation: Bonddimension of ρ = $(dim(space(ρs_normalized[1],1)))")))
        @timeit alg.timer_output "truncate_disorder_MPO" ρs, ϵρ = truncate(ρs_normalized, alg.trunc_method_state)
        info.ρ_trunc_err[ix] = ϵρ
        (alg.verbosity > 1) && (@info(crayon"magenta"("After truncation: Bonddimension of ρ = $(dim(space(ρs[1],1)))")))
        
        @timeit alg.timer_output "gauge" ρs = gauge(ρs)
        @timeit alg.timer_output "Compute error" es, ϵz = entanglement_spectrum_norm(ρs)

        info.ϵsz[ix] = ϵz
        ϵent = sum(es[1:end-1])
        info.ϵsent[ix] = ϵent
        (alg.verbosity > 1) && (@info(crayon"light_blue"("Max. error after normalization: ϵ₁ = $(ϵent), N2 = $(ϵz)")))

        @timeit alg.timer_output "energy_density" E = DisorderKit.energy_density(ρs, Hs)
        ϵ_conv = (abs(Eprev - E)/dτ)
        Eprev = E
        info.ϵsconv[ix] = ϵ_conv
        (alg.verbosity > 0) && (@info(crayon"light_blue"("Convergence error: ϵ_conv = $(ϵ_conv)")))

        (alg.verbosity > 0) && (@info(crayon"cyan"("Finalize")))
        @timeit alg.timer_output "finalizer" push!(data, alg.finalizer.f!(ρs, Hs))

        @timeit alg.timer_output "Copy" ρprev = deepcopy(ρs)
    end
    return ρs, data, info
end
