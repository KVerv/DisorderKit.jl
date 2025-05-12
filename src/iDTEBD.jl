abstract type AbstractAlgorithm end

# Algortihm for computing the density matrix of a disordered system at finite temperatures
struct  iDTEBD <: AbstractAlgorithm
    alg_inversion::AbstractInversionAlgorithm
    alg_trunc_Z::AbstractTruncationAlgorithm
    alg_trunc_disordermpo::AbstractTruncationAlgorithm
    invtol::Float64
    nsteps::Int
    verbosity::Int
    truncfrequency::Int
    timer_output::TimerOutput

    function iDTEBD(alg_inversion::AbstractInversionAlgorithm, alg_trunc_Z::AbstractTruncationAlgorithm, alg_trunc_disordermpo::AbstractTruncationAlgorithm; invtol::Float64 = 1e-8, nsteps::Int = 50, verbosity::Int = 0, truncfrequency::Int = 1,  timer_output::TimerOutput = TimerOutput())
        return new(alg_inversion, alg_trunc_Z, alg_trunc_disordermpo; invtol, nsteps, verbosity, truncfrequency, timer_output)
    end
end

function evolve_densitymatrix(Ts::DisorderMPO, ps::Vector{<:Real}, alg::iDTEBD; ρ0 = nothing)
    ρs = isnothing(ρ0) ? deepcopy(Ts) : ρ0
    χ = alg.alg_inversion.inverse_dim
    mpoZinv = nothing
    for ix in 1:alg.nsteps
        (alg.verbosity > 0) && (@info "Iteration $ix)")
        (alg.verbosity > 1) && (@info(crayon"magenta"("Before normalization: Bonddimension of ρ = $(dim(space(ρs[1])[1]))")))
        (alg.verbosity > 1) && (@info(crayon"magenta"("Using Z⁻¹ bonddimension of χ = $(χ)")))
        @timeit alg.timer_output "normalize_each_disorder_sector" begin
            ρ_normalized, ϵ_acc, mpoZinv = normalize_each_disorder_sector(ρs, alg.alg_trunc_Z, alg.alg_inversion; init_guess = mpoZinv, verbosity = alg.verbosity)
        end
        while ϵ_acc > alg.invtol
            χ *= 2
            (alg.verbosity > 1) && (@info(crayon"magenta"("Using Z⁻¹ bonddimension of χ = $(χ)")))
            @timeit alg.timer_output "normalize_each_disorder_sector" begin
                ρ_normalized, ϵ_acc, mpoZinv = normalize_each_disorder_sector(ρs, alg.alg_trunc_Z, alg.alg_inversion; init_guess = mpoZinv, verbosity = alg.verbosity)
            end
        end
        if mod(ix, alg.truncfrequency) == 0
            (alg.verbosity > 0) && (@info(crayon"magenta"("Truncating ρ")))
            (alg.verbosity > 1) && (@info(crayon"magenta"("Before truncation: Bonddimension of ρ = $(dim(space(ρs_normalized[1])[1]))")))
            @timeit alg.timer_output "truncate_disorder_MPO" ρs = truncate_disorder_MPO(ρ_normalized, ps, alg.alg_trunc_disordermpo)
            (alg.verbosity > 1) && (@info(crayon"magenta"("After truncation: Bonddimension of ρ = $(dim(space(ρs[1])[1]))")))
        end
        (alg.verbosity > 0) && (@info(crayon"magenta"("Evolve")))
        @timeit alg.timer_output "evolve_one_time_step" ρs = evolve_one_time_step(ρs, Ts)
    end
    return ρs
end

# evolve one time step
function evolve_one_time_step(ρ::DisorderMPO, T::DisorderMPO)
    return ρ * T
end