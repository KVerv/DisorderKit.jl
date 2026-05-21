abstract type AbstractAlgorithm end

# Algortihm for computing the density matrix of a disordered system at finite temperatures
struct FiniteTemperature_iDTEBD <: AbstractAlgorithm
    trunc_method::MatrixAlgebraKit.TruncationStrategy
    momenttol::Float64
    verbosity::Int
    timer_output::TimerOutput
    finalizer::Finalizer

    function FiniteTemperature_iDTEBD(trunc_method::MatrixAlgebraKit.TruncationStrategy; momenttol::Float64 = 1e-6, verbosity::Int = 0, timer_output::TimerOutput = TimerOutput(), finalizer::Finalizer = default_Finalizer)
        return new(trunc_method, momenttol, verbosity, timer_output, finalizer)
    end
end

# βspan contains the values of β at which the density matrix is evaluated. βspan[1] is the value of β at which the initial density matrix is evaluated.
function evolve_densitymatrix(ρ0::InfiniteDisorderDensityMatrix, Hs::DisorderMPOHam, βspan::AbstractVector{<:Number}, alg::FiniteTemperature_iDTEBD)
    data = Vector{alg.finalizer.E}(undef, length(βspan))
    ϵs = zeros(length(βspan))

    data[1] = alg.finalizer.f!(ρ0, Hs)
    ϵs[1] = 1e-16
    
    ρs = deepcopy(ρ0)

    nsteps = 2:length(βspan)
    for ix in nsteps
        (alg.verbosity > 0) && (@info "Iteration $ix, β = $(βspan[ix])")
        (alg.verbosity > 0) && (@info(crayon"magenta"("Constructing time evolution operator")))
        @timeit alg.timer_output "construct_time_evolution_operator" begin
            dβ = βspan[ix] - βspan[ix-1]
            Us = time_evolution_MPO(Hs, dβ/2)
        end

        (alg.verbosity > 0) && (@info(crayon"magenta"("Evolve")))
        @timeit alg.timer_output "evolve_one_time_step" ρs = ρs * Us

        @timeit alg.timer_output "normalize_each_disorder_sector" begin
            ρs_normalized, ϵ_acc = normalize_density_matrix(ρs)
            (alg.verbosity > 1) && (@info(crayon"magenta"("Max. error after normalization: ϵ₁ = $(ϵ_acc[1]), ϵ₂ = $(ϵ_acc[2])")))
        end

        (alg.verbosity > 0) && (@info(crayon"magenta"("Truncating ρ")))
        (alg.verbosity > 1) && (@info(crayon"magenta"("Before truncation: Bonddimension of ρ = $(dim(space(ρs_normalized[1],1)))")))
        @timeit alg.timer_output "truncate_disorder_MPO" ρs = truncate(ρs_normalized, alg.trunc_method)
        (alg.verbosity > 1) && (@info(crayon"magenta"("After truncation: Bonddimension of ρ = $(dim(space(ρs[1],1)))")))
        
        @timeit alg.timer_output "finalizer" data[ix] = alg.finalizer.f!(ρs, Hs)
        ϵs[ix] = ϵ_acc[2]

    end
    return ρs, ϵs, data
end


# Algorithm for computing the groundstate density matrix of a disordered system
struct Groundstate_iDTEBD <: AbstractAlgorithm
    trunc_method::MatrixAlgebraKit.TruncationStrategy
    convtol::Float64
    maxiter::Int
    verbosity::Int
    timer_output::TimerOutput
    finalizer::Finalizer

    function Groundstate_iDTEBD(trunc_method::MatrixAlgebraKit.TruncationStrategy; convtol::Float64 = 1e-8, maxiter::Int = 100, verbosity::Int = 0, timer_output::TimerOutput = TimerOutput(), finalizer::Finalizer = default_Finalizer)
        return new(trunc_method, convtol, maxiter, verbosity, timer_output, finalizer)
    end
end

function evolve_densitymatrix(ρ0::InfiniteDisorderDensityMatrix, Hs::DisorderMPOHam, dβ::Float64, alg::Groundstate_iDTEBD)
    data = Vector{alg.finalizer.E}()
    ϵsacc = Float64[]
    ϵsconv = Float64[]

    push!(data, alg.finalizer.f!(ρ0, Hs))
    ϵ_conv = 1.0
    push!(ϵsacc, 1e-16)
    push!(ϵsconv, ϵ_conv)
    
    ρs = deepcopy(ρ0)

    ix = 1
    ϵ_conv = 1.0
    ρprev = deepcopy(ρs)
    Eprev = DisorderKit.energy_density(ρs, Hs)
    while (ϵ_conv > alg.convtol) && (ix <= alg.maxiter)
        ix += 1
        (alg.verbosity > 0) && (@info "Iteration $ix")
        (alg.verbosity > 0) && (@info(crayon"magenta"("Constructing time evolution operator")))
        @timeit alg.timer_output "construct_time_evolution_operator" begin
            Us = time_evolution_MPO(Hs, dβ/2)
        end

        (alg.verbosity > 0) && (@info(crayon"magenta"("Evolve")))
        @timeit alg.timer_output "evolve_one_time_step" ρs = ρs * Us
        # if ix < 20
        #     @timeit alg.timer_output "evolve_one_time_step" ρs = ρs * Us
        # else
        #     @timeit alg.timer_output "evolve_one_time_step" ρs = ρs * ρs
        # end

        @timeit alg.timer_output "normalize_each_disorder_sector" begin
            ρs_normalized, ϵ_acc = normalize_density_matrix(ρs)
            (alg.verbosity > 1) && (@info(crayon"magenta"("Max. error after normalization: ϵ₁ = $(ϵ_acc)")))
        end

        (alg.verbosity > 0) && (@info(crayon"magenta"("Truncating ρ")))
        (alg.verbosity > 1) && (@info(crayon"magenta"("Before truncation: Bonddimension of ρ = $(dim(space(ρs_normalized[1],1)))")))
        @timeit alg.timer_output "truncate_disorder_MPO" ρs = truncate(ρs_normalized, alg.trunc_method)
        (alg.verbosity > 1) && (@info(crayon"magenta"("After truncation: Bonddimension of ρ = $(dim(space(ρs[1],1)))")))
        ρs = gauge!(ρs)

        @timeit alg.timer_output "finalizer" push!(data, alg.finalizer.f!(ρs, Hs))
        # if space(ρs.opp[1], 1) == space(ρprev.opp[1], 1)
        #     # errors = norm.(ρs.opp .- ρprev.opp)
        #     # @show errors
        #     # ϵ_conv = maximum(norm.(ρs.opp .- ρprev.opp))
        #     ϵ_conv = 1-real(fidelity(ρs, ρprev))
        # else
        #     (alg.verbosity > 0) && (@info(crayon"magenta"("Warning: The virtual spaces have changed, cannot compute convergence error. Setting ϵ_conv = 1.0")))
        #     ϵ_conv = 1.0
        # end
        E = DisorderKit.energy_density(ρs, Hs)
        ϵ_conv = abs(Eprev - E)
        Eprev = E

        (alg.verbosity > 0) && (@info(crayon"magenta"("Convergence error: ϵ_conv = $(ϵ_conv)")))

        ρprev = deepcopy(ρs)
        push!(ϵsacc, ϵ_acc)
        push!(ϵsconv, ϵ_conv)
    end
    return ρs, ϵsconv, ϵsacc, data
end
