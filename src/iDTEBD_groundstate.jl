# Algorithm for computing the groundstate of a disordered system through imaginary time evolution
# struct GroundstateInfo
#     ϵsconv::Vector{Float64}
#     ϵsent::Vector{Float64}
#     ϵsz::Vector{Float64}
#     ρ_trunc_err::Vector{Float64}
#     R_trunc_err::Vector{Float64}

#     function GroundstateInfo(maxiter::Int)
#         return new(Vector{Float64}(undef, maxiter), Vector{Float64}(undef, maxiter), Vector{Float64}(undef, maxiter), Vector{Float64}(undef, maxiter), Vector{Float64}(undef, maxiter))
#     end
# end

# function groundstate(ρ0::InfiniteDisorderMPS, Hs::DisorderMPOHam, dτ::Float64, alg::Groundstate_iDTEBD)
#     data = Vector{alg.finalizer.E}()
#     info = GroundstateInfo(alg.maxiter)

#     ϵ_conv = 1.0
    
#     @timeit alg.timer_output "Copy" ρs = deepcopy(ρ0)

#     ix = 0
#     ϵ_conv = 1.0
#     @timeit alg.timer_output "Copy" ρprev = deepcopy(ρs)
#     @timeit alg.timer_output "energy_density" Eprev = DisorderKit.energy_density(ρs, Hs)
#     while (ϵ_conv > alg.convtol) && (ix+1 <= alg.maxiter)
#         @timeit alg.timer_output "gauge" ρs = gauge(ρs)
#         ix += 1
#         (alg.verbosity > 0) && (@info "Iteration $ix")
#         (alg.verbosity > 0) && (@info(crayon"cyan"("Constructing time evolution operator")))
#         @timeit alg.timer_output "construct_time_evolution_operator" begin
#             Us = time_evolution_MPO(Hs, dτ; N = 2)
#         end
#         @timeit alg.timer_output "construct_renormalisation_operator" begin
#             R, ϵR = construct_renormalisation(ρs, Hs, dτ, alg)
#             info.R_trunc_err[ix] = ϵR
#         end

#         (alg.verbosity > 0) && (@info(crayon"cyan"("Evolve")))
#         @timeit alg.timer_output "evolve_one_time_step" ρs_normalized = (ρs * Us) * R

#         (alg.verbosity > 0) && (@info(crayon"magenta"("Truncating ρ")))
#         (alg.verbosity > 1) && (@info(crayon"magenta"("Before truncation: Bonddimension of ρ = $(dim(space(ρs_normalized[1],1)))")))
#         @timeit alg.timer_output "truncate_disorder_MPO" ρs, ϵρ = truncate(ρs_normalized, alg.trunc_method_state; timer=alg.timer_output)
#         info.ρ_trunc_err[ix] = ϵρ
#         (alg.verbosity > 1) && (@info(crayon"magenta"("After truncation: Bonddimension of ρ = $(dim(space(ρs[1],1)))")))
        
#         @timeit alg.timer_output "gauge" ρs = gauge(ρs)
#         # @timeit alg.timer_output "Compute error" es, ϵz = entanglement_spectrum_norm(ρs)
#         # ϵent = sum(es[1:end-1])

#         ϵz = 0.
#         ϵent = 0.
        
#         info.ϵsz[ix] = ϵz
#         info.ϵsent[ix] = ϵent
#         (alg.verbosity > 1) && (@info(crayon"light_blue"("Max. error after normalization: ϵ₁ = $(ϵent), N2 = $(ϵz)")))

#         @timeit alg.timer_output "energy_density" E = DisorderKit.energy_density(ρs, Hs)
#         # ϵ_conv = (abs(Eprev - E)/dτ)
#         # Eprev = E
#         @timeit alg.timer_output "trace_distance" ϵ_conv = average_trace_distance(ρs, ρprev)/dτ^2

#         info.ϵsconv[ix] = ϵ_conv
#         (alg.verbosity > 0) && (@info(crayon"light_blue"("Convergence error: ϵ_conv = $(ϵ_conv)")))

#         (alg.verbosity > 0) && (@info(crayon"cyan"("Finalize")))
#         @timeit alg.timer_output "finalizer" push!(data, alg.finalizer.f!(ρs, Hs))

#         @timeit alg.timer_output "Copy" ρprev = deepcopy(ρs)
#     end
#     return ρs, data, info
# end


# function groundstate(ρ0::InfiniteDisorderMPS, Hs::DisorderMPOHam, dτ::Float64, alg::Groundstate_iDTEBD)
#     data = Vector{alg.finalizer.E}()
#     info = GroundstateInfo(alg.maxiter)

#     ϵ_conv = 1.0
    
#     @timeit alg.timer_output "Copy" ρs = deepcopy(ρ0)

#     ix = 0
#     ϵ_conv = 1.0
#     @timeit alg.timer_output "Copy" ρprev = deepcopy(ρs)
#     @timeit alg.timer_output "energy_density" Eprev = DisorderKit.energy_density(ρs, Hs)
#     while (ϵ_conv > alg.convtol) && (ix+1 <= alg.maxiter)
#         @timeit alg.timer_output "gauge" ρs = gauge(ρs)
#         ix += 1
#         (alg.verbosity > 0) && (@info "Iteration $ix")
#         (alg.verbosity > 0) && (@info(crayon"cyan"("Constructing time evolution operator")))
#         @timeit alg.timer_output "construct_time_evolution_operator" begin
#             Us = time_evolution_MPO(Hs, dτ; N = 2)
#         end

#         (alg.verbosity > 0) && (@info(crayon"cyan"("Evolve")))
#         @timeit alg.timer_output "evolve_one_time_step" ρs_normalized = (ρs * Us) 


#         (alg.verbosity > 0) && (@info(crayon"magenta"("Truncating ρ")))
#         (alg.verbosity > 1) && (@info(crayon"magenta"("Before truncation: Bonddimension of ρ = $(dim(space(ρs_normalized[1],1)))")))
#         @timeit alg.timer_output "truncate_disorder_MPO" ρs, ϵρ = truncate(ρs_normalized, alg.trunc_method_state; timer=alg.timer_output)
#         info.ρ_trunc_err[ix] = ϵρ
#         (alg.verbosity > 1) && (@info(crayon"magenta"("After truncation: Bonddimension of ρ = $(dim(space(ρs[1],1)))")))
        
#         @timeit alg.timer_output "gauge" ρs = gauge(ρs)
#         # @timeit alg.timer_output "Compute error" es, ϵz = entanglement_spectrum_norm(ρs)
#         # ϵent = sum(es[1:end-1])

#         ϵz = 0.
#         ϵent = 0.
        
#         info.ϵsz[ix] = ϵz
#         info.ϵsent[ix] = ϵent
#         (alg.verbosity > 1) && (@info(crayon"light_blue"("Max. error after normalization: ϵ₁ = $(ϵent), N2 = $(ϵz)")))

#         @timeit alg.timer_output "energy_density" E = DisorderKit.energy_density(ρs, Hs)
#         # ϵ_conv = (abs(Eprev - E)/dτ)
#         # Eprev = E
#         @timeit alg.timer_output "trace_distance" ϵ_conv = average_trace_distance(ρs, ρprev)/dτ^2

#         info.ϵsconv[ix] = ϵ_conv
#         (alg.verbosity > 0) && (@info(crayon"light_blue"("Convergence error: ϵ_conv = $(ϵ_conv)")))

#         (alg.verbosity > 0) && (@info(crayon"cyan"("Finalize")))
#         @timeit alg.timer_output "finalizer" push!(data, alg.finalizer.f!(ρs, Hs))

#         @timeit alg.timer_output "Copy" ρprev = deepcopy(ρs)
#     end
#     return ρs, data, info
# end

struct GroundstateInfo
    ϵsconv::Vector{Float64}
    Fid_err::Vector{Float64}

    function GroundstateInfo(maxiter::Int)
        return new(Vector{Float64}(undef, maxiter), Vector{Float64}(undef, maxiter))
    end
end

function groundstate(ψ0::InfiniteDisorderMPS, Hs::DisorderMPOHam, dτ::Float64, alg::Groundstate_iDTEBD; maxD::Int = 2)
    data = Vector{alg.finalizer.E}()
    info = GroundstateInfo(alg.maxiter)

    @timeit alg.timer_output "Copy" ψ = deepcopy(ψ0)

    ix = 0
    ϵ_conv = 1.0
    @timeit alg.timer_output "Copy" ψprev = deepcopy(ψ)
    @timeit alg.timer_output "energy_density" Eprev = DisorderKit.energy_density(ψ, Hs)
    while (ϵ_conv > alg.convtol) && (ix+1 <= alg.maxiter)
        ix += 1
        (alg.verbosity > 0) && (@info "Iteration $ix")
        (alg.verbosity > 0) && (@info(crayon"cyan"("Constructing time evolution operator")))
        @timeit alg.timer_output "construct_time_evolution_operator" begin
            Us = time_evolution_MPO(Hs, dτ; N = 2)
        end

        (alg.verbosity > 0) && (@info(crayon"cyan"("Evolve")))
        @timeit alg.timer_output "evolve_one_time_step" ψ_ev = Us * ψ 


        (alg.verbosity > 0) && (@info(crayon"magenta"("Truncating ρ")))
        (alg.verbosity > 1) && (@info(crayon"magenta"("Before truncation: Bonddimension of ρ = $(dim(space(ψ_ev[1],1)))")))

        D = dim(space(ψ[1],1))
        if D == maxD
            @timeit alg.timer_output "truncate_disorder_MPO" ψ, _ = approximate(ψ_ev, ψ; gradtol = alg.trunc_tol)
        else
            D_dis = length(ψ_ev.ps)
            D_phys = dim(space(ψ_ev[1],2))
            Dev = dim(space(ψ_ev[1],1))
            Dguess = min(maxD, Dev)
            ϕ0 = InfiniteDisorderMPS(ψ_ev.ps, D_dis, D_phys, Dguess)
            @timeit alg.timer_output "truncate_disorder_MPO" ψ, _ = approximate(ψ_ev, ϕ0; gradtol = alg.trunc_tol)
        end
        (alg.verbosity > 1) && (@info(crayon"magenta"("After truncation: Bonddimension of ρ = $(dim(space(ψ[1],1)))")))

        Fid_err = 1 - fidelity(ψprev, ψ)
        (alg.verbosity > 1) && (@info(crayon"light_blue"("Max. error after approximate: ϵ = $(Fid_err)")))

        @timeit alg.timer_output "energy_density" E = DisorderKit.energy_density(ψ, Hs)
        dE = (abs(Eprev - E)/dτ)
        ϵ_conv = Fid_err/dτ^2
        Eprev = E
        # @timeit alg.timer_output "trace_distance" ϵ_conv = average_trace_distance(ρs, ρprev)/dτ^2

        info.ϵsconv[ix] = ϵ_conv
        (alg.verbosity > 0) && (@info(crayon"light_blue"("Convergence error: (E, dE, ε) = ($(E), $(dE), $(ϵ_conv)),")))

        (alg.verbosity > 0) && (@info(crayon"cyan"("Finalize")))
        @timeit alg.timer_output "finalizer" push!(data, alg.finalizer.f!(ψ, Hs))

        @timeit alg.timer_output "Copy" ψprev = deepcopy(ψ)
    end
    return ψ, data, info
end