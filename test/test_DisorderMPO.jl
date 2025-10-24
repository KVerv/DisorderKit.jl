@testset "Reduction_Expansion_DisorderMPO" begin
    Δτ = 0.1 

    trunc_method_Zmpo = StandardTruncation(trunc_method = truncerr(1e-10))
    inversion_alg = VOMPS_Inversion(1; tol = 1e-8, maxiter = 10, verbosity = 2)
    trunc_method_disordermpo =  DisorderTracedTruncation(trunc_method = truncerr(1e-10))

    gs = [0.7,1.0,1.3, 1.6]
    T = RTFIM_time_evolution_Trotter(Δτ, gs, [1.])
    ρ = T

    ρ_reduced = DisorderKit.reduce_disorderindex(ρ)
    ρ_expanded = DisorderKit.expand_disorderindex(ρ_reduced)
    @show ρ_expanded
    @test mpo_fidelity(ρ, ρ_expanded) > 1 - 1e-12
end