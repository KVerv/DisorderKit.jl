
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

function compute_energy_moments(ρ::InfiniteDisorderDensityMatrix, Hs::DisorderMPOHam)
    λ, l, r = environments(ρ)
    @show λ

    @tensor ED[-1; -2] :=  l[6; 1] * ρ[1][1 2 3; 7 8 11] * Hs.D[4 -1; 2 3] * conj(ρ[1][6 4 -2; 7 8 10]) * r[11; 10]
    ED /= λ
    #FIXME Include long range interactions
    @tensor EL[-4 -1; -2 -3 -5] := l[6; 1] * ρ[1][1 2 3; 7 8 -3] * Hs.L[4 -4; 2 3 -2] * conj(ρ[1][6 4 -5; 7 8 -1])
    @tensor ER[-4 -1 -2; -3 -5] := r[6; 1] * ρ[1][-2 2 3; 7 8 6] * Hs.R[-1 4 -4; 2 3] * conj(ρ[1][-3 4 -5; 7 8 1])

    @tensor ECB[-1 -2; -3 -4] := EL[-1 1; 2 3 -2] * ER[-3 2 3; 1 -4]
    ECB /= λ^2
    U, S, V, ϵ = svd_trunc(ECB; trunc = truncerror(atol = 1e-12))
    @show S
    @show ϵ
    Id = id(ComplexF64, space(ρ[1], 2))
    @tensor D[-1 -2; -3 -4] := ED[-2; -4] * Id[-1; -3]
    @tensor L[-1 -2; -3 -4 -5] := U[-2 -4; -5] * Id[-1; -3]
    @tensor R[-1 -2 -3; -4 -5] := S[-1; 1] * V[1 -3; -5] * Id[-2; -4]

    A = zeros(ComplexF64, space(R,1) ⊗ space(R,2) ⊗ space(R,3), space(R,2) ⊗ space(R,3) ⊗ space(R,1))
    return A, D, L, R
end


# function compute_energy_moments(ρ::InfiniteDisorderDensityMatrix, Hs::DisorderMPOHam, dβ::Real)
#     pspace = space(ρ[1], 2)
#     dspace = space(ρ[1], 3)
    
#     λ, l, r = environments(ρ)
#     P = make_DiagonalBlockTensorMap(ρ.ps)
#     @show λ

#     ebar = energy_density(ρ, Hs)

#     @tensor ED1[-1; -2] :=  l[6; 1] * ρ[1][1 2 3; 7 8 11] * Hs.D[4 -1; 2 3] * conj(ρ[1][6 4 -2; 7 8 10]) * r[11; 10]
#     ED1 /= λ
#     @tensor ELR1[-1; -2] := l[1; 2] * ρ[1][2 5 6; 3 4 11] * ρ[1][11 12 13; 9 10 19] * conj(ρ[1][1 7 -2; 3 4 8]) * conj(ρ[1][8 15 17; 9 10 18]) * Hs.L[7 -1; 5 6 14] * Hs.R[14 15 16; 12 13] * P[17; 16] * r[19; 18]
#     ELR1 /= λ^2

#     e1 = ELR1 + ED1

#     M1 = zeros(ComplexF64, BlockTensorKit.boxplus(dspace), BlockTensorKit.boxplus(dspace))
#     for i in 1:length(dspace)
#         M1[i, i] = TensorMap(exp.(dβ*(e1[i,i].data)), ℂ^1, ℂ^1)
#     end

#     @tensor ED2[-1 -2; -3 -4] :=  l[1; 2] * ρ[1][2 5 6; 3 4 11] * ρ[1][11 12 -2; 9 10 14] * conj(ρ[1][1 7 -3; 3 4 8]) * conj(ρ[1][8 12 -4; 9 10 13]) * Hs.D[7 -1; 5 6] * r[14; 13]
#     ED2 /= λ^2
#     @tensor ELR2[-1 -2; -3 -4] := l[1; 2] * ρ[1][2 5 6; 3 4 11] * ρ[1][11 12 13; 9 10 19] * conj(ρ[1][1 7 -3; 3 4 8]) * conj(ρ[1][8 15 -4; 9 10 18]) * Hs.L[7 -1; 5 6 14] * Hs.R[14 15 -2; 12 13] * r[19; 18]
#     ELR2 /= λ^2

#     e2 = ELR2 + ED2
#     M2 = zeros(ComplexF64, BlockTensorKit.boxplus(dspace) ⊗ BlockTensorKit.boxplus(dspace), BlockTensorKit.boxplus(dspace) ⊗ BlockTensorKit.boxplus(dspace))
#     for i in 1:length(dspace)
#         for j in 1:length(dspace)
#             M2[i, j, i, j] = TensorMap(exp.(dβ*(e2[i, j, i, j].data)), ℂ^1⊗ℂ^1, ℂ^1⊗ℂ^1)
#         end
#     end
    
#     @tensor M11[-1 -2; -3 -4] := M1[-1; -3] * M1[-2; -4]
#     M2 = M2 - M11
#     M2perm = permute(M2, ((1, 3), (2, 4)))

#     U, S, V, ϵ = svd_trunc(M2perm; trunc = truncerror(atol = 1e-12))
#     @show S
#     @show ϵ
#     T01 = U*sqrt(S)
#     T01 = permute(T01, ((1,), (2, 3)))
#     T10 = sqrt(S)*V
#     T10 = permute(T10, ((1, 2), (3,)))

#     pId = id(ComplexF64, pspace[1])
#     dId = id(ComplexF64, dspace)


#     @show dspace
#     @show dspace[:]
#     if q == 0
#         Wcodomain = BlockTensorKit.boxplus(ℂ^1) ⊗ BlockTensorKit.boxplus(pspace) ⊗ BlockTensorKit.boxplus(dspace)
#         Wdomain = BlockTensorKit.boxplus(pspace) ⊗ BlockTensorKit.boxplus(dspace) ⊗  BlockTensorKit.boxplus(ℂ^1)
#         W = SparseBlockTensorMap{AbstractTensorMap{ComplexF64, ComplexSpace, 3, 3}}(undef, Wcodomain, Wdomain)


#         Wdata = exp(dβ*ebar) * pId
#         for i in 1:length(dspace)
#             W[1,1,i,1,i,1] = TensorMap(Wdata.data, ℂ^1 ⊗ pspace[1] ⊗ dspace[i], pspace[1] ⊗  dspace[i] ⊗ ℂ^1)
#         end

#     elseif q == 1
#         Wcodomain = BlockTensorKit.boxplus(ℂ^1) ⊗ BlockTensorKit.boxplus(pspace) ⊗ BlockTensorKit.boxplus(dspace)
#         Wdomain = BlockTensorKit.boxplus(pspace) ⊗ BlockTensorKit.boxplus(dspace) ⊗  BlockTensorKit.boxplus(ℂ^1)
#         W = SparseBlockTensorMap{AbstractTensorMap{ComplexF64, ComplexSpace, 3, 3}}(undef, Wcodomain, Wdomain)


#         @tensor Wdata[-1 -2; -3 -4] := M1[-2; -4] * pId[-1; -3]
#         for i in 1:length(dspace)
#             W[1,1,i,1,i,1] = TensorMap(Wdata[1,i,1,i].data, ℂ^1 ⊗ pspace[1] ⊗ dspace[i], pspace[1] ⊗  dspace[i] ⊗ ℂ^1)
#         end
#     end
#     # vspace = space(T10, 1)
#     # Wcodomain = BlockTensorKit.boxplus([ℂ^1, vspace]...) ⊗ BlockTensorKit.boxplus(pspace) ⊗ BlockTensorKit.boxplus(dspace)
#     # Wdomain = BlockTensorKit.boxplus(pspace) ⊗ BlockTensorKit.boxplus(dspace) ⊗  BlockTensorKit.boxplus([ℂ^1, vspace]...)
#     # W = SparseBlockTensorMap{AbstractTensorMap{ComplexF64, ComplexSpace, 3, 3}}(undef, Wcodomain, Wdomain)

#     # @tensor W00[-1 -2; -3 -4] := M1[-2; -4] * Id[-1; -3]
#     # @tensor W01[-1 -2; -3 -4 -5] := T01[-2; -4 -5] * Id[-1; -3]
#     # @tensor W10[-1 -2 -3; -4 -5] := T10[-1 -3; -5] * Id[-2; -4]

#     # for i in 1:length(dspace)
#     #     W[1, 1, i, 1, i, 1] = TensorMap(W00[1,i,1,i].data, ℂ^1 ⊗ pspace[1] ⊗ dspace[1], pspace[1] ⊗  dspace[1] ⊗ ℂ^1)
#     #     # W[1, 1, i, 1, i, 2] = TensorMap(W01[1,i,1,i,1].data, ℂ^1 ⊗ pspace[1] ⊗ dspace[1], pspace[1] ⊗  dspace[1] ⊗ vspace[1])
#     #     # W[2, 1, i, 1, i, 1] = TensorMap(W10[1,1,i,1,i].data, vspace[1] ⊗ pspace[1] ⊗ dspace[1], pspace[1] ⊗  dspace[1] ⊗ ℂ^1)
        
#     #     W[1, 1, i, 1, i, 2] = zeros(ComplexF64, ℂ^1 ⊗ pspace[1] ⊗ dspace[1], pspace[1] ⊗  dspace[1] ⊗ vspace[1])
#     #     W[2, 1, i, 1, i, 1] = zeros(ComplexF64, vspace[1] ⊗ pspace[1] ⊗ dspace[1], pspace[1] ⊗  dspace[1] ⊗ ℂ^1)      
#     #     W[2, 1, i, 1, i, 2] = zeros(ComplexF64, vspace[1] ⊗ pspace[1] ⊗ dspace[1], pspace[1] ⊗  dspace[1] ⊗ vspace[1])
#     # end

#     return InfiniteDisorderMPO([W])
# end


# βspan contains the values of β at which the density matrix is evaluated. βspan[1] is the value of β at which the initial density matrix is evaluated.
function evolve_densitymatrix(ρ0::InfiniteDisorderDensityMatrix, Hs::DisorderMPOHam, βspan::AbstractVector{<:Number}, alg::FiniteTemperature_iDTEBD)
    data = Vector{alg.finalizer.E}(undef, length(βspan))
    ϵs = zeros(length(βspan))

    data[1] = alg.finalizer.f!(ρ0, Hs)
    ϵs[1] = 1e-16
    
    @timeit alg.timer_output "Copy" ρs = deepcopy(ρ0)

    nsteps = 2:length(βspan)
    for ix in nsteps
        (alg.verbosity > 0) && (@info "Iteration $ix, β = $(βspan[ix])")
        (alg.verbosity > 0) && (@info(crayon"cyan"("Constructing time evolution operator")))
        @timeit alg.timer_output "construct_time_evolution_operator" begin
            dβ = βspan[ix] - βspan[ix-1]
            A, D, L, R = compute_energy_moments(ρs, Hs)
            Dcorr = Hs.D - D
            Lcorr = zeros(ComplexF64, space(L,1) ⊗ space(L,2), space(L,1) ⊗ space(L,2) ⊗  BlockTensorKit.boxplus([ℂ^1, space(L,5)']...))
            Rcorr = zeros(ComplexF64, BlockTensorKit.boxplus([ℂ^1, space(R,1)]...) ⊗ space(R,2) ⊗ space(R,3), space(R,2) ⊗ space(R,3))
            Lcorr[1, 1, 1, 1, 1] = Hs.L
            Lcorr[1, 1, 1, 1, 2] = -L
            Rcorr[1, 1, 1, 1, 1] = Hs.R
            Rcorr[2, 1, 1, 1, 1] = -R
            Hcorrected = DisorderMPOHam(A, Lcorr, Rcorr, Dcorr)
            Us1 = time_evolution_MPO(Hcorrected, dβ/2; N = 1)

            # Hcorrected = DisorderMPOHam(A, L, R, D)
            # Us1 = time_evolution_MPO(Hs, dβ/2; N = 2)
            # Us2 = time_evolution_MPO(Hcorrected, -dβ/2; N=1)
        end

        (alg.verbosity > 0) && (@info(crayon"magenta"("Evolve")))
        @timeit alg.timer_output "evolve_one_time_step" ρs = ρs * Us1# * Us2

        # (alg.verbosity > 0) && (@info(crayon"magenta"("Normalize")))
        # @timeit alg.timer_output "normalize_each_disorder_sector" begin
        #     ρs_normalized = normalize(ρs)
        # end

        ρs_normalized = gauge(ρs)
        (alg.verbosity > 0) && (@info(crayon"magenta"("Truncating ρ")))
        (alg.verbosity > 1) && (@info(crayon"magenta"("Before truncation: Bonddimension of ρ = $(dim(space(ρs_normalized[1],1)))")))
        @timeit alg.timer_output "truncate_disorder_MPO" ρs = truncate(ρs_normalized, alg.trunc_method)
        (alg.verbosity > 1) && (@info(crayon"magenta"("After truncation: Bonddimension of ρ = $(dim(space(ρs[1],1)))")))
        
        ρs = gauge(ρs)
        @timeit alg.timer_output "Compute error" es, N2 = entanglement_spectrum_norm(ρs)
        # ϵ_acc = N2 - 1
        # ϵ_acc = real.(N2)
        # ϵ_acc = abs.(1 - N2)
        ϵ_acc = sum(es[1:end-1])
        (alg.verbosity > 1) && (@info(crayon"light_blue"("Max. error after normalization: ϵ₁ = $(ϵ_acc), N2 = $(N2)")))

        (alg.verbosity > 0) && (@info(crayon"cyan"("Finalize")))

        @timeit alg.timer_output "finalizer" data[ix] = alg.finalizer.f!(ρs, Hs)
        ϵs[ix] = ϵ_acc
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
    ϵsent = Float64[]
    ϵsN2 = Float64[]
    ϵsconv = Float64[]
    δs = Float64[]

    push!(data, alg.finalizer.f!(ρ0, Hs))
    ϵ_conv = 1.0
    push!(ϵsent, 1e-16)
    push!(ϵsN2, 1e-16)
    push!(ϵsconv, ϵ_conv)
    push!(δs, 1e-16)
    
    @timeit alg.timer_output "Copy" ρs = deepcopy(ρ0)

    ix = 1
    ϵ_conv = 1.0
    @timeit alg.timer_output "Copy" ρprev = deepcopy(ρs)
    @timeit alg.timer_output "energy_density" Eprev = DisorderKit.energy_density(ρs, Hs)
    while (ϵ_conv > alg.convtol) && (ix <= alg.maxiter)
        ρs = gauge(ρs)
        ix += 1
        (alg.verbosity > 0) && (@info "Iteration $ix")
        (alg.verbosity > 0) && (@info(crayon"cyan"("Constructing time evolution operator")))
        @timeit alg.timer_output "construct_time_evolution_operator" begin
            N_disorder = dim(space(Hs.D, 2))
            # Us = time_evolution_MPO(Hs, dβ/2)

            # h = dβ/2
            # ρ1 = ρs * time_evolution_MPO(Hs, h; N = 1)
            # # ρ2 = ρs * time_evolution_MPO(Hs, 2*h; N = 2)
            # A0, D0, L0, R0 = compute_energy_moments(ρs, Hs)
            # A1, D1, L1, R1 = compute_energy_moments(ρ1, Hs)
            # # A2, D2, L2, R2 = compute_energy_moments(ρ2, Hs)
            # if space(L1) == space(L0) #&& space(L1) == space(L2)
            #     A = A0
            #     D = D0
            #     L = L0
            #     R = R0
            #     # A = (A0 + A1)/2
            #     # D = (D0 + D1)/2
            #     # L = (L0 + L1)/2
            #     # R = (R0 + R1)/2
            #     # A = (A0 + 4*A1 + A2)/3
            #     # D = (D0 + 4*D1 + D2)/3
            #     # L = (L0 + 4*L1 + L2)/3
            #     # R = (R0 + 4*R1 + R2)/3
            # else
            #     @show "L1 and L0 have different spaces"
            #     A = A0
            #     D = D0
            #     L = L0
            #     R = R0
            # end

            # # A, D, L, R = compute_energy_moments(ρs, Hs)
            # # @show space(D)
            # # @show space(L)
            # # @show space(R)

            # # Dcorr = Hs.D - D
            # # Lcorr = zeros(ComplexF64, space(L,1) ⊗ space(L,2), space(L,1) ⊗ space(L,2) ⊗  BlockTensorKit.boxplus([ℂ^1, space(L,5)']...))
            # # Rcorr = zeros(ComplexF64, BlockTensorKit.boxplus([ℂ^1, space(R,1)]...) ⊗ space(R,2) ⊗ space(R,3), space(R,2) ⊗ space(R,3))
            # # @show space(Lcorr[1, 1, 1, 1, 1])
            # # @show space(Hs.L)
            # # for i in 1:N_disorder
            # #     Lcorr[1, i, 1, i, 1] = Hs.L[1, i, 1, i, 1]
            # #     Lcorr[1, i, 1, i, 2] = -L[1, i, 1, i, 1]
            # #     Rcorr[1, 1, i, 1, i] = Hs.R[1, 1, i, 1, i]
            # #     Rcorr[2, 1, i, 1, i] = -R[1, 1, i, 1, i]
            # # end
            # # iso = isomorphism(space(Rcorr,1), oplus(space(Rcorr,1)))
            # # @show space(iso)
            # # @show space(Rcorr)
            # # @tensor Rcorr2[-1 -2 -3; -4 -5] := conj(iso[1; -1]) * Rcorr[1 -2 -3; -4 -5]
            # # @tensor Lcorr2[-1 -2; -3 -4 -5] := Lcorr[-1 -2; -3 -4 1] * iso[1; -5]

            # # Acorr = zeros(ComplexF64, space(Rcorr2,1) ⊗ space(R,2) ⊗ space(R,3), space(R,2) ⊗ space(R,3) ⊗ space(Rcorr2,1))
            # # Hcorrected = DisorderMPOHam(Acorr, Lcorr2, Rcorr2, Dcorr)
            # # Us1 = time_evolution_MPO(Hcorrected, dβ/2; N = 2)

            # Hcorrected = DisorderMPOHam(A, L, R, D)
            # Us2 = time_evolution_MPO(Hcorrected, -h; N=1)
            Us1 = time_evolution_MPO(Hs, dβ/2; N = 1)
            # Us2 = time_evolution_MPO(Hs, dβ/2; N = 1)
            # # Us2 = compute_energy_moments(ρs, Hs, dβ/2; q=1)

            # trunc_methodZ = MatrixAlgebraKit.truncrank(2)
            # fused_space = fuse(space(ρs[1], 1)', space(ρs[1], 1), space(Us2[1], 1))
            # iso = isomorphism(fused_space, space(ρs[1], 1)' ⊗ space(ρs[1], 1) ⊗ space(Us2[1], 1))
            # @tensor Mfull[-1 -2; -3 -4] := iso[-1; 7 4 1] * Us2[1][1 8 -2; 2 3 11] * ρs[1][4 2 3; 5 6 10] * conj(ρs[1][7 8 -3; 5 6 9]) * conj(iso[-4; 9 10 11])
            # Mtrunc, _ = truncate_mpo(Mfull, trunc_methodZ)
            
            # # vspace = space(Mtrunc, 1)
            # # dspace = space(ρs[1], 3)
            # # Wcodomain = BlockTensorKit.boxplus([ℂ^1, vspace]...) ⊗ BlockTensorKit.boxplus(dspace)
            # # Wdomain = BlockTensorKit.boxplus(dspace) ⊗  BlockTensorKit.boxplus([ℂ^1, vspace]...)
            # # W = SparseBlockTensorMap{AbstractTensorMap{ComplexF64, ComplexSpace, 2, 2}}(undef, Wcodomain, Wdomain)

            # # for i in 1:length(dspace)
            # #     W[1, i, i, 1] = BraidingTensor{ComplexF64, ComplexSpace}(dspace[1], ℂ^1)
            # #     W[2, i, i, 2] = BraidingTensor{ComplexF64, ComplexSpace}(dspace[1], ℂ^1)
            # #     W[3, i, i, 3] = TensorMap(-Mtrunc[1, i, i, 1].data, vspace[1] ⊗ dspace[1], dspace[1] ⊗ vspace[1])
            # # end

            # # @show space(W)
            # # Wtrunc, _ = truncate_mpo(W, MatrixAlgebraKit.truncrank(4))

            # # @show space(Wtrunc)

            # Idp = id(ComplexF64, space(ρs[1], 2))
            # @tensor MUtensor[-1 -2 -3; -4 -5 -6] := Idp[-2; -4] * Mtrunc[-1 -3; -5 -6]
            # MU = InfiniteDisorderMPO([MUtensor])

            Us2 = construct_renormalisation(ρs, Hs, dβ/2)
        end

        (alg.verbosity > 0) && (@info(crayon"cyan"("Evolve")))
        @timeit alg.timer_output "evolve_one_time_step" ρs = ρs * Us1 * Us2
        # @timeit alg.timer_output "evolve_one_time_step" ρs = ρs * Us1 #* MU

        # if ix < 20
        #     @timeit alg.timer_output "evolve_one_time_step" ρs = ρs * Us
        # else
        #     @timeit alg.timer_output "evolve_one_time_step" ρs = ρs * ρs
        # end

        # (alg.verbosity > 1) && (@info(crayon"yellow"("Before Normalization: Bonddimension of ρ = $(dim(space(ρs[1],1)))")))
        # (alg.verbosity > 0) && (@info(crayon"yellow"("Normalize Density Matrix")))
        # @timeit alg.timer_output "normalize_each_disorder_sector" begin
        #     # ρs_normalized, δ = normalize(ρs)
        #     # ρs_normalized = gauge(ρs_normalized)

        #     # ρs_normalized = ρs
        #     # ρs_normalized = gauge(ρs)
        # end

        # ρs_normalized = gauge(ρs)
        ρs_normalized = ρs
        (alg.verbosity > 0) && (@info(crayon"magenta"("Truncating ρ")))
        (alg.verbosity > 1) && (@info(crayon"magenta"("Before truncation: Bonddimension of ρ = $(dim(space(ρs_normalized[1],1)))")))
        @timeit alg.timer_output "truncate_disorder_MPO" ρs = truncate(ρs_normalized, alg.trunc_method)
        # ρs = ρs_normalized
        (alg.verbosity > 1) && (@info(crayon"magenta"("After truncation: Bonddimension of ρ = $(dim(space(ρs[1],1)))")))
        
        @timeit alg.timer_output "gauge" ρs = gauge(ρs)
        @timeit alg.timer_output "Compute error" es, N2 = entanglement_spectrum_norm(ρs)
        # ϵ_acc = N2 - 1
        ϵN2 = real.(N2)
        γbar, Vγ, _ = lyapunovexp(ρs; L=10, n_samples=1000)
        ϵN2 = Vγ
        ϵent = γbar
        # ϵ_acc = abs.(1 - N2)
        # ϵent = sum(es[1:end-1])
        (alg.verbosity > 1) && (@info(crayon"light_blue"("Max. error after normalization: ϵ₁ = $(ϵent), N2 = $(ϵN2)")))

        (alg.verbosity > 0) && (@info(crayon"cyan"("Finalize")))
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
        @timeit alg.timer_output "energy_density" E = DisorderKit.energy_density(ρs, Hs)
        ϵ_conv = (abs(Eprev - E)/dβ)
        Eprev = E

        (alg.verbosity > 0) && (@info(crayon"light_blue"("Convergence error: ϵ_conv = $(ϵ_conv)")))

        @timeit alg.timer_output "Copy" ρprev = deepcopy(ρs)
        push!(ϵsent, ϵent)
        push!(ϵsN2, ϵN2)
        push!(ϵsconv, ϵ_conv)
        δ = 0
        push!(δs, δ)
    end
    return ρs, ϵsconv, ϵsent, ϵsN2, δs, data
end
