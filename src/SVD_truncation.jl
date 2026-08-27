function left_orthonormalize_mpo(A::AbstractMPOTensor, L₀::AbstractBondTensor, trunc_method::MatrixAlgebraKit.TruncationStrategy; conv_tol::Real = 1e-6, timer::TimerOutput = TimerOutput())
    L = L₀
    L /= norm(L)
    Lold = L
    Amps = permute(A, ((1,2,3),(4,)))
    @tensor LA[-1 -2 -3; -4] := L[-1; 1] * Amps[1 -2 -3; -4]
    AL, S, V, ϵ = svd_trunc(LA; trunc = (atol = 1e-12,))
    L = S * V
    λ = norm(L)
    L /= λ
    if space(L) == space(L₀)
        ε = norm(L - Lold)
    else
        ε = 1.
    end
    while ε > conv_tol
        L /= norm(L)
        Lold = L
        @tensor LA[-1 -2 -3; -4] := L[-1; 1] * Amps[1 -2 -3; -4]
        AL, S, V = svd_trunc(LA; trunc = (atol = 1e-8,))
        L = S * V
        λ = norm(L)
        L /= λ
        if space(L) == space(Lold)
            ε = norm(L - Lold)
        end
    end
    @info(crayon"red"("Left canonicalization: ε = $ε"))
    AL = permute(AL, ((1,2),(3,4)))
    return AL, L, λ, ε
end

function right_orthonormalize_mpo(A::AbstractMPOTensor, C₀::AbstractBondTensor, trunc_method::MatrixAlgebraKit.TruncationStrategy; conv_tol::Real = 1e-6, timer::TimerOutput = TimerOutput())
    C = C₀
    C /= norm(C)
    Cold = C
    Amps = permute(A, ((1,2,3),(4,)))
    @tensor AC[-1; -2 -3 -4] := Amps[-1; -2 -3 1] * C[1; -4]
    U, S, AR = svd_trunc(AC; trunc = (atol = 1e-9,))
    C = U * S
    λ = norm(C)
    C /= λ
    if space(C) == space(C₀)
        ϵ = norm(C - Cold)
    else
        ϵ = 1.
    end
    while ϵ > conv_tol
        C /= norm(C)
        Cold = C
        @tensor AC[-1; -2 -3 -4] := Amps[-1; -2 -3 1] * C[1; -4]
        U, S, AR = svd_trunc(AC; trunc = (atol = 1e-9,))
        C = U * S
        λ = norm(C)
        C /= λ
        if space(C) == space(Cold)
            ϵ = norm(C - Cold)
        end
    end
    @info(crayon"red"("Right canonicalization: ϵ = $ϵ"))
    AR = permute(AR, ((1,2),(3,4)))
    return AR, C, λ, ϵ
end

function truncate_mpo(O::AbstractMPOTensor, alg::SuccessiveSVD; timer::TimerOutput = TimerOutput())
    @show space(O, 1)
    L₀ = rand(ComplexF64, space(O, 1), space(O, 1))
    OL, L, λ, ϵL = left_orthonormalize_mpo(O, L₀, alg.trunc_method; conv_tol=alg.conv_tol, timer=timer)
    C₀ = rand(ComplexF64, space(OL, 4)', space(OL, 4)')
    OR, C, λ, ϵR = right_orthonormalize_mpo(OL, C₀, alg.trunc_method; conv_tol=alg.conv_tol, timer=timer)
    U, S, V, ϵC = svd_trunc(C; trunc=alg.trunc_method)
    # @show S
    # @show ϵC
    @tensor O_updated[-1 -2; -3 -4] := conj(U[1; -1]) * OL[1 -2; -3 2] * U[2; -4]
    return O_updated, ϵC, ϵL, ϵR
end