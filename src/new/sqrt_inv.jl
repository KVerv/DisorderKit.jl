abstract type InversionStrategy end
const AbstractBentMPOTensor = AbstractTensorMap{T, S, 3, 1} where {T, S}


struct GradientOptimization <: InversionStrategy
    algorithm::OptimKit.OptimizationAlgorithm
end

struct OptimState
    A::AbstractBentMPOTensor
    δ::Float64
end

function fidelity_transfer_left(A::AbstractMPOTensor, O::AbstractMPOTensor)
    function ftransfer(vl)
        @tensor vl[-1; -2 -3] := A[5 6; 4 -3] * O[3 4; 2 -2] * conj(A[1 6; 2 -1]) * vl[1; 3 5]
        return vl
    end
    return ftransfer
end

function fidelity_transfer_right(A::AbstractMPOTensor, O::AbstractMPOTensor)
     function ftransfer(vr)
        @tensor vr[-1 -2; -3] := A[-2 5; 2 1] * O[-1 2; 4 3] * conj(A[-3 5; 4 6]) * vr[3 1; 6]
        return vr
    end
    return ftransfer
end

function fidelity_norm_transfer_left(A::AbstractMPOTensor, O::AbstractMPOTensor)
    function ftransfer(vl)
        @tensor vl[-1 -2 -3; -4 -5 -6] := A[1 12; 2 -6] * O[3 2; 4 -5] * conj(A[5 6; 4 -4]) * A[7 6; 8 -3] * conj(O[9 10; 8 -2]) * conj(A[11 12; 10 -1]) * vl[11 9 7; 5 3 1]
        return vl
    end
    return ftransfer
end

function fidelity_norm_transfer_right(A::AbstractMPOTensor, O::AbstractMPOTensor)
    function ftransfer(vr)
        @tensor vr[-1 -2 -3; -4 -5 -6] := A[-3 12; 2 1] * O[-2 2; 4 3] * conj(A[-1 6; 4 5]) * A[-6 6; 8 7] * conj(O[-5 10; 8 9]) * conj(A[-4 12; 10 11]) * vr[5 3 1; 11 9 7]
        return vr
    end
    return ftransfer
end

function bend(A::AbstractMPOTensor)
    @tensor B[-1 -2 -3; -4] := A[-1 -2; -3 -4]
    return B
end

function unbend(B::AbstractBentMPOTensor)
    @tensor A[-1 -2; -3 -4] := B[-1 -2 -3; -4]
    return A
end

function fidelity_identity(B::AbstractBentMPOTensor, Z::AbstractMPOTensor)
    A = unbend(B)
    vl0 = ones(ComplexF64, space(A, 1), space(Z, 1)⊗space(A, 1))
    vr0 = ones(ComplexF64, space(Z, 1)⊗space(A, 1), space(A, 1))
    ftr = fidelity_transfer_right(A, Z)
    valsr, vrs = eigsolve(x->ftr(x), vr0, 1, :LM)
    ftl = fidelity_transfer_left(A, Z)
    valsl, vls = eigsolve(x->ftl(x), vl0, 1, :LM)

    vln0 = ones(ComplexF64, space(A, 1)⊗space(Z,1)⊗space(A, 1)', space(A, 1)'⊗space(Z,1)⊗space(A, 1))
    vrn0 = ones(ComplexF64, space(A, 1)'⊗space(Z,1)⊗space(A, 1), space(A, 1)⊗space(Z,1)⊗space(A, 1)')
    ftrn = fidelity_norm_transfer_right(A, Z)
    valsnr, vrns = eigsolve(x->ftrn(x), vrn0, 1, :LM)
    ftln = fidelity_norm_transfer_left(A, Z)
    valsnl, vlns = eigsolve(x->ftln(x), vln0, 1, :LM)

    r = vrs[1]  
    l = vls[1]


    ln = vlns[1]
    rn = vrns[1]

    @tensor trO = l[1; 2 3] * ftr(r)[2 3; 1]
    @tensor NO = l[1; 2 3] * r[2 3; 1]

    @tensor trOO = ln[1 2 3; 4 5 6] * ftrn(rn)[4 5 6; 1 2 3]
    @tensor NOO = ln[1 2 3; 4 5 6] * rn[4 5 6; 1 2 3]

    D = dim(space(A, 2))
    F = (trO/NO)*conj((trO/NO))/(D*(trOO/NOO))
    # @show F
    # imag(F) < 1e-6 || @warn("Fidelity has imaginary part: F = $F")

    return real.(1 - F)
end

function entanglement_error(A::AbstractMPOTensor, Z::AbstractMPOTensor)
    vln0 = ones(ComplexF64, space(A, 1)⊗space(Z,1)⊗space(A, 1)', space(A, 1)'⊗space(Z,1)⊗space(A, 1))
    vrn0 = ones(ComplexF64, space(A, 1)'⊗space(Z,1)⊗space(A, 1), space(A, 1)⊗space(Z,1)⊗space(A, 1)')
    ftrn = fidelity_norm_transfer_right(A, Z)
    valsnr, vrns = eigsolve(x->ftrn(x), vrn0, 1, :LM)
    ftln = fidelity_norm_transfer_left(A, Z)
    valsnl, vlns = eigsolve(x->ftln(x), vln0, 1, :LM)

    U, S, V = svd_full((vrns[1] * vlns[1]))
    S = S.data
    S /= sum(S)
    S = sort(S)
    @show S
    ϵ = abs(sum(S[1:end-1]))
    return ϵ
end


# Cost function for the inverse square root of the partition function
function cost_function(Zs::AbstractMPOTensor)
    function fg(As::OptimState)
        target_val = fidelity_identity(As.A, Zs)
        grad = gradient(x -> fidelity_identity(x, Zs), As.A)
        gradp = Stiefel.project(grad[1], As.A)
        return target_val, gradp
    end
end

function precondition(As::OptimState, g::Stiefel.StiefelTangent)
    A = unbend(As.A)
    f_r = transfer_right_mpo(A)
    v_r = rand(ComplexF64, space(A, 1), space(A, 1))
    _, rs = eigsolve(f_r, v_r, 1, :LM)
    r = rs[1]
    δ = As.δ
    Id = id(ComplexF64, space(r,1))
    rinv = inv(sqrt((r^2 + δ^2*Id)))
    U, S, V = svd_full(rinv)
    rinv = V'*S*V

    Z = g.Z * rinv

    A = sylvester(r,r,-2*g.A)

    gnew = Stiefel.StiefelTangent(g.W, A, Z)

    return gnew
end

function finalize!(x::OptimState, f::Float64, g::Stiefel.StiefelTangent, numiter::Int)
    δ = norm(g)
    x = OptimState(x.A, δ)
    return x, f, g
end

function retract(x::OptimState, η::Stiefel.StiefelTangent, α::Real)
    Anew, η2 =  Stiefel.retract(x.A, η, α)
    return OptimState(Anew, x.δ), η2
end

function inner(x::OptimState, η1::Stiefel.StiefelTangent, η2::Stiefel.StiefelTangent)
    return Stiefel.inner(x.A, η1, η2)
end

function transport!(ξ::Stiefel.StiefelTangent, x::OptimState, η::Stiefel.StiefelTangent, α::Real, x2::OptimState)
    ξ = Stiefel.transport(ξ, x.A, η, α, x2.A)
    return ξ
end

# Compute Z^(-1/2)
function inv_sqrt(Zs::AbstractMPOTensor, As::AbstractMPOTensor, alg::GradientOptimization)
    Astate = OptimState(bend(As), 1e-1)
    fg = cost_function(Zs)

    A_opt, _, _, _, gradhist = optimize(fg, Astate, alg.algorithm; (finalize!)=finalize!, (scale!) = Stiefel.scale!, retract=retract, inner=inner, (transport!)=transport!, (add!)=Stiefel.add!,  precondition = precondition)

    return unbend(A_opt.A)
end


