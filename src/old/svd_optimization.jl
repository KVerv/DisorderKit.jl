
function contraction(ρ::AbstractTensorMap{T, R, 3, 3}, X::AbstractTensorMap{T, S, 1, 1}) where {T, R, S}
    @tensor M[-1; -2] := ρ[1 2 3; 2 3 4] * X[-1; 1] * conj(X[-2; 4])
    return M
end

function get_λ(ρ::AbstractTensorMap{T, R, 3, 3}, X::AbstractTensorMap{T, S, 1, 1}) where {T, R, S}
    x0 = Zygote.ignore_derivatives() do
        x0 = Tensor(rand, ComplexF64, space(X)[1])
        return x0
    end
    λs, _ = eigsolve(x -> contraction(ρ,X)*x, x0, 1, :LM)
    return norm(λs[1])
end

function target_λ(ρs::DisorderMPO)
    ρ = ρs[1]
    function fg(X::AbstractTensorMap{T, S, 1, 1}) where {T, S}
        target_val = get_λ(ρ, X)
        grad = gradient(x -> get_λ(ρ, x), X)
        return target_val, grad[1]
    end
    return fg
end

function insert_isometry(ρ::DisorderMPO, X::AbstractTensorMap{T, S, 1, 1}) where {T, S}
    @tensor ρ2[-1 -2 -3; -4 -5 -6] := ρ[1][1 -2 -3; -4 -5 2] * X[-1; 1] * conj(X[-6; 2])
    return ρ2
end

# function mpo_transf(ρ1::AbstractDisorderMPOTensor, ρ2::AbstractDisorderMPOTensor, v::AbstractTensorMap{T,S, 1, 1}, P::AbstractTensorMap{Q, R, 1, 1}) where {T, S, R, Q}
#     @tensor Tv[-1; -2] := P[7; 5] * ρ1[1 3 5; 4 6 -2] * conj(ρ2[2 3 7; 4 6 -1]) * v[2; 1]
#     return Tv
# end

function mpo_transf(ρ1::AbstractDisorderMPOTensor, ρ2::AbstractDisorderMPOTensor, v::AbstractTensorMap{T,S, 1, 1}, P::AbstractTensorMap{Q, R, 1, 1}) where {T, S, R, Q}
    @tensor Tv[-1; -2] := ρ1[1 3 5; 4 6 -2] * conj(ρ2[2 3 5; 4 6 -1]) * v[2; 1]
    return Tv
end

function mpo_ovlp(ρ1::AbstractDisorderMPOTensor, ρ2::AbstractDisorderMPOTensor, ps::Vector{<:Real})
    V1 = space(ρ1, 1)
    V2 = space(ρ2, 1)
    D = space(ρ1, 3)
    P = TensorMap(diagm(ps), D, D)

    v0 = TensorMap(rand, ComplexF64, V2, V1)
    λs, _ = eigsolve(x -> mpo_transf(ρ1,ρ2,x,P), v0, 1, :LM)
    return λs[1]
end

function mpo_fidelity(ρ::DisorderMPO, X::AbstractTensorMap{T, S, 1, 1}, ps::Vector{<:Real}) where {T, S}
    ρ1 = ρ[1]
    ρ2 = insert_isometry(ρ, X)
    return norm(mpo_ovlp(ρ1, ρ2, ps) * mpo_ovlp(ρ2, ρ1, ps) /(mpo_ovlp(ρ1, ρ1, ps) * mpo_ovlp(ρ2, ρ2, ps)))
end

function target_fid(ρs::DisorderMPO, ps::Vector{<:Real})
    function fg(X::AbstractTensorMap{T, S, 1, 1}) where {T, S}
        target_val = mpo_fidelity(ρs, X, ps)
        grad = gradient(x -> mpo_fidelity(ρs, x, ps), X)
        return target_val, grad[1]
    end
    return fg
end

function svd_update(grad)
    U, _, V = tsvd(grad)
    return U*V
end

function optimize_isometry(ρ::DisorderMPO, X::AbstractTensorMap{T, S, 1, 1}, ps::Vector{<:Real}; conv_tol::Float64 = 1e-6, f_tol::Float64 = 1e-6, maxit::Int = 50) where {T, S}
    X1 = X
    ϵ = 1
    ϵ_f = 1
    ix = 1
    val_old = 1
    while (ϵ > conv_tol) && (ix < maxit) && (ϵ_f > f_tol)
        @info(crayon"red"("Iteration $ix:"))
        # target_val, grad = target_λ(ρ)(X1)
        target_val, grad = target_fid(ρ, ps)(X1)
        ϵ_f = abs(1-target_val)
        @info(crayon"red"("Target function: $target_val"))
        @info(crayon"red"("Target convergence: $ϵ_f"))
        X2 = svd_update(grad)
        # Gauge fix
        # U = X2*X1'
        # @show norm(X1'*X1-X2'*X2)
        # ϵ = norm(X2-X1)
        ϵ = abs(target_val - val_old)
        @info(crayon"red"("Convergence: $ϵ"))
        X1 = X2
        val_old = target_val
        ix += 1
    end
    # target_val, grad = target_λ(ρ)(X1)
    target_val, grad = target_fid(ρ, ps)(X1)
    @info(crayon"red"("Final Target function: $target_val"))
    return X1
end