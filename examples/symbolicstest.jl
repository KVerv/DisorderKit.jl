using Symbolics

N = 10
# generate W matrices
@variables A[1:N] R[1:N] L[1:N] D[1:N] U[1:N] t
Ws = map(1:N) do l
    return [1 A[l] 0 0
            0 0 L[l] D[l]
            0 0 0 R[l]
            t*U[l] 0 0 0]
end

Xs = map(1:N) do l
    return [1 A[l] 0 0 0
            0 0 L[l] D[l] 0
            0 0 0 R[l] 0
            0 0 0 0 U[l]
            0 0 0 0 1]
end

# Ys = map(1:N) do l
#     return [1-D[l]^2 L[l]+D[l]*L[l] L[l]-L[l]*D[l] L[l]^2; R[l]+D[l]*R[l] A[l]+D[l]*A[l] -L[l]*R[l] A[l]*L[l];
#             R[l]-D[l]*R[l] -L[l]*R[l] A[l]-D[l]*A[l] A[l]*L[l]; -R[l]^2 A[l]*R[l] A[l]*R[l] A[l]^2]
# end

# Ys = map(1:N) do l
#     return kron([1-D[l]+D[l]^2 L[l]; -R[l] A[l]], [1+D[l] L[l]; +R[l] A[l]])
# end

# Xs = map(1:N) do l
#     return [1 - 1/2*D[l] + 3/8*D[l]^2 L[l]
#             -1/2*R[l] A[l]]
# end

# Xs = map(1:N) do l
#     return [1 - 1*(D[l] + τD[l]) + 1*(D[l]+ τD[l])^2 L[l]+ τL[l]
#             -1*(R[l]+ τR[l]) A[l]+ τA[l]]
# end

# generate boundary vectors
Vₗ = [1, 0, 0, 0]'
Vᵣ = [1, 0, 0, 0]
Vhl = [1, 0, 0, 0, 0]'
Vhr = [0, 0, 0, 0, 1]

# expand the MPO
Z = expand(Vₗ * prod(Ws) * Vᵣ) 
H = expand(Vhl * prod(Xs) * Vhr)
# expId = expand(Zinv*Z*Zinv)
# expId = expand(Zinv*Z)
# expId = expand(Vₗ * prod(Ys) * Vᵣ)

Zt = terms(Z)
Zt = sort(Zt, by = Symbolics.degree)

Ht = terms(H)
Ht = sort(Ht, by = Symbolics.degree)