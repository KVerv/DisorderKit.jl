using Symbolics

N = 3
# generate W matrices
@variables A[1:N] R[1:N] L[1:N] D[1:N]
Ws = map(1:N) do l
    return [1 + D[l] L[l]
            R[l] A[l]]
end

Xs = map(1:N) do l
    return [1 - D[l] L[l]
            -R[l] A[l]]
end

# Ys = map(1:N) do l
#     return [1-D[l]^2 L[l]+D[l]*L[l] L[l]-L[l]*D[l] L[l]^2; R[l]+D[l]*R[l] A[l]+D[l]*A[l] -L[l]*R[l] A[l]*L[l];
#             R[l]-D[l]*R[l] -L[l]*R[l] A[l]-D[l]*A[l] A[l]*L[l]; -R[l]^2 A[l]*R[l] A[l]*R[l] A[l]^2]
# end

Ys = map(1:N) do l
    return kron([1-D[l]+D[l]^2 L[l]; -R[l] A[l]], [1+D[l] L[l]; +R[l] A[l]])
end

# Xs = map(1:N) do l
#     return [1 - 1/2*D[l] + 3/8*D[l]^2 L[l]
#             -1/2*R[l] A[l]]
# end

# Xs = map(1:N) do l
#     return [1 - 1*(D[l] + τD[l]) + 1*(D[l]+ τD[l])^2 L[l]+ τL[l]
#             -1*(R[l]+ τR[l]) A[l]+ τA[l]]
# end

# generate boundary vectors
Vₗ = [1, 0, 0 ,0]'
Vᵣ = [1, 0, 0 ,0]

# expand the MPO
Z = expand(Vₗ * prod(Ws) * Vᵣ) 
Zinv = expand(Vₗ * prod(Xs) * Vᵣ)
# expId = expand(Zinv*Z*Zinv)
# expId = expand(Zinv*Z)
expId = expand(Vₗ * prod(Ys) * Vᵣ)

Zt = terms(Zinv)
Zt = sort(Zt, by = Symbolics.degree)
t = terms(expId)
t = sort(t, by = Symbolics.degree)