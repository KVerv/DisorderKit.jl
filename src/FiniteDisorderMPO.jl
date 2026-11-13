# Finite time evolution MPO of a disordered chain: one MPO tensor per site and disorder sector

# Finite DisorderMPO: opp[i][p] is the MPO tensor at site i for disorder sector p
struct FiniteDisorderMPO{T<:AbstractMPOTensor}
    opp::Vector{Vector{T}}
end

Base.getindex(O::FiniteDisorderMPO, i::Int) = O.opp[i]
Base.getindex(O::FiniteDisorderMPO, i::Int, p::Int) = O.opp[i][p]
Base.size(O::FiniteDisorderMPO) = size(O.opp)
Base.length(O::FiniteDisorderMPO) = length(O.opp)
Base.iterate(O::FiniteDisorderMPO, i=1) = (i > length(O.opp)) ? nothing : (O[i], i + 1)

# Move the disorder dependence of the bond term Ls[p]⊗Rs[p] onto the left site: Rs[p] = α_p R_ref gives
# Ls[p] → α_p Ls[p] and Rs[p] → R_ref, such that the coupling of bond (i, i+1) is set by the sector of site i
function outgoing_coupling(Hs::DisorderMPOHam)
    R_ref = Hs.Rs[1]
    nR = real(dot(R_ref, R_ref))
    αs = map(Hs.Rs) do R
        α = dot(R_ref, R)/nR
        TensorKit.norm(R - α*R_ref) <= 1e-12*max(TensorKit.norm(R), 1.0) || throw(ArgumentError("Rs[p] should all be proportional to each other"))
        α
    end
    return DisorderMPOHam(Hs.As, fill(R_ref, length(Hs.Rs)), [α*L for (α, L) in zip(αs, Hs.Ls)], Hs.Ds)
end

# Time evolution MPO on an open chain with the disorder sectors of every site, sectors[i][p] the Hamiltonian sector of
# sector p on site i (as in FiniteDisorderMPS): bulk tensors of the infinite time evolution MPO of every sector, where
# only the first row is kept on the first site and the first column on the last site (as in MPSKit.open_boundary_conditions).
# The coupling of bond (i, i+1) is set by the sector of site i, as in energy and the free fermion reference.
# compress_tol > 0: SVD compression of the bonds afterwards (compress_mpo), relative cutoff compress_tol
function time_evolution_MPO(Hs::DisorderMPOHam, dt::Real, sectors::Vector{<:Vector{<:Integer}}; N::Int = 1,
                            compress_tol::Real = 1e-12)
    Us = time_evolution_MPO(outgoing_coupling(Hs), dt; N = N)
    L = length(sectors)

    opp = map(1:L) do i
        map(Us.opp[sectors[i]]) do U
            Ui = U
            # The first virtual basis vector is the first one of the trivial sector (the unit block comes first)
            if i == 1
                el = zeros(ComplexF64, oneunit(space(U, 1)), space(U, 1))
                dense_block(el)[1, 1] = 1
                @plansor Ui[-1 -2; -3 -4] := el[-1; 1] * Ui[1 -2; -3 -4]
            end
            if i == L
                er = zeros(ComplexF64, space(U, 4)', oneunit(space(U, 4)))
                dense_block(er)[1, 1] = 1
                @plansor Ui[-1 -2; -3 -4] := Ui[-1 -2; -3 1] * er[1; -4]
            end
            Ui
        end
    end
    U = FiniteDisorderMPO(opp)
    return compress_tol > 0 ? compress_mpo(U; rtol = compress_tol) : U
end

# SVD compression of the bonds of a finite disorder MPO. The Taylor cluster MPO of MPSKit keeps every cluster channel,
# far more than needed: e.g. for the golden chain at order 4 the bond has 78 multiplets of which 9 carry singular values
# above 1e-10. A bond is shared by all disorder sectors of its two sites, so the sectors are compressed jointly: site i
# is stacked to M_i = Σ_u U_{i,u} ⊗ e_u with the disorder index (trivial sector) as an extra physical leg, as the disorder
# leg of the SRC cores. Left QR sweep, then right SVD sweep that drops the singular values below rtol times the largest
# one of the cut. Planar layout M[t h s u; j] (W[h s; t j], t input and s output spin), so it works for every sector type
function compress_mpo(U::FiniteDisorderMPO; rtol::Real = 1e-12)
    L = length(U)
    S = spacetype(U[1, 1])
    Ms = map(1:L) do i
        q = length(U[i])
        sum(enumerate(U[i])) do (u, W)
            transpose(W, ((3, 1, 2), (4,))) ⊗ basis_ket(S, q, u, ComplexF64)    # t h s u ← j
        end
    end
    for i in 1:L-1                                                   # left orthonormal
        Q, R = left_orth(Ms[i])
        Ms[i] = Q
        M = Ms[i+1]
        @plansor Mn[-1 -2 -3 -4; -5] := R[-2; 1] * M[-1 1 -3 -4; -5]
        Ms[i+1] = Mn
    end
    for i in L:-1:2                                                  # truncated right sweep
        Uh, Σ, Vh, = robust_svd_trunc(transpose(Ms[i], ((2,), (1, 5, 4, 3))); trunc = trunctol(; rtol = rtol, p = Inf))   # h ← t j u s
        Ms[i] = transpose(Vh, ((2, 1, 5, 4), (3,)))                  # t m s u ← j
        US = Uh * Σ
        M = Ms[i-1]
        @plansor Mp[-1 -2 -3 -4; -5] := M[-1 -2 -3 -4; 1] * US[1; -5]
        Ms[i-1] = Mp
    end
    opp = map(1:L) do i
        q = length(U[i])
        M = Ms[i]
        map(1:q) do u
            e = basis_ket(S, q, u, ComplexF64)
            @plansor X[-1 -2 -3; -4] := M[-1 -2 -3 1; -4] * conj(e[1])
            transpose(X, ((2, 3), (1, 4)))                           # W[h s; t j]
        end
    end
    return FiniteDisorderMPO(opp)
end

# Multiply FiniteDisorderMPS with FiniteDisorderMPO
function Base.:*(O::FiniteDisorderMPO, ψ::FiniteDisorderMPS)
    (length(ψ) == length(O)) || throw(ArgumentError("O should have the same length as ψ"))
    all(length(ψ[i]) == length(O[i]) for i in 1:length(ψ)) || throw(ArgumentError("O should have the same amount of disorder sectors as ψ on every site"))

    opp = map(1:length(ψ)) do i
        map(enumerate(ψ[i])) do (p, W)
            Op = O[i, p]
            isol = isomorphism(ComplexF64, fuse(space(W, 1), space(Op, 1)), space(W, 1) ⊗ space(Op, 1))
            isor = isomorphism(ComplexF64, fuse(space(W, 3)', space(Op, 4)'), space(W, 3)' ⊗ space(Op, 4)')
            @plansor Wnew[-1 -2; -4] := isol[-1; 1 2] * W[1 3; 4] * Op[2 -2; 3 5] * conj(isor[-4; 4 5])
        end
    end

    return FiniteDisorderMPS(opp, ψ.ps, ψ.sectors)
end
