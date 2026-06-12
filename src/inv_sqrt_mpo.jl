function make_inv_sqrt_mpo(
        H::MPOHamiltonian, factors::Vector, alg::MPSKit.TaylorCluster;
        tol = eps(real(scalartype(H)))
    )
    return _make_inv_sqrt_mpo(H, factors, Val(alg.N), alg.extension, alg.compression; tol)
end

function _make_inv_sqrt_mpo(
        H::MPOHamiltonian, factors::Vector, ::Val{N},
        extension::Bool, compression::Bool;
        tol
    ) where {N}
    H_n = H^N
    virtual_sz = map(0:length(H)) do i
        return i == 0 ? size(H[1], 1) : size(H[i], 4)
    end
    linds = map(virtual_sz) do sz
        return LinearIndices(ntuple(Returns(sz), Val(N)))
    end


    mpo = MPO(map(SparseBlockTensorMap, parent(H_n)))
    _taylor_loopback!(mpo, virtual_sz, linds, Val(N), factors)
    _taylor_remove_equivalents!(mpo, virtual_sz, linds)

    return MPSKit.remove_orphans!(mpo; tol)
end

# Stable partition: elements equal to `sentinel` first, others after, preserving relative order within each group.
@inline function _partition_first(t::NTuple{M, Int}, sentinel::Int) where {M}
    n_match = count(==(sentinel), t)
    return ntuple(Val(M)) do j
        if j <= n_match
            _kth_match(t, sentinel, j, true)
        else
            _kth_match(t, sentinel, j - n_match, false)
        end
    end
end

@inline function _kth_match(t::NTuple{M, Int}, sentinel::Int, k::Int, want_match::Bool) where {M}
    cnt = 0
    for x in t
        if (x == sentinel) == want_match
            cnt += 1
            cnt == k && return x
        end
    end
    return 0
end

# Algorithm 1: project the auxiliary virtual-bond directions onto the physical
# block, completing the Nth-order time-evolution MPO.
function _taylor_loopback!(mpo, virtual_sz, linds, ::Val{N}, factors::Vector) where {N}
    for (i, slice) in enumerate(parent(mpo))
        V_right = virtual_sz[i + 1]
        linds_right = linds[i + 1]
        cinds_right = CartesianIndices(linds_right)
        for b in cinds_right[2:end]
            all(in((1, V_right)), b.I) || continue

            b_lin = linds_right[b]
            a = count(==(V_right), b.I)
            factor = factors[a] * factorial(a) * factorial(N - a) / factorial(N)
            slice[:, 1, 1, 1] = slice[:, 1, 1, 1] + factor * slice[:, 1, 1, b_lin]
            for I in nonzero_keys(slice)
                (I[1] == b_lin || I[4] == b_lin) && delete!(slice, I)
            end
        end
    end
    return mpo
end

# Algorithm 2: collapse rows and columns that are equivalent under the
# permutation symmetry of the Taylor expansion.
function _taylor_remove_equivalents!(mpo, virtual_sz, linds)
    for (i, slice) in enumerate(parent(mpo))
        V_left = virtual_sz[i]
        linds_left = linds[i]
        for c in CartesianIndices(linds_left)
            c_lin = linds_left[c]
            s_c = CartesianIndex(_partition_first(c.I, 1))

            n1 = count(==(1), c.I)
            n3 = count(==(V_left), c.I)

            if n3 <= n1 && s_c != c
                for k in 1:size(slice, 4)
                    I = CartesianIndex(c_lin, 1, 1, k)
                    if I in nonzero_keys(slice)
                        slice[linds_left[s_c], 1, 1, k] += slice[I]
                        delete!(slice, I)
                    end
                end
            end
        end

        V_right = virtual_sz[i + 1]
        linds_right = linds[i + 1]
        for c in CartesianIndices(linds_right)
            c_lin = linds_right[c]
            s_r = CartesianIndex(_partition_first(c.I, V_right))

            n1 = count(==(1), c.I)
            n3 = count(==(V_right), c.I)

            if n3 > n1 && s_r != c
                for k in 1:size(slice, 1)
                    I = CartesianIndex(k, 1, 1, c_lin)
                    if I in nonzero_keys(slice)
                        slice[k, 1, 1, linds_right[s_r]] += slice[I]
                        delete!(slice, I)
                    end
                end
            end
        end
    end
    return mpo
end
