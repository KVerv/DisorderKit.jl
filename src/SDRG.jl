function ising_sdrg(L::Int, nsamples::Int, random_J, random_h)
    Γs_all = Vector{Float64}[]
    Ms_all = Vector{Float64}[]
    J_dist = Dict{Float64, Int}[]
    h_dist = Dict{Float64, Int}[]
    for i in ProgressBar(1:nsamples)

        Js = [random_J() for _ in 1:L-1]
        hs = [random_h() for _ in 1:L]

        μs = ones(Float64, L)
        Nspins = L

        Ω₀ = maximum(vcat(Js, hs))
        Γs = Float64[]
        Ms = Float64[]
        while Nspins > 1
            Ω = maximum(vcat(Js, hs))
            push!(Γs, log(Ω₀/Ω))
            push!(Ms, sum(μs)/L)
            Js, hs, μs, Nspins = ising_sdrg_step(Js, hs, μs, Nspins)
            # @show countmap(Js)
            # @show countmap(hs)
            if Nspins == L/4
                push!(J_dist, countmap(Js))
                push!(h_dist, countmap(hs))
            end
        end

        push!(Γs_all, Γs)
        push!(Ms_all, Ms)
    end

    return hcat(Γs_all...), hcat(Ms_all...), J_dist, h_dist
end

function ising_sdrg_step(Js::Vector{Float64}, hs::Vector{Float64}, μs::Vector{Float64}, Nspins::Int)
    if Nspins == 1
        return Js, hs, μs, Nspins
    end

    max_J = maximum(Js)
    max_h = maximum(hs)

    if max_J > max_h
        # Decimate bond
        index = findfirst(==(max_J), Js)
        if index == 1
            new_h = hs[index] * hs[index+1] / Js[index]
            new_μ = μs[index] + μs[index+1]
            new_Js = Js[2:end]
            new_hs = vcat(new_h, hs[3:end])
            new_μs = vcat(new_μ, μs[3:end])
        elseif index == length(Js)
            new_h = hs[index] * hs[index+1] / Js[index]
            new_μ = μs[index] + μs[index+1]
            new_Js = Js[1:end-1]
            new_hs = vcat(hs[1:end-2], new_h)
            new_μs = vcat(μs[1:end-2], new_μ)
        else
            new_h = hs[index] * hs[index+1] / Js[index]
            new_μ = μs[index] + μs[index+1]
            new_Js = vcat(Js[1:index-1], Js[index+1:end])
            new_hs = vcat(hs[1:index-1], new_h, hs[index+2:end])
            new_μs = vcat(μs[1:index-1], new_μ, μs[index+2:end])
        end
    else
        # Decimate site
        index = findfirst(==(max_h), hs)
        if index == 1
            new_Js = Js[2:end]
            new_hs = hs[2:end]
            new_μs = μs[2:end]
        elseif index == length(hs)
            new_hs = hs[1:index-1]
            new_μs = μs[1:index-1]
            new_Js = Js[1:index-2]
        else
            new_J = Js[index-1] * Js[index] / hs[index]
            new_Js = vcat(Js[1:index-2], new_J, Js[index+1:end])
            new_hs = vcat(hs[1:index-1], hs[index+1:end])
            new_μs = vcat(μs[1:index-1], μs[index+1:end])
        end
    end

    Nspins -= 1

    return new_Js, new_hs, new_μs, Nspins
end
