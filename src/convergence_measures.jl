struct ConvergenceMeasure
    E::Type
    f!::Function

    function ConvergenceMeasure(E::Type, f::Function)
        return new(E, f)
    end
end


function E_gradient(ρ::InfiniteDisorderDensityMatrix, ρprev::InfiniteDisorderDensityMatrix, Hs::DisorderMPOHam, dτ::Float64)
    E = energy_density(ρ, Hs)
    Eprev = energy_density(ρprev, Hs)
    ϵ_conv = (abs(Eprev - E)/dτ)
    return ϵ_conv
end
