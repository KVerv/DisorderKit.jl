struct Finalizer
    E::Type
    f!::Function

    function Finalizer(E::Type, f::Function)
        return new(E, f)
    end
end


function finalize!(ρ::InfiniteDisorderDensityMatrix, Hs::DisorderMPOHam)
    E = energy_density(ρ, Hs)
    return E
end

default_Finalizer = Finalizer(Float64, finalize!)