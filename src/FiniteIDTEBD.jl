# Groundstate of a FiniteDisorderMPS through imaginary time evolution:
# apply the disorder time evolution MPO and approximate the result with a gradient optimization of the overlap

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

# Time evolution MPO on a chain of length L with open boundary conditions:
# bulk tensors of the infinite time evolution MPO of every sector, where only the first row is kept
# on the first site and the first column on the last site (as in MPSKit.open_boundary_conditions).
# The coupling of bond (i, i+1) is set by the sector of site i, as in energy and the free fermion reference
function time_evolution_MPO(Hs::DisorderMPOHam, dt::Real, L::Int; N::Int = 1)
    Us = time_evolution_MPO(outgoing_coupling(Hs), dt; N = N)

    opp = map(1:L) do i
        map(Us.opp) do U
            Ui = U
            if i == 1
                el = zeros(ComplexF64, ℂ^1, space(U, 1))
                el[1, 1] = 1
                @tensor Ui[-1 -2; -3 -4] := el[-1; 1] * Ui[1 -2; -3 -4]
            end
            if i == L
                er = zeros(ComplexF64, space(U, 4)', ℂ^1)
                er[1, 1] = 1
                @tensor Ui[-1 -2; -3 -4] := Ui[-1 -2; -3 1] * er[1; -4]
            end
            Ui
        end
    end
    return FiniteDisorderMPO(opp)
end

# Multiply FiniteDisorderMPS with FiniteDisorderMPO
function Base.:*(O::FiniteDisorderMPO, ψ::FiniteDisorderMPS)
    (length(ψ) == length(O)) || throw(ArgumentError("O should have the same length as ψ"))
    (length(ψ[1]) == length(O[1])) || throw(ArgumentError("O should have the same amount of disorder sectors as ψ"))

    opp = map(1:length(ψ)) do i
        map(enumerate(ψ[i])) do (p, W)
            Op = O[i, p]
            isol = isomorphism(ComplexF64, fuse(space(W, 1), space(Op, 1)), space(W, 1) ⊗ space(Op, 1))
            isor = isomorphism(ComplexF64, fuse(space(W, 3)', space(Op, 4)'), space(W, 3)' ⊗ space(Op, 4)')
            @tensor Wnew[-1 -2; -4] := isol[-1; 1 2] * W[1 3; 4] * Op[2 -2; 3 5] * conj(isor[-4; 4 5])
        end
    end

    return FiniteDisorderMPS(opp, ψ.ps)
end

# Approximate ψ by a left-isometric FiniteDisorderMPS by maximizing the normalized overlap, starting from ϕ0
function approximate(ψ::FiniteDisorderMPS, ϕ0::FiniteDisorderMPS; gradtol=1e-12, maxiter=1000, verbosity=5)
    Nψ = sqrt(abs(overlap(ψ, ψ)))

    function fg(ρ::FiniteDisorderMPS)
        target_val = -abs(overlap(ρ, ψ))/Nψ
        grad = gradient(x -> -abs(overlap(x, ψ))/Nψ, ρ)
        gradp = project(grad[1].opp, ρ)

        return target_val, gradp
    end

    ψ_opt, f_opt, _, _, gradhist = optimize(fg, ϕ0, LBFGS(;maxiter=maxiter,verbosity=verbosity, gradtol = gradtol); retract = retract, inner = inner, (scale!) = scale!, (transport!) = transport!, (add!) = add!)
    return ψ_opt, 1 + f_opt, gradhist
end

struct Groundstate_FiniteDTEBD <: AbstractAlgorithm
    trunc_tol::Float64
    convtol::Float64
    maxiter::Int
    approx_maxiter::Int
    verbosity::Int
    timer_output::TimerOutput
    finalizer::Finalizer

    function Groundstate_FiniteDTEBD(; trunc_tol::Float64 = 1e-8, convtol::Float64 = 1e-8, maxiter::Int = 100, approx_maxiter::Int = 1000, verbosity::Int = 0, timer_output::TimerOutput = TimerOutput(), finalizer::Finalizer = Finalizer(Float64, energy_density))
        return new(trunc_tol, convtol, maxiter, approx_maxiter, verbosity, timer_output, finalizer)
    end
end

function groundstate(ψ0::FiniteDisorderMPS, Hs::DisorderMPOHam, dτ::Float64, alg::Groundstate_FiniteDTEBD; maxD::Int = 2)
    data = Vector{alg.finalizer.E}()
    info = GroundstateInfo(alg.maxiter)

    L = length(ψ0)
    D_dis = length(ψ0.ps)
    D_phys = dim(space(ψ0[1,1], 2))

    @timeit alg.timer_output "Copy" ψ = deepcopy(ψ0)
    @timeit alg.timer_output "construct_time_evolution_operator" Us = time_evolution_MPO(Hs, dτ, L; N = 4)

    ix = 0
    ϵ_conv = 1.0
    @timeit alg.timer_output "energy_density" Eprev = DisorderKit.energy_density(ψ, Hs)
    while (ϵ_conv > alg.convtol) && (ix+1 <= alg.maxiter)
        ix += 1
        (alg.verbosity > 0) && (@info "Iteration $ix")

        (alg.verbosity > 0) && (@info(crayon"cyan"("Evolve")))
        @timeit alg.timer_output "evolve_one_time_step" ψ_ev = Us * ψ

        (alg.verbosity > 0) && (@info(crayon"magenta"("Approximating ψ")))
        (alg.verbosity > 1) && (@info(crayon"magenta"("Before approximation: Bonddimensions of ψ = $([dim(space(ψ_ev[i,1],1)) for i in 2:L])")))

        # Start from the previous state once it has reached the maximal bond dimensions, otherwise from a random one
        ϕ0 = FiniteDisorderMPS(ψ.ps, L, D_dis, D_phys, maxD)
        if all(space(ϕ0[i,1]) == space(ψ[i,1]) for i in 1:L)
            ϕ0 = ψ
        end
        @timeit alg.timer_output "approximate" ψ, ϵ_approx, _ = approximate(ψ_ev, ϕ0; gradtol = alg.trunc_tol, maxiter = alg.approx_maxiter, verbosity = max(alg.verbosity-1, 0))
        info.ρ_trunc_err[ix] = ϵ_approx
        (alg.verbosity > 1) && (@info(crayon"magenta"("After approximation: Bonddimensions of ψ = $([dim(space(ψ[i,1],1)) for i in 2:L]), error = $(ϵ_approx)")))

        @timeit alg.timer_output "energy_density" E = DisorderKit.energy_density(ψ, Hs)
        dE = abs(Eprev - E)/dτ
        ϵ_conv = dE
        Eprev = E

        info.ϵsconv[ix] = ϵ_conv
        (alg.verbosity > 0) && (@info(crayon"light_blue"("Convergence error: (E, dE) = ($(E), $(dE))")))

        (alg.verbosity > 0) && (@info(crayon"cyan"("Finalize")))
        @timeit alg.timer_output "finalizer" push!(data, alg.finalizer.f!(ψ, Hs))
    end

    for v in (info.ϵsconv, info.ϵsent, info.ϵsz, info.ρ_trunc_err, info.R_trunc_err)
        resize!(v, ix)
    end
    return ψ, data, info
end
