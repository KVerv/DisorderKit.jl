
using TensorKit, CairoMakie, LsqFit

using JLD2, TimerOutputs
using DisorderKit


Δτ = 0.025 # Step size for imaginary time evolution
W = 0.3
δ = 0.0
D_R = 2
maxiter = 3000

Ds = [8]

hs = exp(δ) * [1., exp(-2W)]
Js = [1., exp(-2W)]
w1 = 0.5
w2 = 1-w1
ps = [w1*w1, w1*w2, w2*w1, w2*w2]

Hs = DisorderKit.random_transverse_field_ising(Js, hs)


set_theme!(theme_latexfonts())
fig1 = Figure(backgroundcolor=:white, fontsize=40, size=(3000, 2000))
ax1 = Axis(fig1[1, 1], 
        xlabel = L"τ",
        ylabel = L"$E$",
        xscale = log10,
        # yscale = log10
        )
ax2 = Axis(fig1[1, 2], 
        xlabel = L"τ",
        ylabel = L"$ϵ_{conv}$",
        xscale = log10,
        yscale = log10
        )
ax3 = Axis(fig1[2, 1], 
        xlabel = L"τ",
        ylabel = L"$M$",
        # xscale = log10,
        # yscale = log10
        )
ax4 = Axis(fig1[2, 2], 
        xlabel = L"τ",
        ylabel = L"$ξ$",
        # xscale = log10,
        # yscale = log10
        )

    fig3 = Figure(backgroundcolor=:white, fontsize=40, size=(3000, 2000))
    ax31 = Axis(fig3[1, 1], 
            xlabel = L"\ln r",
            ylabel = L"$\ln (C_r)$",
            # xscale = log10,
            # yscale = log10
    )
    ax32 = Axis(fig3[1, 2], 
            xlabel = L"\ln r",
            ylabel = L"$η$",
            # xscale = log10,
            # yscale = log10
    )


τs = Δτ:Δτ:maxiter*Δτ

SDs = []
ξDs = []
for D in Ds

    folder_name1 = "ρ_δ$(δ)_W$(W)_D$(D)_Δτ$(Δτ)_DR$(D_R)"

    println(folder_name1)
    isdir("data/$folder_name1") || error("Directory data/$folder_name1 does not exist. Please create it first.")

    τ = maxiter*Δτ
    state = load_object("data/$folder_name1/ρs_$(τ).jld2")
    data = load_object("data/$folder_name1/data_$(τ).jld2")
    info = load_object("data/$folder_name1/info_$(τ).jld2")


    Es = getindex.(data, 1)
    Ms = getindex.(data, 3)
    ξs = getindex.(data, 2)
    Ss = getindex.(data, 4)

    # E = Es[end]
    # ξ = ξs[end]
    # S = Ss[end]
    # M = Ms[end]
#     ix = min(1500, length(Es))
        ix = length(Es)
    E = Es[ix]
    ξ = ξs[ix]
    S = Ss[ix]
    M = Ms[ix]


    @show (E, ξ, M, S)

    Z = TensorMap([1. 0.; 0. -1.], ℂ^2, ℂ^2)
    rmax = round(Int, 3*ξ)
    Cs = real.(DisorderKit.two_point_correlator(state, Z, Z, rmax))
#     Cs .-= M^2
#     Cs ./= Cs[1]
    rs = 1:rmax

    η = -diff(log.(abs.(Cs)))./diff(log.(rs))
    rs *= 1/ξ
    # yss .*= rs.^0.38
#     Cs .*= ξ^0.35


    scatter!(ax31, log.(rs), log.(Cs), label=L"$D=%$D$", markersize=20)
    @show ξ
    @show η[1:3]
    scatter!(ax32, log.(rs)[1:end-1], η, label=L"$D=%$D$", markersize=20)

    if D==Ds[end]
            lines!(ax31, log.(rs), -0.25 *(log.(rs).-log.(rs[1])).+log.(Cs[1]), color=:black, linewidth=2)
            lines!(ax31, log.(rs), -0.38 *(log.(rs).-log.(rs[1])).+log.(Cs[1]), color=:red, linewidth=2)
        end



    colors = Makie.wong_colors()
    scatterlines!(ax1, τs[1:length(Es)], Es, label=L"$Δτ=%$Δτ$", markersize=20)
#     scatterlines!(ax2, τs[1:length(Es)], info.ϵsconv[1:length(Es)], label=L"$ϵ_{conv}$", markersize=20)
        scatterlines!(ax3, τs[1:length(Ms)], Ms, label=L"$ϵ_{acc}$", markersize=20)
    scatterlines!(ax4, τs[1:length(ξs)], ξs, label=L"$ϵ_{acc}$", markersize=20)
    push!(SDs, S)
    push!(ξDs, ξ)
end


fig2 = Figure(backgroundcolor=:white, fontsize=40, size=(3000, 2000))
ax1 = Axis(fig2[1, 1], 
        xlabel = L"\ln ξ",
        ylabel = L"$S$",
        # xscale = log10
        # yscale = log10
        )
scatterlines!(ax1, log.(ξDs), SDs, label=L"Data", markersize=20)

p0q = [1., 1.]
linmodel(t, p) = p[1] .+ p[2] * t
xs = log.(ξDs)
ys = SDs
linfit = curve_fit(linmodel, xs, ys, p0q)
@show linfit.param
c = linfit.param[2]*6

lines!(ax1, xs, linmodel(xs, linfit.param), label=L"$c=%$c$", color=:black, linewidth=2)

save("plots/D_ts.pdf",fig1)
save("plots/S_xi.pdf",fig2)
save("plots/C_r.pdf",fig3)
