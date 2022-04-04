using BSON, UnicodePlots, Plots, LinearAlgebra, Statistics
using ColorSchemes
Ns = [8]
"""
data_loc should be the location of the folder containing the bson files with the entropy data
"""
function readspatialentropydata(data_loc="./entropy/", unicode=false)  
    all_data = Dict()
    for datapath = readdir(data_loc)
        N, t_max, p = split(splitext(datapath)[1], ',')
        #println(N, ',', t_max, ',', p)

        data = BSON.load(data_loc*datapath)[:entropy]
        if (N, p) in keys(all_data)
            all_data[(N, p)] = cat(all_data[(N, p)], data, dims=[1])
        else
            all_data[(N, p)] = data
        end
    end
    pk = plot()
    for N in Ns
        for p in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
            data = all_data[(string(N), string(p))]

            #data = BSON.load(data_loc*datapath)[:entropy]
            av_entropy = mean(data, dims=1)
            # places where uninitialized data didn't get overwritten
            av_entropy[av_entropy .> 100] .= NaN
            av_entropy[av_entropy .< -100] .= NaN
            av_entropy = av_entropy[:, :, end]

            if unicode
                plt = UnicodePlots.lineplot(1: size(av_entropy, 3), vec(av_entropy[:, size(av_entropy, 2) ÷ 2, :]))
                show(plt)
                println()
            else
                ## plot temporal entanglement
                #display(plot!(0:size(av_entropy, 3)-1, vec(av_entropy[:, size(av_entropy, 2) ÷ 2+1, :]), lab="$N, $p", legend=true))
                
                ## plot spatial entanglement
                plot!(pk, 1:2:size(av_entropy, 2), av_entropy'[1:2:end], lab="$N, $p", legend=:outertopleft, palette=palette(:seaborn_bright, 11, rev=false), marker=:x, linewidth=1.5)
                plot!(pk, xlabel="\$n\$", ylabel="\$S_E(n)\$")
                plot!(pk, legendtitle="\$L, p_m, p_u\$")
                plot!(pk, xscale=:log10)
                Plots.savefig(pk, "figs/spatial_entanglement.pdf")
            end
        end
    end
end

function readcorrelationsdata(data_loc="./correlations/", unicode=false)  
    plot()
    for datapath = readdir(data_loc)
        N, t_max, p = split(splitext(datapath)[1], ',')
        N, t_max = map(a->parse(Int64, a), [N, t_max])
        p = parse(Float64, p)
        if !(N == Ns[1])
            continue
        end

        CCs = BSON.load(data_loc*datapath)[:correlations]
        zs = BSON.load(data_loc*datapath)[:zs]
        a, b = (N÷4, 3*(N÷4))
        println(a, ',', b)
        c1 = mean(CCs, dims=1)[1, :, :]
        c2 = mean(zs.*permutedims(zs, (1, 3, 2)), dims=1)[1, :, :]
        av_correlations_bulk = c1-c2
        #av_correlations = mean(data.^2, dims=1)
        #av_correlations = mean(data, dims=1)

        function corr_vs_r(A)
            K = size(A, 1)
            diags = map(k->diag(A, k), 1:(K-1))
            corrs = map(x -> mean(x), diags)
            #corrs = map(x -> x[1], diags)
            return (1:(K-1), corrs)
        end

        if unicode
            plt = UnicodePlots.lineplot(1: size(av_correlations, 3), vec(av_correlations[:, size(av_correlations, 2) ÷ 2, :]))
            show(plt)
            println()
        else
            ## plot temporal entanglement
            #display(plot!(0:size(av_correlations, 3)-1, vec(av_correlations[:, size(av_correlations, 2) ÷ 2+1, :]), lab="$N, $p", legend=true))
            
            ## plot spatial entanglement
            x, corrs = corr_vs_r(c1)
            p1 = plot(x, corrs, lab="$N, $p", marker=:x)#, palette=palette(:viridis, 3)))
            plot!(xlabel="\$r\$", ylabel="\$\\overline{C(r)}\$", title="<ZZ>")
            
            x, corrs = corr_vs_r(c2)
            p2 = plot(x, corrs, lab="$N, $p", marker=:x)#, palette=palette(:viridis, 3)))
            plot!(xlabel="\$r\$", ylabel="\$\\overline{C(r)}\$", title="<Z><Z>")
            x, corrs = corr_vs_r(c1-c2)
            p3 = plot(x, corrs, lab="$N, $p", marker=:x)#, palette=palette(:viridis, 3)))
            plot!(xlabel="\$r\$", ylabel="\$\\overline{C(r)}\$", title="<ZZ>-<Z><Z>")

            plot(p1, p2, p3)
        end
    end
    Plots.savefig("figs/correlations.pdf")
end

function readtemporalentropydata(data_loc="./entropy/", unicode=false)  
    all_data = Dict()
    for datapath = readdir(data_loc)
        N, t_max, p = split(splitext(datapath)[1], ',')

        data = BSON.load(data_loc*datapath)[:entropy]
        if (N, p) in keys(all_data)
            all_data[(N, p)] = cat(all_data[(N, p)], data, dims=[1])
        else
            all_data[(N, p)] = data
        end
    end
    pk = plot()
    for N in Ns
        p = 1.0
        arr = all_data[(string(N), string(p))]

        n_samples = size(arr, 1)
        av_entropy = mean(arr, dims=1)
        
        # places where uninitialized data didn't get overwritten
        av_entropy[av_entropy .> 100] .= NaN
        av_entropy[av_entropy .< -100] .= NaN
        if unicode
            plt = UnicodePlots.lineplot(1: size(av_entropy, 3), vec(av_entropy[:, size(av_entropy, 2) ÷ 2, :]))
            show(plt)
            println()
        else
            ## plot temporal entanglement
            plot!(pk, 1:size(av_entropy, 3), vec(av_entropy[:, size(av_entropy, 2) ÷ 2+1, :])[1:end], lab="$n_samples, $N, $p", legend=:outertopleft, xscale=:log10, marker=:x)
            plot!(pk, xlabel="\$t\$", ylabel="\$S_E(t)\$")
            plot!(pk, legendtitle="\$n, L, p_m, p_u\$")
            display(pk)
        end
    end
    Plots.savefig("figs/measurement_only_temporal_entanglement.pdf")

    pk = plot()
    for N in Ns
        for p in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
            if (string(N), string(p)) in keys(all_data)
                arr = all_data[(string(N), string(p))]

                n_samples = size(arr, 1)
                av_entropy = mean(arr, dims=1)
                
                # places where uninitialized data didn't get overwritten
                av_entropy[av_entropy .> 100] .= NaN
                av_entropy[av_entropy .< -100] .= NaN
                if unicode
                    plt = UnicodePlots.lineplot(1: size(av_entropy, 3), vec(av_entropy[:, size(av_entropy, 2) ÷ 2, :]))
                    show(plt)
                    println()
                else
                    ## plot temporal entanglement
                    plot!(pk, 1:size(av_entropy, 3), vec(av_entropy[:, size(av_entropy, 2) ÷ 2+1, :])[1:end], lab="$n_samples, $N, $p", legend=:outertopleft, xscale=:log10, marker=:x)
                    plot!(pk, xlabel="\$t\$", ylabel="\$S_E(t)\$")
                    plot!(pk, legendtitle="\$n, L, p_m, p_u\$")
                    display(p)
                end
            end
        end
    end
    Plots.savefig("figs/temporal_entanglement.pdf")
end

readtemporalentropydata("./data/entropy/")
readcorrelationsdata("./data/correlations/")
