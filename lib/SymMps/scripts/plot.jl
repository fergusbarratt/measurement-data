using BSON, UnicodePlots, Plots, LinearAlgebra, Statistics
using ColorSchemes

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
    for N in [8, 24, 32]
        for p in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
            if !((string(N), string(p)) in keys(all_data))
                continue
            end
            data = all_data[(string(N), string(p))]

            #data = BSON.load(data_loc*datapath)[:entropy]
            av_entropy = mean(data, dims=1)
            # places where uninitialized data didn't get overwritten
            av_entropy[av_entropy .> 100] .= NaN
            av_entropy[av_entropy .< -100] .= NaN
            av_entropy = (av_entropy[:, :, end]+av_entropy[:, :, end-1])/2

            if unicode
                plt = UnicodePlots.lineplot(1: size(av_entropy, 3), vec(av_entropy[:, size(av_entropy, 2) ÷ 2, :]))
                show(plt)
                println()
            else
                ## plot temporal entanglement
                #display(plot!(0:size(av_entropy, 3)-1, vec(av_entropy[:, size(av_entropy, 2) ÷ 2+1, :]), lab="$N, $p", legend=true))
                
                ## plot spatial entanglement
                plot!(pk, 1:size(av_entropy, 2), av_entropy'[1:end], lab="$N, $p", legend=:outertopleft, palette=palette(:seaborn_bright, 11, rev=false), marker=:x, linewidth=1.5)
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
        if !(N==24)
            continue
        end

        #zs1 = BSON.load(data_loc*datapath)[:eswaps][:, :, 1:2:end]
        zs1 = BSON.load(data_loc*datapath)[:eswaps]
        mzs1 = mean(zs1, dims=1)

        #zs2 = BSON.load(data_loc*datapath)[:oswaps][:, :, 1:2:end]
        zs2 = BSON.load(data_loc*datapath)[:oswaps]
        mzs2 = mean(zs2, dims=1)

        println(zs1[1, :, :])
        println(mzs1)
        println(mzs2)


        CC1 = -(zs1.*permutedims(zs1, (1, 3, 2)))#.-mzs1.*permutedims(mzs1, (1, 3, 2)))
        CC2 = -(zs2.*permutedims(zs2, (1, 3, 2)))#.-mzs2.*permutedims(mzs2, (1, 3, 2)))
        #CC = (CC1+CC2)/2
        CC = CC1


        av_correlations = mean(CC, dims=1)
        #Plots.heatmap(av_correlations[1, :, :], yflip=true)
        # display(diag(av_correlations[1, :, :], 0))
        # display(diag(av_correlations[1, :, :], 2))
        # plot!(xlabel='x')
        # Plots.savefig("figs/heatmap.pdf")
        #av_correlations = mean(data.^2, dims=1)
        #av_correlations = mean(data, dims=1)
        K = size(CC, 2)
        f = 4 # middle (k-2/k) of system (K=4 is half)
        a, b = (K÷f, (f-1)*(K÷f))
        av_correlations_bulk = av_correlations[1, a+1:b, a+1:b]

        function corr_vs_r(A)
            K = size(A, 1)
            rs = 1:(K-1)
            diags = map(k->diag(A, k), rs)
            corrs = map(x -> mean(x), diags)
            #corrs = map(x -> x[1], diags)
            return rs, corrs
        end

        x, corrs = corr_vs_r(av_correlations_bulk)
        if unicode
            plt = UnicodePlots.lineplot(1: size(av_correlations, 3), vec(av_correlations[:, size(av_correlations, 2) ÷ 2, :]))
            show(plt)
            println()
        else
            ## plot temporal entanglement
            #display(plot!(0:size(av_correlations, 3)-1, vec(av_correlations[:, size(av_correlations, 2) ÷ 2+1, :]), lab="$N, $p", legend=true))
            
            ## plot spatial entanglement
            display(plot!(x, -corrs, lab="$N, $p", marker=:x, legend=:outertopleft))#, palette=palette(:viridis, 3)))
            plot!(xlabel="\$r\$", ylabel="\$\\overline{C(r)}\$")
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
    for N in [8, 24, 32]
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
    for N in [8, 24]
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
readspatialentropydata("./data/entropy/")
readcorrelationsdata("./data/correlations/")
#readcorrelationsdata("./data/correlations/")
