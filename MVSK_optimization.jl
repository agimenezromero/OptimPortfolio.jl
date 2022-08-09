λs = parse.(Float64, [ARGS[1], ARGS[2], ARGS[3], ARGS[4]])
idx = Int(parse.(Float64, ARGS[5]) - 1)

using Statistics
using DelimitedFiles

include("OptimPortfolio.jl")
include("Utils.jl")

#LOAD DATA
returns, assets = readdlm("returns.txt", header=true)

#CONSTRUCT PORTFOLIO
portfolio = create_portfolio(returns, assets)

#OPTIMIZE
result_MVSK, w_opt_MVSK = MVSK(portfolio, λs)

#WRITE RESULTS
results_string = ["returns" "risk" "skewness" "kurtosis"]
λs_string = ["λ1" "λ2" "λ3" "λ4"]

w_string = string.(zeros((1, 10)))

for i in 1 : length(portfolio.assets)
   
    w_string[i] = "w_$(i)"
    
end

header = hcat(results_string, λs_string, w_string)
total_info = hcat(transpose(result_MVSK), transpose(λs), transpose(w_opt_MVSK))

f = open("HD_efficient_frontier/results_$(idx).txt", "w")

writedlm(f, header)
writedlm(f, total_info)

close(f)
