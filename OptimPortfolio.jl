using Convex
using SCS
using Evolutionary

#Portfolio object
mutable struct Portfolio
   
    μ::Array{Float64} #Array of mean returns for each asset in the portolio
    
    Σ::Matrix{Float64} #Covariance matrix for the assets in the portfolio
    
    w::Array{Float64} #Weights to be optimized
    
    Portfolio(μ, Σ) = new(μ, Σ)
        
end

function MPT_fixed_return(portfolio::Portfolio, target_return; w_lower=0.0, w_upper=1.0)
   
    w = Variable(length(portfolio.μ))
    
    ret = dot(w, portfolio.μ)
    
    risk = quadform(w, portfolio.Σ)
        
    p = minimize(risk, ret >= target_return, sum(w)==1, w>=w_lower, w<=w_upper)
    
    solve!(p, SCS.Optimizer, silent_solver = true)
    
    portfolio.w = evaluate(w)
        
end

function MPT_fixed_risk(portfolio::Portfolio, target_risk, w_lower=0.0, w_upper=1.0)
   
    w = Variable(length(portfolio.μ))
    
    ret = dot(w, portfolio.μ)
    
    risk = quadform(w, portfolio.Σ)
    
    p = maximize(ret, risk <= target_risk^2, sum(w)==1, w>=w_lower, w<=w_upper)
    
    solve!(p, SCS.Optimizer, silent_solver = true)
    
    portfolio.w = evaluate(w)
    
end

function MPT_efficient_frontier(portfolio, points, w_lower=0.0, w_upper=1.0)
    
    w = Variable(length(portfolio.μ))
    
    ret = dot(w, portfolio.μ)
    
    risk = quadform(w, portfolio.Σ)
    
    λs = range(0.0, stop=1, length=points)
    
    MeanVar = zeros(points, 2)
    
    weights = zeros(points, length(portfolio.μ))
    
    for i in 1:points
        
        λ = λs[i]
        
        p = minimize(λ*risk - (1-λ)*ret, sum(w) == 1,  w>=w_lower, w<=w_upper)
        
        solve!(p, SCS.Optimizer; silent_solver=true)
        
        MeanVar[i, :] = [evaluate(ret), sqrt(evaluate(risk))]
        weights[i, :] = evaluate(w)
        
    end
    
    return MeanVar, weights
    
end

#This function will be for fixed constrains, but latter we can implement a more general function in which the
#constrains are provided by the user in the format PenaltyConstrains(...)
function EO_fixed_return(f, target_return, portfolio)
	    
    ## 0<w_i<1
    lower = zeros(6)
    upper = ones(6)

    ## sum(w) = 1
    c(w) = [sum(w), sum(w.*portfolio.μ)]
    lc   = [1.0, target_return] # lower bound for constraint function
    uc   = [1.0, Inf]   # upper bound for constraint function

    #con = WorstFitnessConstraints(lower, upper, lc, uc, c) 
    con = PenaltyConstraints(1e5, lower, upper, lc, uc, c)

    #Define initial condition
    x0 = ones(length(portfolio.µ)) #./ length(portfolio.µ)

    #Optimize using Genetic Algorithm
    result = Evolutionary.optimize(f, con, x0, CMAES(μ=100, sigma0=0.1), 
                    Evolutionary.Options(iterations=Int(1e5), abstol=1e-12, reltol=1e-6))
    
    ret_GA = sum(result.minimizer.*portfolio.μ)
    risk_GA = sqrt(transpose(result.minimizer)*portfolio.Σ*result.minimizer)
    
    return ret_GA, risk_GA, result.minimizer
    
end
