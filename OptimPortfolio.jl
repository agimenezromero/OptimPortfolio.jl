using Convex
using SCS

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

function Efficient_frontier(portfolio, points, w_lower=0.0, w_upper=1.0)
    
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
        
        MeanVar[i, :] = [evaluate(ret), evaluate(risk)]
        weights[i, :] = evaluate(w)
        
    end
    
    return MeanVar, weights
    
end
