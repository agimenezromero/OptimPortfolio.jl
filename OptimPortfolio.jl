using Convex
using SCS
using Evolutionary

#Portfolio object
mutable struct Portfolio
    
    assets::Array{String}
   
    μ::Matrix{Float64} #Array of mean returns for each asset in the portolio
    
    Σ::Matrix{Float64} #Covariance matrix for the assets in the portfolio
    
    CSK::Array{Float64} #Coskewness tensor
    
    CK::Array{Float64} #Cokurtosis
    
    w::Array{Float64} #Weights to be optimized
    
    Portfolio(assets, μ, Σ, CSK, CK) = new(assets, μ, Σ, CSK, CK)
        
end

function create_portfolio(data, assets)
   
    μ = mean(data, dims=1)
    
    Σ = cov(data)
    
    CSK = coskewness(data)
    
    CK = cokurtosis(data)
    
    portfolio = Portfolio(assets, μ, Σ, CSK, CK)
    
    return portfolio
    
end

function MV_fixed_return(portfolio::Portfolio, target_return; method="DCP", w_lower=0.0, w_upper=1.0)

	if method == "DCP"
   
		w = Variable(length(portfolio.μ))
		
		ret = dot(w, portfolio.μ)
		
		risk = quadform(w, portfolio.Σ)
		    
		p = minimize(risk, ret >= target_return, sum(w)==1, w>=w_lower, w<=w_upper)
		
		solve!(p, SCS.Optimizer, silent_solver = true)
		
		w_opt = evaluate(w)
		
		portfolio.w = w_opt
		
		ret_opt = dot(w_opt, portfolio.μ)
		risk_opt = sqrt(transpose(w_opt)*portfolio.Σ*w_opt)
		
		return ret_opt, risk_opt, w_opt
		
	elseif method == "EO" 
	
		f(w) = transpose(w)*portfolio.Σ*w
	
		## 0<w_i<1
		lower = ones(length(portfolio.µ)) .* w_lower
		upper = ones(length(portfolio.µ)) .* w_upper

		## sum(w) = 1
		c(w) = [sum(w), dot(w, portfolio.μ)]
		lc   = [1.0, target_return] # lower bound for constraint function
		uc   = [1.0, Inf]   # upper bound for constraint function

		#con = WorstFitnessConstraints(lower, upper, lc, uc, c) 
		con = PenaltyConstraints(1e5, lower, upper, lc, uc, c)

		#Define initial condition
		x0 = ones(length(portfolio.µ)) #./ length(portfolio.µ)

		#Optimize using Genetic Algorithm
		result = Evolutionary.optimize(f, con, x0, CMAES(μ=100, sigma0=0.1), 
		                Evolutionary.Options(iterations=Int(1e5), abstol=1e-12, reltol=1e-6))
		                
		portfolio.w = result.minimizer
		
		ret_opt = dot(result.minimizer, portfolio.μ)
		risk_opt = sqrt(transpose(result.minimizer)*portfolio.Σ*result.minimizer)
		
		#Sanity Check
		if sum(result.minimizer) < 0.99
		
			println("Optimization did not converge...returning NaN.")
		
			return NaN, NaN, ones(length(portfolio.assets)) .* NaN
		
		else
		
			return ret_opt, risk_opt, result.minimizer
		
		end	
	
	end
        
end

function MV_fixed_risk(portfolio::Portfolio, target_risk; method="DCP", w_lower=0.0, w_upper=1.0)

	if method == "DCP"
   
		w = Variable(length(portfolio.μ))
		
		ret = dot(w, portfolio.μ)
		
		risk = quadform(w, portfolio.Σ)
		
		p = maximize(ret, risk <= target_risk^2, sum(w)==1, w>=w_lower, w<=w_upper)
		
		solve!(p, SCS.Optimizer, silent_solver = true)
		
		w_opt = evaluate(w)
		
		portfolio.w = w_opt
		
		ret_opt = dot(w_opt, portfolio.μ)
		risk_opt = sqrt(transpose(w_opt)*portfolio.Σ*w_opt)
		
		return ret_opt, risk_opt, w_opt
		
	elseif method == "EO"
	
		f(w) = -sum(w.*portfolio.μ)

		## 0<w_i<1
		lower = ones(length(portfolio.µ)) .* w_lower
		upper = ones(length(portfolio.µ)) .* w_upper

		## sum(w) = 1
		c(w) = [sum(w), transpose(w)*portfolio.Σ*w]
		lc   = [1.0, 0] # lower bound for constraint function
		uc   = [1.0, target_risk^2]   # upper bound for constraint function

		#con = WorstFitnessConstraints(lower, upper, lc, uc, c) 
		con = PenaltyConstraints(1e5, lower, upper, lc, uc, c)

		#Define initial condition
		x0 = ones(length(portfolio.µ)) #./ length(portfolio.µ)

		#Optimize using Genetic Algorithm
		result = Evolutionary.optimize(f, con, x0, CMAES(μ=100, sigma0=0.1), 
				        Evolutionary.Options(iterations=Int(1e5), abstol=1e-12, reltol=1e-6))
				        
		portfolio.w = result.minimizer

		ret_opt = dot(result.minimizer, portfolio.μ)
		risk_opt = sqrt(transpose(result.minimizer)*portfolio.Σ*result.minimizer)
		
		#Sanity Check
		if sum(result.minimizer) < 0.99
		
			println("Optimization did not converge...returning NaN.")
		
			return NaN, NaN, ones(length(portfolio.assets)) .* NaN
		
		else
		
			return ret_opt, risk_opt, result.minimizer
		
		end	
	
	end
    
end

function MV_efficient_frontier(portfolio, points; method="DCP", w_lower=0.0, w_upper=1.0, pop=500, sigma=1)
    
    λs = range(0.0, stop=1.0, length=points)
    
    MeanVar = zeros(points, 2)
		
	weights = zeros(points, length(portfolio.μ))
    
    if method == "DCP"
    
		w = Variable(length(portfolio.μ))
		
		ret = dot(w, portfolio.μ)
		
		risk = quadform(w, portfolio.Σ)
		
		for i in 1:points
		    
		    λ = λs[i]
		    
		    p = minimize(λ*risk - (1-λ)*ret, sum(w) == 1,  w>=w_lower, w<=w_upper)
		    
		    solve!(p, SCS.Optimizer; silent_solver=true)
		    
		    MeanVar[i, :] = [evaluate(ret), sqrt(evaluate(risk))]
		    weights[i, :] = evaluate(w)
		    
		end
		
	elseif method == "EO"
	
		## 0<w_i<1
		lower = ones(length(portfolio.µ)) .* w_lower
		upper = ones(length(portfolio.µ)) .* w_upper

		## sum(w) = 1
		c(w) = [sum(w)]
		lc   = [0.99] # lower bound for constraint function
		uc   = [1.01]   # upper bound for constraint function

		con = PenaltyConstraints(1e6, lower, upper, lc, uc, c)

		#Define initial condition
		x0 = ones(length(portfolio.µ)) #./ length(portfolio.µ)
	
		for i in 1:points
	
			#Define objective function
			f(w) = λs[i] * (transpose(w)*portfolio.Σ*w)  - (1-λs[i]) * dot(w, portfolio.μ)

			#Optimize using Genetic Algorithm
			result = Evolutionary.optimize(f, con, x0, CMAES(μ=pop, sigma0=sigma), 
				            Evolutionary.Options(iterations=Int(1e6), abstol=1e-10))

			ret_opt = dot(result.minimizer, portfolio.μ)
			risk_opt = sqrt(transpose(result.minimizer)*portfolio.Σ*result.minimizer)
		            
			ϵ = 0.001
			
			#Sanity Check
			if (sum(result.minimizer) + ϵ < lc[1]) || (sum(result.minimizer) - ϵ > uc[1]) || result.converged == false
			
				println("Optimization failed for λ=$(λs[i])...returning NaN.")
				
				MeanVar[i, :] = [NaN, NaN]
		    	weights[i, :] = ones(length(portfolio.assets)) .* NaN
			
			else
			
				MeanVar[i, :] = [ret_opt, risk_opt]
		    	weights[i, :] = result.minimizer
			
			end	
	
		end
		
	end
    
    return MeanVar, weights
    
end

function MVSK(portfolio, λs; w_lower=0.0, w_upper=1.0, pop=500, sigma=1)
   
    ## 0<w_i<1
    lower = ones(length(portfolio.assets)) .* w_lower
    upper = ones(length(portfolio.assets)) .* w_upper

    ## sum(w) = 1
    c(w) = [sum(w)]
    lc   = [0.99] # lower bound for constraint function
    uc   = [1.01]   # upper bound for constraint function

    con = PenaltyConstraints(1e6, lower, upper, lc, uc, c)

    #Define initial condition
    x0 = ones(length(portfolio.µ)) #./ length(portfolio.µ)
    
    #Define objective function
    f(w) = -λs[1] * dot(w, portfolio.μ) + λs[2] * (transpose(w)*portfolio.Σ*w) - λs[3] * skewness_portfolio(portfolio, w) + λs[4] * kurtosis_portfolio(portfolio, w)

    #Optimize using Genetic Algorithm
    result = Evolutionary.optimize(f, con, x0, CMAES(μ=pop, sigma0=sigma), 
                    Evolutionary.Options(iterations=Int(1e6), abstol=1e-10))

    ret_opt = dot(result.minimizer, portfolio.μ)
    risk_opt = sqrt(transpose(result.minimizer)*portfolio.Σ*result.minimizer)
    skew_opt = skewness_portfolio(portfolio, result.minimizer)
    kurt_opt = kurtosis_portfolio(portfolio, result.minimizer)

      ϵ = 0.001
    
    #Sanity Check
    if (sum(result.minimizer) + ϵ < lc[1]) || (sum(result.minimizer) - ϵ > uc[1])

        println("Optimization did not converge...returning NaN.")

        ret_opt = NaN
        risk_opt = NaN
        skew_opt = NaN
        kurt_opt = NaN

        weights = ones(length(portfolio.assets)) .* NaN

        return [ret_opt, risk_opt, skew_opt, kurt_opt], weights, result

    else

        return [ret_opt, risk_opt, skew_opt, kurt_opt], result.minimizer

    end
    
end

function EO(f, con, portfolio)
	    
    #Define initial condition
    x0 = ones(length(portfolio.µ)) #./ length(portfolio.µ)

    #Optimize using Genetic Algorithm
    result = Evolutionary.optimize(f, con, x0, CMAES(μ=100, sigma0=0.1), 
                    Evolutionary.Options(iterations=Int(1e5), abstol=1e-12, reltol=1e-6))
    
    ret_GA = dot(result.minimizer, portfolio.μ)
    risk_GA = sqrt(transpose(result.minimizer)*portfolio.Σ*result.minimizer)
    
    return ret_GA, risk_GA, result.minimizer
    
end
