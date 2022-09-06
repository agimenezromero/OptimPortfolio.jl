using Statistics
using StatsBase

function coskewness(data)

    N = size(data)[2]
    L = size(data)[1]

    M3 = zeros((N, N, N))
    
    avgs = mean(data, dims=1)
    
    cntr = data .- avgs
    
	@inbounds @simd for i in 1 : N

	    for j in 1:N

	        for k in 1:N

				M3[i, j, k] = sum(cntr[:, i] .* cntr[:, j] .* cntr[:, k])

	        end

	    end

	end

	M3 = M3 ./ L

    return M3
    
end

function cokurtosis(data)

    N = size(data)[2]
    L = size(data)[1]

    M4 = zeros((N, N, N, N))
    
    avgs = mean(data, dims=1)
    
    cntr = data .- avgs
    	
	@inbounds @simd for i in 1 : N

	    for j in 1:N

	        for k in 1:N

	            for l in 1:N

	                M4[i,j,k,l] += sum(cntr[:,i] .* cntr[:,j] .* cntr[:,k] .* cntr[:,l])

	            end

	        end

	    end

	end
		
    M4 = M4 ./ L

    return M4
    
end

function return_portfolio(portfolio, weights)

	return dot(weights,  portfolio.μ)

end

function risk_portfolio(portfolio, weights)

	return sqrt(transpose(weights) * portfolio.Σ * weights)

end

function skewness_portfolio(portfolio, weights)
    
    N = length(portfolio.assets)
    
    M3 = reshape(portfolio.CSK, (N, N^2))
    
    coskew_p = transpose(weights) * M3 * kron(weights, weights)
    
    return coskew_p
    
end

function kurtosis_portfolio(portfolio, weights)
    
    N = length(portfolio.assets)
    
    M4 = reshape(portfolio.CK, (N, N^3))
    
    cokurt_p = transpose(weights) * M4 * kron(weights, weights, weights)
    
    return cokurt_p
    
end

function standarized_skewness_portfolio(portfolio, weights)
    
    return skewness_portfolio(portfolio, weights) / risk_portfolio(portfolio, weights)^3
    
end

function standarized_kurtosis_portfolio(portfolio, weights)
    
    return kurtosis_portfolio(portfolio, weights) / risk_portfolio(portfolio, weights)^4 
    
end

function standarized_excess_kurtosis_portfolio(portfolio, weights)
    
    return kurtosis_portfolio(portfolio, weights) / risk_portfolio(portfolio, weights)^4 - 3
    
end

function SharpeRatio(ret, risk)
   
    return @. ret / risk
    
end

function mVaR(ret, risk, skew, kurt, α)
   
    Zα = quantile(Normal(0,1), α)
    
    return @. -ret + risk*(-Zα-(1/6)*(Zα^2-1)*skew - (1/24)*(Zα^3-3*Zα)*kurt + (1/36)*(2*Zα^3-5*Zα)*skew^2)
    
end

function mSharpeRatio(ret, risk, skew, kurt, α)
   
    return @. ret / mVaR(ret, risk, skew, kurt, α)
    
end

function maximum_drawdown(x)
    
    MDD = zeros(1, size(x)[2])
    peak = ones(size(x)[2]) .* -99999
    
    DD = zeros(size(x))
    
    for j in 1 : size(x)[2]
        
        for i in 1 : size(x)[1]
        
            if (x[i, j] > peak[j])

                peak[j] = x[i, j]

            end

            DD[i, j] = 100.0 * (peak[j] - x[i, j]) / peak[j]

            if (DD[i, j] > MDD[1, j])

                MDD[1, j] = DD[i, j]

            end
            
        end

    end
        
    return -DD, -MDD
    
end
