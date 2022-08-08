using Statistics
using StatsBase

function coskewness(data)

    N = size(data)[2]
    L = size(data)[1]

    M3 = zeros((N, N, N))
    
    cntr = data .- mean(data, dims=1)

    @inbounds @simd for i in 1 : N

        for j in 1:N

            for k in 1:N

                for t in 1 : L

                    M3[i,j,k] += cntr[t,i] * cntr[t,j] * cntr[t,k]

                end

                σiσjσk = sqrt(var(data[:, i])*var(data[:, j]) * var(data[:, k]))

                M3[i,j,k] = M3[i,j,k] / (σiσjσk)

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
    
    cntr = data .- mean(data, dims=1)

    @inbounds @simd for i in 1 : N

        for j in 1:N

            for k in 1:N

                for l in 1:N

                    for t in 1 : L

                        M4[i,j,k,l] += cntr[t,i] * cntr[t,j] * cntr[t,k] * cntr[t,l]

                    end

                    σiσjσkσl = sqrt(var(data[:, i])*var(data[:, j]) * var(data[:, k]) * var(data[:, l]))

                    M4[i,j,k,l] = M4[i,j,k,l] / (σiσjσkσl)

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

	return transpose(weights) * portfolio.Σ * weights

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
