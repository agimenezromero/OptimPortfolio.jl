using Statistics
using StatsBase

function coskewness(data)

    N = size(data)[2]
    L = size(data)[1]

    M3 = zeros((N, N, N))

    @inbounds @simd for i in 1 : N

        for j in 1:N

            for k in 1:N

                for t in 1 : L

                    M3[i,j,k] += (data[t,i] - mean(data[:,i])) * (data[t,j] - mean(data[:,j])) * (data[t,k] - mean(data[:,k]))

                end

                σiσjσk = sqrt(var(data[:, i])*var(data[:, j]) * var(data[:, k]))

                M3[i,j,k] = M3[i,j,k] / (L*σiσjσk)

            end

        end

    end

    return M3
    
end

function cokurtosis(data)

	N = size(data)[2]
	L = size(data)[1]

	M4 = zeros((N, N, N, N))

	@inbounds @simd for i in 1 : N

		for j in 1:N

			for k in 1:N
				
				for l in 1:N

				    for t in 1 : L

				        M4[i,j,k,l] += (data[t,i] - mean(data[:,i])) * (data[t,j] - mean(data[:,j])) * (data[t,k] - mean(data[:,k])) * (data[t,l] - mean(data[:,l]))

				    end

				    σiσjσkσl = sqrt(var(data[:, i])*var(data[:, j]) * var(data[:, k])) * var(data[:, l])

				    M4[i,j,k, l] = M4[i,j,k] / (L*σiσjσkσl)
				    
				end

			end

		end
	
	end

    return M4
    
end

function coskewness_portfolio(data, weights)
    
    N = size(data)[2]
    
    M3 = reshape(coskewness(data), (N, N^2))
    
    coskew_p = transpose(weights) * M3 * kron(weights, weights)
    
    return coskew_p
    
end

function cokurtosis_portfolio(data, weights)
    
    N = size(data)[2]
    
    M4 = reshape(cokurtosis(data), (N, N^3))
    
    cokurt_p = transpose(weights) * M4 * kron(weights, weights, weights)
    
    return cokurt_p
    
end
