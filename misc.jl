# Define a "model" type, that just needs a predict function
type GenericModel
	predict # Function that makes predictions
end

type LinearModel
	predict # Funcntion that makes predictions
	w # Weight vector
end

type CompressModel
	compress # Function that compresses
	expand # Function that de-compresses
	W # weight matrix
end

# Function to compute the mode of a vector
function mode(x)
	# Returns mode of x
	# if there are multiple modes, returns the smallest
	x = sort(x[:]);

	commonVal = [];
	commonFreq = 0;
	x_prev = NaN;
	freq = 0;
	for i in 1:length(x)
		if(x[i] == x_prev)
			freq += 1;
		else
			freq = 1;
		end
		if(freq > commonFreq)
			commonFreq = freq;
			commonVal = x[i];
		end
		x_prev = x[i];
	end
	return commonVal
end

# Return element-wise log, but set log(0)=0
function log0(x)
	y = copy(x)
	y[y.==0] = 1
	return log.(y)
end


# Return squared Euclidean distance all pairs of rows in X1 and X2
function distancesSquared(X1,X2)
	(n,d) = size(X1)
	(t,d2) = size(X2)
	assert(d==d2)
	return X1.^2*ones(d,t) + ones(n,d)*(X2').^2 - 2X1*X2'
end


# Subtract mean of each column and divide by standard deviation
# (or call it with mu and sigma to use these specific mean/std)
function standardizeCols(X;mu=[],sigma=[])
	(n,d) = size(X)

	if isempty(mu)
		mu_j = mean(X,1)
	else
		mu_j = mu
	end

	Xstd = zeros(n,d)
	for j in 1:d
		Xstd[:,j] = X[:,j] - mu_j[j]
	end

	if isempty(sigma)
		sigma_j = std(Xstd,1)
	else
		sigma_j = sigma
	end

	for j in 1:d
		Xstd[:,j] /= sigma_j[j]
	end

	if isempty(mu) & isempty(sigma)
		return (Xstd,mu_j,sigma_j)
	else
		return Xstd
	end
end


### A function to compute the gradient numerically
function numGrad(func,x)
	n = length(x);
	delta = 2*sqrt(1e-12)*(1+norm(x));
	g = zeros(n);
	e_i = zeros(n)
	for i = 1:n
		e_i[i] = 1;
		(fxp,) = func(x + delta*e_i)
		(fxm,) = func(x - delta*e_i)
		g[i] = (fxp - fxm)/2delta;
		e_i[i] = 0
	end
	return g
end

### Check if number is a real-finite number
function isfinitereal(x)
	return (imag(x) == 0) & (!isnan(x)) & (!isinf(x))
end
