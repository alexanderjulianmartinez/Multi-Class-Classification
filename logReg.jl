include("misc.jl")
include("findMin.jl")

function logReg(X,y)

	(n,d) = size(X)

	# Initial guess
	w = zeros(d,1)

	# Function we're going to minimize (and that computes gradient)
	funObj(w) = logisticObj(w,X,y)

	# Solve least squares problem
	w = findMin(funObj,w,derivativeCheck=false)

	# Make linear prediction function
	predict(Xhat) = sign.(Xhat*w)

	# Return model
	return LinearModel(predict,w)
end

function logisticObj(w,X,y)
	yXw = y.*(X*w)
	f = sum(log.(1 + exp.(-yXw)))
	g = -X'*(y./(1+exp.(yXw)))
	return (f,g)
end

# Multi-class Softmax version 
function softmaxClassifier(X,y)
	(n,d) = size(X)
	k = maximum(y)

	# Initial guess
	w = zeros(d*k,1)

	# Function we're going to minimize (and that computes gradient)
	funObj(w) = softmaxObj(w,X,y)

	# Solve least squares problem
	w = findMin(funObj,w,derivativeCheck=true)

	W = reshape(w,(d,k))

	# Make linear prediction function
	predict(Xhat) = mapslices(indmax,Xhat*W,2)

	return LinearModel(predict,W)
end

function softmaxObj(w,X,y)
	(n,d) = size(X)
	k = maximum(y)
	# reshape W to d x k
	W = reshape(w,(d,k))
	
	f = 0
	g = zeros(d,k)
	for i in 1:n 
		h = 0
		yi = y[i]

		for c in 1:k
			h += exp(dot(X[i,:],W[:,c])) 
		end

		f += -dot(X[i,:],W[:,yi]) + log(h)
	end

	for j in 1:d
		for c in 1:k
			m = 0
			for i in 1:n
				term1 = -(y[i] == c)*X[i,j]
			
				top = X[i,j]*exp(dot(X[i,:],W[:,c]))
				bottom = 0
				
				for s in 1:k
					bottom += exp(dot(X[i,:],W[:,s]))
				end
					
				m += term1 + (top/bottom)
			end
            #@printf("Found g[%d,%d]\n",j,c)
            g[j,c] = m
		end
	end
			
	return (f,g[:])
end



# Multi-class one-vs-all version (assumes y_i in {1,2,...,k})
function logRegOnevsAll(X,y)
	(n,d) = size(X)
	k = maximum(y)

	# Each column of 'w' will be a logistic regression classifier
	W = zeros(d,k)

	for c in 1:k
		yc = ones(n,1) # Treat class 'c' as +1
		yc[y .!= c] = -1 # Treat other classes as -1

		# Each binary objective has the same features but different labels
		funObj(w) = logisticObj(w,X,yc)

		W[:,c] = findMin(funObj,W[:,c],verbose=false)
	end

	# Make linear prediction function
	predict(Xhat) = mapslices(indmax,Xhat*W,2)

	return LinearModel(predict,W)
end


