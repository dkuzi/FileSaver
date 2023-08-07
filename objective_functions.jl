using LinearAlgebra

"""
creates and returns f and grad(f) for the L2-loss

TODO: Look at 'call_oracle' and adjust accordingly! Maybe do 'L2Loss' as struct with constructor to construct it
-> function takes data, labels, lmbda as input and creates struct with necessary values + functions.

# Arguments
- 'data::
"""
function construct_L2Loss(
    data::Matrix{Float64},
    labels::Vector{Float64};
    lmbda=0.0,
    data_squared=nothing,
    data_labels=nothing,
    labels_squared=nothing,
    data_squared_inverse=nothing)

    A = data
    m, n = size(A)
    b = reshape(labels, m, 1) 
    lmbda = Float64(lmbda)

    # A squared
    if data_squared != nothing
        A_squared = (2 / m) * data_squared
    else
        A_squared = (2 / m) * (A' * A)
    end

    if lmbda != 0.0
        A_squared = A_squared + lmbda * Matrix(I, n, n)
    end

    @assert size(A_squared) == (n, n)

    # A.Tb
    if data_labels != nothing
        A_b = (2 / m) * data_labels
        A_b = reshape(A_b, n, 1)
    else
        A_b = (2 / m) * (A' * b)
        A_b = reshape(A_b, n, 1)
    end
    
    
    # A_squared_inv
    A_squared_inv = nothing
    solution = nothing
    if data_squared_inverse != nothing
        A_squared_inverse = (m / 2) * data_squared_inverse
        solution = data_squared_inverse * data_labels
        @assert lmbda == 0. "Regularization not implemented for hessian-based algorithms."
    end

    # b.Tb
    if labels_squared != nothing
        b_squared = (2 / m) * labels_squared
    else
        b_squared = (2 / m) * (b' * b)
    end
    
    """
    evaluates f at x
    """
    function evaluate_function(x::Vector{Float64})
        x = hcat(x)
        return ((1 / 2) * (x' * A_squared * x) .+ (A_b' * x) .+ (1 / 2) * b_squared)[1]
    end

    """
    evaluates gradient of f at x
    """
    function evaluate_gradient!(storage::Vector{Float64}, x::Vector{Float64})
        x = hcat(x)
        return storage .= A_squared * x + A_b
    end

    return L2Loss(A, A_squared, A_b, b_squared, A_squared_inv, solution), evaluate_function, evaluate_gradient!
    
end

struct L2Loss
    A
    A_squared
    A_b
    b_squared
    A_squared_inv
    solution    
end