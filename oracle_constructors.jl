"""
Struct for Conditional-Gradients based oracle.

# Fields
- 'solution::Union{Vector{Float64}, Nothing}': solution vector (only if inverse hessian boosting enabled, else nothing)
- 'f': objective function. Takes as input 'x::Vector{Float64}' and outputs evaluation 'f(x)' at point 'x'.
- 'grad!': gradient of f. Input: mutable 'storage::Vector{Float64}' and 'x::Vector{Float64}', output: assigns to 'storage' gradient of 'f' at 'x'.
- 'solver': CG-based algorithm to solve convex problem. Input: (f, grad!, region, x0; kwargs...), output: (x_opt, _)
"""
struct ConditionalGradients
    solution::Union{Vector{Float64}, Nothing}
    f
    grad!
    solver
end


"""
Constructs objective function and gradient for CG-based oracle and returns oracle, f and grad!

# Arguments
- 'oracle_type::String': string denoting which oracle to construct
- 'data::Union{Matrix{Float64}, Matrix{Int64}}': data (O_evaluations)
- 'labels::Union{Matrix{Float64}, Matrix{Int64}, Vector{Float64}, Vector{Int64}}': labels (term_evaluated)
- 'lambda::Union{Float64, Int64}': regularization parameter (if applicable)
- 'data_squared::Union{Matrix{Float64}, Matrix{Int64}}': squared data 
- 'data_labels::Union{Matrix{Float64}, Vector{Float64}}': data' * labels 
- 'labels_squared::Float64': labels' * labels
- 'data_squared_inverse::Union{Matrix{Float64}, Matrix{Int64}, Nothing}': inverse of data_squared (default is nothing)

# Returns
- 'ConditionalGradients(solution, evaluate_function, evaluate_gradient!, solver)': struct keeping data and functions needed

"""
function conditional_gradients(oracle_type::String, 
        data::Union{Matrix{Float64}, Matrix{Int64}}, 
        labels::Union{Matrix{Float64}, Matrix{Int64}, Vector{Float64}, Vector{Int64}},
        lambda::Union{Float64, Int64}, 
        data_squared::Union{Matrix{Float64}, Matrix{Int64}}, 
        data_labels::Union{Matrix{Float64}, Vector{Float64}},
        labels_squared::Float64;
        data_squared_inverse::Union{Matrix{Float64}, Matrix{Int64}, Nothing}=nothing)
           
    A = data
    m, n = size(data)
    b = labels
        
    # A_squared
    if data_squared != nothing
        A_squared = 2/m * data_squared
    else
        A_squared = 2/m * A' * A
    end
        
    if lambda != 0.
        A_squared = A_squared + lambda * Matrix(I, n, n)
    end
        
    # A_b
    if data_labels != nothing
        A_b = 2/m * data_labels
    else
        A_b = 2/m * (A' * b)
    end
        
    # A_squared_inv
    A_squared_inv = nothing
    solution = nothing
    if data_squared_inverse != nothing
        A_squared_inv = m/2 * data_squared_inverse
        solution = data_squared_inverse * data_labels
        @assert lambda == 0. "Regularization not implemented for hessian-based algorithms."
    end
        
    # b_squared
    if labels_squared != nothing
        b_squared = 2/m * labels_squared
    else
        b_squared = 2/m * b' * b
    end
        
    """
    objective function
    """
    function evaluate_function(x::Vector{Float64})
        return ((1 / 2) * (x' * A_squared * x) + (A_b' * x) + (1 / 2) * b_squared)
    end
        
    """
    gradient of f
    """
    function evaluate_gradient!(storage::Vector{Float64}, x::Vector{Float64})
        return storage .= A_squared * x + A_b
    end
        
    if oracle_type == "CG"
        oracle = frank_wolfe
    elseif oracle_type == "BCG"
        oracle = blended_conditional_gradient
    elseif oracle_type == "BPCG"
        oracle = FrankWolfe.blended_pairwise_conditional_gradient
    end
        
    return ConditionalGradients(solution, evaluate_function, evaluate_gradient!, oracle)     
end


"""
Runs ABM algorithm to find coefficient vector and computes loss.

# Arguments
- 'oracle_type::String': string denoting which oracle to construct
- 'data::Union{Matrix{Float64}, Matrix{Int64}}': data (O_evaluations)
- 'labels::Union{Matrix{Float64}, Matrix{Int64}, Vector{Float64}, Vector{Int64}}': labels (term_evaluated)
- 'lambda::Union{Float64, Int64}': regularization parameter (if applicable)
- 'data_squared::Union{Matrix{Float64}, Matrix{Int64}}': squared data 
- 'data_labels::Union{Matrix{Float64}, Vector{Float64}}': data' * labels 
- 'labels_squared::Float64': labels' * labels
- 'data_squared_inverse::Union{Matrix{Float64}, Matrix{Int64}, Nothing}': inverse of data_squared (default is nothing)

# Returns
- 'coefficient_vector::Vector{Float64}': coefficient vector minimizing ABM optimization problem
- 'loss::Float64': loss using 'coefficient_vector' 
"""
function abm(data::Union{Matrix{Float64}, Matrix{Int64}}, 
        labels::Union{Matrix{Float64}, Matrix{Int64}, Vector{Float64}, Vector{Int64}},
        data_squared::Union{Matrix{Float64}, Matrix{Int64}}, 
        data_labels::Union{Matrix{Float64}, Vector{Float64}},
        labels_squared::Float64;
        data_squared_inverse::Union{Matrix{Float64}, Matrix{Int64}, Nothing}=nothing)
    data_with_labels = hcat(data, labels)
    m = size(data_with_labels, 1)
    
    if size(data_with_labels, 1) > size(data_with_labels, 2)
        data_squared_with_labels = hcat(data_squared, data_labels)
        bottom_row = vcat(data_labels, labels_squared)
        bottom_row = bottom_row'
        data_squared_with_labels = vcat(data_squared_with_labels, bottom_row)
        F = svd(data_squared_with_labels)
    else
        F = svd(data_with_labels)
    end
    
    U, S, Vt = F.U, F.S, F.Vt
    coefficient_vector = Vt[:, end]
    loss = 1/size(data, 1) * norm(data_with_labels * coefficient_vector, 2)^2
    
    return coefficient_vector, loss
end