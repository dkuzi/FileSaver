"""
Given A, A.T.A and (A.T.A)^-1 efficiently compute B = [A, a], B.T.B and (B.T.B)^-1.
Necessary for fast inverse hessian boosting.
"""
function streaming_matrix_updates(A, A_squared, A_a, a, a_squared; A_squared_inv=nothing, built_in::Bool=false)
    B_squared_inv = nothing

    if built_in
        B = hcat(A, a)
        B_squared = B' * B
        if A_squared_inv != nothing
            B_squared_inv = inv(B_squared)
        end
    else
        B = hcat(A, a)

        b = A_a

        B_squared = hcat(A_squared, b)
        B_squared = vcat(B_squared, vcat(b, a_squared)')

        if A_squared_inv != nothing 
            # write B_squared_inv as S = | S_1, s_2|
            #                            | s_2.T s_3|

            A_squared_inv_b = A_squared_inv * b
            b_A_squared_inv_b = (b' * A_squared_inv_b)[1]

            s_2 = A_squared_inv + ((A_squared_inv_b * A_squared_inv_b') ./ (a_squared - b_A_squared_inv_b))
            s_2 = (s_2 * b) ./ a_squared

            s_3 = (1 - (b' * s_2)[1]) / a_squared

            S_1 = A_squared_inv - (A_squared_inv_b * s_2')

            B_squared_inv = hcat(S_1, s_2)
            B_squared_inv = vcat(B_squared_inv, vcat(s_2, s_3)')            
        end
    end

    return B, B_squared, B_squared_inv
    
end


"""
Projects x onto the L1 Ball with radius 'radius'.

Reference: 
"Efficient Projections onto the ℓ1-Ball for Learning in High Dimensions", https://stanford.edu/~jduchi/projects/DuchiShSiCh08.pdf
"""
function l1_projection(x; radius=1.)
    @assert radius > 0 "Radius must be positive."
    
    if norm(x, 1) <= radius
        return x
    end
    
    n = size(x, 1)
    x = reshape(x, n, 1)
    v = abs.(x)
    u = sort(v, dims=1)
    u = reverse(u)
    
    csum = cumsum(u, dims=1)
    
    p = findlast(u .* collect(1:n) .> csum .- radius)[1]
    
    theta = (1 / p) * (csum[p] - radius)
    
    w = reshape([max(v[i]-theta, 0) for i in 1:size(v, 1)], n, 1)
    return sign.(x) .* w
end


"""
Calls ORACLE for computing coefficient vector

# Arguments
- 'oracle::String': Name of ORACLE to use
- 'f': function to optimize
- 'grad': gradient of f
- 'feasible_region': feasible region over which to optimize for coefficient vector
- 'initial_point::Vector{Float64}': starting point; must be in feasible_region

# Returns
- 'x_opt::Vector{Float64}': coefficient vector minimizing f over feasible_region
"""
function call_oracle(f, grad, feasible_region, initial_point::Vector{Float64}; oracle::String="BPCG")
    if oracle == "CG"
        x_opt, _ = frank_wolfe(f, grad, feasible_region, initial_point)
    elseif oracle == "BCG"
        x_opt, _ = blended_conditional_gradient(f, grad, feasible_region, initial_point)
    elseif oracle == "BPCG"
        x_opt, _ = FrankWolfe.blended_pairwise_conditional_gradient(f, grad, feasible_region, initial_point)
    else
        println("Oracle not implemented.")
        return nothing
    end
    return x_opt
end
