using LinearAlgebra
using SparseArrays

"""
Creates and keeps track of sets O and G for OAVI.
"""
mutable struct SetsOandG
    
# border terms before purging by degree
border_terms_raw
# evaluations of border_terms_raw over X by degree
border_evaluations_raw
# border terms after purging by degree
border_terms_purged
# evaluations of border_terms_purged over X by degree
border_evaluations_purged
# indices such that border_terms_raw[i][non_purging_indices[i], :] = border_terms_purged[i]
non_purging_indices   

# array of O terms by degree
O_terms
# array of evaluation of O terms over X by degree
O_evaluations
# indices in the border that get appended to O
O_indices

# sets for vanishing polynomials
G_coefficient_vectors
G_evaluations

# leading terms
leading_terms
    
end


"""
updates border sets
"""
function update_border(sets, border_terms_raw, border_evaluations_raw, non_purging_indices)
    sets.non_purging_indices = append!(sets.non_purging_indices, non_purging_indices)
    sets.border_terms_raw = append!(sets.border_terms_raw, border_terms_raw)
    sets.border_evaluations_raw = append!(sets.border_evaluations_raw)
    sets.border_terms_purged = append!(sets.border_terms_purged, border_terms_raw[:, non_purging_indices])
    sets.border_evaluations_purged = append!(sets.border_evaluations_purged, border_evaluations_raw[:, non_purging_indices])
end    


"""
updates O sets
"""
function update_O(sets, O_terms, O_evaluations, O_indices)
    sets.O_terms = hcat(sets.O_terms, O_terms)
    sets.O_evaluations = hcat(sets.O_evaluations, O_evaluations)
    sets.O_indices = append!(sets.O_indices, O_indices)
end


"""
updates G sets
"""
function update_G(sets, G_coefficient_vectors=nothing)
    if G_coefficient_vectors != nothing
        if size(sets.G_evluations, 2) == 0
            sets.G_evaluations = hcat(sets.O_evaluations, sets.border_evaluations_purged[end] * G_coefficient_vectors)
        else
            current_G_evaluations = hcat(sets.O_evaluations, sets.border_evaluations_purged[end] * G_coefficient_vectors)
            sets.G_evaluations = hcat(sets.G_evaluations, current_G_evaluations)
        end
    end
    sets.G_coefficient_vectors = append!(sets.G_coefficient_vectors, G_coefficient_vectors)
end


"""
updates leading terms
"""
function update_leading_terms(sets, leading_terms=nothing)
    if sets.leading_terms == nothing
        if leading_terms != nothing
            sets.leading_terms = leading_terms
        end
    else
        sets.leading_terms = hcat(sets.leading_terms, leading_terms)
    end
end




"""
constructs the border of 'terms'

# Arguments
- 'terms::Matrix{Int64}': Matrix with monomial terms as columns
- 'terms_evaluated::Matrix{Float64}': Matrix with evaluations of 'terms' over X
- 'X_train::Vector{Vector{Float64}}': data
- 'degree_1_terms::Matrix{Int64}': Matrix with degree 1 monomials as columns
- 'degree_1_terms_evaluated::Matrix{Float64}': evaluations of 'degree_1_terms' over X
- 'purging_terms::Matrix{Int64}': purge terms in 'terms' divisible by any of these 

# Returns
- 'border_terms_raw::Matrix{Int64}': non-purged border constructed from 'terms'
- 'border_evaluations_raw::Matrix{Float64}': non-purged evaluations of border terms over X
- 'non_purging_indices::Vector{Int64}': array of non-purging indices
"""
function construct_border(terms::Matrix{Int64}, terms_evaluated::Matrix{Float64}, X_train::Vector{Vector{Float64}}, 
    degree_1_terms::Matrix{Int64}=zeros(Int64, 0, 0), degree_1_terms_evaluated::Matrix{Float64}=zeros(Float64, 0, 0), 
    purging_terms::Matrix{Int64}=zeros(Int64, 0, 0))

    X_train = vecvec_to_mat(X_train)
    dim = length(X_train[1, :])
    
    if size(degree_1_terms, 2) == 0
        
        border_terms_raw = 1 * Matrix(I, dim, dim)
        border_evaluations_raw = X_train  # This constitutes evaluation as evals[1] = [term[1](x1), term[2](x1), ..., term[m](x1)]
    
    else
        # terms
        len_deg1 = length(degree_1_terms)
        len_terms = length(terms)
        
        terms_repeat = transpose(repeat(terms', outer=len_deg1))
        degree_1_tile = tile(degree_1_terms, len_terms)
        
        border_terms_raw = degree_1_tile + terms_repeat
        
        # evaluations
        len_deg1_eval = length(degree_1_terms_evaluated)
        len_terms_eval = length(terms_evaluated)
        
        terms_evaluated_repeat = transpose(repeat(terms', outer=len_deg1_eval))
        degree_1_evaluated_tile = tile(degree_1_evaluated, len_terms_eval)
        
        border_evaluations_raw = degree_1_evaluated_tile .* terms_evaluated_repeat
        
    end
    
    border_terms_purged, border_evaluations_purged, unique_indices = get_unique_columns(
        border_terms_raw, border_evaluations_raw)
    
    if size(purging_terms, 2) != 0
        border_terms_purged, border_evaluations_purged, unique_indices_2 = purge(
            border_terms_purged, border_evaluations_purged, purging_terms)
        
        if unique_indices_2 != []
            non_purging_indices = [unique_indices[i] for i in unique_indices_2]
        else
            non_purging_indices = unique_indices
        end
    else
        non_purging_indices = unique_indices
    end
    
    return border_terms_raw, border_evaluations_raw, non_purging_indices
    
end


"""
purges each term in 'terms' that is divisible by at least one term 'purging_terms'

# Arguments
- 'terms::Matrix{Int64}': Matrix with monomial terms as columns
- 'terms_evaluated::Matrix{Float64}': evaluations of 'terms' over data
- 'purging_terms::Matrix{Int64}': Matrix with purging terms as columns

# Returns
- 'terms[:, inidces]::Matrix{Int64}': purged version of terms
- 'terms_evaluated[:, indices]::Matrix{Float64}': purged evaluations
- 'indices::Vector{Int64}': array with non-purging indices
"""
function purge(terms::Matrix{Int64}, terms_evaluated::Matrix{Float64}, purging_terms::Matrix{Int64})
    purging_indices = []
    indices = [x for x in 1:size(terms, 2)]
    
    for i in 1:size(terms, 2)
        for j in 1:size(purging_terms, 2)
            if all(terms[:, i] - purging_terms[:, j] .>= 0)
                append!(purging_indices, i)
                break
            end
        end
    end
    
    indices = deleteat!(indices, purging_indices)
    return terms[:, indices], terms_evaluated[:, indices], indices
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
function call_oracle(f, grad, feasible_region, initial_point::Vector{Float64}, oracle::String="BPCG")
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