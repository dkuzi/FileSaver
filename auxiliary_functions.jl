using LinearAlgebra

"""
finds unique columns in matrix x1 and returns only unique elements in x1 as well as corresponding columns in x2.
"""
function get_unique_columns(x1::Matrix, x2=zeros(0, 0))
    cx1, cx2 = copy(x1), copy(x2)
    sorted_x1, sorted_x2 = deg_lex_sort(cx1, cx2)
    unique_indices = unique(i -> sorted_x1[:, i], 1:size(sorted_x1, 2))
    x1_unique = sorted_x1[:, unique_indices]
    if size(x2) != (0, 0)
        x2_unique = sorted_x2[:, unique_indices]
    else
        x2_unique = cx2
    end

    return x1_unique, x2_unique, unique_indices
end

"""
sorts matrix1 degree-lexicographically and matrix2 accordingly
"""
function deg_lex_sort(matrix1::Matrix{Int64}, matrix2=zeros(0, 0))
    sorted_matrix2 = nothing
    
    rev_mat1 = matrix1[end:-1:1, :]
    mat = vcat(sum(matrix1, dims=1), rev_mat1)
    sorted_list = sortperm(collect(eachcol(mat)))

    sorted_matrix1 = matrix1[:, sorted_list]

    if size(matrix2) != (0, 0)
        sorted_matrix2 = matrix2[:, sorted_list]
    end
    return sorted_matrix1, sorted_matrix2, sorted_list
end


"""
Transforms an object of type Matrix{} to an Array of Arrays where each row is an individual Array.
"""
function mat_to_arr_of_arrs(A::Matrix{}, col_is_row::Int64=0)

    if  col_is_row == 0
        # converts matrix to array of arrays with rows of matrix being the individual entries of array,
        # i.e. first row of matrix A becomes the array result[1]
        transformed_A = [A[i,:] for i in 1:size(A,1)]
    
    
    elseif col_is_row == 1
        # converts matrix to array of arrays with columns of matrix being the individual entries of array,
        # i.e. first column of matrix A becomes the array result[1]
        transformed_A = [A[:,i] for i in 1:size(A,2)]
       
    end
    
    return transformed_A
    
end

                
"""
converts Array of Arrays to Matrix where A[i] becomes row [i] in output Matrix. 
If optional parameter 'arr_is_col' = 1 convert arrays into columns instead of rows.
"""
function vecvec_to_mat(A, arr_is_col::Int64=0)
    elem_type = eltype(A[1])
    A_mat = zeros(elem_type, length(A), length(A[1]))
    for i in eachindex(A)
        A_mat[i, :] = A[i]
    end
    
    if arr_is_col == 0
        return A_mat
    end
    
    if arr_is_col == 1
        return A_mat'
    end
    
    println("Argument 'arr_is_col' needs to be in {0, 1}.")
    return nothing
end


"""
tiles each column in A k-times
"""
function tile(A, k)
    tile_A = zeros(size(A, 1), 0)
    for i in 1:size(A, 2)
        tile_Ai = reshape(repeat(A[:, i], k), size(A, 1), k)
        tile_A = hcat(tile_A, tile_Ai)
    end
    return tile_A
end                            