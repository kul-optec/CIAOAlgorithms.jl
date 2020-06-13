function findfst(ind::Array{Int64}, idx::Int64)
    x = falses(length(ind))
    for k in eachindex(ind)
        if ind[k] == idx
            x[k] = true
            return x
        end
    end     
end 