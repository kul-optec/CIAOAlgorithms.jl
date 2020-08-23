function findfst(ind::Array{Int}, idx::Int)
    x = falses(length(ind))
    for k in eachindex(ind)
        if ind[k] == idx
            x[k] = true
            return x
        end
    end     
end 