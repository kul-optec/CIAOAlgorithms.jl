module CIAOAlgorithms

const RealOrComplex{R} = Union{R,Complex{R}}
const Maybe{T} = Union{T, Nothing}

# utulities 
include("utilities/indexingUtilities.jl")

# algorithms 
include("algorithms/Finito/Finito.jl")
include("algorithms/ProShI/ProShI.jl")
include("algorithms/SVRG/SVRG.jl")
include("algorithms/SAGA_SAG/SAGA.jl")

end # module