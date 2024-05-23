
module PerceptronJulia
include("model/Model.jl")
include("function/Activation.jl")
include("function/Optimiser.jl")
include("utils/Math.jl")
include("function/Trainer.jl")
include("data/Data.jl")
include("utils/Plot.jl")
# --------------------
using .Model
using .Trainer
using .Data
using .Plot
using .Activation
using .Math
using .Optimiser
# --------------------
end # module PerceptronJulia
