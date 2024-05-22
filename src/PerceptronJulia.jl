
module PerceptronJulia
include("model/Model.jl")
include("function/Trainer.jl")
include("data/Data.jl")
include("utils/Plot.jl")
# --------------------
using .Model
using .Trainer
using .Data
using .Plot
# --------------------
end # module PerceptronJulia
