
module PerceptronJulia
include("model/Model.jl")
include("function/Trainer.jl")
include("data/Data.jl")
include("utils/Plot.jl")
include("utils/Symbol.jl")
# --------------------
using .Model
using .Trainer
using .Data
using .Plot
using .Symbol
# --------------------
end # module PerceptronJulia
