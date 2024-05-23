include("../src/utils/Plot.jl")
include("../src/data/Data.jl")
include("../src/model/Model.jl")
include("../src/function/Trainer.jl")
include("../src/PerceptronJulia.jl")
# --------------------
using Test
using .PerceptronJulia
using .TrainerJulia
using .Model
using .Data
# --------------------
include("utils/test_plot.jl")
include("data/test_data.jl")
include("model/test_model.jl")
include("function/test_trainer.jl")
