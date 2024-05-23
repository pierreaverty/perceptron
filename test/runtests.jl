include("../src/utils/Plot.jl")
include("../src/data/Data.jl")
include("../src/model/Model.jl")
include("../src/function/Activation.jl")
include("../src/function/Optimiser.jl")
include("../src/function/Trainer.jl")
include("../src/PerceptronJulia.jl")
include("../src/utils/Math.jl")
# --------------------
using Test
# --------------------
include("utils/test_plot.jl")
include("data/test_data.jl")
include("model/test_model.jl")
include("function/test_trainer.jl")
include("function/test_activation.jl")
include("function/test_optimiser.jl")
