abstract type AbstractAlgorithm end

struct Groundstate_iDTEBD <: AbstractAlgorithm
    trunc_method_state::MatrixAlgebraKit.TruncationStrategy
    trunc_method_norm::MatrixAlgebraKit.TruncationStrategy
    convtol::Float64
    maxiter::Int
    verbosity::Int
    timer_output::TimerOutput
    finalizer::Finalizer

    function Groundstate_iDTEBD(trunc_method::MatrixAlgebraKit.TruncationStrategy, trunc_method_norm::MatrixAlgebraKit.TruncationStrategy; convtol::Float64 = 1e-8, maxiter::Int = 100, verbosity::Int = 0, timer_output::TimerOutput = TimerOutput(), finalizer::Finalizer = default_Finalizer)
        return new(trunc_method, trunc_method_norm, convtol, maxiter, verbosity, timer_output, finalizer)
    end
end