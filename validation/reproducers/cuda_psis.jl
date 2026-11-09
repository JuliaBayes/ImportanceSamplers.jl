module CUDAPSISValidation

using CUDA, ImportanceSamplers, Statistics, Test
import MLDataDevices

function main(device_id = 0)
    CUDA.functional() || error("CUDA is required for this reproducer")
    CUDA.device!(device_id)
    CUDA.allowscalar(false)
    device = MLDataDevices.CUDADevice{typeof(CUDA.device()),Nothing}(CUDA.device())
    records = []
    @testset "CUDA PSIS residency and numerical parity" begin
        for T in (Float32, Float64)
            n = 100_000
            p = ((0:(n-1)) .+ 0.5) ./ n
            raw = T.(log1p.(expm1.(-0.7 .* log1p.(-p)) ./ 0.7))
            raw = raw[mod.(37 .* (0:(n-1)), n).+1]
            samples = WeightedSamples(
                T.(range(-3, 3; length = n)),
                raw;
                provenance = (round = fill(2, n),),
            )
            expected = psis(samples)
            resident = device(samples)
            actual = psis(resident)
            @test actual.samples === resident.samples
            @test actual.provenance === resident.provenance
            @test actual.diagnostics.transfers.bytes == sizeof(Bool) + 4sizeof(T)
            host = MLDataDevices.cpu_device()(actual)
            tolerance = T === Float32 ? 5e-5 : 2e-11
            @test host.logweights ≈ expected.logweights rtol = tolerance atol = tolerance
            @test actual.diagnostics.pareto_k ≈ expected.diagnostics.pareto_k rtol =
                tolerance
            @test Array(resident.logweights) == raw
            @test mean(actual) ≈ mean(expected) rtol = tolerance
            tied_logs = T.(range(-5, 0; length = 100))
            tied_logs[87:92] .= T(-0.5)
            tied = psis(device(tied_logs))
            @test Array(tied.logweights) ≈ psis(tied_logs).logweights rtol = tolerance atol =
                tolerance
            push!(
                records,
                (;
                    precision = T,
                    count = n,
                    status = actual.diagnostics.status,
                    maximum_error = maximum(abs.(host.logweights .- expected.logweights)),
                ),
            )
        end
    end
    return records
end

end

if abspath(PROGRAM_FILE) == @__FILE__
    CUDAPSISValidation.main()
end
