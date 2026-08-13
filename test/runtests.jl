import Test: @test, @test_throws, @testset
import Logging: @info

testdir = dirname(@__FILE__)

import Luna: set_fftw_mode
set_fftw_mode(:estimate)

# Test suite layout: testset name => files, in execution order.
# Run a subset by passing substring patterns as test arguments, e.g.
#   Pkg.test("Luna", test_args=["rk45", "freespace"])
#   julia --project test/runtests.jl rk45 freespace
# A file runs if any pattern occurs in its filename; no arguments runs everything.
const TESTFILES = [
    "Maths" => ["test_maths.jl"],
    "PhysData" => ["test_physdata.jl"],
    "Capillary" => ["test_capillary.jl"],
    "Rectangular Modes" => ["test_rect_modes.jl"],
    "ODE Solver" => ["test_rk45.jl"],
    "Ionisation" => ["test_ionisation.jl"],
    "Output" => ["test_output.jl"],
    "Multimode" => ["test_multimode.jl"],
    "Polarisation" => ["test_polarisation.jl", "test_polarisation_field.jl",
                       "test_polarisation_env.jl"],
    "Tools" => ["test_tools.jl"],
    "Utils" => ["test_utils.jl"],
    "Gradients" => ["test_gradient.jl"],
    "Tapers" => ["test_tapers.jl"],
    "Scans" => ["test_scans.jl"],
    "Raman" => ["test_raman.jl"],
    "Kerr" => ["test_kerr.jl"],
    "LinearOps" => ["test_linops.jl"],
    "Modes" => ["test_modes.jl"],
    "Radial Propagation" => ["test_radial.jl"],
    "Full 3D Propagation" => ["test_full_freespace.jl"],
    "Performance fast paths" => ["test_perf_bitident.jl"],
    "Antiresonant modes" => ["test_antiresonant.jl"],
    "Fields" => ["test_fields.jl"],
    "Processing" => ["test_processing.jl"],
    "Vector plasma" => ["test_vectorplasma.jl"],
    "Statistics" => ["test_stats.jl"],
    "Gas mixtures" => ["test_mixtures.jl"],
    "Interface" => ["test_interface.jl"],
    "Linear propagation" => ["test_linearprop.jl"],
    "GNLSE interface" => ["test_gnlse.jl"],
    "Noise model" => ["test_noise.jl"],
]

const testpatterns = String.(ARGS)
matches(fname) = isempty(testpatterns) || any(p -> occursin(p, fname), testpatterns)

@testset "All" begin
for (setname, files) in TESTFILES
    torun = filter(matches, files)
    isempty(torun) && continue
    @testset "$setname" begin
        for fname in torun
            @info("================= $fname")
            include(joinpath(testdir, fname))
        end
    end
end
end
