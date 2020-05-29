using AMP
using Test, LinearAlgebra, BenchmarkTools
using JuLIP, JuLIP.Testing
using AMP: amp_acsf

@testset "AMP.jl" begin
    include("test.jl")
end

