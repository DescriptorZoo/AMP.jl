@testset "AMP ACSF Descriptor" begin

@info("Testing AMP ACSF Descriptor for dc Si with cutoff=6.5 and Behler2011 parameters.")
using AMP, JuLIP, Test

at = bulk(:Si, cubic=true)
desc = amp_acsf(at)
#round.(desc[1], digit=7) == acsfpy
acsf_ref = [9.89642686254e+00, 6.33890586962e+00,
            4.09478636081e+00, 2.32524172255e+00,
            1.08689635509e+00, 4.18793801933e-01,
            5.49166080698e-02, 1.05677667385e-03, 
            5.48207184853e+00, 1.46814261897e+01,
            2.33489680545e+00, 1.15342511467e+01, 
            3.68647068294e+00, 9.83117086198e+00, 
            1.59917860535e+00, 7.74387878439e+00, 
            1.91527626092e+00, 5.07034523923e+00, 
            8.57808585905e-01, 4.01287756422e+00, 
            8.09806838507e-01, 2.12270653881e+00, 
            3.78501769332e-01, 1.69140146963e+00, 
            1.35537274890e-01, 1.22679889188e+00, 
            2.17936760608e-03, 3.46608033973e-01, 
            2.59711785058e-01, 6.72593672234e-01, 
            1.27751356608e-01, 5.40633243783e-01, 
            4.87646991566e-02, 4.01895952394e-01, 
            5.18718040910e-04, 1.17867171366e-01, 
            3.30515305924e-02, 8.43383109191e-02,
            1.72801778808e-02, 6.85669582075e-02, 
            7.09381911607e-03, 5.25105488730e-02,
            5.61170353167e-05, 1.60425743126e-02, 
            1.18713390808e-03, 3.00606908300e-03, 
            6.40398948138e-04, 2.45933412306e-03, 
            2.72582239093e-04, 1.91258708755e-03, 5.95430642794e-04]
acsf_now = vcat(desc[1,:]...)
println("AMP returns:",acsf_now)
println("Reference:",acsf_ref)
println("AMP desc size:",length(acsf_now))
println(@test acsf_now  ≈  acsf_ref)

desc = amp_acsf(at, Behler2011=false, cuttype="Tanhyper3")
acsf_now = vcat(desc[1,:]...)
println("AMP desc size:",length(acsf_now))
end
