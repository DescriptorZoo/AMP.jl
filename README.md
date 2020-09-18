![CI](https://github.com/DescriptorZoo/AMP.jl/workflows/CI/badge.svg)

# AMP.jl
Julia wrappers for [Atomistic Machine-learning Package (AMP)](https://amp.readthedocs.io/en/latest/index.html)

## Dependencies:

- [AMP](https://bitbucket.org/andrewpeterson/amp)
- [JuLIP.jl](https://github.com/JuliaMolSim/JuLIP.jl)
- [PyCall.jl](https://github.com/JuliaPy/PyCall.jl)
- [ASE.jl](https://github.com/JuliaMolSim/ASE.jl)

## Installation:

First, install AMP Python code following the code's [documentation](https://amp.readthedocs.io/en/latest/installation.html).
Either stable package with
```
pip3 install numpy
pip3 install amp-atomistics
```
or development with
```
pip3 install git+https://bitbucket.org/andrewpeterson/amp
```

Once you have installed the Python package that is used by your Julia installation, you can simply add this package to your Julia environment with the following command in Julia package manager and test AMP ACSF descriptor for Si:
```
] add https://github.com/DescriptorZoo/AMP.jl.git
] test AMP
```

### How to cite:

You can read about Amp in the first paper below. Although this code uses Amp functions in some functions, it also has significant other code parts that generate descriptors based on symmetry functions. If you use this code, we would appreciate if you cite the following papers:
- Alireza Khorshidi and Andrew A. Peterson, Computer Physics Communications 207:310-324, 2016. [DOI:10.1016/j.cpc.2016.05.010](http://dx.doi.org/10.1016/j.cpc.2016.05.010)
- Berk Onat, Christoph Ortner, James R. Kermode, 	[arXiv:2006.01915 (2020)](https://arxiv.org/abs/2006.01915)

If you use this code and hence dependent code [AMP](https://bitbucket.org/andrewpeterson/amp), you need to accept any license of [AMP](https://amp.readthedocs.io/en/latest/index.html) and cite both the code and the reference papers as they are described in code's [webpage](https://amp.readthedocs.io/en/latest/credits.html).

