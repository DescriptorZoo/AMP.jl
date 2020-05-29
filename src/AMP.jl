module AMP

using ASE
using JuLIP
using PyCall

function __init__()
    py"""
    import math
    import numpy as np
    import itertools
    
    from amp import Amp
    from amp.descriptor.gaussian import Gaussian, make_symmetry_functions
    from amp.descriptor.gaussian import NeighborlistCalculator, FingerprintCalculator
    from amp.descriptor.gaussian import calculate_G2, calculate_G4, Cosine
    
    import ase
    from ase.build import bulk
    from ase.neighborlist import neighbor_list
    from ase.data import atomic_numbers, chemical_symbols
    
    PI = 3.14159265359
    unit_BOHR = 0.52917721067
    
    # DEFINITONS FOR SUPPORT FUNCTIONS
    # Of course these are slower functions than the ones in Fortran library 
    # -- ahem -- but these versions are not in the Fortran library unless they
    #   are also added to the library and Fortran library is compiled in AENET/AEPY.
    def dict2cutoff(dct):
        return globals()[dct['name']](**dct['kwargs'])
    
    class Cosine(object):
        def __init__(self, Rc):
    
            self.Rc = Rc
    
        def __call__(self, Rij):
            if isinstance(Rij, np.ndarray):
                rtn = np.zeros(len(Rij), dtype=np.float)
                cutoff_indices = Rij > self.Rc
                rtn[np.invert(cutoff_indices)] =  0.5 * (np.cos(
                    np.pi * Rij[np.invert(cutoff_indices)] / self.Rc) + 1.)
                return rtn
            else:
                if Rij > self.Rc:
                    return 0.
                else:
                    return 0.5 * (np.cos(np.pi * Rij / self.Rc) + 1.)
    
        def prime(self, Rij):
            if isinstance(Rij, np.ndarray):
                rtn = np.zeros(len(Rij), dtype=np.float)
                cutoff_indices = Rij > self.Rc
                rtn[np.invert(cutoff_indices)] =  -0.5 * np.pi / self.Rc * np.sin(
                        np.pi * Rij[np.invert(cutoff_indices)] / self.Rc)
                return rtn
            else:
                if Rij > self.Rc:
                    return 0.
                else:
                    return -0.5 * np.pi / self.Rc * np.sin(np.pi * Rij / self.Rc)
    
        def todict(self):
            return {'name': 'Cosine',
                    'kwargs': {'Rc': self.Rc}}
    
        def __repr__(self):
            return ('<Cosine cutoff with Rc=%.3f from amp.descriptor.cutoffs>'
                    % self.Rc)
    
    class Tanhyper3(object):
        def __init__(self, Rc):
    
            self.Rc = Rc
    
        def __call__(self, Rij):
            if isinstance(Rij, np.ndarray):
                rtn = np.zeros(len(Rij), dtype=np.float)
                cutoff_indices = Rij > self.Rc
                rtn[np.invert(cutoff_indices)] = (np.tanh(1. - (Rij[np.invert(cutoff_indices)] / self.Rc)))**3
                return rtn
            else:
                if Rij > self.Rc:
                    return 0.
                else:
                    return (np.tanh(1. - (Rij / self.Rc)))**3
    
        def prime(self, Rij):
            if Rij > self.Rc:
                return 0.
            else:
                in_tanh = np.tanh(1. - (Rij / self.Rc))
                return 3. * in_tanh**2 * (1. - in_tanh)**2
    
        def todict(self):
            return {'name': 'Tanhyper3',
                    'kwargs': {'Rc': self.Rc}}
    
        def __repr__(self):
            return ('<Tanhyper3 cutoff with Rc=%.3f from DescriptorZoo>'
                    % self.Rc)
    
    def get_fingerprint(self, index, symbol,
                        neighborsymbols, neighborpositions):
        Ri = self.atoms[index].position
    
        num_symmetries = len(self.globals.Gs[symbol])
        fingerprint = [None] * num_symmetries
    
        for count in range(num_symmetries):
            G = self.globals.Gs[symbol][count]
    
            if G['type'] == 'G2':
                ridge = calculate_G2(neighborsymbols, neighborpositions,
                                     G['element'], G['eta'], G['Rs'], G['Rc_scale'],
                                     G['Rc'], Ri, self.fortran, G['integral_scale'])
            elif G['type'] == 'G4':
                ridge = calculate_G4(neighborsymbols, neighborpositions,
                                     G['elements'], G['gamma'],
                                     G['zeta'], G['eta'], G['Rc'],
                                     Ri, self.fortran, G['integral_scale'])
            else:
                raise NotImplementedError('Unknown G type: %s' % G['type'])
            fingerprint[count] = ridge
    
        return symbol, fingerprint
    
    def _get_key_values_list(data, find_key=None):
        rv = []
        founds = []
        for item in data:
            found = False
            if isinstance(item, list):
                found, item = _get_key_values_list(item, find_key=find_key)
            elif isinstance(item, dict):
                found, item = _get_key_values_dict(item, find_key=find_key)
            if found:
                founds.append(found)
                rv.extend(item)
        if True in founds:
            found = True
        return found, list(set(rv))
    
    def _get_key_values_dict(data, find_key=None):
        rv = []
        founds = {}
        for key, value in data.items():
            found = False
            founds[key] = False
            if isinstance(value, list):
                found, value = _get_key_values_list(value, find_key=find_key)
            elif isinstance(value, dict):
                found, value = _get_key_values_dict(value, find_key=find_key)
            if found or find_key == key:
                founds[key] = True
            if founds[key]:
                if isinstance(value, list):
                    rv.extend(value)
                else:
                    rv.append(value)
        if True in founds.values():
            found = True
        return found, list(set(rv))
    
    def calc_desc(ase_database, desc, Gs, max_cutoff=None, AMP_only=False):
        out_desc = []
        found, cutoffs_list = _get_key_values_dict(Gs, find_key='Rc')
        if found:
            max_cutoff = np.max(np.asarray(cutoffs_list))
        else:
            if max_cutoff is None:
                max_cutoff = 12.
        for count, ase_structure in enumerate(ase_database):
            # Calculate descriptor
            amp_desc = []
            desc_struct = {1:ase_structure}
            amp_out = amp_gaussian_desc_calc(desc, desc_struct) 
            for amp_item in amp_out[0]:
                amp_desc.append(amp_item[1])
            rtn_desc = np.array(amp_desc)
            out_desc.append({
                'descriptors' : rtn_desc,
                'symbols'     : ase_structure.get_chemical_symbols()
                })
        return out_desc
    
    def set_symmetry_functions(elements, gtype, etas, Rs=None, Rc_scale=None, 
                               zetas=None, gammas=None, Rc=None):
        if Rc is None:
            Rc = 6.5
        if isinstance(Rc, (float, int)):
            set_cutoff_type = Cosine(Rc)
            Rc = set_cutoff_type.todict()
    
        if gtype == 'G2':
            G = [{'type': 'G2', 
                  'element': element, 
                  'eta': float(eta), 
                  'Rs' : float(Rs[ei]), 
                  'Rc_scale': Rc_scale[ei],
                  'Rc': Rc}
                 if Rs is not None and Rc_scale is not None 
                 else
                 {'type': 'G2', 
                  'element': element, 
                  'eta': float(eta), 
                  'Rs' : float(Rs[ei]), 
                  'Rc_scale': False,
                  'Rc': Rc}
                 if Rs is not None
                 else
                 {'type': 'G2', 
                  'element': element, 
                  'eta': float(eta), 
                  'Rs' : 0., 
                  'Rc_scale': Rc_scale[ei],
                  'Rc': Rc}
                 if Rc_scale is not None
                 else 
                 {'type': 'G2', 
                  'element': element, 
                  'eta': float(eta), 
                  'Rs' : 0., 
                  'Rc_scale': False,
                  'Rc': Rc}
                 for ei, eta in enumerate(etas)
                 for element in elements]
            return G
        elif gtype == 'G4' or gtype == 'G5':
            G = []
            for eta in etas:
                for zeta in zetas:
                    for gamma in gammas:
                        for i1, el1 in enumerate(elements):
                            for el2 in elements[i1:]:
                                els = sorted([el1, el2])
                                G.append({'type': gtype,
                                          'elements': els,
                                          'eta': float(eta),
                                          'gamma': float(gamma),
                                          'zeta': int(zeta),
                                          'Rc': Rc})
            return G
        raise NotImplementedError('Unknown type: {}.'.format(gtype))
    
    
    # DEFINITONS FOR SUPPORT FUNCTIONS
    def amp_fingerprint_calc(atom, image, f_calc, n_list):
        symbol = atom.symbol
        index = atom.index
        neighborindices, neighboroffsets = n_list[index]
        neighborsymbols = [image[_].symbol for _ in neighborindices]
        neighborpositions = \
            [image.positions[neighbor] + np.dot(offset, image.cell)
                for (neighbor, offset) in zip(neighborindices,
                                              neighboroffsets)]
        return f_calc.get_fingerprint(
            index, symbol, neighborsymbols, neighborpositions)           
    
    def amp_gaussian_desc_calc(amp_desc, image_dict, desc_atoms=[]):    
        allfingerprints = []
        for key, image in image_dict.items():
            n_calc = NeighborlistCalculator(cutoff=amp_desc.parameters.cutoff['kwargs']['Rc'])
            neighborlist = {key: n_calc.calculate(image, key)}
            f_calc = FingerprintCalculator(neighborlist=neighborlist,
                                           Gs=amp_desc.parameters.Gs,
                                           cutoff=amp_desc.parameters.cutoff,
                                           fortran=amp_desc.fortran)
            f_calc.atoms = image
            fingerprints = []
            if desc_atoms:
                for atom_index in desc_atoms:
                    fingerprints.append(amp_fingerprint_calc(image[atom_index], image, f_calc, neighborlist[key]))
            else:
                for atom in image:
                    fingerprints.append(amp_fingerprint_calc(atom, image, f_calc, neighborlist[key]))
                
            allfingerprints.append(fingerprints)
        return allfingerprints
    
    def generate_sf_parameters(n=5, cutoff=6.0, bohr=False):
        #Default values: cutoff = 6.0
        # n = 5 , number of intervals
        if bohr:
            cutoff = cutoff/unit_BOHR
        m = np.array(range(n+1), dtype=np.float32)
        n_pow = n**(m/n)
        eta_m = (n_pow/cutoff)**2
        R_s = cutoff/n_pow
        eta_s = np.zeros(n+1)
        for mi in range(n):
            eta_s[mi] = 1. / (R_s[n - mi] - R_s[n - mi - 1])**2
        # !!!!!! FLIPED eta_s ARRAY HERE !!!! REVERSED ORDER NOT MENTIONED IN PAPER !!!!!!
        # ALSO R_s and eta_s shifted !!! NOT MENTIONED AGAIN !!!
        return eta_m, R_s[1:n+1], np.flip(eta_s[0:n],0)
    
    
    def get_all_symbols(ase_data):
        symbols = []
        all_elements = list(itertools.chain.from_iterable([
            m.numbers for m in ase_data]))  # flattened list of all proton numbers Z in the dataset
        numbers = np.unique(all_elements)
        bins = np.bincount(all_elements)
        fractions = np.asarray([float(bin_num)/float(len(all_elements)) for bin_num in bins])
        for num in numbers:
            symbols.append(chemical_symbols[num])
        return symbols, numbers, fractions
    
    def set_SF_paramaters_of_AMP_G(typeslist, G_types=None, G_settings=None, G_cutoff_types=None, 
                                   Behler2011=False, cuttype='Tanhyper3', AMPformat=True):
        if Behler2011:
            max_cutoff = 6.5
            sympack = [ 
                np.array([   9.,  100.,  200.,  350.,  600., 1000., 2000., 4000.]),
                np.array([   1.,   1.,   1.,   1.,  30.,  30.,  30.,  30.,  80.,  80.,  80.,  80., 
                           150., 150., 150., 150., 150., 150., 150., 150., 250., 250., 250., 250., 
                           250., 250., 250., 250., 450., 450., 450., 450., 450., 450., 450., 450., 
                           800., 800., 800., 800., 800., 800., 800.]),
                np.array([  -1.,   1.,  -1.,   1.,  -1.,   1.,  -1.,   1.,  -1.,   1.,  -1.,   1., 
                            -1.,   1.,  -1.,   1.,  -1.,   1.,  -1.,   1.,  -1.,   1.,  -1.,   1.,  
                            -1.,   1.,  -1.,   1.,  -1.,   1.,  -1.,   1.,  -1.,   1.,  -1.,   1.,
                            -1.,   1.,  -1.,   1.,  -1.,   1.,   1.]),
                np.array([   1.,   1.,   2.,   2.,   1.,   1.,   2.,   2.,   1.,   1.,   2.,   2.,  
                             1.,   1.,   2.,   2.,   4.,   4.,  16.,  16.,   1.,   1.,   2.,   2.,   
                             4.,   4.,  16.,  16.,   1.,   1.,   2.,   2.,   4.,   4.,  16.,  16.,
                             1.,   1.,   2.,   2.,   4.,   4.,  16.])
                ]
            if AMPformat:
                amp_r_etas = (max_cutoff*max_cutoff)*sympack[0]/(10000.0*unit_BOHR*unit_BOHR)
                amp_a_etas = (max_cutoff*max_cutoff)*sympack[1]/(10000.0*unit_BOHR*unit_BOHR)
            else:
                amp_r_etas = sympack[0]/(10000.0*unit_BOHR*unit_BOHR) 
                amp_a_etas = sympack[1]/(10000.0*unit_BOHR*unit_BOHR)
            G = set_symmetry_functions(elements=typeslist, gtype='G2',
                                       etas=amp_r_etas)
            ang_typ = 'G4'
            for ang_i in range(len(amp_a_etas)):
                G += set_symmetry_functions(elements=typeslist, gtype=ang_typ,
                                            etas=[amp_a_etas[ang_i]],
                                            zetas=[sympack[3][ang_i]],
                                            gammas=[sympack[2][ang_i]])
    
        else:
            if G_types is None:
                rad_typ = 'G2'
                ang_typ = 'G4'
                G_types = [rad_typ, rad_typ, rad_typ,
                           rad_typ, rad_typ, rad_typ,
                           ang_typ, 
                           ang_typ]
            if G_settings is None:
                G_settings = [[8, 4., 0], [8, 8., 0], [8, 12., 0],
                              [8, 4., 1], [8, 8., 1], [8, 12., 1],
                              [8, 4., [1, 2, 4, 8, 16], [-1.0, 1.0]], 
                              [8, 8., [1, 2, 4],        [-1.0, 1.0]]]
            if G_cutoff_types is None:
                G_cutoff_types = [cuttype, cuttype, cuttype,
                                  cuttype, cuttype, cuttype,
                                  cuttype, cuttype]
            
            G = None
    
            max_cutoff = 0.0
            for Gi, Gtyp in enumerate(G_types):
                cutf = G_settings[Gi][1]*unit_BOHR
                if cutf > max_cutoff:
                    max_cutoff = cutf
                set_eta_m, set_R_s, set_eta_s = generate_sf_parameters(n=G_settings[Gi][0], 
                                                                       cutoff=cutf)
                if 'Tanhyper3' in G_cutoff_types[Gi]:
                    set_cutoff_type = Tanhyper3(cutf)
                    set_cutoff = set_cutoff_type.todict()
                else:
                    set_cutoff = cutf
    
                Gadd = None
                if 'G2' in Gtyp:
                    if G_settings[Gi][2] < 1:
                        Gadd = set_symmetry_functions(elements=typeslist, gtype='G2',
                                                      etas=set_eta_m, Rc=set_cutoff)
                    else:
                        Gadd = set_symmetry_functions(elements=typeslist, gtype='G2',
                                                      etas=set_eta_s,
                                                      Rs=set_R_s,
                                                      Rc=set_cutoff)
                elif 'G4' in Gtyp:
                        Gadd = set_symmetry_functions(elements=typeslist, gtype=Gtyp,
                                                      etas=set_eta_m,
                                                      zetas=G_settings[Gi][2],
                                                      gammas=G_settings[Gi][3],
                                                      Rc=set_cutoff)
                else:
                    print('WARNING! Unrecognized SF G type : ',Gtyp)
    
                if G is None:
                    G = Gadd
                else:
                    G += Gadd
    
    
        # Use same descriptor setting for all species
        GG = {}
        for element in typeslist:
            if G is not None:
                GG.update({element : G})
    
        return GG, max_cutoff
    

    def AMP_ACSF_desc(at, AMPonly=True, 
                      Behler2011=False, 
                      AMPformat=False, 
                      cut_type='Tanhyper3'):
        all_syms, all_nums, all_fractions = get_all_symbols([at])
    
        G, max_cutoff = set_SF_paramaters_of_AMP_G(all_syms, Behler2011=Behler2011, 
                                                   cuttype=cut_type, AMPformat=AMPformat)
      
        desc = Gaussian(cutoff=max_cutoff, Gs=G, elements=all_syms, fortran=True)
        fingerprints = calc_desc([at], desc, G, AMP_only=AMPonly)
        return fingerprints[0]['descriptors']
    """
end
    
# To Do: add it as a package
#amppy = pyimport("AMP.py")

export amp_acsf

function amp_acsf(at; AMPonly=true, Behler2011=true, AMPformat=true, cuttype="Cosine")
    atom_struct = ASEAtoms(at)
    # Calculate descriptor
    acsf_desc = py"AMP_ACSF_desc"(atom_struct.po, 
                                  AMPonly=AMPonly, 
                                  Behler2011=Behler2011,
                                  AMPformat=AMPformat,
                                  cut_type=cuttype)
    return acsf_desc
end

end # module
