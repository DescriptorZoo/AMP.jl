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
    """This function converts a dictionary (which was created with the
    to_dict method of one of the cutoff classes) into an instantiated
    version of the class. Modeled after ASE's dict2constraint function.
    """
    if len(dct) != 2:
        raise RuntimeError('Cutoff dictionary must have only two values,'
                           ' "name" and "kwargs".')
    return globals()[dct['name']](**dct['kwargs'])

class Cosine(object):
    """Cube Tangent Hyperbolic functional form in arXiv:1804.02150v1

    Parameters
    ---------
    Rc : float
        Radius above which neighbor interactions are ignored.
    """

    def __init__(self, Rc):

        self.Rc = Rc

    def __call__(self, Rij):
        """
        Parameters
        ----------
        Rij : float or numpy.float
            Distance between pair atoms.

        Returns
        -------
        float or numpy.float array
            The value of the cutoff function.
        """
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
        """Derivative (dfc_dRij) of the Cube Tangent Hyperbolic cutoff function with respect to Rij.

        Parameters
        ----------
        Rij : float
            Distance between pair atoms.

        Returns
        -------
        float
            The value of derivative of the cutoff function.
        """
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
    """Cube Tangent Hyperbolic functional form in arXiv:1804.02150v1

    Parameters
    ---------
    Rc : float
        Radius above which neighbor interactions are ignored.
    """

    def __init__(self, Rc):

        self.Rc = Rc

    def __call__(self, Rij):
        """
        Parameters
        ----------
        Rij : float or numpy.float
            Distance between pair atoms.

        Returns
        -------
        float or numpy.float array
            The value of the cutoff function.
        """
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
        """Derivative (dfc_dRij) of the Cube Tangent Hyperbolic cutoff function with respect to Rij.

        Parameters
        ----------
        Rij : float
            Distance between pair atoms.

        Returns
        -------
        float
            The value of derivative of the cutoff function.
        """
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

def fp_calculate_G2(neighborsymbols,
                 neighborpositions, G_element, eta, Rs, Rc_scale, 
                 cutoff, Ri, fortran, integral_scale=None):
    """Calculate G2 symmetry function.
    
    MODIFIED THE FUNCTION!!

    Returns
    -------
    ridge : float
        G2 fingerprint.
    """

    Rc = cutoff['kwargs']['Rc']
    cutoff_fxn = dict2cutoff(cutoff)
    ridge = 0.  # One aspect of a fingerprint :)
    num_neighbors = len(neighborpositions)   # number of neighboring atoms
    for count in range(num_neighbors):
        symbol = neighborsymbols[count]
        Rj = neighborpositions[count]
        if symbol == G_element:
            Rij = np.linalg.norm(Rj - Ri)
            Ra = (Rij - Rs) ** 2.
            if Rc_scale:
                Ra = Ra / (Rc ** 2.)
            args_cutoff_fxn = dict(Rij=Rij)
            if cutoff['name'] == 'Polynomial':
                args_cutoff_fxn['gamma'] = cutoff['kwargs']['gamma']
            pinch = (np.exp(-eta * Ra) *
                      cutoff_fxn(**args_cutoff_fxn)) # One pinch of a fingerprint :)
            ridge += pinch
    return ridge

def fp_calculate_G4(neighborsymbols, neighborpositions,
                 G_elements, gamma, zeta, eta, cutoff,
                 Ri, fortran, integral_scale=None):
    """Calculate G4 symmetry function.

    Returns
    -------
    ridge : float
        G4 fingerprint.
    """
    Rc = cutoff['kwargs']['Rc']
    cutoff_fxn = dict2cutoff(cutoff)
    ridge = 0.
    counts = range(len(neighborpositions))
    for j in counts:
        for k in counts[(j + 1):]:
            els = sorted([neighborsymbols[j], neighborsymbols[k]])
            if els != G_elements:
                continue
            Rij_vector = neighborpositions[j] - Ri
            Rij = np.linalg.norm(Rij_vector)
            Rik_vector = neighborpositions[k] - Ri
            Rik = np.linalg.norm(Rik_vector)
            Rjk_vector = neighborpositions[k] - neighborpositions[j]
            Rjk = np.linalg.norm(Rjk_vector)
            cos_theta_ijk = np.dot(Rij_vector, Rik_vector) / Rij / Rik
            term = (1. + gamma * cos_theta_ijk) ** zeta
            term *= np.exp(-eta * (Rij ** 2. + Rik ** 2. + Rjk ** 2.) /
                            (Rc ** 2.))
            _Rij = dict(Rij=Rij)
            _Rik = dict(Rij=Rik)
            _Rjk = dict(Rij=Rjk)
            if cutoff['name'] == 'Polynomial':
                _Rij['gamma'] = cutoff['kwargs']['gamma']
                _Rik['gamma'] = cutoff['kwargs']['gamma']
                _Rjk['gamma'] = cutoff['kwargs']['gamma']
            term *= cutoff_fxn(**_Rij)
            term *= cutoff_fxn(**_Rik)
            term *= cutoff_fxn(**_Rjk)
            ridge += term
    ridge *= 2. ** (1. - zeta)
    return ridge


def get_fingerprint(self, index, symbol,
                    neighborsymbols, neighborpositions):
    """Returns the fingerprint of symmetry function values for atom
    specified by its index and symbol.

    Returns
    -------
    symbol, fingerprint : list of float
            fingerprints for atom specified by its index and symbol.
    """
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

def group_symmetry_functions(Gs):
    """ This function groups symmetry fuctions
        according to their types and cutoffs.
    """
    groups = {}
    for elm in Gs.keys():
        G2cuts = {}
        G4cuts = {}
        G5cuts = {}
        for Gi, G in enumerate(Gs[elm]):
            gtype = G['type']
            rcut = G['Rc']['kwargs']['Rc']
            if gtype == 'G2':
                G2pack = [
                    G['element'], 
                    G['eta'], 
                    G['Rs'], 
                    G['Rc_scale']
                    ]
                if rcut in G2cuts.keys():
                    G2cuts[rcut][1].append(G2pack)
                else:
                    G2cuts[rcut] = [G['Rc'],[G2pack]]
            elif gtype == 'G4':
                G4pack = [
                    G['elements'], 
                    G['gamma'],
                    G['zeta'], 
                    G['eta']
                    ]
                if rcut in G4cuts.keys():
                    G4cuts[rcut][1].append(G4pack)
                else:
                    G4cuts[rcut] = [G['Rc'],[G4pack]]
            elif gtype == 'G5':
                G5pack = [
                    G['elements'], 
                    G['gamma'],
                    G['zeta'], 
                    G['eta']
                    ]
                if rcut in G5cuts.keys():
                    G5cuts[rcut][1].append(G5pack)
                else:
                    G5cuts[rcut] = [G['Rc'],[G5pack]]
        groups[elm] = [G2cuts,G4cuts,G5cuts]
    return groups


def calc_fingerprints(image, Gs, max_cutoff):
    """Returns all fingerprints of symmetry function values for atoms
    specified by image.

    Returns
    -------
    symbol, fingerprint : list of float
            fingerprints for atom specified by its index and symbol.
    """
    fortran=False
    desc = []
    ni, nj, nd, nvec = neighbor_list('ijdD', image, max_cutoff)
    img_syms = np.asarray(image.get_chemical_symbols())

    for i, atom in enumerate(image):
        Ri = atom.position
        symbol = atom.symbol
        index = atom.index
        neighborsymbols = img_syms[nj[np.isin(ni, index)]]
        neighborpositions = image.positions[nj[np.isin(ni, index)]]
        #neighVecs = nvec[nj[np.isin(ni, index)]]

        num_symmetries = len(Gs[symbol])
        fingerprint = [None] * num_symmetries

        for count in range(num_symmetries):
            G = Gs[symbol][count]
            ridge = 0.
            if G['Rc']['kwargs']['Rc'] < max_cutoff:
                neighsym = neighborsymbols[
                            nd[np.isin(ni, index)] < G['Rc']['kwargs']['Rc']]
                neighpos = neighborpositions[
                            nd[np.isin(ni, index)] < G['Rc']['kwargs']['Rc']]
            else:
                neighsym = neighborsymbols
                neighpos = neighborpositions
            if G['type'] == 'G2':
                ridge = fp_calculate_G2(neighsym, neighpos,
                                     G['element'], G['eta'], G['Rs'],
                                     G['Rc_scale'], G['Rc'],
                                     Ri, fortran)
            elif G['type'] == 'G4':
                ridge = fp_calculate_G4(neighsym, neighpos,
                                     G['elements'], G['gamma'],
                                     G['zeta'], G['eta'], G['Rc'],
                                     Ri, fortran)
            elif G['type'] == 'G5':
                ridge = fp_calculate_G5(neighsym, neighpos,
                                     G['elements'], G['gamma'],
                                     G['zeta'], G['eta'], G['Rc'],
                                     Ri, fortran)
            else:
                raise NotImplementedError('Unknown G type: %s' % G['type'])
            fingerprint[count] = ridge
        desc.append(fingerprint)

    return np.array(desc)

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
    for key, value in data.iteritems():
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
        if AMP_only:
            amp_desc = []
            desc_struct = {1:ase_structure}
            amp_out = amp_gaussian_desc_calc(desc, desc_struct) 
            for amp_item in amp_out[0]:
                amp_desc.append(amp_item[1])
            rtn_desc = np.array(amp_desc)
        else:
            rtn_desc = desc(ase_structure, Gs, max_cutoff)
        out_desc.append({
            "descriptors" : rtn_desc,
            "symbols"     : ase_structure.get_chemical_symbols()
            })
    return out_desc

def select_db_element(fingerprints, element=None):
    dbase = []
    if element:
        for item in fingerprints:
            for symi, sym in enumerate(item['symbols']):
                if element == sym:
                    dbase.append(item['descriptors'][symi])
    else:
        for item in fingerprints:
            for row in item['descriptors']:
                dbase.append(row)
    return np.array(dbase)


def set_symmetry_functions(elements, gtype, etas, Rs=None, Rc_scale=None, 
                           zetas=None, gammas=None, Rc=None):
    """Helper function to create Gaussian symmetry functions.
    Returns a list of dictionaries with symmetry function parameters
    in the format expected by the Gaussian class.

    Parameters
    ----------
    elements : list of str
        List of element types. The first in the list is considered the
        central element for this fingerprint. #FIXME: Does that matter?
    type : str
        Either G2 or G4.
    etas : list of floats
        eta values to use in G2 or G4 fingerprints
    Rs : list of floats
        R_s shift values to use in G2 fingerprints
    Rc_scale : list of Booleans
        Whether use Rc to scale in G2 fingerprints
    zetas : list of floats
        zeta values to use in G4 fingerprints
    gammas : list of floats
        gamma values to use in G4 fingerprints

    Returns
    -------
    G : list of dicts
        A list, each item in the list contains a dictionary of fingerprint
        parameters.
    """
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

def amp_calc_desc(ase_conf, desc):
    out_desc = None
    amp_desc = []
    desc_struct = {1:ase_conf}
    amp_out = amp_gaussian_desc_calc(desc, desc_struct)
    sys.stdout.flush()
    for amp_item in amp_out[0]:
        amp_desc.append(amp_item[1])
    out_desc = np.asarray(amp_desc)
    return out_desc

def fp_calc_desc(ase_database, desc, Gs):
    out_desc = []
    total_stime = time.time()
    for count, ase_structure in enumerate(ase_database):
        # Calculate descriptor
        rtn_desc = desc(ase_structure, Gs)
        out_desc.append({
            "descriptors" : rtn_desc,
            "symbols"     : ase_structure.get_chemical_symbols()
            })
    return out_desc

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

def remap_nonstring_keys(map_dct):
    dct = {} 
    for k, v in map_dct.items():
        if isinstance(v, dict):
            dct[k] = remap_nonstring_keys(v)
        elif isinstance(v, (list, tuple)):
            dct[k] = []
            for item in v:
                if isinstance(item, dict):
                    dct[k].append(remap_nonstring_keys(item))
                else:
                    dct[k].append(item)
        if isinstance(k, (tuple)):
            dct['keys'] = map_dct.keys()
            dct['values'] = map_dct.values()
            return dct
        else:
            if k not in dct:
                dct[k] = v
    return dct

def remap_keys_values(map_dct):
    dct = {} 
    if 'keys' in map_dct.keys() and 'values' in map_dct.keys():
        for item_i, item_k in enumerate(map_dct['keys']):
            dct[tuple(item_k)] = map_dct['values'][item_i]
        return dct
    else:
        for k, v in map_dct.items():
            if isinstance(v, dict):
                dct[k] = remap_keys_values(v)
            elif isinstance(v, (list, tuple)):
                dct[k] = [] 
                for item in v:
                    if isinstance(item, dict):
                        dct[k].append(remap_keys_values(item))
                    else:
                        dct[k].append(item)
            if k not in dct: 
                dct[k] = v
    return dct

def _decode_list(data):
    rv = []
    for item in data:
        if isinstance(item, unicode):
            item = item.encode('utf-8')
        elif isinstance(item, list):
            item = _decode_list(item)
        elif isinstance(item, dict):
            item = _decode_dict(item)
        rv.append(item)
    return rv

def _decode_dict(data):
    rv = {}
    for key, value in data.iteritems():
        if isinstance(key, unicode):
            key = key.encode('utf-8')
        if isinstance(value, unicode):
            value = value.encode('utf-8')
        elif isinstance(value, list):
            value = _decode_list(value)
        elif isinstance(value, dict):
            value = _decode_dict(value)
        rv[key] = value
    return rv

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

def select_db_element(fingerprints, element=None):
    dbase = []
    if element:
        for item in fingerprints:
            for symi, sym in enumerate(item['symbols']):
                if element == sym:
                    dbase.append(item['descriptors'][symi])
    else:
        for item in fingerprints:
            for row in item['descriptors']:
                dbase.append(row)
    return np.array(dbase)

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

    print_G(GG, typeslist)

    return GG, max_cutoff


def AMP_ACSF_desc(at, AMPonly=True, 
                  Behler2011=False, 
                  AMPformat=False, 
                  cut_type='Tanhyper3'):
    all_syms, all_nums, all_fractions = get_all_symbols([at])
    all_elements = [e for e in all_nums]

    G, max_cutoff = set_SF_paramaters_of_AMP_G(all_syms, Behler2011=args.Behler2011, 
                                               cuttype=cut_type, AMPformat=AMPformat)
  
    desc = Gaussian(cutoff=max_cutoff, Gs=G, elements=all_syms, fortran=True)
    return calc_desc([at], desc, G, AMP_only=AMPonly)

