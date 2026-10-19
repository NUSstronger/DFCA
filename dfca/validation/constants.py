"""
Magnetic Constants for DFT Validation.

Magnetic element classifications and initial magnetic moment reference values.
These constants are scientific constants derived from experimental and DFT data.

References:
    - C2DB magnetic classification scheme
    - C2DB MAGMOM values
"""

# Magnetic element classifications
MAG_3D = {'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu'}
MAG_RE = {'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd',
          'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu'}
MAG_4D5D = {'Zr', 'Nb', 'Mo', 'Ru', 'Rh', 'Pd',
            'Hf', 'Ta', 'W',  'Re', 'Os', 'Ir', 'Pt'}

# Initial magnetic moment reference values (μB)
# Aligned with C2DB standards
MAGMOM_REF = {
    # 3d transition metals (high-spin)
    'Ti': 2.0, 'V': 3.0, 'Cr': 4.0, 'Mn': 5.0,
    'Fe': 4.0, 'Co': 3.0, 'Ni': 2.0, 'Cu': 1.0,
    # 4d transition metals (normal-spin)
    'Zr': 1.0, 'Nb': 1.0, 'Mo': 2.0, 'Ru': 2.0,
    'Rh': 1.0, 'Pd': 1.0,
    # 5d transition metals (normal-spin)
    'Hf': 1.0, 'Ta': 1.0, 'W': 2.0,  'Re': 2.0,
    'Os': 2.0, 'Ir': 1.0, 'Pt': 1.0,
    # Rare earth elements (strong magnetic)
    'Ce': 1.0, 'Pr': 2.0, 'Nd': 3.0, 'Pm': 4.0,
    'Sm': 5.0, 'Eu': 6.0, 'Gd': 7.0, 'Tb': 6.0,
    'Dy': 5.0, 'Ho': 4.0, 'Er': 3.0, 'Tm': 2.0,
    'Yb': 1.0, 'Lu': 0.5,
}

__all__ = [
    'MAG_3D',
    'MAG_RE',
    'MAG_4D5D',
    'MAGMOM_REF',
]
