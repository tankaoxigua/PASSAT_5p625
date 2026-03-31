from .fd import FiniteDifferenceOperator
from .fvm import FiniteVolumeOperator
from .sho import SpectralSHOperator

def build_space_operator(config, mesh):
    method = config.UPDATE.SPACE_METHOD

    if method == "FDM":
        return FiniteDifferenceOperator(mesh)
    elif method == "FVM":
        return FiniteVolumeOperator(mesh)
    elif method == "SPECTRAL_SH":
        return SpectralSHOperator(mesh=mesh, lmax=config.UPDATE.LMAX)

    else:
        raise ValueError(f"Unknown space method: {method}")
