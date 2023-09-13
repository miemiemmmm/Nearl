from .. import _usegpu
from . import _geometry, _map, _surface

if _usegpu:
  from . import interpolate_g as interpolate
else:
  from . import interpolate_c as interpolate


__all__ = [
  "_geometry",
  "_map",
  "_surface",
  "interpolate",
]


