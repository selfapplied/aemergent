Here’s a lightweight roadmap so you can see exactly what will change before we touch any code.  

────────────────────────────────────────
Phase 0 (1 file, no behaviour change)
────────────────────────────────────────
Add a tiny helper module `aemergent/numtypes.py`

```python
from enum import Enum, auto
from dataclasses import dataclass
from jaxtyping import Float, Int, Array
import jax.numpy as jnp

class Encoding(Enum):
    MAG       = auto()        # unsigned magnitude
    TC        = auto()        # two’s-complement
    SPLIT_IJK = auto()        # stacked quaternion channels

@dataclass(slots=True)
class NumArray:
    data: Array               # carries dtype & shape
    eps:  float               # tolerance / granularity
    enc:  Encoding            # representation tag

    # helper views
    def f(self) -> Float[Array, "..."]:
        return self.data.astype(jnp.float32) * self.eps
```

Nothing else changes yet—no imports broken.

────────────────────────────────────────
Phase 1 (core ops gain shape-checked signatures)
────────────────────────────────────────
Edit `operators.py` (or `combinion.py`) to use jaxtyping:

```python
from jaxtyping import Float, Int, Array
from .numtypes import NumArray, Encoding

VecN   = Float[Array, "N"]    # generic 1-D vector of length N

def duplicator(v: NumArray[VecN]) -> NumArray[Float[Array, "2N"]]:
    out = jnp.kron(v.data, jnp.array([1, 1], dtype=v.data.dtype))
    return NumArray(out, v.eps, v.enc)
```

Mypy / Pyright will now yell if we wire the wrong shape or dtype.

────────────────────────────────────────
Phase 2 (entropy, optimiser, combinion)
────────────────────────────────────────
• `Combinion.state` becomes `NumArray`.  
• `entropy()` casts via `.f()` so underlying ints stay intact.  
• Optimiser returns new `NumArray`s, recording every cast in
  `TRACE.append(("to-float", len, eps, enc))`.

────────────────────────────────────────
Phase 3 (graceful migration)
────────────────────────────────────────
Add simple shims:

```python
def as_numarray(x) -> NumArray:
    return x if isinstance(x, NumArray) else NumArray(jnp.asarray(x), 1.0, Encoding.MAG)
```

so existing tests keep passing during the transition.

────────────────────────────────────────
What you’ll get immediately
────────────────────────────────────────
• Static shape/type errors caught at edit-time.  
• Runtime `beartype` checks so no silent shape mismatch.  
• Cup-lifting trace: run any routine and inspect `numtypes.TRACE`
  to see every tolerance / encoding boundary.

────────────────────────────────────────
Next step
────────────────────────────────────────
If the plan looks good I’ll:

1.  drop in `numtypes.py`  
2.  update `combinion.py` ops & entropy  
3.  run linter to show zero new errors.

Let me know and I’ll start the code pass.