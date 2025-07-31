"""Combinion – self-similar Pascal vectors

A *combinion* is a real, palindromic vector built from successive
Pascal‐row duplications ('*') and carries ('+').  Depth 0 is unary
⟨1⟩, depth 1 is ⟨1 1⟩, depth 2 is ⟨1 2 1⟩, depth 3 is
⟨1 3 3 1⟩ … – the same rows that generate Sierpiński’s gasket when printed
mod 2.

For depth ≥ 3 the four outer-edge coefficients hold the familiar
quaternion components (w,x,y,z).  This module exposes helpers to:

• expand – perform one *+ step (duplication then carry)
• entropy – Shannon entropy of magnitudes
• optimise – Gibbs-style optimiser (U − τS) re-used from the old Combit
• conversions   combinion ⇆ quaternion, integer, float
"""

from collections.abc import Iterable
from typing import List, Tuple
import numpy as np
import jax.numpy as jnp

__all__ = [
    "duplicator",
    "carry",
    "expand",
    "Combinion",
    "to_quaternion",
    "from_quaternion",
]

# -------------------------------------------------------------
# Core ops  *  (Kronecker duplication) and +  (carry)
# -------------------------------------------------------------

def duplicator(v: jnp.ndarray) -> jnp.ndarray:  # '*' step
    """Kronecker-duplicate `v` with [1,1]."""
    return jnp.kron(v, jnp.array([1, 1]))


def carry(v: jnp.ndarray) -> jnp.ndarray:  # '+' step
    """Pascal carry: add left-shifted copy into place."""
    return v.at[1:].add(v[:-1])


def expand(v: jnp.ndarray, steps: int = 1) -> jnp.ndarray:
    """Repeated *+ expansion."""
    out = v
    for _ in range(steps):
        out = carry(duplicator(out))
    return out

# -------------------------------------------------------------
# Quaternion helpers (palindromic indices 0,1,-2,-1)
# -------------------------------------------------------------

def to_quaternion(v: jnp.ndarray) -> Tuple[float, float, float, float]:
    assert v.size >= 4, "Need depth ≥3 to extract quaternion"
    w, x, y, z = float(v[0]), float(v[1]), float(v[-2]), float(v[-1])
    return w, x, y, z


def from_quaternion(q: Tuple[float, float, float, float], depth: int = 3) -> jnp.ndarray:
    """Embed quaternion as palindromic combinion of 2**depth length."""
    assert depth >= 3, "Depth must be ≥3 to fit quaternion"
    base = jnp.array(q)
    v = base
    while v.size < 2 ** depth:
        v = jnp.concatenate([v, v[::-1]])
    return v

# -------------------------------------------------------------
# Combinion class – minimal, focused
# -------------------------------------------------------------

class Combinion:
    def __init__(self, state: Iterable[float]):
        self.state = jnp.array(state, dtype=jnp.float32)

    # ---------- representation ----------
    def __repr__(self):  # concise
        return f"<Combinion len={len(self.state)} energy={self.energy:.3f}>"

    # ---------- algebra ----------
    def expand(self, steps: int = 1) -> "Combinion":
        self.state = expand(self.state, steps)
        return self

    # ---------- metrics ----------
    @property
    def energy(self) -> float:
        return float(jnp.mean(jnp.abs(self.state)))

    def entropy(self) -> float:
        mag = jnp.abs(self.state)
        p = mag / (jnp.sum(mag) + 1e-12)
        return float(-jnp.sum(p * jnp.log(p + 1e-12)))

    # ---------- optimisation ----------
    def optimise(self, target: jnp.ndarray, *, steps: int = 200, alpha: float = 0.5, step_size: float = 0.1) -> List[float]:
        """Gibbs-free-energy (U − τS) minimisation."""
        losses: List[float] = []

        def mse(a, b):
            return float(jnp.mean((a - b) ** 2))

        curr_loss = mse(self.state, target)
        tau = alpha * curr_loss / max(self.entropy(), 1e-12)

        rng = np.random.default_rng()
        for _ in range(steps):
            perturb = jnp.array(rng.normal(scale=step_size, size=self.state.shape))
            prev_state = self.state
            prev_entropy = self.entropy()
            self.state = self.state + perturb
            trial_loss = mse(self.state, target)
            new_entropy = self.entropy()
            delta_g = (trial_loss - curr_loss) - tau * (new_entropy - prev_entropy)
            if delta_g < 0 or rng.random() < np.exp(-delta_g):
                curr_loss = trial_loss
                losses.append(curr_loss)
            else:
                self.state = prev_state
        return losses

    # ---------- conversions ----------
    def as_quaternion(self):
        return to_quaternion(self.state)

    @classmethod
    def from_quaternion(cls, q: Tuple[float, float, float, float], depth: int = 3):
        return cls(from_quaternion(q, depth=depth))

    def as_triangle(self):  # right-justified view
        return np.roll(self.state, len(self.state)//2)
