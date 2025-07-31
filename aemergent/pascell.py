#!/usr/bin/env python3
"""
Pascell - Compact Vectorized CA Engine

A high-performance, vectorized cellular automata engine using numpy and Kronecker products
for efficient pattern generation and analysis.

Features:
- Vectorized rule application using numpy operations
- Kronecker-based pattern generation for mathematical elegance
- Bitmask signature analysis for pattern diversity measurement
- Performance tracking and evolutionary parameter adjustment
- Backward compatibility with traditional CA approaches

Example:
    >>> from aemergent.src.pascell import CAEngine
    >>> engine = CAEngine(max_rows=20)
    >>> grid, signatures, params, performance = engine.generate()
    >>> print(f"Final performance: {performance[-1]:.3f}")
"""

import functools
from hmac import new
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple, Optional
from collections import deque
import math

@dataclass(slots=True)
class CASetup:
    @staticmethod
    def by_rows(n: int):
        complexity = 0.5 / n
        return CASetup( max_rows=n, mod_base=n, capacity=n * 2, complexity=complexity)

    @staticmethod
    def by_complexity(complexity: float):
        return CASetup.by_rows(int(1 / complexity) * 2)

    max_rows: int = 20
    mod_base: int = 10
    capacity: int = 10
    complexity: float = 0.5

@dataclass(slots=True)
class CARule:
    name: str
    func: Callable = field(default_factory=lambda: play_pascal)
    active: bool = True
    history: deque[int] = field(default_factory=lambda: deque(maxlen=10))
    points: float = 0.0
    mass: float = 1.0


@dataclass(slots=True)
class CAEngine:
    """
    Compact vectorized CA engine with evolutionary scoring.
    
    Uses numpy vectorization and Kronecker products for high-performance
    cellular automata generation and analysis.
    
    Attributes:
        max_rows (int): Maximum number of rows to generate
        params (dict): Engine parameters including mod_base, capacity, and weights
    
    Example:
        >>> engine = CAEngine(max_rows=50)
        >>> grid, signatures, params, performance = engine.generate()
        >>> print(f"Generated {len(grid)} rows with {len(set(signatures))} unique patterns")
    """
    
    setup: CASetup
    players: list[CARule] = field(default_factory=lambda: [CARule(name='pascal', func=play_pascal, active=True)])
    objectives: list[CARule] = field(default_factory=lambda: [])
    weights: list[float] = field(default_factory=lambda: [1.0])
    history_len: int = 10
    sigma: float = 1.0
    zoom: Optional['GoldenSpiralZoom'] = None

    def __post_init__(self):
        self._rebalance_weights()
        for p in self.players:
            p.history = deque(maxlen=self.history_len)
            p.points = 0.0
            p.mass = 1.0
            
    def player(self, name: str) -> CARule | None:
        """Get a player by name"""
        return next(p for p in self.players if p.name == name)

    @property
    def active_players(self) -> list[CARule]:
        """Get all active players"""
        return [p for p in self.players if p.active]
    
    def add_player(self, name: str, player_func: Callable):
        """Add a new player to the CA game"""
        self.players.append(CARule(name=name, func=player_func))
        self._rebalance_weights()

    def remove_player(self, name: str):
        """Remove a player from the CA game"""
        self.players = [p for p in self.players if p.name != name]
        self._rebalance_weights()
    
    def objective(self, name: str) -> CARule | None:
        """Get an objective by name"""
        return next(o for o in self.objectives if o.name == name)
    
    @property
    def active_objectives(self) -> list[CARule]:
        """Get all active objectives"""
        return [o for o in self.objectives if o.active]

    def _rebalance_weights(self):
        """Rebalance weights when players are added/removed"""
        n_players = len([p for p in self.players if p.active])
        if n_players > 0:
            self.weights = n_players * [1.0 / n_players]
    
    def rule(self, history: np.ndarray, neighborhood: np.ndarray) -> np.ndarray:
        """
        Vectorized application of complementary CA rules.
        
        Combines multiple players using weighted vectorized operations
        for maximum efficiency and complexity.
        
        Args:
            history: Column's full evolutionary history
            neighborhood: Immediate spatial context (left/right neighbors)
            
        Returns:
            Vectorized rule application results
        """
        if not self.active_players:
            return neighborhood
        
        # Get results from all active players
        results = []
        for player in self.active_players:
            player_result = player.func(history, neighborhood)
            results.append(player_result)
        
        # Weighted combination of all players
        weights = self.weights[:len(results)]
        return np.inner(weights, results)
        result = np.zeros_like(neighborhood[:, 0], dtype=np.float32)
        for i, player_result in enumerate(results):
            result += weights[i] * player_result
        return result.astype(np.int32)
    
    def evolve_parameters(self, performance: float):
        """Evolve parameters based on performance"""
        if performance > self.setup.complexity:
            # Increase complexity
            self.setup = CASetup.by_complexity(self.setup.complexity + 0.1)
        else:
            # Decrease complexity
            self.setup = CASetup.by_complexity(self.setup.complexity - 0.1)
    
    def bitmasks(self, grid: np.ndarray) -> np.ndarray:
        """
        Generate signatures using modular objective system.
        
        Uses configurable objectives to evaluate grid patterns.
        
        Args:
            grid: 2D numpy array of CA grid
            
        Returns:
            Array of signatures based on active objectives
        """
        if self.active_objectives:
            return np.any([a.func(grid) for a in self.active_objectives], axis=0, keepdims=True)
        return np.packbits([1])
    
    def generate(self, place: Optional[np.ndarray] = None) -> "CAGame":
        """
        Generate CA evolution using player rules.
        
        Uses the active players to generate each row based on the previous row.
        
        Returns:
            CAGame object with evolution results
        """
        grid = np.zeros((self.setup.max_rows, self.setup.max_rows + 1), dtype=np.int32)

        if place:
            assert (width := place.shape[0]) <= self.setup.max_rows, "Place is too wide"
            assert len(place.shape) == 1, "Set the initial row starting conditions"
            grid[:width] = place

        start = time.time()
        # Pre-allocate grid for maximum efficiency
        grid[0, :1] = [1]  # Initialize first row with just 1
        
        # Generate each row using player rules
        for i in range(1, self.setup.max_rows):
            prev_row = grid[i-1]
            
            # Create history (column evolution) and neighborhood (left/right)
            history = grid[:i]  # Full column history up to current row
            left = prev_row[:-1]
            right = prev_row[1:]
            neighborhood = np.column_stack([left, right])  # 2-column array: [left, right]
            
            # Apply player rules
            middle = self.rule(history, neighborhood)            
            # Create extended arrays for rule application
            # For left end: use (0, left[0])
            # For middle: use (left, right) 
            # For right end: use (right[-1], 0)
            extended_left = np.concatenate([[0], left, [right[-1] if len(right) > 0 else 0]])
            extended_right = np.concatenate([[left[0] if len(left) > 0 else 0], right, [0]])
            extended_neighborhood = np.column_stack([extended_left, extended_right])
            
            # Apply rule to entire extended row
            full_result = self.rule(history, extended_neighborhood)
            
            # Apply objectives as bitmasks to transform the result
            new_row = full_result[:i+1]
            grid[i, :len(new_row)] = new_row
        
        # Evaluate objectives
        signatures, performance_score = self.evaluate_objectives(grid)
        
        # Calculate performance evolution
        unique_counts = np.sum(np.unique_counts(signatures))
        performance = unique_counts / np.arange(1, len(signatures) + 1)
        
        # Evolve parameters based on objective performance
        if len(performance) > 0:
            self.evolve_parameters(performance_score)
        # Cooperative gain distribution
        distribute_points(self.active_players, performance_score, self.sigma)
        
        return CAGame(signatures, performance, grid, time.time() - start)
    
    def add_objective(self, name: str, objective_func: Callable):
        """Add a new objective to the CA game"""
        self.objectives.append(CARule(name=name, func=objective_func))
    
    def remove_objective(self, name: str):
        """Remove an objective from the CA game"""
        self.objectives = [o for o in self.objectives if o.name != name]
    
    def evaluate_objectives(self, grid: np.ndarray) -> Tuple[np.ndarray, float]:
        """Evaluate all active objectives and return combined score"""
        if not self.active_objectives:
            return np.array([]), 0.0
        
        results = []
        for objective in self.active_objectives:
            obj_result = objective.func(grid)
            results.append(obj_result)
        
        if not results:
            return np.array([]), 0.0
        
        # Combine objective results
        combined = np.mean(results, axis=0)
        
        # Calculate performance score
        unique_count = len(np.unique(combined))
        total_count = len(combined)
        performance = unique_count / total_count if total_count > 0 else 0.0
        
        return combined, performance

@dataclass(slots=True)
class CAGame:
    signatures: np.ndarray
    performance: np.ndarray
    grid: np.ndarray
    elapsed: float

    def __getitem__(self, slice: slice):
        return "\n".join(
            "".join(str(x) for x in row) for row in self.grid[slice]
        )

    def __str__(self):
        final_performance = self.performance[-1] if len(self.performance) > 0 else 0.0
        return (
            f"n={self.grid.shape[-1]}: {self[-1:]}\n"
            f"Generation time: {self.elapsed:.3f}s\n"
            f"Final performance: {final_performance:.3f}"
        )


def clip(rule, base: int = 10):
    """Clip values to base"""
    def clipped_rule(history, neighborhood):
        result = rule(history, neighborhood)
        return np.where(result >= base, base - 1, result)
    return clipped_rule


def play_pascal(history: np.ndarray, neighborhood: np.ndarray) -> np.ndarray:
    """Pascal's triangle rule: add adjacent elements without modular arithmetic"""
    # Handle both 1D and 2D neighborhood arrays
    return neighborhood[:, 0] + neighborhood[:, 1]


def play_pascal911(history: np.ndarray, neighborhood: np.ndarray) -> np.ndarray:
    """Overflow prevention 9 + 1 -> 1 rule in base 10"""
    return clip(play_pascal, 10)(history, neighborhood)

# --------------------------------------------------------------------
# Advanced Engine Utilities – golden-spiral zoom, decorators, scoring
# --------------------------------------------------------------------


class GoldenSpiralZoom:
    """Golden-spiral expansion for cellular automata grids."""

    def __init__(self, a: float = 1.0, b: float = 0.05, dtheta: float = math.radians(137.5)):
        self.a = a
        self.b = b
        self.dtheta = dtheta

    def _offset(self, step: int) -> tuple[int, int]:
        theta = step * self.dtheta
        r = self.a * math.exp(self.b * theta)
        dx = int(round(r * math.cos(theta)))
        dy = int(round(r * math.sin(theta)))
        return dy, dx  # row, col order

    def map(self, coords: np.ndarray, step: int) -> np.ndarray:
        """Vectorised coordinate shift for the given step."""
        if coords.size == 0:
            return coords
        dy, dx = self._offset(step)
        return coords + np.array([dy, dx])

    def mask(self, coords: np.ndarray, step: int) -> np.ndarray:
        """All-pass mask by default – override for pruning."""
        return np.ones(len(coords), dtype=bool)


def view(radius: int):
    """Decorator to annotate rule functions with a view radius."""
    def decorator(f):
        f._view_radius = radius
        return f
    return decorator


def distribute_points(players: list[CARule], global_gain: float, sigma: float = 1.0):
    """Softmax-like distribution of cooperative gain across players."""
    if not players:
        return
    scores = np.array([p.points for p in players], dtype=float)
    if np.allclose(scores, 0):
        weights = np.full_like(scores, 1 / len(players), dtype=float)
    else:
        weights = np.exp(scores / (sigma + 1e-9))
        weights_sum = weights.sum()
        weights /= weights_sum if weights_sum else 1.0
    for p, w in zip(players, weights):
        p.mass += global_gain * w
        p.points += global_gain * w


# Back-compatibility aliases
pascal_player = play_pascal
pascal911 = play_pascal911

def diversity_objective(grid: np.ndarray) -> np.ndarray:
    """Default objective: maximize pattern diversity"""
    sums = np.sum(grid, axis=1)
    unique_counts = np.array(
        [len(np.unique(row[row != 0])) for row in grid])
    palindromes = np.array(
        [np.array_equal(row[:len(row)//2], row[len(row)//2-1::-1]) for row in grid])
    symmetries = np.array([
        abs(np.sum(row[:len(row)//2]) - np.sum(row[len(row)//2:])) <= 1
        for row in grid
    ])

    signatures = (sums % 2).astype(np.int32) << 0
    signatures |= (unique_counts % 2).astype(np.int32) << 1
    signatures |= palindromes.astype(np.int32) << 2
    signatures |= symmetries.astype(np.int32) << 3

    return signatures

def symmetry_objective(grid: np.ndarray) -> np.ndarray:
    """Objective: maximize symmetry patterns"""
    symmetries = np.array([
        np.sum(np.abs(row - row[::-1])) for row in grid
    ])
    return symmetries.astype(np.int32)

def complexity_objective(grid: np.ndarray) -> np.ndarray:
    """Objective: maximize computational complexity"""
    complexities = np.array([
        np.sum(np.abs(np.diff(row))) for row in grid
    ])
    return complexities.astype(np.int32)

def detection_objective(f: Callable, grid: np.ndarray) -> np.ndarray:
    """Objective: detect patterns matching a function"""
    scores = np.array([
        sum(1 for x in row if f(x)) for row in grid
    ])
    return scores.astype(np.int32)

def fractal_objective(grid: np.ndarray) -> np.ndarray:
    """Objective: maximize fractal-like patterns"""
    fractal_scores = np.array([
        len(set(np.diff(row))) for row in grid
    ])
    return fractal_scores.astype(np.int32)


def demo_ca_evolution(n: int = 20):
    engine = CAEngine(
        CASetup.by_rows(n),
        [CARule(name='pascal', func=play_pascal911, active=True)],
    )
    game = engine.generate()
    print(game[:], "\n", game)
    return game

if __name__ == "__main__":
    # Run demo when executed directly
    demo_ca_evolution()

__all__ = ["CAEngine", "CASetup", "pascal_player", "clip", "pascal911", "diversity_objective", "symmetry_objective", "complexity_objective", "detection_objective", "fractal_objective"]