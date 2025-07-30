#!/usr/bin/env python3
"""
Tests for pascell module - Compact Vectorized CA Engine
"""

import pytest
import numpy as np
from aemergent.pascell import (
    CAEngine, CASetup, CARule, CAGame,
    pascal_player, clip, pascal911,
    diversity_objective, symmetry_objective, complexity_objective,
    fractal_objective, detection_objective
)


class TestCASetup:
    """Test CASetup configuration"""
    
    def test_by_rows(self):
        setup = CASetup.by_rows(10)
        assert setup.max_rows == 10
        assert setup.mod_base == 10
        assert setup.capacity == 20
        assert setup.complexity == 0.05
    
    def test_by_complexity(self):
        setup = CASetup.by_complexity(0.1)
        assert setup.max_rows == 20
        assert setup.complexity == 0.025  # 0.5 / 20


class TestCARule:
    """Test CARule functionality"""
    
    def test_rule_creation(self):
        rule = CARule(name="test", func=pascal_player)
        assert rule.name == "test"
        assert rule.active is True
    
    def test_rule_function(self):
        rule = CARule(name="test", func=pascal_player)
        left = np.array([1, 2, 3])
        right = np.array([4, 5, 6])
        result = rule.func(left, right)
        assert np.array_equal(result, np.array([5, 7, 9]))


class TestCAEngine:
    """Test CAEngine core functionality"""
    
    def setup_method(self):
        self.setup = CASetup.by_rows(5)
        self.engine = CAEngine(self.setup)
    
    def test_engine_initialization(self):
        assert len(self.engine.active_players) == 1
        assert self.engine.weights == [1.0]
    
    def test_add_player(self):
        self.engine.add_player("test", pascal_player)
        assert len(self.engine.players) == 2
        assert len(self.engine.weights) == 2
        assert np.allclose(self.engine.weights, [0.5, 0.5])
    
    def test_remove_player(self):
        self.engine.add_player("test", pascal_player)
        self.engine.remove_player("test")
        assert len(self.engine.players) == 1
        assert self.engine.weights == [1.0]
    
    def test_weight_normalization_fix(self):
        """Test that weights are properly normalized to prevent division by zero"""
        # Add multiple players to test normalization
        self.engine.add_player("player1", pascal_player)
        self.engine.add_player("player2", pascal911)
        
        # Test that weights sum to 1.0
        assert np.isclose(np.sum(self.engine.weights), 1.0)
        assert len(self.engine.weights) == 3
        
        # Test rule application with multiple players
        history = np.array([[1], [2], [3]])  # Column history
        neighborhood = np.column_stack([np.array([1, 2, 3]), np.array([4, 5, 6])])  # [left, right]
        result = self.engine.rule(history, neighborhood)
        
        # Should not raise division by zero error
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.int32
    
    def test_rule_application_single_player(self):
        history = np.array([[1], [2], [3]])  # Column history
        neighborhood = np.column_stack([np.array([1, 2, 3]), np.array([4, 5, 6])])  # [left, right]
        result = self.engine.rule(history, neighborhood)
        expected = pascal_player(history, neighborhood)
        assert np.array_equal(result, expected)
    
    def test_rule_application_multiple_players(self):
        self.engine.add_player("player1", pascal911)
        history = np.array([[9], [8], [7]])  # Column history
        neighborhood = np.column_stack([np.array([9, 8, 7]), np.array([1, 2, 3])])  # [left, right]
        result = self.engine.rule(history, neighborhood)
        
        # Should handle overflow correctly
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.int32
    
    def test_no_active_players(self):
        """Test behavior when no players are active"""
        self.engine.players[0].active = False
        history = np.array([[1], [2], [3]])  # Column history
        neighborhood = np.column_stack([np.array([1, 2, 3]), np.array([4, 5, 6])])  # [left, right]
        result = self.engine.rule(history, neighborhood)
        
        # Should use default rule
        expected = (neighborhood[:, 0] + neighborhood[:, 1]) % self.setup.mod_base
        assert np.array_equal(result, expected)
    
    def test_add_objective(self):
        self.engine.add_objective("diversity", diversity_objective)
        assert len(self.engine.objectives) == 1
        assert self.engine.objective("diversity") is not None
    
    def test_remove_objective(self):
        self.engine.add_objective("diversity", diversity_objective)
        self.engine.remove_objective("diversity")
        assert len(self.engine.objectives) == 0
    
    def test_evaluate_objectives(self):
        self.engine.add_objective("diversity", diversity_objective)
        grid = np.array([[1, 2, 3], [4, 5, 6]])
        signatures, performance = self.engine.evaluate_objectives(grid)
        
        assert isinstance(signatures, np.ndarray)
        assert isinstance(performance, float)
        assert 0.0 <= performance <= 1.0
    
    def test_generate_basic(self):
        """Test basic CA generation"""
        game = self.engine.generate()
        
        assert isinstance(game, CAGame)
        assert game.grid.shape[0] == self.setup.max_rows
        assert isinstance(game.elapsed, float)
        assert game.elapsed >= 0.0


class TestPlayerFunctions:
    """Test individual player functions"""
    
    def test_pascal_player(self):
        left = np.array([1, 2, 3])
        right = np.array([4, 5, 6])
        result = pascal_player(left, right)
        assert np.array_equal(result, np.array([5, 7, 9]))
    
    def test_pascal911(self):
        left = np.array([9, 8, 7])
        right = np.array([1, 2, 3])
        result = pascal911(left, right)
        # Should clip to base 10
        assert np.all(result < 10)
    
    def test_clip_function(self):
        clipped_pascal = clip(pascal_player, 5)
        left = np.array([3, 4, 5])
        right = np.array([2, 1, 0])
        result = clipped_pascal(left, right)
        assert np.all(result < 5)


class TestObjectiveFunctions:
    """Test objective functions"""
    
    def test_diversity_objective(self):
        grid = np.array([[1, 2, 3], [4, 5, 6], [1, 1, 1]])
        result = diversity_objective(grid)
        assert isinstance(result, np.ndarray)
        assert len(result) == 3
    
    def test_symmetry_objective(self):
        grid = np.array([[1, 2, 1], [3, 4, 3], [1, 1, 1]])
        result = symmetry_objective(grid)
        assert isinstance(result, np.ndarray)
        assert len(result) == 3
    
    def test_fractal_objective(self):
        grid = np.array([[1, 2, 3], [4, 5, 6], [1, 1, 1]])
        result = fractal_objective(grid)
        assert isinstance(result, np.ndarray)
        assert len(result) == 3
    
    def test_detection_objective(self):
        def is_even(x):
            return x % 2 == 0
        
        grid = np.array([[1, 2, 3], [4, 5, 6], [1, 1, 1]])
        result = detection_objective(is_even, grid)
        assert isinstance(result, np.ndarray)
        assert len(result) == 3


class TestCAGame:
    """Test CAGame functionality"""
    
    def test_game_creation(self):
        signatures = np.array([1, 2, 3])
        performance = np.array([0.5, 0.6, 0.7])
        grid = np.array([[1, 2], [3, 4]])
        game = CAGame(signatures, performance, grid, 1.5)
        
        assert game.signatures is signatures
        assert game.performance is performance
        assert game.grid is grid
        assert game.elapsed == 1.5
    
    def test_game_string_representation(self):
        signatures = np.array([1, 2, 3])
        performance = np.array([0.5, 0.6, 0.7])
        grid = np.array([[1, 2], [3, 4]])
        game = CAGame(signatures, performance, grid, 1.5)
        
        result = str(game)
        assert "n=2" in result
        assert "Generation time: 1.500s" in result
        assert "Final performance: 0.700" in result
    
    def test_game_slicing(self):
        signatures = np.array([1, 2, 3])
        performance = np.array([0.5, 0.6, 0.7])
        grid = np.array([[1, 2], [3, 4]])
        game = CAGame(signatures, performance, grid, 1.5)
        
        result = game[0:1]
        assert isinstance(result, str)
        assert "12" in result


    def test_demo_ca_evolution(self):
        """Test the demo function"""
        from aemergent.pascell import demo_ca_evolution
        
        game = demo_ca_evolution(5)
        assert isinstance(game, CAGame)
        assert game.grid.shape[0] == 5


if __name__ == "__main__":
    pytest.main([__file__]) 