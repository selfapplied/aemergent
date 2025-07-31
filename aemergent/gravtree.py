"""
Geometric Text Convolution GravTree

A quantum quadtree that generates convolution kernels and attention masks
for pattern detection in geometric text space. Treats text as a geometric
line of symbols where patterns can be detected through convolution and
focused through attention masks.

ARCHITECTURE:
- Text as geometric space: symbols positioned along geometric line
- Convolution kernels: detect patterns in symbolic geometry
- Attention masks: focus on specific regions of symbolic space
- QKV attention: creates "physical world" for geometric reasoning

USAGE:
    qq = GravTree()
    kernels = qq.generate_kernels(text)
    masks = qq.generate_masks(text)
    patterns = qq.detect_patterns(text, kernels, masks)
"""

import numpy as np
import struct
import pickle
from collections import defaultdict
from typing import TypeVar, Generic, Protocol, Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

# Type variable for geometric patterns
T = TypeVar('T')

class QNode:
    """
    Quantum Quaternion Node for geometric text processing
    
    Attributes:
        quat (np.ndarray): 4D quaternion vector [w, x, y, z]
        children (list): Child QNode instances
    """
    __slots__ = ('quat', 'children')
    
    def __init__(self, w: float, x: float, y: float, z: float):
        self.quat = np.array([w, x, y, z], dtype=np.float64)
        self.children: list[QNode] = []
        
    def rotate(self, axis: str, angle: float) -> None:
        """Rotate node and children in geometric space"""
        v = angle/2
        q = np.array([np.cos(v), 0, 0, 0], dtype=np.float64)
        
        # Map axis to quaternion component
        axis_map = {'i': 1, 'j': 2, 'k': 3, 'x': 1, 'y': 2, 'z': 3}
        if axis in axis_map:
            q[axis_map[axis]] = np.sin(v)
        
        # Rotate current node using vectorized quaternion multiplication
        self.quat = quat_mult_vec(self.quat, q)
        
        # Normalize using numpy
        mag = np.linalg.norm(self.quat)
        if mag > 0:
            self.quat /= mag
        
        # Rotate children if entropy threshold passed
        if abs(self.quat[3]) > 0.7:  # z component
            for child in self.children:
                child.rotate(axis, angle * 0.5)
    
    def encode(self) -> bytes:
        """Encode node to bytes"""
        quat_data = struct.pack('4d', *self.quat)
        children_data = b''
        for child in self.children:
            child_bytes = child.encode()
            children_data += struct.pack('I', len(child_bytes)) + child_bytes
        return quat_data + struct.pack('I', len(children_data)) + children_data

def quat_mult_vec(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Vectorized quaternion multiplication"""
    w, x, y, z = p
    W, X, Y, Z = q
    return np.array([
        w*W - x*X - y*Y - z*Z,
        w*X + x*W + y*Z - z*Y,
        w*Y - x*Z + y*W + z*X,
        w*Z + x*Y - y*X + z*W
    ], dtype=np.float64)

class GravTree(Generic[T]):
    """
    Geometric Text Convolution GravTree
    
    Generates convolution kernels and attention masks for pattern detection
    in geometric text space. Treats text as a geometric line of symbols
    where patterns can be detected through convolution and focused through
    attention masks.
    """
    
    def __init__(self):
        # Root node for geometric processing
        self.root = QNode(0.854, -0.522, -1.020, -0.458)
        
        # Core quadrants for geometric operations
        self.root.children = [
            QNode(0.2, 0.8, -0.3, 0.4),   # Kernel generation
            QNode(0.0, 0.0, 1.0, 0.0),     # Mask generation
            QNode(0.1, 0.3, 0.5, 0.7),     # Pattern detection
            QNode(0.0, 0.0, 0.0, 0.0)      # Geometric reasoning
        ]
        
        # Geometric subdivisions
        self.root.children[0].children = [
            QNode(0.4, 0.3, 0.2, 0.1),    # Edge detection
            QNode(0.6, 0.2, 0.1, 0.1),     # Feature detection
            QNode(0.1, 0.8, 0.05, 0.05),   # Structure detection
            QNode(0.0, 0.0, 0.0, 0.0)      # Pattern composition
        ]
        
        self.op_attractor: defaultdict[str, int] = defaultdict(int)
        self.entropy: defaultdict[str, int] = defaultdict(int)
        self.patterns: Dict[str, T] = {}
    
    def text_to_geometry(self, text: str) -> np.ndarray:
        """Convert text to geometric space"""
        # Convert text to geometric coordinates
        geometry = np.array([ord(c) for c in text], dtype=np.float64)
        # Normalize to geometric space
        if len(geometry) > 0:
            geometry = (geometry - np.min(geometry)) / (np.max(geometry) - np.min(geometry))
        return geometry
    
    def generate_kernel(self, kernel_size: int = 3) -> np.ndarray:
        """Generate convolution kernel based on quantum state"""
        root_state = self.root.quat
        
        # Use quaternion components to determine kernel parameters
        # w: kernel intensity, x: kernel frequency, y: kernel phase, z: kernel width
        intensity = abs(root_state[0])
        frequency = abs(root_state[1])
        phase = root_state[2] * np.pi
        width = max(1, int(abs(root_state[3]) * kernel_size))
        
        # Generate geometric kernel
        kernel = np.zeros(kernel_size, dtype=np.float64)
        center = kernel_size // 2
        
        for i in range(kernel_size):
            distance = abs(i - center)
            # Create geometric kernel based on quantum state
            kernel[i] = intensity * np.exp(-((distance - frequency) ** 2) / (2 * width ** 2))
            kernel[i] *= np.cos(2 * np.pi * i / kernel_size + phase)
        
        # Normalize kernel
        if np.sum(kernel) > 0:
            kernel = kernel / np.sum(kernel)
        
        return kernel
    
    def generate_attention_mask(self, text_length: int) -> np.ndarray:
        """Generate a smooth attention mask (Gaussian window)."""
        root_state = self.root.quat

        # Parameters derived from quaternion
        focus_intensity = abs(root_state[0])
        focus_center = int(abs(root_state[1]) * (text_length - 1))
        focus_width = max(1, int(abs(root_state[2]) * text_length / 4))

        positions = np.arange(text_length)
        mask = focus_intensity * \
            np.exp(-((positions - focus_center) ** 2) / (2 * focus_width ** 2))
        return mask
    
    def convolve_text(self, text: str, kernel: np.ndarray) -> np.ndarray:
        """Apply convolution kernel to geometric text"""
        geometry = self.text_to_geometry(text)
        
        # Apply convolution in geometric space
        return np.convolve(geometry, kernel, mode="same")
    
    def detect_patterns(self, text: str) -> List[Tuple[int, float]]:
        """Detect patterns using convolution kernels and attention masks"""
        kernel = self.generate_kernel()
        mask = self.generate_attention_mask(len(text))
        convolved = self.convolve_text(text, kernel)
        focused = convolved * mask

        # Detect peaks above threshold
        threshold = np.mean(focused) + np.std(focused)
        peak_indices = np.where(focused > threshold)[0]
        return [(int(idx), float(focused[idx])) for idx in peak_indices]

    def generate_patterns(self, text: str) -> List[Tuple[str, float]]:
        """Return (character, score) pairs for detected pattern peaks."""
        peaks = self.detect_patterns(text)
        return [(text[idx], score) for idx, score in peaks]

    def gainloss(self, target_kernel: np.ndarray, *, steps, step_size: float = 0.15, T: float = 1.0) -> List[float]:
        """Simulated-annealing optimiser that tweaks the root quaternion so the
        generated kernel matches ``target_kernel``.

        Returns the loss history (MSE per step). The routine is deliberately
        tiny—no external dependencies, single quaternion rotation per step.
        """
        losses: list[float] = []
        klen = len(target_kernel)
        axes = ('i', 'j', 'k')

        def mse(a, b):
            return float(np.mean((a - b) ** 2))

        curr_kernel = self.generate_kernel(klen)
        curr_loss = mse(curr_kernel, target_kernel)
        losses.append(curr_loss)

        for _ in range(steps):
            axis = np.random.choice(axes)
            angle = np.random.normal(scale=step_size)

            # Backup quaternion, apply trial rotation
            prev_quat = self.root.quat.copy()
            self.root.rotate(axis, angle)
            trial_kernel = self.generate_kernel(klen)
            trial_loss = mse(trial_kernel, target_kernel)

            accept = trial_loss < curr_loss or np.random.rand(
            ) < np.exp(-(trial_loss - curr_loss) / T
                       if accept:
                       curr_loss=trial_loss
                       losses.append(curr_loss)
                       else:
                       # Revert rotation
                       self.root.quat=prev_quat

                       return losses

    def evolve(self, loss: float) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Evolve using geometric pattern detection"""
        if self.entropy:
            axis = max(self.entropy.items(), key=lambda x: x[1])[0]
        else:
            axis = 'j'
        
        self.root.rotate(axis, loss/np.pi)
        self.entropy[axis] += 1
        
        # Check for structural changes
        struct = self.root.children[0]
        if abs(struct.quat[1]) > 0.9:  # x component
            # Generate test kernel and mask
            test_text = "quantum geometric convolution"
            kernel = self.generate_kernel()
            mask = self.generate_attention_mask(len(test_text))
            return kernel, mask
        return None
    
    def encode(self) -> bytes:
        """Encode the entire quadtree to bytes"""
        root_data = self.root.encode()
        patterns_data = pickle.dumps(self.patterns)
        attractor_data = pickle.dumps(dict(self.op_attractor))
        entropy_data = pickle.dumps(dict(self.entropy))
        
        data = (
            struct.pack('I', len(root_data)) + root_data +
            struct.pack('I', len(patterns_data)) + patterns_data +
            struct.pack('I', len(attractor_data)) + attractor_data +
            struct.pack('I', len(entropy_data)) + entropy_data
        )
        
        return data
    
    def reset(self) -> None:
        """Reset the quadtree to initial state"""
        self.root = QNode(0.854, -0.522, -1.020, -0.458)
        self.root.children = [
            QNode(0.2, 0.8, -0.3, 0.4),   # Kernel generation
            QNode(0.0, 0.0, 1.0, 0.0),     # Mask generation
            QNode(0.1, 0.3, 0.5, 0.7),     # Pattern detection
            QNode(0.0, 0.0, 0.0, 0.0)      # Geometric reasoning
        ]
        self.root.children[0].children = [
            QNode(0.4, 0.3, 0.2, 0.1),     # Edge detection
            QNode(0.6, 0.2, 0.1, 0.1),     # Feature detection
            QNode(0.1, 0.8, 0.05, 0.05),   # Structure detection
            QNode(0.0, 0.0, 0.0, 0.0)   
        ]
        self.op_attractor.clear()
        self.entropy.clear()
        self.patterns.clear()


                if __name__ == "__main__":

                qq=GravTree()
                sample="quantum geometric convolution"
                print("Detected patterns:", qq.generate_patterns(sample))
                evolved=qq.evolve(0.1)
                if evolved is not None:
        kernel, mask=evolved
        print("Kernel:", kernel, "Mask:", mask)

                __all__=["GravTree"]

                qq=GravTree()
                sample="quantum geometric convolution"
                print("Detected patterns:", qq.generate_patterns(sample))
                evolved=qq.evolve(0.1)
                if evolved is not None:
        kernel, mask=evolved
        print("Kernel:", kernel, "Mask:", mask)

                qq=GravTree()
                (data.decode("utf-8"))

                evolved=qq.evolve(0.1)
                if evolved is not None:
        kernel, mask=evolved
        print("Kernel:", kernel, "Mask:", mask)




__all__ = ["GravTree"]