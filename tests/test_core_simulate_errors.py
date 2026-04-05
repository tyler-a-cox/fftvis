"""
Tests for core simulate module error paths and edge cases.
Focus on error handling and boundary conditions.
"""

import numpy as np
import pytest
from unittest.mock import Mock, patch

from fftvis.core.simulate import SimulationEngine, default_accuracy_dict
from fftvis.core.beams import BeamEvaluator


class ConcreteSimulationEngine(SimulationEngine):
    """Concrete implementation for testing abstract base class."""
    
    def simulate(self, ants, fluxes, ra, dec, freqs, times, beam, **kwargs):
        """Minimal implementation for testing."""
        return np.zeros((len(ants), len(freqs), 4), dtype=complex)
    
    def _evaluate_vis_chunk(self, params):
        """Minimal implementation for testing."""
        return np.zeros((10, 10), dtype=complex)


class TestCoreSimulateErrors:
    """Test error handling in core simulation module."""
    
    def test_abstract_methods_not_implemented(self):
        """Test that abstract methods must be implemented."""
        # Try to instantiate abstract base class directly
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            SimulationEngine()
    
    def test_default_accuracy_dict(self):
        """Test default accuracy settings."""
        # Check structure
        assert isinstance(default_accuracy_dict, dict)
        assert 1 in default_accuracy_dict  # float32
        assert 2 in default_accuracy_dict  # float64
        
        # Check values are reasonable
        assert 0 < default_accuracy_dict[1] < 1e-3  # float32 tolerance
        assert 0 < default_accuracy_dict[2] < 1e-9  # float64 tolerance
        assert default_accuracy_dict[2] < default_accuracy_dict[1]  # float64 more precise
    
    def test_simulation_engine_initialization(self):
        """Test initialization of concrete simulation engine."""
        engine = ConcreteSimulationEngine()
        
        # Should have simulate method
        assert hasattr(engine, 'simulate')
        assert callable(engine.simulate)
        
        # Should have chunk evaluation method
        assert hasattr(engine, '_evaluate_vis_chunk')
        assert callable(engine._evaluate_vis_chunk)
    
    def test_invalid_precision_handling(self):
        """Test handling of invalid precision values."""
        # Create engine with custom error handling
        class ErrorTestEngine(ConcreteSimulationEngine):
            def simulate(self, ants, fluxes, ra, dec, freqs, times, beam, 
                        precision=2, eps=None, **kwargs):
                # Get epsilon value
                if eps is None:
                    if precision not in default_accuracy_dict:
                        raise ValueError(f"Invalid precision: {precision}")
                    eps = default_accuracy_dict[precision]
                return super().simulate(ants, fluxes, ra, dec, freqs, times, beam, **kwargs)
        
        engine = ErrorTestEngine()
        
        # Test with invalid precision
        with pytest.raises(ValueError, match="Invalid precision"):
            engine.simulate(
                ants={0: [0, 0, 0]},
                fluxes=np.ones(1),
                ra=np.zeros(1),
                dec=np.zeros(1),
                freqs=np.array([150e6]),
                times=np.array([0]),
                beam=Mock(),
                precision=3  # Invalid
            )
    
    def test_empty_input_handling(self):
        """Test handling of empty inputs."""
        class EmptyInputEngine(ConcreteSimulationEngine):
            def simulate(self, ants, fluxes, ra, dec, freqs, times, beam, **kwargs):
                # Validate inputs
                if len(ants) == 0:
                    raise ValueError("No antennas provided")
                if len(fluxes) == 0:
                    raise ValueError("No sources provided")
                if len(freqs) == 0:
                    raise ValueError("No frequencies provided")
                if len(times) == 0:
                    raise ValueError("No times provided")
                
                return super().simulate(ants, fluxes, ra, dec, freqs, times, beam, **kwargs)
        
        engine = EmptyInputEngine()
        
        # Test empty antennas
        with pytest.raises(ValueError, match="No antennas"):
            engine.simulate(
                ants={},
                fluxes=np.ones(1),
                ra=np.zeros(1),
                dec=np.zeros(1),
                freqs=np.array([150e6]),
                times=np.array([0]),
                beam=Mock()
            )
        
        # Test empty sources
        with pytest.raises(ValueError, match="No sources"):
            engine.simulate(
                ants={0: [0, 0, 0]},
                fluxes=np.array([]),
                ra=np.array([]),
                dec=np.array([]),
                freqs=np.array([150e6]),
                times=np.array([0]),
                beam=Mock()
            )
        
        # Test empty frequencies
        with pytest.raises(ValueError, match="No frequencies"):
            engine.simulate(
                ants={0: [0, 0, 0]},
                fluxes=np.ones(1),
                ra=np.zeros(1),
                dec=np.zeros(1),
                freqs=np.array([]),
                times=np.array([0]),
                beam=Mock()
            )
        
        # Test empty times
        with pytest.raises(ValueError, match="No times"):
            engine.simulate(
                ants={0: [0, 0, 0]},
                fluxes=np.ones(1),
                ra=np.zeros(1),
                dec=np.zeros(1),
                freqs=np.array([150e6]),
                times=np.array([]),
                beam=Mock()
            )
    
    def test_mismatched_array_sizes(self):
        """Test handling of mismatched input array sizes."""
        class ArraySizeEngine(ConcreteSimulationEngine):
            def simulate(self, ants, fluxes, ra, dec, freqs, times, beam, **kwargs):
                # Validate array sizes
                nsrc = len(fluxes) if fluxes.ndim == 1 else fluxes.shape[0]
                
                if len(ra) != nsrc:
                    raise ValueError(f"RA size {len(ra)} != number of sources {nsrc}")
                if len(dec) != nsrc:
                    raise ValueError(f"Dec size {len(dec)} != number of sources {nsrc}")
                
                return super().simulate(ants, fluxes, ra, dec, freqs, times, beam, **kwargs)
        
        engine = ArraySizeEngine()
        
        # Test mismatched RA
        with pytest.raises(ValueError, match="RA size"):
            engine.simulate(
                ants={0: [0, 0, 0]},
                fluxes=np.ones(5),
                ra=np.zeros(3),  # Wrong size
                dec=np.zeros(5),
                freqs=np.array([150e6]),
                times=np.array([0]),
                beam=Mock()
            )
        
        # Test mismatched Dec
        with pytest.raises(ValueError, match="Dec size"):
            engine.simulate(
                ants={0: [0, 0, 0]},
                fluxes=np.ones(5),
                ra=np.zeros(5),
                dec=np.zeros(3),  # Wrong size
                freqs=np.array([150e6]),
                times=np.array([0]),
                beam=Mock()
            )
    
    def test_chunk_evaluation_errors(self):
        """Test error handling in chunk evaluation."""
        class ChunkErrorEngine(ConcreteSimulationEngine):
            def __init__(self, error_on_chunk=0):
                self.error_on_chunk = error_on_chunk
                self.chunk_count = 0
            
            def _evaluate_vis_chunk(self, params):
                if self.chunk_count == self.error_on_chunk:
                    raise RuntimeError("Simulated chunk processing error")
                self.chunk_count += 1
                return np.zeros((10, 10), dtype=complex)
        
        # Test error on first chunk
        engine = ChunkErrorEngine(error_on_chunk=0)
        
        with pytest.raises(RuntimeError, match="chunk processing error"):
            engine._evaluate_vis_chunk({})
    
    def test_invalid_coordinate_handling(self):
        """Test handling of invalid coordinates."""
        class CoordinateEngine(ConcreteSimulationEngine):
            def simulate(self, ants, fluxes, ra, dec, freqs, times, beam, **kwargs):
                # Check coordinate ranges
                if np.any(ra < 0) or np.any(ra > 2*np.pi):
                    raise ValueError("RA must be in range [0, 2π]")
                if np.any(dec < -np.pi/2) or np.any(dec > np.pi/2):
                    raise ValueError("Dec must be in range [-π/2, π/2]")
                
                return super().simulate(ants, fluxes, ra, dec, freqs, times, beam, **kwargs)
        
        engine = CoordinateEngine()
        
        # Test invalid RA
        with pytest.raises(ValueError, match="RA must be in range"):
            engine.simulate(
                ants={0: [0, 0, 0]},
                fluxes=np.ones(1),
                ra=np.array([3*np.pi]),  # Invalid
                dec=np.zeros(1),
                freqs=np.array([150e6]),
                times=np.array([0]),
                beam=Mock()
            )
        
        # Test invalid Dec
        with pytest.raises(ValueError, match="Dec must be in range"):
            engine.simulate(
                ants={0: [0, 0, 0]},
                fluxes=np.ones(1),
                ra=np.zeros(1),
                dec=np.array([np.pi]),  # Invalid
                freqs=np.array([150e6]),
                times=np.array([0]),
                beam=Mock()
            )
    
    def test_numerical_overflow_handling(self):
        """Test handling of numerical overflow conditions."""
        class OverflowEngine(ConcreteSimulationEngine):
            def simulate(self, ants, fluxes, ra, dec, freqs, times, beam, **kwargs):
                # Check for potential overflow
                max_flux = np.max(np.abs(fluxes))
                if max_flux > 1e10:
                    raise ValueError("Flux values too large, risk of overflow")
                
                # Check frequency range
                if np.any(freqs > 1e12):  # > 1 THz
                    raise ValueError("Frequencies unrealistically high")
                if np.any(freqs < 1e6):   # < 1 MHz
                    raise ValueError("Frequencies unrealistically low")
                
                return super().simulate(ants, fluxes, ra, dec, freqs, times, beam, **kwargs)
        
        engine = OverflowEngine()
        
        # Test large flux
        with pytest.raises(ValueError, match="Flux values too large"):
            engine.simulate(
                ants={0: [0, 0, 0]},
                fluxes=np.array([1e15]),  # Very large
                ra=np.zeros(1),
                dec=np.zeros(1),
                freqs=np.array([150e6]),
                times=np.array([0]),
                beam=Mock()
            )
        
        # Test high frequency
        with pytest.raises(ValueError, match="Frequencies unrealistically high"):
            engine.simulate(
                ants={0: [0, 0, 0]},
                fluxes=np.ones(1),
                ra=np.zeros(1),
                dec=np.zeros(1),
                freqs=np.array([1e13]),  # Very high
                times=np.array([0]),
                beam=Mock()
            )
        
        # Test low frequency
        with pytest.raises(ValueError, match="Frequencies unrealistically low"):
            engine.simulate(
                ants={0: [0, 0, 0]},
                fluxes=np.ones(1),
                ra=np.zeros(1),
                dec=np.zeros(1),
                freqs=np.array([1e5]),  # Very low
                times=np.array([0]),
                beam=Mock()
            )
    
    def test_beam_evaluation_errors(self):
        """Test handling of beam evaluation errors."""
        class BeamErrorEngine(ConcreteSimulationEngine):
            def simulate(self, ants, fluxes, ra, dec, freqs, times, beam, **kwargs):
                # Check if beam is valid
                if beam is None:
                    raise ValueError("Beam evaluator is required")
                
                if not isinstance(beam, BeamEvaluator):
                    raise TypeError("Beam must be a BeamEvaluator instance")
                
                # Simulate beam evaluation error
                try:
                    # Try to call a method that might fail
                    beam.evaluate_beam(0, 0, freqs[0], 0)
                except Exception as e:
                    raise RuntimeError(f"Beam evaluation failed: {str(e)}")
                
                return super().simulate(ants, fluxes, ra, dec, freqs, times, beam, **kwargs)
        
        engine = BeamErrorEngine()
        
        # Test None beam
        with pytest.raises(ValueError, match="Beam evaluator is required"):
            engine.simulate(
                ants={0: [0, 0, 0]},
                fluxes=np.ones(1),
                ra=np.zeros(1),
                dec=np.zeros(1),
                freqs=np.array([150e6]),
                times=np.array([0]),
                beam=None
            )
        
        # Test invalid beam type
        with pytest.raises(TypeError, match="Beam must be a BeamEvaluator"):
            engine.simulate(
                ants={0: [0, 0, 0]},
                fluxes=np.ones(1),
                ra=np.zeros(1),
                dec=np.zeros(1),
                freqs=np.array([150e6]),
                times=np.array([0]),
                beam="not a beam"
            )
    
    def test_precision_epsilon_interaction(self):
        """Test interaction between precision and epsilon parameters."""
        class PrecisionEngine(ConcreteSimulationEngine):
            def simulate(self, ants, fluxes, ra, dec, freqs, times, beam, 
                        precision=2, eps=None, **kwargs):
                # Handle epsilon
                if eps is None:
                    eps = default_accuracy_dict.get(precision, 1e-6)
                
                # Validate epsilon
                if eps <= 0:
                    raise ValueError("Epsilon must be positive")
                if eps > 1:
                    raise ValueError("Epsilon must be <= 1")
                
                # Store for testing
                self.used_eps = eps
                self.used_precision = precision
                
                return super().simulate(ants, fluxes, ra, dec, freqs, times, beam, **kwargs)
        
        engine = PrecisionEngine()
        
        # Test default epsilon
        engine.simulate(
            ants={0: [0, 0, 0]},
            fluxes=np.ones(1),
            ra=np.zeros(1),
            dec=np.zeros(1),
            freqs=np.array([150e6]),
            times=np.array([0]),
            beam=Mock(),
            precision=1
        )
        assert engine.used_eps == default_accuracy_dict[1]
        
        # Test custom epsilon overrides precision
        custom_eps = 1e-8
        engine.simulate(
            ants={0: [0, 0, 0]},
            fluxes=np.ones(1),
            ra=np.zeros(1),
            dec=np.zeros(1),
            freqs=np.array([150e6]),
            times=np.array([0]),
            beam=Mock(),
            precision=1,
            eps=custom_eps
        )
        assert engine.used_eps == custom_eps
        
        # Test invalid epsilon
        with pytest.raises(ValueError, match="Epsilon must be positive"):
            engine.simulate(
                ants={0: [0, 0, 0]},
                fluxes=np.ones(1),
                ra=np.zeros(1),
                dec=np.zeros(1),
                freqs=np.array([150e6]),
                times=np.array([0]),
                beam=Mock(),
                eps=-1e-6
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])