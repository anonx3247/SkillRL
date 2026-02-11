"""Convergence detection for early stopping in evolution loop."""


class ConvergenceDetector:
    """Detects when evolution has converged via patience-based early stopping.

    Tracks best success rate and stops when no improvement for N consecutive iterations.
    """

    def __init__(self, patience: int = 5, min_delta: float = 0.01):
        """Initialize convergence detector.

        Args:
            patience: Number of iterations without improvement before convergence
            min_delta: Minimum improvement to count as progress
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_value = 0.0
        self.no_improvement_count = 0

    def check(self, current_value: float) -> bool:
        """Check if evolution has converged.

        Args:
            current_value: Current success rate

        Returns:
            True if converged (should stop), False if should continue
        """
        # Check if we improved by at least min_delta
        if current_value > self.best_value + self.min_delta:
            # Progress made - update best and reset counter
            self.best_value = current_value
            self.no_improvement_count = 0
            return False
        else:
            # No improvement - increment counter
            self.no_improvement_count += 1
            return self.no_improvement_count >= self.patience

    def reset(self):
        """Reset detector state."""
        self.best_value = 0.0
        self.no_improvement_count = 0
