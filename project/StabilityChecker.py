from scipy.linalg import solve_continuous_lyapunov as lyap
import numpy as np

class StabilityChecker:
    def __init__(self, A, B):
        self.A = A
        self.B = B

    def check_stability(self, G):
        """
        Check closed-loop stability for a given gain G.
        """
        A_cl = self.A - self.B * G
        eigenvalues = np.linalg.eigvals(A_cl)
        stable = all(ev.real < 0 for ev in eigenvalues)
        return stable, eigenvalues

    def validate_lyapunov(self, G):
        """
        Use Lyapunov function to validate asymptotic stability.
        """
        A_cl = self.A - self.B * G
        Q = np.eye(A_cl.shape[0])  # Positive definite Q
        try:
            P = lyap(A_cl.T, -Q)  # Solve Lyapunov equation
            return np.all(np.linalg.eigvals(P) > 0)  # Check if P is positive definite
        except np.linalg.LinAlgError:
            return False

    def is_stable(self, G):
        """
        Check both eigenvalue stability and Lyapunov stability.
        """
        eigen_stable, eigenvalues = self.check_stability(G)
        lyapunov_stable = self.validate_lyapunov(G)
        return eigen_stable and lyapunov_stable, eigenvalues
