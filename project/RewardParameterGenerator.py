class RewardParameterGenerator:
    def __init__(self, w1_range, lam_range, lam2_range, w2_range):
        self.w1_range = w1_range
        self.lam_range = lam_range
        self.lam2_range = lam2_range
        self.w2_range = w2_range

    def generate_combinations(self):
        """
        Generate all possible combinations of reward parameters.
        """
        from itertools import product
        return list(product(self.w1_range, self.lam_range, self.lam2_range, self.w2_range))