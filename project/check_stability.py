A = np.array([[-2.0]])
B = np.array([[3.0]])

# Initialize stability checker
checker = StabilityChecker(A, B)

# Define reward parameter ranges
reward_gen = RewardParameterGenerator(
    w1_range=np.linspace(0.1, 2.0, 5), 
    lam_range=np.linspace(0.1, 2.0, 5), 
    lam2_range=np.linspace(0.1, 2.0, 5), 
    w2_range=np.linspace(0.01, 0.1, 3)
)

# Generate reward combinations
reward_combinations = reward_gen.generate_combinations()

# Check stability for each combination
stable_combinations = []
for w1, lam, lam2, w2 in reward_combinations:
    G = w1 + lam + lam2  # Approximation for proportional gain
    stable, eigenvalues = checker.is_stable(G)
    if stable:
        stable_combinations.append((w1, lam, lam2, w2))

print("Stable Reward Combinations:", stable_combinations)
