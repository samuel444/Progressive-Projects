import numpy as np

# Possible states
states = ["Sunny", "Rainy"]

# Transition matrix
# Rows = current state
# Columns = next state
#
# If today is Sunny:
# 80% chance tomorrow is Sunny
# 20% chance tomorrow is Rainy
#
# If today is Rainy:
# 30% chance tomorrow is Sunny
# 70% chance tomorrow is Rainy
transition_matrix = np.array([
    [0.8, 0.2],
    [0.3, 0.7]
])

print("Transition Matrix:")
print(transition_matrix)


# Simulate a Markov chain

# 0 = Sunny
# 1 = Rainy
current_state = 0

weather = [states[current_state]]

# Simulate the next 20 days
for _ in range(20):

    # Select the next state using the probabilities
    # from the current state's row
    current_state = np.random.choice(
        [0, 1],
        p=transition_matrix[current_state]
    )

    weather.append(states[current_state])

print("\nSimulated Weather:")
print(weather)


# Work with probabilities instead of one simulation

# Today is definitely Sunny
# 100% Sunny, 0% Rainy
state_probabilities = np.array([1.0, 0.0])

# Multiply by the transition matrix
# to get tomorrow's probabilities
tomorrow = state_probabilities @ transition_matrix

print("\nTomorrow:")
print(
    f"Sunny = {tomorrow[0]:.3f}, "
    f"Rainy = {tomorrow[1]:.3f}"
)

# Calculate probabilities two days ahead
two_days = tomorrow @ transition_matrix

print("\nTwo Days Ahead:")
print(
    f"Sunny = {two_days[0]:.3f}, "
    f"Rainy = {two_days[1]:.3f}"
)


# See how the probabilities change over time
state_probabilities = np.array([1.0, 0.0])

print("\nProbabilities Over Time:")

for day in range(1, 11):

    state_probabilities = (
        state_probabilities @ transition_matrix
    )

    print(
        f"Day {day}: "
        f"Sunny = {state_probabilities[0]:.3f}, "
        f"Rainy = {state_probabilities[1]:.3f}"
    )

# Long-run stationary distribution
state_probabilities = np.array([1.0, 0.0])

for _ in range(100):

    state_probabilities = (
        state_probabilities @ transition_matrix
    )

print("\nLong-Run Probabilities:")
print(
    f"Sunny = {state_probabilities[0]:.3f}, "
    f"Rainy = {state_probabilities[1]:.3f}"
)

