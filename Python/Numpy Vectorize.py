import numpy as np

a = np.array([1, 2, 3, 4, 5])

# Vectorization
def power(x):
    return x ** 4

v_power = np.vectorize(power)

print(v_power(a)) 

# Vectorization with excluded arguments
def power_exclude(x, powers):
    return x ** powers

v_power_exclude = np.vectorize(power_exclude, excluded=['powers'])
result = v_power_exclude(a, 5)
print(result)

# Vectorization with output types
def power_otype(x):
    return (x % 2)

v_power_otype = np.vectorize(power_otype, otypes=[bool])
print(v_power_otype(a))

