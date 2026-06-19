import numpy as np

def power(x):
    return x ** 4

v_power = np.vectorize(power)

a = np.array([1, 2, 3, 4, 5])
result = v_power(a)
print(result) 

def power_exclude(x, powers):
    return x ** powers

v_power_exclude = np.vectorize(power_exclude, excluded=['powers'])
result = v_power_exclude(a, 5)
print(result)

def power_otype(x):
    return (x % 2) == 0

v_power_otype = np.vectorize(power_otype, otypes=[bool])
result = v_power_otype(a)
print(result)

