import numpy as np

arr = np.arange(10001)
# arr = np.linspace(0, 10000, 10001)
sum = np.sum((arr % 3 != 0) & (arr % 8 != 0))
print(sum)