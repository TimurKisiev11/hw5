import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from findiff import FinDiff
from scipy.misc import derivative
from sympy.solvers import solve
from sympy import Symbol
from matplotlib.pyplot import axhline, text, annotate


def y(x):
    return np.exp(-x / 10) * np.sin(x)


def dy(x):
    return np.exp(-x / 10) * np.sin(x) * (-1 / 10) + np.exp(-x / 10) * np.cos(x)


# Находим производную в виде формулы и корни y'(x)=0
xx = Symbol('x')
dyy = sp.diff(sp.exp(-xx / 10) * sp.sin(xx), xx)
root = solve(dyy, Symbol('x'))
print("y'(x) = " + str(dyy))
number_of_roots = 3
root = [root[0]+np.pi*k for k in range(0, number_of_roots)]
print(root)


# Находим значения производной на иксах
# x = np.linspace(0, 10, 10000)
# derivative_values = [derivative(y, i, 0.0001) for i in x]

x = np.linspace(0, 10, 10000)
dx = x[1] - x[0]
df = FinDiff(0, dx)
yy = np.exp(-x / 10) * np.sin(x)
derivative_values = df(yy)

# График зависимости y от x
plt.plot(x, y(x), label='y(x)')

# График зависимости y от x, при x \in [4,7]
x_segment = np.where((7 >= x) & (x >= 4), x, np.nan)
plt.plot(x_segment, y(x_segment), lw=2, color="red", linestyle='--')
y_segment = y(x[4000:7000])
# plt.plot(x[4000:7000], y_segment, lw = 2, color = "red", linestyle='--')

# График производной y'(x)
plt.plot(x, derivative_values, color="purple", label="$y'(x)$")
# Корни y'(x) = 0
plt.scatter(root, np.zeros(number_of_roots), lw=1, color="green")
for root in root:
    annotate('', xy=(root, y(float(root))),
             xytext=(root, 0),
             arrowprops=dict(arrowstyle='->', color='red'))

print("Среднеквадратичное отклонение: " + str(np.std(y_segment)))
print("Среднее: " + str(np.mean(y_segment)))

# y_m такой, что 70% значений y меньше
y_m = sorted(y_segment)[(len(y_segment) * 7) // 10]
plt.axhline(y_m, color="green", linestyle='--', label="y = $y_m$")
print("y_m: " + str(y_m))
x_m = x[4000 + (len(y_segment) * 7) // 10]
annotate("$y_m$", xy=(x_m, y_m),
         xytext=(x_m, y_m + 0.5),
         arrowprops=dict(arrowstyle='->', color='red'))
plt.legend()
plt.show()
