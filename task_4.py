import sympy as sp
from scipy.integrate import quad
import numpy as np
from numpy import sqrt, pi
import matplotlib.pyplot as plt
from sympy import lambdify


def r(theta):
    return 1 + 3 / 4 * np.sin(3 * theta)


def dr(theta):
    return 9 / 4 * np.cos(3 * theta)


def f_1(theta):
    return r(theta) ** 2


def f_2(theta):
    arg = sp.symbols('x')
    r_func = (1 + 3 / 4 * sp.sin(3 * arg))
    dtdtheta = sp.diff(r_func, arg)
    dtdtheta_val = lambdify(arg, dtdtheta)
    return sqrt(r(theta) ** 2 + dtdtheta_val(theta) ** 2)


# График полярных координат
plt.axes(projection='polar')
theta = np.arange(0, (2 * np.pi), 0.01)
plt.polar(theta, r(theta), color='green')
plt.title('Rosenrot')
plt.show()
# График декартовых координат
x = r(theta) * np.cos(theta)
y = r(theta) * np.sin(theta)
plt.plot(x, y, color='green')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Rosenrot')
plt.show()

res = quad(f_1, 0, 2 * pi)
print("Площадь, ограниченная фигурой = " + str(res[0] / 2))
res = quad(f_2, 0, 2 * pi)
print("Длина кривой = " + str(res[0]))

xx = sp.symbols('x')
yy = (1 + 3 / 4 * sp.sin(3 * xx))
dyy = sp.diff(yy, xx)
print(dyy)
der = lambdify(xx, dyy)
print(str(dr(3)) + " " + str(der(3)))

