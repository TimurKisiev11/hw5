import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.text import Annotation

def lorenz_system(y, t, sigma, rho, beta):
    x, y, z = y
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]


# Задаем параметры системы Лоренца
sigma, rho, beta = 10, 28, 2.667

# Задаем начальные условия
y0 = (0., 1., 1.05)

# Задаем временные точки для решения системы
t = np.linspace(0, 50, 15000)

# Решаем систему Лоренца
solution = odeint(lorenz_system, y0, t, args=(sigma, rho, beta))

# Получаем решение системы
begin_from = 5000
x = solution[begin_from:, 0]
y = solution[begin_from:, 1]
z = solution[begin_from:, 2]

# Рисуем траекторию решения
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, z, lw=0.5)
ax.scatter(0., 1., 1.05, lw=1, color ="green")
ax.text(0., 1., 1.05, 'Начальная точка', size=10, zorder=1,
    color='g')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Траектория системы Лоренца')
plt.show()
