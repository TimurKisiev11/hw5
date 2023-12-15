from scipy.optimize import minimize

f = lambda x: x[3] * (x[0] - x[1] + x[2] ** 2) + x[2] * (x[0] - x[3])

# список словарей; один словарь - одно ограничение
cons = ({'type': 'ineq', 'fun': lambda x: x[0] * x[1] * x[2] * x[3] - 25},
        {'type': 'eq', 'fun': lambda x: x[0] ** 2 + x[1] ** 2 + x[2] ** 2 + x[3] ** 2 - 40})
bounds = ((1, 5), (1, 5), (1, 5), (1, 5))

result = minimize(f, (2, 4, 4, 2), bounds = bounds, constraints = cons)
print(result)
