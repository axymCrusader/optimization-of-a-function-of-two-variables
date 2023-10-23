import numpy as np

def objective_function(x):
    return 4*(x[0] - 5)**2 + (x[1] - 6)**2

def hooke_jeeves_minimize(initial_point, increment=0.5, epsilon=1e-6, max_iter=1000):
    x = np.array(initial_point)
    n = len(x)

    iteration = 0
    while iteration < max_iter:
        # Шаг 1: Исследование в позитивном направлении
        x_positive = x.copy()
        for i in range(n):
            x_positive[i] += increment
            if objective_function(x_positive) < objective_function(x):
                x_positive[i] += increment  # Двигайтемся дальше в положительном направлении

        # Шаг 2: Исследование в отрицательном направлении
        x_negative = x.copy()
        for i in range(n):
            x_negative[i] -= increment
            if objective_function(x_negative) < objective_function(x):
                x_negative[i] -= increment  # Идём дальше в отрицательном направлении

        # Шаг 3: Переход на лучшую точку или уменьшение приращения
        if objective_function(x_positive) < objective_function(x):
            x = x_positive
        elif objective_function(x_negative) < objective_function(x):
            x = x_negative
        else:
            increment /= 2

        # Проверка на сходимость
        if np.linalg.norm(increment) < epsilon:
            break

        iteration += 1

    return x, objective_function(x)

# Начальное предположение о минимуме
initial_guess = [0.0, 0.0]


min_point, min_value = hooke_jeeves_minimize(initial_guess)

print("Minimum found at (x, y) =", min_point)
print("Minimum value =", min_value)



