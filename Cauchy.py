import numpy as np

def objective_function(x):
    return 4*(x[0] - 5)**2 + (x[1] - 6)**2

def grad_objective_function(x, delta=1e-8):
    # Численный градиент функции
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_plus_delta = x.copy()
        x_minus_delta = x.copy()
        x_plus_delta[i] += delta
        x_minus_delta[i] -= delta
        grad[i] = (objective_function(x_plus_delta) - objective_function(x_minus_delta)) / (2 * delta)
    return grad

def cauchy_minimize(initial_point, delta=0.1, tol=1e-6, max_iter=1000):
    current_point = initial_point
    n = len(initial_point)
    
    iteration = 0
    while iteration < max_iter:
        gradient = grad_objective_function(current_point)
        norm_gradient = np.linalg.norm(gradient)
        
        # Проверка сходимости по норме градиента
        if norm_gradient < tol:
            break
        
        # Вычислить точку Коши
        cauchy_point = current_point - (delta * gradient) / norm_gradient
        
        # Поиск минимума на отрезке от текущей_точки до точки Коши
        t = 1.0
        while objective_function(cauchy_point) > objective_function(current_point - t * delta * gradient):
            t *= 0.5
            cauchy_point = current_point - t * delta * gradient
        
        # Обновление текущей точки
        current_point = cauchy_point
        
        iteration += 1
    
    return current_point, objective_function(current_point)

# Начальное предположение о минимуме
initial_guess = np.array([0.0, 0.0])


min_point, min_value = cauchy_minimize(initial_guess)

print("Minimum found at (x, y) =", min_point)
print("Minimum value =", min_value)