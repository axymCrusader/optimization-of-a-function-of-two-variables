import numpy as np

def objective_function(x):
    return 4*(x[0] - 5)**2 + (x[1] - 6)**2

def gradient(x, epsilon=1e-8):
    # Градиент вычесленный с помощью центральных разностей
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = x.copy()
        x_plus[i] += epsilon
        x_minus = x.copy()
        x_minus[i] -= epsilon
        grad[i] = (objective_function(x_plus) - objective_function(x_minus)) / (2 * epsilon)
    return grad

def fletcher_reeves_minimize(initial_point, tol=1e-6, max_iter=1000, eta=0.01):
    current_point = initial_point
    n = len(initial_point)
    current_gradient = gradient(current_point)
    current_search_direction = -current_gradient
    current_value = objective_function(current_point)
    
    iteration = 0
    while iteration < max_iter:
        # Выполнить поиск в строке вдоль направления поиска
        alpha = 1.0  # Начальный размер шага
        beta = 0.5   # Коэффициент уменьшения размера шага
        
        while objective_function(current_point + alpha * current_search_direction) > current_value + eta * alpha * np.dot(current_gradient, current_search_direction):
            alpha *= beta
        
        # Обновление точки и градиента
        next_point = current_point + alpha * current_search_direction
        next_gradient = gradient(next_point)
        
        # Проверка на сходимость
        grad_norm = np.linalg.norm(next_gradient)
        if grad_norm < tol:
            break
        
        # Обновление направления поиска по формуле Флетчера-Ривза
        beta_fr = np.dot(next_gradient, next_gradient) / np.dot(current_gradient, current_gradient)
        next_search_direction = -next_gradient + beta_fr * current_search_direction
        
        current_point = next_point
        current_gradient = next_gradient
        current_search_direction = next_search_direction
        current_value = objective_function(current_point)
        iteration += 1
    
    return current_point, current_value

# Начальное предположение о минимуме
initial_guess = np.array([0.0, 0.0])


min_point, min_value = fletcher_reeves_minimize(initial_guess)

print("Minimum found at (x, y) =", min_point)
print("Minimum value =", min_value)
