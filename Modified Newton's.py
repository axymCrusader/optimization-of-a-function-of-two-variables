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

def hessian(x, epsilon=1e-8):
    # Матрица Гессе вычесленная с помощью центральных разностей
    n = len(x)
    hess = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            x_plus_i = x.copy()
            x_plus_i[i] += epsilon
            x_plus_j = x.copy()
            x_plus_j[j] += epsilon
            x_minus_i = x.copy()
            x_minus_i[i] -= epsilon
            x_minus_j = x.copy()
            x_minus_j[j] -= epsilon
            
            hess[i, j] = (objective_function(x_plus_i) - objective_function(x_minus_i) -
                          objective_function(x_plus_j) + objective_function(x_minus_j)) / (4 * epsilon**2)
    
    return hess

def modified_newton_minimize(initial_point, tol=1e-6, max_iter=1000):
    current_point = initial_point
    
    iteration = 0
    while iteration < max_iter:
        grad = gradient(current_point)
        hess = hessian(current_point)
        
        # Проверка сходимости по норме градиента
        grad_norm = np.linalg.norm(grad)
        if grad_norm < tol:
            break
        
        # Решиаем линейную систему для получения направления поиска
        search_direction = np.linalg.solve(hess, -grad)
        
        # Выполняем поиск в строке вдоль направления поиска
        alpha = 1.0  # Начальный размер шага
        beta = 0.5   # Коэффициент уменьшения размера шага
        
        while objective_function(current_point + alpha * search_direction) > objective_function(current_point):
            alpha *= beta
        
        current_point = current_point + alpha * search_direction
        iteration += 1
    
    return current_point, objective_function(current_point)

# Начальное предположение о минимуме
initial_guess = np.array([0.0, 0.0])


min_point, min_value = modified_newton_minimize(initial_guess)

print("Minimum found at (x, y) =", min_point)
print("Minimum value =", min_value)
