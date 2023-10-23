import numpy as np

def objective_function(x):
    return 4*(x[0] - 5)**2 + (x[1] - 6)**2

def simplex_minimize(initial_vertex, alpha=1.0, beta=0.5, gamma=2.0, tol=1e-6, max_iter=1000):
    n = len(initial_vertex)
    vertices = np.zeros((n + 1, n))
    vertices[0] = initial_vertex

    for i in range(n):
        vertex = initial_vertex.copy()
        vertex[i] += alpha
        vertices[i + 1] = vertex

    iteration = 0
    while iteration < max_iter:
        # Оцениваем функцию в каждой вершине
        values = np.array([objective_function(vertex) for vertex in vertices])
        sorted_indices = np.argsort(values)
        best_vertex = vertices[sorted_indices[0]]
        worst_vertex = vertices[sorted_indices[-1]]
        second_worst_vertex = vertices[sorted_indices[-2]]

        # Вычисляем центроид (исключая худшую вершину)
        centroid = np.mean(vertices[sorted_indices[:-1]], axis=0)

        # Отражаем наихудшую вершину через центроид
        reflected_vertex = centroid + alpha * (centroid - worst_vertex)

        if objective_function(reflected_vertex) < objective_function(second_worst_vertex):
            # Если отраженная вершина лучше, чем вторая худшая, -> расширение
            expanded_vertex = centroid + gamma * (reflected_vertex - centroid)
            if objective_function(expanded_vertex) < objective_function(second_worst_vertex):
                vertices[sorted_indices[-1]] = expanded_vertex
            else:
                vertices[sorted_indices[-1]] = reflected_vertex
        else:
            # Если отраженная вершина хуже второй худшей, -> сужение
            contracted_vertex = centroid + beta * (worst_vertex - centroid)
            if objective_function(contracted_vertex) < objective_function(worst_vertex):
                vertices[sorted_indices[-1]] = contracted_vertex
            else:
                # Сокращение симплекса в направлении наилучшей вершины
                for i in range(1, n + 1):
                    vertices[sorted_indices[i]] = best_vertex + 0.5 * (vertices[sorted_indices[i]] - best_vertex)

        # Проверка на сходимость
        max_diff = np.max(np.abs(vertices - vertices[sorted_indices[0]]))
        if max_diff < tol:
            break

        iteration += 1

    return best_vertex, objective_function(best_vertex)

# Начальное предположение о минимуме
initial_guess = np.array([0.0, 0.0])


min_vertex, min_value = simplex_minimize(initial_guess)

print("Minimum found at (x, y) =", min_vertex)
print("Minimum value =", min_value)