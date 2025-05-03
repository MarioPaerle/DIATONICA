import matplotlib.pyplot as plt


points_y = [65, 65, 65, 57, 65]
points_x = [1, 4, 8, 49, 65]


import numpy as np

def polyntepolate(x_points, y_points):
    x_points = np.array(x_points)
    y_points = np.array(y_points)
    n = len(x_points)

    def p(x):
        x = np.array(x, ndmin=1)  # Allow scalar or array input
        total = np.zeros_like(x, dtype=float)
        for i in range(n):
            xi, yi = x_points[i], y_points[i]
            li = np.ones_like(x, dtype=float)
            for j in range(n):
                if i != j:
                    li *= (x - x_points[j]) / (xi - x_points[j])
            total += yi * li
        return total if len(total) > 1 else total[0]

    return p

def melodic_interpolate(x_points, y_points, lenght, steps, mask=None):
    if mask is not None:
        assert len(mask) == lenght
    else:
        mask = [1 for _ in range(lenght)]
    approx = polyntepolate(x_points, y_points)
    X = np.linspace(0, lenght, steps)
    notes = [int(approx(x))*m for m, x in zip(mask, X)]
    return notes

if __name__ == '__main__':
    approx = polyntepolate(points_x, points_y)
    X = np.linspace(0, 64, 64)
    notes = [int(approx(x)) for x in X]
    print(notes)
    plt.plot(X, [approx(x) for x in X], 'o')
    plt.plot(points_x, points_y, 'o')
    plt.show()