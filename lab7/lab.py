import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

class Lab3Elliptic:
    Lx = 1.0
    Ly = np.pi / 2.0
    
    @staticmethod
    def exact(x, y):
        # Аналитическое решение U(x, y) = exp(x) * cos(y)
        return np.exp(x) * np.cos(y)

def solve_elliptic(Nx, Ny, method='liebmann', omega=1.5, tol=1e-5, max_iter=20000):
    hx = Lab3Elliptic.Lx / Nx
    hy = Lab3Elliptic.Ly / Ny
    
    x = np.linspace(0, Lab3Elliptic.Lx, Nx + 1)
    y = np.linspace(0, Lab3Elliptic.Ly, Ny + 1)
    u = np.zeros((Nx + 1, Ny + 1))
    
    # 1. Граничные условия 1-го рода (Дирихле)
    u[0, :] = np.cos(y)
    u[Nx, :] = np.exp(1.0) * np.cos(y)
    
    # Линейная интерполяция начального приближения для ускорения сходимости
    for i in range(1, Nx):
        u[i, :] = u[0, :] + (u[Nx, :] - u[0, :]) * (x[i] / Lab3Elliptic.Lx)
        
    hx2, hy2 = hx**2, hy**2
    denom = 2 * (hx2 + hy2)
    
    iters = 0
    err = 1.0
    u_new = np.copy(u)
    
    # Список для сохранения истории ошибок
    history_err = []
    exact_grid = np.array([[Lab3Elliptic.exact(xi, yj) for yj in y] for xi in x])
    
    while err > tol and iters < max_iter:
        err = 0.0
        
        for i in range(1, Nx):
            for j in range(1, Ny):
                if method == 'liebmann':
                    val = (hy2 * (u[i+1, j] + u[i-1, j]) + hx2 * (u[i, j+1] + u[i, j-1])) / denom
                elif method == 'seidel':
                    val = (hy2 * (u[i+1, j] + u_new[i-1, j]) + hx2 * (u[i, j+1] + u_new[i, j-1])) / denom
                elif method == 'sor':
                    val_seidel = (hy2 * (u[i+1, j] + u_new[i-1, j]) + hx2 * (u[i, j+1] + u_new[i, j-1])) / denom
                    val = (1 - omega) * u[i, j] + omega * val_seidel
                
                err = max(err, abs(val - u[i, j]))
                u_new[i, j] = val
                
        # 3. Аппроксимация граничных условий 2-го рода (Неймана)
        for i in range(1, Nx):
            # u_y(x, 0) = 0
            val_0 = u_new[i, 1]
            err = max(err, abs(val_0 - u_new[i, 0]))
            u_new[i, 0] = val_0
            
            # u_y(x, pi/2) = -exp(x)
            val_Ny = u_new[i, Ny-1] - hy * np.exp(x[i])
            err = max(err, abs(val_Ny - u_new[i, Ny]))
            u_new[i, Ny] = val_Ny
            
        u = np.copy(u_new)
        
        # Сохраняем реальную ошибку относительно точного решения для графика
        current_max_err = np.max(np.abs(u - exact_grid))
        history_err.append(current_max_err)
        
        iters += 1
        
    return x, y, u, iters, history_err

if __name__ == '__main__':
    Nx, Ny = 30, 30
    methods = [
        ('liebmann', 'Метод Либмана'),
        ('seidel', 'Метод Зейделя'),
        ('sor', 'SOR (omega=1.5)')
    ]
    
    results = {}
    
    # Сбор данных
    for code, name in methods:
        x, y, u_num, iters, hist = solve_elliptic(Nx, Ny, method=code, omega=1.5)
        results[code] = {'u': u_num, 'iters': iters, 'hist': hist, 'name': name}

    # --- Вывод таблицы результатов в консоль ---
    print(f"Сетка: Nx={Nx}, Ny={Ny}")
    print("-" * 65)
    print(f"{'Метод':<30} | {'Итерации':<10} | {'Макс. погрешность'}")
    print("-" * 65)
    for code, name in methods:
        iters = results[code]['iters']
        max_err = results[code]['hist'][-1]  # Берем последнюю ошибку из истории
        print(f"{name:<30} | {iters:<10} | {max_err:.6e}")
    print("-" * 65)

    # Данные для графиков
    exact_grid = np.array([[Lab3Elliptic.exact(xi, yj) for yj in y] for xi in x])
    error_grid = np.abs(results['seidel']['u'] - exact_grid) # Ошибка для метода Зейделя
    X, Y = np.meshgrid(y, x)

    # --- График 1: 3D Поверхности (Аналитическое vs Численное) ---
    fig1 = plt.figure(figsize=(14, 6))
    ax1 = fig1.add_subplot(121, projection='3d')
    ax1.plot_surface(X, Y, exact_grid, cmap=cm.viridis)
    ax1.set_title('Аналитическое решение U(x,y)')
    ax1.set_xlabel('y'); ax1.set_ylabel('x'); ax1.set_zlabel('U')
    
    ax2 = fig1.add_subplot(122, projection='3d')
    ax2.plot_surface(X, Y, results['seidel']['u'], cmap=cm.plasma)
    ax2.set_title('Численное решение (Зейдель)')
    ax2.set_xlabel('y'); ax2.set_ylabel('x'); ax2.set_zlabel('u')
    plt.tight_layout()

    # --- График 2: Двумерные изолинии (Контуры) и Распределение ошибки ---
    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(14, 6))
    
    cp = ax3.contourf(X, Y, results['seidel']['u'], levels=20, cmap=cm.coolwarm)
    fig2.colorbar(cp, ax=ax3)
    ax3.set_title('Изолинии численного решения')
    ax3.set_xlabel('y'); ax3.set_ylabel('x')
    
    cp2 = ax4.contourf(X, Y, error_grid, levels=20, cmap=cm.Reds)
    fig2.colorbar(cp2, ax=ax4)
    ax4.set_title('Распределение абсолютной погрешности ||U - u||')
    ax4.set_xlabel('y'); ax4.set_ylabel('x')
    plt.tight_layout()

    # --- График 3: 3D Поверхность ошибки ---
    fig3 = plt.figure(figsize=(8, 6))
    ax5 = fig3.add_subplot(111, projection='3d')
    surf_err = ax5.plot_surface(X, Y, error_grid, cmap=cm.magma)
    fig3.colorbar(surf_err, shrink=0.5, aspect=5)
    ax5.set_title('3D профиль абсолютной погрешности')
    ax5.set_xlabel('y'); ax5.set_ylabel('x'); ax5.set_zlabel('Ошибка')
    plt.tight_layout()

    # --- График 4: История сходимости итерационных методов ---
    fig4, ax6 = plt.subplots(figsize=(8, 6))
    for code in ['liebmann', 'seidel', 'sor']:
        data = results[code]
        ax6.plot(range(1, data['iters'] + 1), data['hist'], label=data['name'], linewidth=2)
        
    ax6.set_yscale('log')
    ax6.set_title('Скорость сходимости итерационных методов')
    ax6.set_xlabel('Номер итерации')
    ax6.set_ylabel('Максимальная ошибка (log scale)')
    ax6.grid(True, which="both", ls="--", alpha=0.7)
    ax6.legend()
    plt.tight_layout()

    plt.show()