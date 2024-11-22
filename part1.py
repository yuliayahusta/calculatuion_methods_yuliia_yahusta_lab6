import sympy as sp
import matplotlib.pyplot as plt
import numpy as np


# Функція для вибору диференціальної рівняння
def select_function():
    print("\nОберіть функцію для обчислення:")
    print("1. y' = x + sin(y / sqrt(2))")
    print("2. y' = x + cos(y / sqrt(3))")
    print("3. Введіть власну функцію")

    choice = int(input("Ваш вибір (1/2/3): "))

    if choice == 1:
        # Введення даних для першої функції
        a = float(input("\nВведіть проміжок [a; b]:\na (за умовою завдання 0.8) = "))
        b = float(input("b (за умовою завдання 1.8) = "))
        h = float(input("Крок h (0.1) = "))
        y0 = float(input(f"Початкова умова: y({a}) (за умовою завдання 1.3)= "))
        return lambda x, y: x + np.sin(y / np.sqrt(2)), a, b, h, y0
    elif choice == 2:
        # Введення даних для другої функції
        a = float(input("\nВведіть проміжок [a; b]:\na (за умовою завдання 1.2) = "))
        b = float(input("b (за умовою завдання 2.2) = "))
        h = float(input("Крок h (0.1) = "))
        y0 = float(input(f"Початкова умова: y({a}) (за умовою завдання 2.1) = "))
        return lambda x, y: x + np.cos(y / np.sqrt(3)), a, b, h, y0
    elif choice == 3:
        # Введення даних для власної функції
        user_function_str = input("\nВведіть функцію (наприклад x + sin(y / sqrt(2)):\n y'= ")
        a = float(input("Введіть проміжок [a; b]:\na = "))
        b = float(input("b = "))
        h = float(input("Крок h = "))
        y0 = float(input(f"Початкова умова: y({a}) = "))

        # Використовуємо sympy для розбору функції
        x, y = sp.symbols('x y')
        user_function = sp.sympify(user_function_str)

        # Перетворюємо функцію в Python-вираз
        def func(x_val, y_val):
            return float(user_function.subs({x: x_val, y: y_val}))

        return func, a, b, h, y0
    else:
        print("Неправильний вибір!")
        return None, None, None, None, None


# Метод Ейлера
def euler_method(f, a, b, h, y0):
    x = [a]
    y = [y0]
    n = int((b - a) / h)
    for i in range(n):
        x.append(x[i] + h)
        y.append(y[i] + h * f(x[i], y[i]))
    return x, y


# Метод Ейлера-Коші
def euler_cauchy_method(f, a, b, h, y0):
    x = [a]
    y = [y0]
    n = int((b - a) / h)
    for i in range(n):
        x_i = x[i]
        y_i = y[i]

        # Крок передбачення
        y_pred = y_i + h * f(x_i, y_i)

        # Обчислення x[i+1]
        x_next = x_i + h

        # Крок корекції
        y_corr = y_i + (h / 2) * (f(x_i, y_i) + f(x_next, y_pred))

        # Додавання нових значень
        x.append(x_next)
        y.append(y_corr)
    return x, y


# Метод Рунге-Кутти 4-го порядку
def runge_kutta_method(f, a, b, h, y0):
    """
    Найбільш точний метод в розрахунку ЗДР. В коді використовується для показу наближеного графіку функціїю
    Метод розраховує нове значення y[i+1] , усереднюючи кілька оцінок похідної f(x,y):
    k1 - миттєва швидкість (градієнт) на початку інтервалу
    k2 - швидкість на середині інтервалу, використовуючи k1 для уточнення
    k3 - ще одна оцінка на середині інтервалу, уточнена за k2
    k4 - швидкість на кінці інтервалу, уточнена за k3
    """
    x = [a]
    y = [y0]
    n = int((b - a) / h)
    for i in range(n):
        x_i = x[i]
        y_i = y[i]

        k1 = h * f(x_i, y_i)
        k2 = h * f(x_i + h / 2, y_i + k1 / 2)
        k3 = h * f(x_i + h / 2, y_i + k2 / 2)
        k4 = h * f(x_i + h, y_i + k3)

        x.append(x_i + h)
        y.append(y_i + (k1 + 2 * k2 + 2 * k3 + k4) / 6)
    return x, y


# Функція для виведення таблиці
def print_table(x, y, method_name):
    print(f"\nТаблиця значень ({method_name}):")
    for i in range(len(x)):
        print(f"x = {x[i]:.2f}, y = {y[i]:.4f}")


# Функція для побудови графіка
def plot_results(x_euler, y_euler, x_euler_cauchy, y_euler_cauchy, x_rk, y_rk):
    plt.figure(figsize=(10, 6))
    plt.plot(x_euler, y_euler, label="Метод Ейлера", color='fuchsia')
    plt.plot(x_euler_cauchy, y_euler_cauchy, label="Метод Ейлера-Коші", color='blue')
    plt.plot(x_rk, y_rk, label="Метод Рунге-Кутти 4-го порядку", color='black')
    plt.title("Розв'язок ЗДР різними методами")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(which='both', linestyle='--', linewidth=0.5)
    plt.minorticks_on()
    plt.legend()
    plt.show()


# Функція для обчислення похибки
def calculate_error(y_method, y_rk):
    return abs(y_method - y_rk)


# Головна функція
def main():
    print("Звичайні диференційні рівняння. Методи Ейлера, Ейлера-Коші:")

    # Вибір функції
    f, a, b, h, y0 = select_function()
    if not f:
        return

    # Обчислення результатів
    x_euler, y_euler = euler_method(f, a, b, h, y0)
    x_euler_cauchy, y_euler_cauchy = euler_cauchy_method(f, a, b, h, y0)
    x_rk, y_rk = runge_kutta_method(f, a, b, h, y0)

    # Виведення таблиць
    print_table(x_euler, y_euler, "Метод Ейлера")
    print_table(x_euler_cauchy, y_euler_cauchy, "Метод Ейлера-Коші")

    # Обчислення похибки для останнього значення
    error_euler = calculate_error(y_euler[-1], y_rk[-1])
    error_euler_cauchy = calculate_error(y_euler_cauchy[-1], y_rk[-1])

    # Виведення похибок
    print(f"\nПохибка методу Ейлера порівняно з методом Рунге-Кутти: {error_euler:.4f}")
    print(f"Похибка методу Ейлера-Коші порівняно з методом Рунге-Кутти: {error_euler_cauchy:.4f}")

    # Побудова графіків
    plot_results(x_euler, y_euler, x_euler_cauchy, y_euler_cauchy, x_rk, y_rk)


# Виклик головної функції
if __name__ == "__main__":
    main()