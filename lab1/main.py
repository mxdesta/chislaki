"""
Лабораторная работа 1: Методы решения задач линейной алгебры

"""
import os

# Создаем папки для логов если их нет
if not os.path.exists("logs"):
    os.makedirs("logs")

print("=" * 70)
print(" " * 15 + "ЛАБОРАТОРНАЯ РАБОТА 1")
print(" " * 10 + "Методы решения задач линейной алгебры")
print("=" * 70)
print("\nВсе итерации будут записаны в файлы логов в папке 'logs/'")
print("Финальные результаты будут выведены в консоль")

print("\n\nВыберите задачу для выполнения:")
print("1 - LU-разложение (решение СЛАУ, определитель, обратная матрица)")
print("2 - Метод прогонки (трехдиагональные матрицы)")
print("3 - Итерационные методы (простые итерации и Зейдель)")
print("4 - Метод вращений (собственные значения симметрических матриц)")
print("5 - QR-алгоритм (собственные значения произвольных матриц)")
print("0 - Выполнить все задачи")

choice = input("\nВведите номер задачи: ")

if choice == "1":
    print("\n" + "="*50)
    print("ВЫПОЛНЕНИЕ ЗАДАЧИ 1: LU-РАЗЛОЖЕНИЕ")
    print("="*50)
    import task1_lu_decomposition
elif choice == "2":
    print("\n" + "="*50)
    print("ВЫПОЛНЕНИЕ ЗАДАЧИ 2: МЕТОД ПРОГОНКИ")
    print("="*50)
    import task2_tridiagonal
elif choice == "3":
    print("\n" + "="*50)
    print("ВЫПОЛНЕНИЕ ЗАДАЧИ 3: ИТЕРАЦИОННЫЕ МЕТОДЫ")
    print("="*50)
    import task3_iterative
elif choice == "4":
    print("\n" + "="*50)
    print("ВЫПОЛНЕНИЕ ЗАДАЧИ 4: МЕТОД ВРАЩЕНИЙ")
    print("="*50)
    import task4_rotation
elif choice == "5":
    print("\n" + "="*50)
    print("ВЫПОЛНЕНИЕ ЗАДАЧИ 5: QR-АЛГОРИТМ")
    print("="*50)
    import task5_qr_algorithm_student
elif choice == "0":
    print("\n" + "=" * 70)
    print("ВЫПОЛНЕНИЕ ВСЕХ ЗАДАЧ")
    print("=" * 70)
    
    tasks = [
        ("ЗАДАЧА 1: LU-РАЗЛОЖЕНИЕ", "task1_lu_decomposition"),
        ("ЗАДАЧА 2: МЕТОД ПРОГОНКИ", "task2_tridiagonal"),
        ("ЗАДАЧА 3: ИТЕРАЦИОННЫЕ МЕТОДЫ", "task3_iterative"),
        ("ЗАДАЧА 4: МЕТОД ВРАЩЕНИЙ", "task4_rotation"),
        ("ЗАДАЧА 5: QR-АЛГОРИТМ", "task5_qr_algorithm_student")
    ]
    
    for i, (name, module) in enumerate(tasks):
        print(f"\n{'='*50}")
        print(f"ВЫПОЛНЕНИЕ {name}")
        print('='*50)
        exec(f"import {module}")
        if i < len(tasks) - 1:
            input("\nНажмите Enter для продолжения...")
else:
    print("Неверный выбор!")

print(f"\n{'='*70}")
print("РАБОТА ЗАВЕРШЕНА")
print(f"Все подробные логи сохранены в папке: {os.path.abspath('logs')}")
print('='*70)
