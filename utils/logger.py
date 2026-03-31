"""
Модуль для логирования итераций
"""
import os
import numpy as np
from datetime import datetime


class IterationLogger:
    def __init__(self, task_name, log_dir="logs"):
        self.task_name = task_name
        self.log_dir = log_dir
        self.log_file = None
        self.iteration_count = 0
        
        # Создаем папку для логов
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        
        # Создаем файл лога с временной меткой
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(self.log_dir, f"{task_name}_{timestamp}.log")
        
        # Инициализируем файл лога
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write(f"=== ЛАБОРАТОРНАЯ РАБОТА 1 ===\n")
            f.write(f"Задача: {task_name}\n")
            f.write(f"Время начала: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 60 + "\n\n")
    
    def log_iteration(self, iteration, data):
        """Логирование одной итерации"""
        self.iteration_count += 1
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f"ИТЕРАЦИЯ {iteration}\n")
            f.write("-" * 30 + "\n")
            
            for key, value in data.items():
                if isinstance(value, np.ndarray):
                    f.write(f"{key}:\n{value}\n")
                else:
                    f.write(f"{key}: {value}\n")
            f.write("\n")
    
    def log_matrix(self, name, matrix):
        """Логирование матрицы"""
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f"{name}:\n{matrix}\n\n")
    
    def log_final_result(self, result_data):
        """Логирование финального результата"""
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("ФИНАЛЬНЫЙ РЕЗУЛЬТАТ\n")
            f.write("=" * 60 + "\n")
            
            for key, value in result_data.items():
                if isinstance(value, np.ndarray):
                    f.write(f"{key}:\n{value}\n")
                else:
                    f.write(f"{key}: {value}\n")
            
            f.write(f"\nВсего итераций: {self.iteration_count}\n")
            f.write(f"Время завершения: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    def get_log_file_path(self):
        """Возвращает путь к файлу лога"""
        return self.log_file


def print_final_summary(task_name, summary_data):
    """Вывод финального резюме в консоль"""
    print("\n" + "=" * 60)
    print(f"ФИНАЛЬНЫЙ РЕЗУЛЬТАТ: {task_name}")
    print("=" * 60)
    
    for key, value in summary_data.items():
        if isinstance(value, np.ndarray):
            if value.size <= 16:  # Выводим только небольшие массивы
                print(f"{key}:")
                print(value)
            else:
                print(f"{key}: [массив размера {value.shape}]")
        else:
            print(f"{key}: {value}")
    
    print("=" * 60)