# This is a sample Python script.
import random
import sys
from sympy import *
from PyQt5 import QtWidgets, uic
import sympy
import PyQt5
from PyQt5.QtCore import QThread
from PyQt5.QtWidgets import QFileDialog

from sympy import symbols, integrate, simplify, latex, Integral

Form, _ = uic.loadUiType("generator.ui")


class WriteThread(QThread):
    pass


def random_not_null(a: int, b: int):
    while True:
        value = random.randint(a, b)
        if value != 0:
            break
    return value


def generate_a_b(min: int, max: int):
    a = random.randint(min, max)
    b = random.randint(min, max)
    while a == b:
        b = random.randint(min, max)
    if a > b:
        a, b = b, a
    return a, b


class TaskGenerator(QtWidgets.QMainWindow, Form):
    def __init__(self):
        super(TaskGenerator, self).__init__()
        self.setupUi(self)

    @staticmethod
    def direct_integration_1(number_of_tasks):
        number_of_tasks = min(100, number_of_tasks)
        tasks = set()

        x, a, b = symbols('x a b')

        while len(tasks) < number_of_tasks:
            # Задаем функцию, которую интегрируем
            f = ((a - b * x) ** 2) / (x ** 1.5)

            # Применяем интегрирование по частям
            u = (a - b * x) ** 2
            dv = 1 / x ** 1.5
            v = integrate(dv, x)

            integral = u * v - integrate(v * sympy.diff(u, x), x)

            # Упростим результат
            integral_simplified = simplify(integral)
            a_value = random.randint(1, 10)
            b_value = random.randint(1, 10)
            # Генерация неверных ответов
            wrong_answers = set()
            while len(wrong_answers) < 3:
                # Генерируем случайные значения для a и b
                wrong_a = random.randint(1, 10)
                wrong_b = random.randint(1, 10)

                # Проверяем, чтобы неверные a и b не совпадали с верными
                while wrong_a == a_value and wrong_b == b_value:
                    wrong_a = random.randint(1, 10)
                    wrong_b = random.randint(1, 10)

                # Вычисляем интеграл с неверными a и b
                wrong_integral = latex(integral_simplified.subs({a: wrong_a, b: wrong_b}).simplify(rational=True))
                wrong_answers.add(f"$${wrong_integral.replace("{", "\{").replace("}", "\}")}$$")
            # Сохраняем задачу и ответ
            tasks.add((
                      f"Вычислите $${latex(Integral(f.subs({a: a_value, b: b_value}), x)).replace("{", "\{").replace("}", "\}")}$$",
                      f"$${latex(integral_simplified.subs({a: a_value, b: b_value}).simplify(rational=True)).replace("{", "\{").replace("}", "\}")}$$",
                      tuple(wrong_answers)))
        return tasks

    @staticmethod
    def direct_integration_2(number_of_tasks):
        number_of_tasks = min(20 ** 4, number_of_tasks)
        tasks = set()

        x = symbols('x')

        while len(tasks) < number_of_tasks:
            a = random.randint(1, 20)
            b = random.randint(1, 20)
            c = random.randint(1, 20)
            d = random.randint(1, 20)

            # Задаем функцию, которую интегрируем
            f = a * x ** 2 + b * x + c / x + d

            # Вычисляем интеграл
            integral = integrate(f, x)

            # Генерация неверных ответов
            wrong_answers = set()
            while len(wrong_answers) < 3:
                # Генерируем случайные значения для коэффициентов
                wrong_a = random.randint(1, 20)
                wrong_b = random.randint(1, 20)
                wrong_c = random.randint(1, 20)
                wrong_d = random.randint(1, 20)

                # Проверяем, чтобы неверные коэффициенты не совпадали с верными
                while (wrong_a == a and wrong_b == b and wrong_c == c and wrong_d == d):
                    wrong_a = random.randint(1, 20)
                    wrong_b = random.randint(1, 20)
                    wrong_c = random.randint(1, 20)
                    wrong_d = random.randint(1, 20)

                # Вычисляем интеграл с неверными коэффициентами
                wrong_integral = integrate(wrong_a * x ** 2 + wrong_b * x + wrong_c / x + wrong_d, x)
                wrong_answers.add(f"$${latex(wrong_integral).replace("{", "\{").replace("}", "\}")}$$")

            # Сохраняем задачу, верный и неверные ответы
            tasks.add((f"Вычислите $${latex(Integral(f)).replace("{", "\{").replace("}", "\}")}$$",
                       f"$${latex(integral).replace("{", "\{").replace("}", "\}")}$$", tuple(wrong_answers)))
            print(tasks)
        return tasks

    @staticmethod
    def __random_not_null(a: int, b: int):
        while True:
            value = random.randint(a, b)
            if value != 0:
                break
        return value

    @staticmethod
    def generate_a_b(min: int, max: int):
        a = random.randint(min, max)
        b = random.randint(min, max)
        while a == b:
            b = random.randint(min, max)
        if a > b:
            a, b = b, a
        return a, b

    @staticmethod
    def generate_task_parts(amount_of_tasks=1, f_x_num=0):
        x, k, P_x, f_x = symbols('x k Pn(x) f(x)')
        functions = [exp(x), log(x), sin(x), cos(x), atan(x)]
        f_x_value = functions[f_x_num]
        f = k * P_x * f_x_value
        tasks = set()
        for _ in range(amount_of_tasks):

            # Генерация многочлена P(x) степени не выше 2
            coefs = [0, 0, 0]
            while 0 in coefs:
                coefs = [random.randint(-4, 4) for _ in range(3)]
            null_indexes = random.sample(range(1, 3), random.randint(0, 2))
            for i in null_indexes:
                coefs[i] = 0
            P_x_value = coefs[0] * (x ** 2)
            for i in range(1, len(coefs)):
                P_x_value += coefs[i] * (x ** (len(coefs) - i - 1))

            # Генерация чисел a, b, k
            if f_x_value == log(x):
                a_value, b_value = generate_a_b(1, 6)
            else:
                a_value, b_value = generate_a_b(-4, 4)
            k_value = random_not_null(-10, 10)

            # Готовый интеграл
            integral = (f.subs({k: k_value, P_x: P_x_value}))

            # Интегрирование по частям
            u = k_value * P_x_value
            dv = f_x_value
            du = diff(u, x)
            v = integrate(dv, x)

            mid_answer_text = f'$${latex(u * v - Integral(v * du)).replace("{", "\{").replace("}", "\}")}$$'
            mid_answer = u * v - integrate(v * du, x)
            answer = mid_answer.subs(x, b_value) - mid_answer.subs(x, a_value)

            mid_wrong = u * v - integrate(diff(dv, x) * integrate(u, x), x)
            false_answers_main = [
                mid_answer.subs(x, a_value) - mid_answer.subs(x, b_value),
                mid_wrong.subs(x, a_value) - mid_wrong.subs(x, b_value),
                mid_wrong.subs(x, b_value) - mid_wrong.subs(x, a_value)
            ]
            false_answers_add = [
                f'$${latex(u * v - Integral(v * du, (x, a_value, b_value))).replace("{", "\{").replace("}", "\}")}$$',
                # f'$${latex(u * dv - Integral(diff(integrate(dv, (x,a_value,b_value)) * diff(u, x)))).replace("{","\{").replace("}","\}")}$$',
                f'$${latex(u * v - Integral(v * dv)).replace("{", "\{").replace("}", "\}")}$$'
            ]

            task = f"Вычислите $${latex(Integral(integral, (x, a_value, b_value))).replace("{", "\{").replace("}", "\}")}$$"
            add_question = f'Как выглядит пример после применения формулы интегрирования по частям?'
            answers = [f'{answer} (Основной вопрос)', f'{mid_answer_text} (Дополнительный вопрос)']
            all_wrong_answers = [f'{i} (Основной вопрос)' for i in false_answers_main] + [f'{i} (Дополнительный вопрос)'
                                                                                          for i in false_answers_add]

            tasks.add((task, add_question, tuple(answers), tuple(all_wrong_answers)))
            print((task, add_question, tuple(answers), tuple(all_wrong_answers)))
            print(task)
        return tasks

    @staticmethod
    def write_tasks(tasks,number, multi=False):
        file_path = QFileDialog.getExistingDirectory(None, "Выбрать путь для сохранения")
        if file_path:
            with open(file_path + "/Задачи.txt", 'w', encoding='utf-8') as file:
                if not multi:
                    for task in tasks:
                        wrong = ""
                        for i in task[2]:
                            wrong += f" ~{i}\n"
                        file.write(
                            f"::МА2 Задание №1:: {task[0]} \n{{={task[1]}\n{wrong}}}")
                        file.write("\n")
                        file.write("\n")
                else:
                    for task in tasks:
                        right = ""
                        wrong = ""
                        for i in task[2]:
                            right += f" =%50.0%{i}\n"
                        for i in task[3]:
                            wrong += f" ~%-50.0%{i}\n"
                        file.write(
                            f"::МА2 Задание №{number}:: {task[0]} {{{right}{wrong}}}")
                        file.write("\n")
                        file.write("\n")


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    application = QtWidgets.QApplication(sys.argv)
    program = TaskGenerator()
    program.write_tasks(program.direct_integration_2(1,3,))
    program.generate_task_parts(1)
    program.direct_integration_2(1)
    program.show()
    application.exec_()
