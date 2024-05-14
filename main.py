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

from scipy.integrate import quad

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
    n1, n2 = "\{", "\}"

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
                wrong_answers.add(f"$${wrong_integral.replace('{', TaskGenerator.n1).replace('}', TaskGenerator.n2)}$$")
            # Сохраняем задачу и ответ
            tasks.add((
                      f"Вычислите $${latex(Integral(f.subs({a: a_value, b: b_value}), x)).replace('{', TaskGenerator.n1).replace('}', TaskGenerator.n2)}$$",
                      f"$${latex(integral_simplified.subs({a: a_value, b: b_value}).simplify(rational=True)).replace('{', TaskGenerator.n1).replace('}', TaskGenerator.n2)}$$",
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
                wrong_answers.add(f"$${latex(wrong_integral).replace('{', TaskGenerator.n1).replace('}', TaskGenerator.n2)}$$")

            # Сохраняем задачу, верный и неверные ответы
            tasks.add((f"Вычислите $${latex(Integral(f)).replace('{', TaskGenerator.n1).replace('}', TaskGenerator.n2)}$$",
                       f"$${latex(integral).replace('{', TaskGenerator.n1).replace('}', TaskGenerator.n2)}$$", tuple(wrong_answers)))
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
    def generate_task_parts(number_of_tasks=1, f_x_num=0):
        x, k, P_x, f_x = symbols('x k Pn(x) f(x)')
        functions = [exp(x), log(x), sin(x), cos(x), atan(x)]
        f_x_value = functions[f_x_num]
        f = k * P_x * f_x_value
        tasks = set()
        while len(tasks)<number_of_tasks:
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

            mid_answer_text = f"$${latex(u * v - Integral(v * du)).replace('{', TaskGenerator.n1).replace('}', TaskGenerator.n2)}$$"
            mid_answer = u * v - integrate(v * du, x)
            answer = mid_answer.subs(x, b_value) - mid_answer.subs(x, a_value)

            mid_wrong = u * v - integrate(diff(dv, x) * integrate(u, x), x)
            false_answers_main = [
                mid_answer.subs(x, a_value) - mid_answer.subs(x, b_value),
                mid_wrong.subs(x, a_value) - mid_wrong.subs(x, b_value),
                mid_wrong.subs(x, b_value) - mid_wrong.subs(x, a_value)
            ]

            false_answers_add = [
                f"$${latex(u * v - Integral(v * du, (x, a_value, b_value))).replace('{', TaskGenerator.n1).replace('}', TaskGenerator.n2)}$$",
                f"$${latex(u * dv - Integral(diff(integrate(dv, (x, a_value, b_value)) * 
                     diff(u, x)),(x, a_value, b_value))).replace('{', TaskGenerator.n1).replace('}', TaskGenerator.n2)}$$",
                f"$${latex(u * v - Integral(v * dv)).replace('{', TaskGenerator.n1).replace('}', TaskGenerator.n2)}$$"
            ]

            task = f"Вычислите $${latex(Integral(integral, (x, a_value, b_value))).replace('{', TaskGenerator.n1).replace('}', TaskGenerator.n2)}$$"
            add_question = f"Как выглядит пример после применения формулы интегрирования по частям?"
            answers = [f"{answer} (Основной вопрос)", f"{mid_answer_text} (Дополнительный вопрос)"]
            all_wrong_answers = [f"{i} (Основной вопрос)' for i in false_answers_main] + [f'{i} (Дополнительный вопрос)"
                                                                                          for i in false_answers_add]

            tasks.add((task, add_question, tuple(answers), tuple(all_wrong_answers)))
        return tasks

    @staticmethod
    def rational_function_1(number_of_tasks=1):
        # number_of_tasks1 = min(100, number_of_tasks)
        tasks = set()
        for _ in range(number_of_tasks):
            A = random.randint(-10, 10)
            while A == 0:
                A = random.randint(-10, 10)  # Генерация случайного целого числа A
            n = random.choice(list(range(-10, 0)) + list(range(1, 11)))  # Генерация случайного целого числа n
            x = symbols('x')  # Создание переменной x
            a = random.randint(-10, 10)  # Генерация случайного целого числа a из интервала [-10;10]
            while a == 0:  # Проверка, что a не равно нулю
                a = random.randint(-10, 10)  # Если a равно нулю, повторить генерацию
            b = random.randint(-10, 10)  # Генерация случайного целого числа b из интервала [-10;10]
            while b == 0:  # Проверка, что b не равно нулю
                b = random.randint(-10, 10)  # Если b равно нулю, повторить генерацию

            # Обмен значений a и b, если a больше b
            if a > b:
                a, b = b, a

            function = A / (x - n)  # Функция A / (x - n)
            integral = integrate(function, (x, a, b))  # Вычисление определенного интеграла

            # Проверка на бесконечный ответ или NaN
            if not (isinstance(integral, sympy.core.numbers.NaN) or integral == sympy.oo or integral == -sympy.oo):
                # Генерация неверных ответов
                wrong_answers = set()
                while len(wrong_answers) < 3:
                    wrong_integral = integral + random.randint(-10, 10)  # Генерация неверного ответа
                    wrong_answers.add(f"$${latex(wrong_integral).replace('{', TaskGenerator.n1).replace('}', TaskGenerator.n2)}$$")

                # Возврат значений функции, пределов, интеграла и неверных ответов
                tasks.add((f"Вычислите $${latex(Integral(function, (x, a, b))).replace('{', TaskGenerator.n1).replace('}', TaskGenerator.n2)}$$",
                           f"$${latex(integral).replace('{', TaskGenerator.n1).replace('}', TaskGenerator.n2)}$$",
                           tuple(wrong_answers)))
        return tasks

    @staticmethod
    def rational_function_2(number_of_tasks=1):
        number_of_tasks1 = min(100, number_of_tasks)
        tasks = set()
        while True:
            A = random.randint(-10, 10)
            while A == 0:
                A = random.randint(-10, 10)  # Генерация случайного целого числа A
            n = random.choice(list(range(-10, 0)) + list(range(1, 11)))  # Генерация случайного целого числа n
            x = symbols('x')  # Создание переменной x
            a = random.randint(-10, 10)  # Генерация случайного целого числа a из интервала [-10;10]
            while a == 0:  # Проверка, что a не равно нулю
                a = random.randint(-10, 10)  # Если a равно нулю, повторить генерацию
            b = random.randint(-10, 10)  # Генерация случайного целого числа b из интервала [-10;10]
            while b == 0:  # Проверка, что b не равно нулю
                b = random.randint(-10, 10)  # Если b равно нулю, повторить генерацию

            # Обмен значений a и b, если a больше b
            if a > b:
                a, b = b, a

            function = A / (x - n)  # Функция A / (x - n)
            integral = integrate(function, (x, a, b))  # Вычисление определенного интеграла

            # Проверка на бесконечный ответ или NaN
            if not (isinstance(integral, sympy.core.numbers.NaN) or integral == sympy.oo or integral == -sympy.oo):
                # Генерация неверных ответов
                wrong_answers = set()
                while len(wrong_answers) < 3:
                    wrong_integral = integral + random.randint(-10, 10)  # Генерация неверного ответа
                    wrong_answers.add(f"$${latex(wrong_integral).replace('{', TaskGenerator.n1).replace('}', TaskGenerator.n2)}$$")

                # Формирование задачи в формате (задача, правильный ответ, неправильные ответы)
                tasks.add((f"Вычислите $${latex(Integral(function, (x, a, b))).replace('{', TaskGenerator.n1).replace('}', TaskGenerator.n2)}$$",
                           f"$${latex(integral).replace('{', TaskGenerator.n1).replace('}', TaskGenerator.n2)}$$",
                           tuple(wrong_answers)))

                if len(tasks) >= number_of_tasks1:
                    return tasks

    @staticmethod
    def rational_function_3(num_tasks):
        def generate_problem():
            while True:
                # Генерация случайных параметров
                B, D = random.randint(-10, 10), random.randint(-10, 10)
                p, q = random.randint(-10, 10), random.randint(-10, 10)

                # Проверка, что многочлен в знаменателе не имеет действительных корней
                if p ** 2 - 4 * q >= 0:
                    continue

                # Проверка на ненулевые значения B, D, p и q
                if B == 0 or D == 0 or p == 0 or q == 0:
                    continue

                a = random.randint(-5, 5)
                b = random.randint(-5, 5)

                # Проверка на равенство 0
                if a == 0 or b == 0:
                    continue

                # Убеждаемся, что a меньше b
                if a >= b:
                    a, b = b, a

                # Создание функции
                x = symbols('x')
                function = (B * x + D) / (x ** 2 + p * x + q)

                # Проверка на бесконечный ответ или NaN
                integral = integrate(function, (x, a, b))
                if integral in [sympy.oo, -sympy.oo] or str(integral) == "nan" or len(
                        str(integral)) > 20 or integral == 0:
                    continue  # Генерируем новую задачу

                # Генерация 5 неправильных ответов
                wrong_answers = tuple(
                    set([f"$${latex(integral + random.randint(-10, 10)).replace('{', TaskGenerator.n1).replace('}', TaskGenerator.n2)}$$" for _ in
                         range(5)]))

                task = (f"Вычислите $${latex(Integral(function, (x, a, b))).replace('{', TaskGenerator.n1).replace('}', TaskGenerator.n2)}$$",
                        f"$${latex(integral).replace('{', TaskGenerator.n1).replace('}', TaskGenerator.n2)}$$",
                        wrong_answers)
                return task

        tasks = []
        for _ in range(num_tasks):
            tasks.append(generate_problem())
        return tasks

    @staticmethod
    def generate_task_square(amount_of_tasks=1):
        x, y, a_, b_, g = symbols('x y a b g')
        f = sqrt(((g - a_ * x ** 2) / b_))
        abc = [[4, 4, 1], [1, 1, 1], [4, 4, 4], [9, 9, 9], [1, 1, 4], [1, 1, 9], [4, 4, 9]]
        tasks = set()
        for _ in range(amount_of_tasks):
            alpha, beta, gamma = random.choice(abc)

            # Интервал интегрирования
            b = sqrt(((gamma) / alpha))
            a = -b
            mid_answer_text = f'от {a:.{2}f} до {b:.{2}f}'

            f_x = f.subs({g: gamma, a_: alpha, b_: beta})
            f_diff = diff(f_x, x)

            tmp1 = simplify(f_x * sqrt(1 + f_diff ** 2))
            func = lambdify(x, tmp1, modules=['numpy'])
            integral = quad(func, a, b)[0]
            answer = 2 * 2 * pi * integral

            false_answers_main = [pi * f_diff, pi * integral / 2, pi * integral]
            false_answers_add = [f'от {(a - 1):.{2}f} до {(b + 1):.{2}f}',
                                 f'от {(a - 0.5):.{2}f} до {(b + 0.5):.{2}f}',
                                 f'от {(a - 2 / 3):.{2}f} до {(b + 2 / 3):.{2}f}']

            task = (f"Вычислите площадь поверхности фигуры, полученной путем вращения окружности {alpha} * "
                    f"$${latex(x ** 2).replace('{', TaskGenerator.n1).replace('}', TaskGenerator.n2)}$$ + {beta} * "
                    f"$${latex(y ** 2).replace('{', TaskGenerator.n1).replace('}', TaskGenerator.n2)}$$ = {gamma} вокруг оси OX")
            add_question = f"Выберите верный интервал интегрирования"
            answers = [
                f"$${latex(answer).replace('{', TaskGenerator.n1).replace('}', TaskGenerator.n2)}$$ (Основной вопрос)",
                f"{mid_answer_text} (Дополнительный вопрос)"]
            all_wrong_answers = [f"$${latex(i).replace('{', TaskGenerator.n1).replace('}', TaskGenerator.n2)}$$ (Основной вопрос)" for i in
                                 false_answers_main] + [f"{i} (Дополнительный вопрос)" for i in false_answers_add]
            tasks.add((task, add_question, tuple(answers), tuple(all_wrong_answers)))
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
                            f"::МА2 Задание №{number}:: {task[0]} \n{{={task[1]}\n{wrong}}}")
                        file.write("\n")
                        file.write("\n")
                else:
                    for task in tasks:
                        right = ""
                        wrong = ""
                        for i in task[2]:
                            right += f" =%50.0%{i}\n"
                        for i in task[3]:
                            wrong += f" ~%-25.0%{i}\n"
                        file.write(
                            f"::МА2 Задание №{number}:: {task[0]} {{{right}{wrong}}}")
                        file.write("\n")
                        file.write("\n")


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    application = QtWidgets.QApplication(sys.argv)
    program = TaskGenerator()
    tasks = program.rational_function_2(3)
    print(tasks)
    program.write_tasks(tasks, 1)
    program.generate_task_parts(1)
    program.direct_integration_2(1)
    program.show()
    application.exec_()
