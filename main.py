# This is a sample Python script.
import random
import sys
from sympy import *
from PyQt5 import QtWidgets, uic
import sympy
import PyQt5
from PyQt5.QtCore import QThread
from PyQt5.QtWidgets import QFileDialog
from fractions import Fraction
from sympy import symbols, integrate, simplify, latex, Integral, Rational
import numpy as np
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
        self.pushButton.clicked.connect(self.kekw)

    def kekw(self):
        match self.comboBox.currentIndex():
            case 0:
                tasks = self.direct_integration_1(self.spinBox.value())
                program.write_tasks(tasks, self.comboBox.currentIndex(), multi=False)
            case 1:
                tasks = self.direct_integration_2(self.spinBox.value())
                program.write_tasks(tasks, self.comboBox.currentIndex(), multi=False)
            case 2:
                tasks = self.parts_integration(self.spinBox.value(),0)
                program.write_tasks(tasks, self.comboBox.currentIndex(), multi=False)
            case 3:
                tasks = self.parts_integration(self.spinBox.value(),1)
                program.write_tasks(tasks, self.comboBox.currentIndex(), multi=False)
            case 4:
                tasks = self.parts_integration(self.spinBox.value(),2)
                program.write_tasks(tasks, self.comboBox.currentIndex(), multi=False)
            case 5:
                tasks = self.parts_integration(self.spinBox.value(),3)
                program.write_tasks(tasks, self.comboBox.currentIndex(), multi=False)
            case 6:
                tasks = self.parts_integration(self.spinBox.value(),4)
                program.write_tasks(tasks, self.comboBox.currentIndex(), multi=False)
            case 7:
                tasks = self.direct_integration_1(self.spinBox.value())
                program.write_tasks(tasks, self.comboBox.currentIndex(), multi=False)
            case 8:
                tasks = self.direct_integration_1(self.spinBox.value())
                program.write_tasks(tasks, self.comboBox.currentIndex(), multi=False)
            case 9:
                tasks = self.direct_integration_1(self.spinBox.value())
                program.write_tasks(tasks, self.comboBox.currentIndex(), multi=False)
            case 10:
                tasks = self.direct_integration_1(self.spinBox.value())
                program.write_tasks(tasks, self.comboBox.currentIndex(), multi=False)
            case 11:
                tasks = self.direct_integration_1(self.spinBox.value())
                program.write_tasks(tasks, self.comboBox.currentIndex(), multi=False)
            case 12:
                tasks = self.direct_integration_1(self.spinBox.value())
                program.write_tasks(tasks, self.comboBox.currentIndex(), multi=False)
            case 13:
                tasks = self.direct_integration_1(self.spinBox.value())
                program.write_tasks(tasks, self.comboBox.currentIndex(), multi=False)


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
    def parts_integration(number_of_tasks=1, f_x_num=0):
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
            answer = f"$${latex(mid_answer.subs(x, b_value) - mid_answer.subs(x, a_value)).replace('{', TaskGenerator.n1).replace('}', TaskGenerator.n2)}$$"

            mid_wrong = u * v - integrate(diff(dv, x) * integrate(u, x), x)
            false_answers_main = set([
                f"$${latex(mid_answer.subs(x, a_value) - mid_answer.subs(x, b_value)).replace('{', TaskGenerator.n1).replace('}', TaskGenerator.n2)}$$",
                f"$${latex(mid_wrong.subs(x, a_value) - mid_wrong.subs(x, b_value)).replace('{', TaskGenerator.n1).replace('}', TaskGenerator.n2)}$$",
                f"$${latex(mid_wrong.subs(x, b_value) - mid_wrong.subs(x, a_value)).replace('{', TaskGenerator.n1).replace('}', TaskGenerator.n2)}$$"])

            false_answers_add = set([
                f"$${latex(u * v - Integral(v * du, (x, a_value, b_value))).replace('{', TaskGenerator.n1).replace('}', TaskGenerator.n2)}$$",
                f"$${latex(u * dv - Integral(diff(integrate(dv, (x, a_value, b_value)) * 
                     diff(u, x)),(x, a_value, b_value))).replace('{', TaskGenerator.n1).replace('}', TaskGenerator.n2)}$$",
                f"$${latex(u * v - Integral(v * dv)).replace('{', TaskGenerator.n1).replace('}', TaskGenerator.n2)}$$"
            ])

            task = f"Вычислите $${latex(Integral(integral, (x, a_value, b_value))).replace('{', TaskGenerator.n1).replace('}', TaskGenerator.n2)}$$"
            add_question = f"Как выглядит пример после применения формулы интегрирования по частям?"
            answers = [f"{answer} (Основной вопрос)", f"{mid_answer_text} (Дополнительный вопрос)"]
            all_wrong_answers = [f"{i} (Основной вопрос)" for i in false_answers_main] + [f"{i} (Дополнительный вопрос)"
                                                                                          for i in false_answers_add]

            tasks.add((task, add_question, tuple(answers), tuple(all_wrong_answers)))
        return tasks

    @staticmethod
    def rational_function_1(number_of_tasks=1):
        tasks = set()
        while len(tasks) < number_of_tasks:
            A = random.randint(-10, 10)
            while A == 0:
                A = random.randint(-10, 10)
            n = random.choice(list(range(-10, 0)) + list(range(1, 11)))
            x = symbols('x')
            a = random.randint(-10, 10)
            while a == 0:
                a = random.randint(-10, 10)
            b = random.randint(-10, 10)
            while b == 0:
                b = random.randint(-10, 10)

            if a > b:
                a, b = b, a

            function = A / (x - n)
            integral = integrate(function, (x, a, b))

            if not (isinstance(integral, sympy.core.numbers.NaN) or integral == sympy.oo or integral == -sympy.oo):
                # Генерация неверных ответов
                wrong_answers = set()
                correct_integral_latex = latex(integral).replace('{', TaskGenerator.n1).replace('}', TaskGenerator.n2)
                offset = random.randint(-10, -5)
                # Генерация неверного ответа
                wrong_integral = integral * offset
                wrong_answers.add(
                    f"$${latex(wrong_integral).replace('{', TaskGenerator.n1).replace('}', TaskGenerator.n2)}$$")
                offset = random.randint(-5, -1)
                # Генерация неверного ответа
                wrong_integral = integral * offset
                wrong_answers.add(
                    f"$${latex(wrong_integral).replace('{', TaskGenerator.n1).replace('}', TaskGenerator.n2)}$$")
                offset = random.randint(2, 5)
                # Генерация неверного ответа
                wrong_integral = integral * offset
                wrong_answers.add(
                    f"$${latex(wrong_integral).replace('{', TaskGenerator.n1).replace('}', TaskGenerator.n2)}$$")

                # Возврат значений функции, пределов, интеграла и неверных ответов
                tasks.add((f"Вычислите $${latex(Integral(function, (x, a, b))).replace('{', TaskGenerator.n1).replace('}', TaskGenerator.n2)}$$",
                           f"$${correct_integral_latex}$$",
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
                # Генерация неверных ответов
                wrong_answers = set()
                correct_integral_latex = latex(integral).replace('{', TaskGenerator.n1).replace('}', TaskGenerator.n2)
                offset = random.randint(-10, -5)
                # Генерация неверного ответа
                wrong_integral = integral * offset
                wrong_answers.add(
                    f"$${latex(wrong_integral).replace('{', TaskGenerator.n1).replace('}', TaskGenerator.n2)}$$")
                offset = random.randint(-5, -1)
                # Генерация неверного ответа
                wrong_integral = integral * offset
                wrong_answers.add(
                    f"$${latex(wrong_integral).replace('{', TaskGenerator.n1).replace('}', TaskGenerator.n2)}$$")
                offset = random.randint(2, 5)
                # Генерация неверного ответа
                wrong_integral = integral * offset
                wrong_answers.add(
                    f"$${latex(wrong_integral).replace('{', TaskGenerator.n1).replace('}', TaskGenerator.n2)}$$")

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
                B, D = random_not_null(-10,10), random_not_null(-10,10)
                p, q = random_not_null(-10,10), random_not_null(-10,10)

                # Проверка, что многочлен в знаменателе не имеет действительных корней
                if p ** 2 - 4 * q >= 0:
                    continue

                # Проверка на ненулевые значения B, D, p и q
                if B == 0 or D == 0 or p == 0 or q == 0:
                    continue

                a = random_not_null(-5,5)
                b = random_not_null(-5,5)

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
                wrong_answers = set()
                correct_integral_latex = latex(integral).replace('{', TaskGenerator.n1).replace('}', TaskGenerator.n2)
                offset = random.randint(-10, -5)
                # Генерация неверного ответа
                wrong_integral = integral * offset
                wrong_answers.add(
                    f"$${latex(wrong_integral).replace('{', TaskGenerator.n1).replace('}', TaskGenerator.n2)}$$")
                offset = random.randint(-5, -1)
                # Генерация неверного ответа
                wrong_integral = integral * offset
                wrong_answers.add(
                    f"$${latex(wrong_integral).replace('{', TaskGenerator.n1).replace('}', TaskGenerator.n2)}$$")
                offset = random.randint(2, 5)
                # Генерация неверного ответа
                wrong_integral = integral * offset
                wrong_answers.add(
                    f"$${latex(wrong_integral).replace('{', TaskGenerator.n1).replace('}', TaskGenerator.n2)}$$")

                task = (f"Вычислите $${latex(Integral(function, (x, a, b))).replace('{', TaskGenerator.n1).replace('}', TaskGenerator.n2)}$$",
                        f"$${latex(integral).replace('{', TaskGenerator.n1).replace('}', TaskGenerator.n2)}$$",
                        tuple(wrong_answers))
                return tuple(task)

        tasks = set()
        while len(tasks)<num_tasks:
            tasks.add(generate_problem())
        return tasks

    @staticmethod
    def geom_integration(amount_of_tasks=1):
        x, y, a_, b_, g = symbols('x y a b g')
        f = sqrt(((g - a_ * x ** 2) / b_))
        abc = [[4, 4, 1], [1, 1, 1], [4, 4, 4], [9, 9, 9], [1, 1, 4], [1, 1, 9], [4, 4, 9]]
        tasks = set()
        while len(tasks)<amount_of_tasks:
            alpha, beta, gamma = [random.randint(1,6),random.randint(1,6),random.randint(1,6)]#random.choice(abc)

            # Интервал интегрирования
            b = sqrt(((gamma) / alpha))
            a = -b
            mid_answer_text = f'от {a:.{2}f} до {b:.{2}f}'

            f_x = f.subs({g: gamma, a_: alpha, b_: beta})
            f_diff = diff(f_x, x)

            tmp1 = simplify(f_x * sqrt(1 + f_diff ** 2))
            func = lambdify(x, tmp1, modules=['numpy'])
            integral = quad(func, a, b)[0]
            answer = round(2 * 2 * integral,2) * pi

            false_answers_main = [pi *f_diff,f'{pi * integral / 2:.{2}f}',f'{pi * integral:.{2}f}']
            false_answers_add = [f'от {(a - 1):.{2}f} до {(b + 1):.{2}f}',
                                 f'от {(a - 0.5):.{2}f} до {(b + 0.5):.{2}f}',
                                 f'от {(a - 2 / 3):.{2}f} до {(b + 2 / 3):.{2}f}']

            task = (f"Вычислите площадь поверхности фигуры, полученной путем вращения окружности "
                    f"$${latex(alpha* x ** 2 + beta * y ** 2).replace('{', TaskGenerator.n1).replace('}', TaskGenerator.n2)} = {gamma}$$ вокруг оси OX")
            add_question = f"\nВыберите верный интервал интегрирования"
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
                            f"::МА2 Задание №{number}:: {task[0]} \n{{={task[1]}\n{wrong}}}".replace("log", "ln"))
                        file.write("\n")
                        file.write("\n")
                else:
                    for task in tasks:
                        right = ""
                        wrong = ""
                        for i in task[2]:
                            right += f" ~%50.0% {i}\n"
                        for i in task[3]:
                            wrong += f" ~%-50.0% {i}\n"
                        file.write(
                            f"::МА2 Задание №{number}:: {task[0]} {task[1]} (Дополнительный вопрос) \n{{{right}{wrong}}}".replace("log", "ln"))
                        file.write("\n")
                        file.write("\n")
    @staticmethod
    def arc_length(num_tasks):
        def generate_func():
            x = Symbol('x')
            n = np.random.randint(2, 4)  # Случайная степень многочлена от 2 до 3 чтобы было поинтереснее
            count = np.random.randint(1, 4)  # Случайное количество одночленов от 1 до 3

            coefficients = []
            for i in range(count):
                coefficient = np.random.randint(-5, 6)  # Случайный коэффициент от -5 до 5
                while coefficient == 0:
                    coefficient = np.random.randint(-5, 6)  # Повторяем, пока не получим ненулевой коэффициент
                coefficients.append(coefficient)

            f = sum(coefficients[i] * (x ** max(0, n - i)) for i in range(len(coefficients)))

            a = np.random.randint(-10, 11)  # Генерация a
            b = np.random.randint(-10, 11)  # Генерация b
            while a == b:  # Проверка на равенство границ
                b = np.random.randint(-10, 11)
            if a > b:  # Гарантия того, что a < b
                a, b = b, a

            return f, a, b

        tasks = set()
        while len(tasks) < num_tasks:
            f, a, b = generate_func()
            x = Symbol('x')
            task = f"Выберите интеграл описывающий длину дуги $$y = {latex(f).replace('{', TaskGenerator.n1).replace('}', TaskGenerator.n2)}$$ на отрезке [{a}, {b}]. Вычислять интеграл не нужно "
            proiz = diff(f, x)
            sqr_proiz = proiz * proiz
            answer = f"$${latex(Integral(sqrt(1 + sqr_proiz), (x, a, b))).replace('{', TaskGenerator.n1).replace('}', TaskGenerator.n2)}$$"  # правильная формула длины дуги
            wrong_answers = []
            wrong_answers.append(
                f"$${latex(pi * Integral(f * f, (x, a, b))).replace('{', TaskGenerator.n1).replace('}', TaskGenerator.n2)}$$")  # неправильный ответ формула объёма тела вращения
            wrong_answers.append(
                f"$${latex(Integral(f, (x, a, b))).replace('{', TaskGenerator.n1).replace('}', TaskGenerator.n2)}$$")  # неправильный ответ формула площади плоской фигуры
            wrong_answers.append(
                f"$${latex(2 * pi * Integral(f * sqrt(1 + sqr_proiz), (x, a, b))).replace('{', TaskGenerator.n1).replace('}', TaskGenerator.n2)}$$")  # неправильный ответ формула площади поверхности вращения
            random.shuffle(wrong_answers)  # Перемешиваем неправильные ответы
            tasks.add((task, answer, tuple(wrong_answers)))
        return tasks

    @staticmethod
    def area_of_plane_figure(num_tasks):
        def fast_answer(alpha, beta):
            # Получение ответа быстрым способом не через интеграл
            pred_answer_1 = beta ** 3  # Предполагаемый числитель, у него меняется знак в зависимости от условия
            pred_answer_2 = 6 * alpha ** 2  # Знаменатель
            return abs(pred_answer_1), pred_answer_2

        def generate_alpha_beta():
            alpha = np.random.randint(-5, 6)
            while alpha == 0:  # Проверка на ноль
                alpha = np.random.randint(-5, 6)

            beta = np.random.randint(-5, 6)
            while beta == 0 or beta == alpha:  # Проверка на ноль и равенство alpha
                beta = np.random.randint(-5, 6)

            return alpha, beta

        def generate_wrong_answers(alpha, beta):  # создание неверных ответов близких к вычислениям
            wrong_answers = []
            numerator, denominator = fast_answer(alpha, beta)
            wrong_answers.append(f'{Fraction(numerator, abs(alpha * beta))}')
            numerator, denominator = fast_answer(beta,
                                                 alpha)  # меняем местами альфу и бету чтобы получить два новых неправильных ответа
            wrong_answers.append(f'{Fraction(numerator, denominator)}')  #
            wrong_answers.append(f'{Fraction(abs(alpha * beta), numerator)}')
            return wrong_answers

        def GetArrayWrongAnswers(basic_arr,
                                 additional_arr):  # вспомогательная функция для формирования массива из элементов массива неправильных ответов основного и дополнительного вопроса
            all_wrong_answers = []
            for elem in basic_arr:
                all_wrong_answers.append(f'{elem} (Основной вопрос)')
            for elem in additional_arr:
                all_wrong_answers.append(f'{elem} (Дополнительный вопрос)')
            return (all_wrong_answers)

        tasks = set()
        while len(tasks) < num_tasks:
            x = Symbol('x')
            alpha, beta = generate_alpha_beta()
            task = f"Вычислите площадь фигуры, ограниченной параболой $$y = {latex(alpha * x ** 2).replace('{', TaskGenerator.n1).replace('}', TaskGenerator.n2)}$$ и прямой $$y = {beta} x$$."
            numerator, denominator = fast_answer(alpha, beta)  # Правильный ответ
            correct_answer = f'{Fraction(numerator, denominator)}'  # сохраняем правильный ответ в виде строки
            wrong_answers = generate_wrong_answers(alpha, beta)  # 3 неправильных ответа
            if correct_answer in wrong_answers:  # Удаляем правильный ответ из неправильных, если он там есть
                wrong_answers.remove(correct_answer)
                wrong_answers.append(
                    f'{Fraction(denominator, numerator)}')  # добавляем перевёрнутую дробь в случае если удалили один ответ
            random.shuffle(wrong_answers)  # Перемешиваем неправильные ответы

            # Формируем уточняющий дополнительный вопрс на знание формулы, ответ и неправильные ответы для него
            add_question = f'Найдите пределы интегрирования a(левая граница) и b(правая граница) для решения данной задачи.'
            border_integral = Fraction(beta, alpha)  # вводим переменную чтобы она не считалась по несколько раз
            if border_integral < 0:  # меняем местами границы в пошаговом решении в зависимости от условия, а также ставим границы в дополнительном ответе в правильном порядке
                add_answer = f'{border_integral} и 0'  # ответ на дополнительный вопрос в зависимости от знака
                add_wrong_answers = [f'0 и {border_integral * (-1)}', f'{2 * border_integral} и 0',
                                     f'{border_integral} и {border_integral * (-1)}']  # Создаём неправильные ответы
            else:
                add_answer = f'0 и {border_integral}'
                add_wrong_answers = [f'{border_integral * (-1)} и 0', f'0 и {2 * border_integral}',
                                     f'{border_integral * (-1)} и {border_integral}']  # Создаём неправильные ответы
            if add_answer in add_wrong_answers:  # Удаляем правильный ответ из неправильных, если он там есть
                add_wrong_answers.remove(add_answer)
            random.shuffle(add_wrong_answers)  # Перемешиваем неправильные ответы
            answers = [f'{correct_answer} (Основной вопрос)', f'{add_answer} (Дополнительный вопрос)']
            all_wrong_answers = GetArrayWrongAnswers(wrong_answers, add_wrong_answers)
            tasks.add((task, add_question, tuple(answers), tuple(all_wrong_answers)))
        return tasks

    @staticmethod
    def volume_of_body(num_tasks):
        # Задаем переменную
        x = symbols('x')
        n1=TaskGenerator.n1
        n2=TaskGenerator.n2
        def GenerateBorder():  # функция, которая генерирует одну из границ интервала задачи
            bord_num = random.randint(-3,
                                      3)  # (числитель) выбирается рандомно целое число от -3 до 3 т.к по шаблону промежуток задан чтобы было в целых пи
            bord_denom = random.randint(1,
                                        3)  # (знаменатель) т.к наши функции синусы и косинусы добавляем деление чтобы получат пи/2 3пи/2 пи/3 табличные значения для синусов и косинусов
            bord = Rational(bord_num,
                            bord_denom) * pi  # сокращаем нашу дробь, используя Rational, чтобы она осталась обыкновенной, и домнажаем на пи
            return bord

        def GetAllTaskParametrs():  # функция, которая возрашает все параметры задачи:
            # a-левая граница, b-правая граница, alpha -множитель(скаляр) f(x), f(x)-рандомная функция из sin(x), cos(x), (sin(x))^2, (cos(x))^2]
            a = GenerateBorder()
            b = GenerateBorder()
            while a == b:
                a = GenerateBorder()
            # Проверяем, если a > b, меняем их местами
            if a > b:
                a, b = b, a
            # Выбор случайной функции f(x)
            functions = [sin(x), cos(x), sin(x) ** 2, cos(x) ** 2]
            f_x = random.choice(functions)
            # Генеррация множителя f(x) alpha
            alpha = 0
            while alpha == 0:  # Проверяем, чтобы alpha не было равно нулю
                alpha = random.randint(-10, 10)
            # Выражаем y через функцию f(x)
            y = alpha * f_x
            return y, a, b

        def GetArrayWrongAnswers(basic_arr,
                                 additional_arr):  # вспомогательная функция для формирования массива из элементов массива неправильных ответов основного и дополнительного вопроса
            all_wrong_answers = []
            for elem in basic_arr:
                all_wrong_answers.append(f'{elem} (Основной вопрос)')
            for elem in additional_arr:
                all_wrong_answers.append(f'{elem} (Дополнительный вопрос)')
            return (all_wrong_answers)

        tasks = set()
        while len(tasks) < num_tasks:
            # Получаем параметры
            y, a, b = GetAllTaskParametrs()
            # Создаём текст задания
            task = f"Дана функция $$y = {latex(y).replace('{', TaskGenerator.n1).replace('}', TaskGenerator.n2)}$$ на отрезке $$[{latex(a).replace('{', TaskGenerator.n1).replace('}', TaskGenerator.n2)}, {latex(b).replace('{', TaskGenerator.n1).replace('}', TaskGenerator.n2)}]$$. Используя определенный интеграл, найдите объем тела, образованного вращением относительно оси OX плоской фигуры"
            answer = integrate(y ** 2, x)  # это пред ответ -неопределённый интеграл
            answer = integrate(y ** 2, (x, a, b))  # определённый интеграл
            answer = abs(answer * pi)  # конечный ответ
            # создание неверных ответов, их фильтрация и перемешка
            wrong_answers = [a, b, integrate(y, (x, a, b)),
                             integrate(y ** 2, (x, a, b))]  # создаём список неправильных ответов
            if answer in wrong_answers:  # Удаляем правильный ответ из неправильных, если он там есть
                wrong_answers.remove(answer)
            for i in range(len(wrong_answers)):
                wrong_answers[
                    i] = f"$${latex(wrong_answers[i]).replace('{', TaskGenerator.n1).replace('}', TaskGenerator.n2)}$$"
            random.shuffle(wrong_answers)  # Перемешиваем неправильные ответы

            # Формируем уточняющий дополнительный вопрс на знание формулы, ответ и неправильные ответы для него
            add_question = f'Выберите верное подинтегральное выражение (Если в вашей формуле интеграл умножается на скаляр, то внесите скаляр под интеграл)'
            add_answer = (y ** 2) * pi
            length_formula = sqrt(1 + (diff(y,
                                            x)) ** 2)  # формула длины кривой в неправильных ответах будет частью других неправильных ответов
            add_wrong_answers = [y, length_formula, y * length_formula,
                                 y * length_formula * 2 * pi]  # здесь используется 4 остальных формулы из учебника
            if add_answer in add_wrong_answers:  # Удаляем правильный ответ из неправильных, если он там есть
                add_wrong_answers.remove(add_answer)
            for i in range(len(add_wrong_answers)):
                add_wrong_answers[
                    i] = f"$${latex(add_wrong_answers[i]).replace('{', TaskGenerator.n1).replace('}', TaskGenerator.n2)}$$"
            random.shuffle(add_wrong_answers)  # Перемешиваем неправильные ответы
            answers = [
                f"$${latex(answer).replace('{', TaskGenerator.n1).replace('}', TaskGenerator.n2)}$$ (Основной вопрос)",
                f"$${latex(add_answer).replace('{', n1).replace('}', n2)}$$ (Дополнительный вопрос)"]
            all_wrong_answers = GetArrayWrongAnswers(wrong_answers, add_wrong_answers)
            tasks.add((task, add_question, tuple(answers), tuple(all_wrong_answers)))
        return tasks



# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    application = QtWidgets.QApplication(sys.argv)
    program = TaskGenerator()
    program.show()
    application.exec_()
