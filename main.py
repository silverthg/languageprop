import sys
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QFileDialog, QTextEdit, QSplitter, QLabel
)
from PyQt5.QtCore import Qt
import matplotlib.pyplot as plt


class WordProcessingApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.words_set1 = []
        self.words_set2 = []
        self.last_matrix = None
        self.sorted_matrix = None
        self.row_indices = []
        self.dictionary = []
        self.sorted_indices = None  # Для хранения индексов после сортировки

        # Классификация согласных для русского и бурятского языков
        self.consonant_classes = {
            'Р': ['П', 'Б', 'В', 'Ф'],
            'Т': ['Т', 'Д'],
            'S': ['С', 'З', 'Ц', 'Ч', 'Ш', 'Щ'],
            'М': ['М'],
            'N': ['Н'],
            'R': ['Р', 'Л'],
            'К': ['К', 'Г', 'Х', 'Һ'],
            'Н': ['А', 'Е', 'Ё', 'И', 'О', 'У', 'Ы', 'Э', 'Ю', 'Я', 'Ө', 'Ү']  # Гласные
        }

    def initUI(self):
        self.setWindowTitle("Приложение для сравнения слов")
        self.resize(1200, 700)

        main_layout = QVBoxLayout()

        button_layout = QHBoxLayout()
        self.load_button1 = QPushButton("Загрузить список A")
        self.load_button1.clicked.connect(self.load_list1)
        button_layout.addWidget(self.load_button1)

        self.load_button2 = QPushButton("Загрузить список B")
        self.load_button2.clicked.connect(self.load_list2)
        button_layout.addWidget(self.load_button2)

        self.export_button = QPushButton("Экспорт матрицы в Excel")
        self.export_button.clicked.connect(self.export_to_excel)
        button_layout.addWidget(self.export_button)

        self.export_sorted_matrix_button = QPushButton("Экспорт отсортированной матрицы")
        self.export_sorted_matrix_button.clicked.connect(self.export_sorted_matrix_to_excel)
        button_layout.addWidget(self.export_sorted_matrix_button)

        self.export_dict_button = QPushButton("Экспорт словаря в Excel")
        self.export_dict_button.clicked.connect(self.export_dictionary_to_excel)
        button_layout.addWidget(self.export_dict_button)

        self.process_button = QPushButton("Обработать слова")
        self.process_button.clicked.connect(self.process_words)
        button_layout.addWidget(self.process_button)

        self.graph_button = QPushButton("Построить график средних")
        self.graph_button.clicked.connect(self.plot_averages_graph)
        button_layout.addWidget(self.graph_button)

        main_layout.addLayout(button_layout)

        splitter = QSplitter(Qt.Horizontal)

        self.words_area1 = QTextEdit()
        self.words_area1.setReadOnly(True)
        splitter.addWidget(self.create_labeled_area("Список A", self.words_area1))

        self.words_area2 = QTextEdit()
        self.words_area2.setReadOnly(True)
        splitter.addWidget(self.create_labeled_area("Список B", self.words_area2))

        self.matrix_area = QTextEdit()
        self.matrix_area.setReadOnly(True)
        splitter.addWidget(self.create_labeled_area("Результаты", self.matrix_area))

        self.dictionary_area = QTextEdit()
        self.dictionary_area.setReadOnly(True)
        splitter.addWidget(self.create_labeled_area("Словарь", self.dictionary_area))

        main_layout.addWidget(splitter)
        self.setLayout(main_layout)

    def create_labeled_area(self, label_text, text_edit):
        area_layout = QVBoxLayout()
        label = QLabel(label_text)
        area_layout.addWidget(label)
        area_layout.addWidget(text_edit)
        container = QWidget()
        container.setLayout(area_layout)
        return container

    def load_list1(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Загрузить список A", "", "CSV Files (*.csv)")
        if file_path:
            self.words_set1 = pd.read_csv(file_path, header=None)[0].tolist()
            self.words_area1.setText("\n".join(self.words_set1))

    def load_list2(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Загрузить список B", "", "CSV Files (*.csv)")
        if file_path:
            self.words_set2 = pd.read_csv(file_path, header=None)[0].tolist()
            self.words_area2.setText("\n".join(self.words_set2))

    def process_words(self):
        if not self.words_set1 or not self.words_set2:
            self.matrix_area.setText("Пожалуйста, загрузите оба списка.")
            return

        # Преобразуем слова перед сравнением
        transformed_set1 = [self.dolgopolsky_transform(word, transform_type=1) for word in self.words_set1]
        transformed_set2 = [self.dolgopolsky_transform(word, transform_type=1) for word in self.words_set2]

        self.last_matrix = self.create_similarity_matrix(transformed_set1, transformed_set2)

        # Сортировка строк матрицы по убыванию максимальных значений
        self.row_indices = np.argsort(-np.max(self.last_matrix, axis=1))
        self.sorted_matrix = self.last_matrix[self.row_indices]

        # Сортировка каждой строки по убыванию с сохранением индексов
        self.sorted_indices = np.argsort(-self.sorted_matrix, axis=1)
        self.sorted_matrix = np.take_along_axis(self.sorted_matrix, self.sorted_indices, axis=1)

        # Вычисление метрик
        avg_value = np.mean(self.last_matrix)
        row_maxes = np.max(self.last_matrix, axis=1)
        col_maxes = np.max(self.last_matrix, axis=0)
        avg_row_max = np.mean(row_maxes)
        avg_col_max = np.mean(col_maxes)
        var_matrix = np.var(self.last_matrix)
        var_row_max = np.var(row_maxes)
        var_col_max = np.var(col_maxes)

        avg_col_sorted = np.mean(self.sorted_matrix, axis=0)
        avg_col_sorted_text = "\n".join(
            f"Столбец {i + 1}: {value:.2f}" for i, value in enumerate(avg_col_sorted)
        )

        top_columns = self.sorted_matrix[:, :10]

        flat_indices = np.dstack(np.unravel_index(np.argsort(-top_columns.ravel()), top_columns.shape))[0]
        self.dictionary = [
            {
                "Слово A": self.words_set1[self.row_indices[i]],
                "Слово B": self.words_set2[self.sorted_indices[i, j]],
                "Коэффициент": self.sorted_matrix[i, j],
            }
            for i, j in flat_indices[:10]
        ]

        # Текстовое отображение метрик
        results_text = (
            f"Среднее по всему полю матрицы 〈ρ〉: {avg_value:.2f}\n"
            f"Максимальные значения по строкам: {', '.join(map(str, row_maxes))}\n"
            f"Максимальные значения по столбцам: {', '.join(map(str, col_maxes))}\n"
            f"Средние от максимумов по строкам 〈ρ_i^max〉: {avg_row_max:.2f}\n"
            f"Средние от максимумов по столбцам 〈ρ_j^max〉: {avg_col_max:.2f}\n"
            f"Дисперсия матрицы D(ρ_ij): {var_matrix:.2f}\n"
            f"Дисперсия максимумов по строкам D(ρ_i^max): {var_row_max:.2f}\n"
            f"Дисперсия максимумов по столбцам D(ρ_j^max): {var_col_max:.2f}\n"
            f"Средние по столбцам от отсортированной матрицы:\n{avg_col_sorted_text}"
        )
        self.matrix_area.setText(results_text)

        dictionary_text = "\n".join(
            f"{entry['Слово A']} - {entry['Слово B']}: {entry['Коэффициент']:.2f}" for entry in self.dictionary
        )
        self.dictionary_area.setText(dictionary_text)

    def create_similarity_matrix(self, list1, list2):
        matrix = np.zeros((len(list1), len(list2)))
        for i, word1 in enumerate(list1):
            for j, word2 in enumerate(list2):
                matrix[i, j] = self.ratcliff_similarity(word1, word2)
        return matrix

    def ratcliff_similarity(self, word1, word2):
        # Алгоритм поиска наибольшей общей подстроки без учета порядка букв
        common = []
        chars1 = list(word1)
        chars2 = list(word2)
        for char in chars1:
            if char in chars2:
                common.append(char)
                chars2.remove(char)
        return 2 * len(common) / (len(word1) + len(word2))

    def dolgopolsky_transform(self, word, transform_type):
        """
        Преобразование Долгопольского для слова.
        :param word: Слово для преобразования.
        :param transform_type: Тип преобразования (1, 2 или 3).
        :return: Преобразованное слово.
        """
        if transform_type == 1:
            # Преобразование D1: учитывает гласные и согласные
            transformed_word = []
            for char in word:
                if char.upper() in self.consonant_classes['Н']:
                    transformed_word.append('H')  # Гласные заменяем на H
                else:
                    for cls, letters in self.consonant_classes.items():
                        if char.upper() in letters:
                            transformed_word.append(cls)
                            break
            return ''.join(transformed_word)

        elif transform_type == 2:
            # Преобразование D2: учитывает только согласные, гласные удаляются
            transformed_word = []
            for char in word:
                if char.upper() not in self.consonant_classes['Н']:
                    for cls, letters in self.consonant_classes.items():
                        if char.upper() in letters:
                            transformed_word.append(cls)
                            break
            return ''.join(transformed_word)

        elif transform_type == 3:
            # Преобразование D3: учитывает только первые две согласные
            transformed_word = []
            consonant_count = 0
            for char in word:
                if char.upper() not in self.consonant_classes['Н']:
                    for cls, letters in self.consonant_classes.items():
                        if char.upper() in letters:
                            transformed_word.append(cls)
                            consonant_count += 1
                            if consonant_count == 2:
                                return ''.join(transformed_word)
                    break
            return ''.join(transformed_word)

        else:
            raise ValueError("Неверный тип преобразования. Допустимые значения: 1, 2, 3.")

    def export_to_excel(self):
        if self.last_matrix is None:
            self.matrix_area.setText("Нет данных для экспорта. Пожалуйста, обработайте слова.")
            return

        file_path, _ = QFileDialog.getSaveFileName(self, "Экспорт матрицы в Excel", "", "Excel Files (*.xlsx)")
        if file_path:
            # Формируем строки и столбцы в формате "слово:классы"
            index_labels = [f"{word}:{self.dolgopolsky_transform(word, transform_type=1)}" for word in self.words_set1]
            column_labels = [f"{word}:{self.dolgopolsky_transform(word, transform_type=1)}" for word in self.words_set2]

            df = pd.DataFrame(self.last_matrix,
                              index=index_labels,
                              columns=column_labels)
            df.to_excel(file_path, sheet_name="Матрица")

    def export_sorted_matrix_to_excel(self):
        if self.sorted_matrix is None:
            self.matrix_area.setText("Нет данных для экспорта. Пожалуйста, обработайте слова.")
            return

        file_path, _ = QFileDialog.getSaveFileName(self, "Экспорт отсортированной матрицы в Excel", "",
                                                   "Excel Files (*.xlsx)")
        if file_path:
            # Формируем строки и столбцы в формате "слово:классы"
            index_labels = [f"{self.words_set1[i]}:{self.dolgopolsky_transform(self.words_set1[i], transform_type=1)}"
                           for i in self.row_indices]
            column_labels = [f"Столбец {j + 1}" for j in range(self.sorted_matrix.shape[1])]

            sorted_df = pd.DataFrame(self.sorted_matrix,
                                     index=index_labels,
                                     columns=column_labels)
            sorted_df.to_excel(file_path, sheet_name="Отсортированная матрица")

    def export_dictionary_to_excel(self):
        if not self.dictionary:
            self.matrix_area.setText("Нет данных для экспорта словаря. Пожалуйста, обработайте слова.")
            return

        file_path, _ = QFileDialog.getSaveFileName(self, "Экспорт словаря в Excel", "", "Excel Files (*.xlsx)")
        if file_path:
            dict_df = pd.DataFrame(self.dictionary)
            dict_df.to_excel(file_path, sheet_name="Словарь", index=False)

    def plot_averages_graph(self):
        if self.sorted_matrix is None:
            self.matrix_area.setText("Нет данных для построения графика. Пожалуйста, обработайте слова.")
            return

        col_means = np.mean(self.sorted_matrix, axis=0)
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(col_means) + 1), col_means, marker="o")
        plt.title("Средние значения по столбцам отсортированной матрицы")
        plt.xlabel("Номер столбца")
        plt.ylabel("Среднее значение")
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = WordProcessingApp()
    ex.show()
    sys.exit(app.exec_())