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

        self.last_matrix = self.create_similarity_matrix(self.words_set1, self.words_set2)

        # Вычисление дополнительных метрик
        avg_value = np.mean(self.last_matrix)
        row_maxes = np.max(self.last_matrix, axis=1)
        col_maxes = np.max(self.last_matrix, axis=0)
        avg_row_max = np.mean(row_maxes)
        avg_col_max = np.mean(col_maxes)
        var_row_max = np.var(row_maxes)
        var_col_max = np.var(col_maxes)

        row_variances = np.var(self.last_matrix, axis=1)
        col_variances = np.var(self.last_matrix, axis=0)

        # Сортировка строк матрицы
        self.row_indices = np.argsort(-row_maxes)
        self.sorted_matrix = self.last_matrix[self.row_indices]

        # Создание словаря на основе 10 наиболее похожих пар слов
        flat_indices = np.dstack(np.unravel_index(np.argsort(-self.last_matrix.ravel()), self.last_matrix.shape))[0]
        self.dictionary = [
            (self.words_set1[i], self.words_set2[j], self.last_matrix[i, j])
            for i, j in flat_indices[:10]
        ]

        # Отображение результатов
        metrics_text = (f"Среднее значение по матрице: {avg_value:.2f}\n"
                        f"Среднее максимумов по строкам: {avg_row_max:.2f}\n"
                        f"Среднее максимумов по столбцам: {avg_col_max:.2f}\n"
                        f"Дисперсия максимумов по строкам: {var_row_max:.2f}\n"
                        f"Дисперсия максимумов по столбцам: {var_col_max:.2f}\n"
                        f"Дисперсия строк матрицы: {np.mean(row_variances):.2f}\n"
                        f"Дисперсия столбцов матрицы: {np.mean(col_variances):.2f}")
        self.matrix_area.setText(metrics_text)

        dictionary_text = "\n".join(f"{word1} {word2} {coef:.2f}" for word1, word2, coef in self.dictionary)
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

    def export_to_excel(self):
        if self.last_matrix is None:
            self.matrix_area.setText("Нет данных для экспорта. Пожалуйста, обработайте слова.")
            return

        file_path, _ = QFileDialog.getSaveFileName(self, "Экспорт матрицы в Excel", "", "Excel Files (*.xlsx)")
        if file_path:
            df = pd.DataFrame(self.last_matrix,
                              index=self.words_set1,
                              columns=self.words_set2)
            df.to_excel(file_path, sheet_name="Матрица")

    def export_dictionary_to_excel(self):
        if not self.dictionary:
            self.matrix_area.setText("Нет данных для экспорта словаря. Пожалуйста, обработайте слова.")
            return

        file_path, _ = QFileDialog.getSaveFileName(self, "Экспорт словаря в Excel", "", "Excel Files (*.xlsx)")
        if file_path:
            dict_df = pd.DataFrame(self.dictionary, columns=["Слово A", "Слово B", "Коэффициент"])
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
