import sys
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QFileDialog,
    QLabel,
    QTextEdit,
    QSplitter,
    QComboBox, QTableWidget, QTableWidgetItem, QDialog,
)
from PyQt5.QtCore import Qt
from nltk.stem import SnowballStemmer
import re
import pymorphy2
from rapidfuzz.fuzz import ratio, partial_ratio
from rapidfuzz.distance import JaroWinkler
import matplotlib.pyplot as plt
import seaborn as sns
import openpyxl


class WordProcessingApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.words_set1 = set()
        self.words_set2 = set()
        self.morph = pymorphy2.MorphAnalyzer()
        self.stemmer = SnowballStemmer("russian")

    def lemmatize_word(self, word):
        return self.morph.parse(word)[0].normal_form

    def initUI(self):
        self.setWindowTitle("Language Prop")
        self.resize(1000, 600)

        main_layout = QVBoxLayout()

        button_layout = QHBoxLayout()
        self.load_button1 = QPushButton("Загрузить первый CSV файл")
        self.load_button1.clicked.connect(self.load_file1)
        button_layout.addWidget(self.load_button1)

        self.load_button2 = QPushButton("Загрузить второй CSV файл")
        self.load_button2.clicked.connect(self.load_file2)
        button_layout.addWidget(self.load_button2)

        self.metric_selector = QComboBox()
        self.metric_selector.addItems(["Ratcliff", "Levenshtein", "Jaro-Winkler"])
        button_layout.addWidget(self.metric_selector)

        self.process_button = QPushButton("Обработать слова")
        self.process_button.clicked.connect(self.process_words)
        button_layout.addWidget(self.process_button)

        self.export_button = QPushButton("Экспорт матрицы в Excel")
        self.export_button.clicked.connect(self.export_matrix_to_excel)
        button_layout.addWidget(self.export_button)

        self.clear_button = QPushButton("Очистить результаты")
        self.clear_button.clicked.connect(self.clear_results)
        button_layout.addWidget(self.clear_button)

        main_layout.addLayout(button_layout)

        splitter = QSplitter(Qt.Horizontal)

        self.words_area1 = QTextEdit()
        self.words_area1.setReadOnly(True)
        splitter.addWidget(self.create_labeled_area("Слова из файла 1", self.words_area1))

        self.words_area2 = QTextEdit()
        self.words_area2.setReadOnly(True)
        splitter.addWidget(self.create_labeled_area("Слова из файла 2", self.words_area2))

        self.result_area = QTextEdit()
        self.result_area.setReadOnly(True)
        splitter.addWidget(self.create_labeled_area("Сопоставление слов", self.result_area))

        self.matrix_area = QTextEdit()
        self.matrix_area.setReadOnly(True)
        splitter.addWidget(self.create_labeled_area("Матрица схожести", self.matrix_area))

        self.heatmap_button = QPushButton("Показать тепловую карту")
        self.heatmap_button.clicked.connect(self.show_heatmap)
        button_layout.addWidget(self.heatmap_button)

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

    def load_file1(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Выберите CSV файл", "", "CSV files (*.csv)")
        if file_path:
            self.words_set1 = self.load_words_from_csv(file_path)
            self.words_area1.setText("\n".join(self.words_set1))

    def load_file2(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Выберите CSV файл", "", "CSV files (*.csv)")
        if file_path:
            self.words_set2 = self.load_words_from_csv(file_path)
            self.words_area2.setText("\n".join(self.words_set2))

    def load_words_from_csv(self, file_path):
        df = pd.read_csv(file_path, header=None)
        words = set(df[0].tolist())
        cleaned_words = {self.clean_word(word) for word in words if self.clean_word(word)}
        lemmatized_words = {self.lemmatize_word(word) for word in cleaned_words}
        return set(list(lemmatized_words)[:100])

    def clean_word(self, word):
        word = re.sub(r"[^\w\s]", "", word)
        word = re.sub(r"\d+", "", word)
        return word.strip().lower()

    def process_words(self):
        selected_metric = self.metric_selector.currentText().lower()

        set1_list = list(self.words_set1)
        set2_list = list(self.words_set2)

        matrix = self.create_similarity_matrix(set1_list, set2_list, metric=selected_metric)

        self.display_matrix(matrix, set1_list, set2_list)

        sorted_matches = self.find_best_matches(matrix, set1_list, set2_list)

        self.last_matrix = matrix
        self.last_set1 = set1_list
        self.last_set2 = set2_list
        result = "\n".join(f"{w1} <-> {w2}: {score:.2f}" for w1, w2, score in sorted_matches)
        self.result_area.setText(result)

    def display_matrix(self, matrix, set1_list, set2_list):
        max_word_length = max(max(map(len, set1_list)), max(map(len, set2_list)), 8)
        col_width = max_word_length + 2

        header = " " * col_width + "".join(f"{word:<{col_width}}" for word in set2_list)
        matrix_str = header + "\n"

        for i, row in enumerate(matrix):
            row_str = f"{set1_list[i]:<{col_width}}" + "".join(f"{val:<{col_width}.2f}" for val in row)
            matrix_str += row_str + "\n"

        self.matrix_area.setText(matrix_str)

    def create_similarity_matrix(self, set1_list, set2_list, metric="ratcliff"):
        matrix = np.zeros((len(set1_list), len(set2_list)))

        for i, word1 in enumerate(set1_list):
            for j, word2 in enumerate(set2_list):
                if metric == "ratcliff":
                    matrix[i][j] = self.ratcliff_obershelp_coefficient(word1, word2)
                elif metric == "levenshtein":
                    matrix[i][j] = self.levenshtein_similarity(word1, word2)
                elif metric == "jaro_winkler":
                    matrix[i][j] = self.jaro_winkler_similarity(word1, word2)

        return matrix

    def show_matrix_window(self):
        if not hasattr(self, "last_matrix") or self.last_matrix is None:
            self.result_area.setText("Сначала обработайте слова для создания матрицы.")
            return

        matrix_dialog = QDialog(self)
        matrix_dialog.setWindowTitle("Матрица схожести")
        matrix_dialog.resize(800, 600)

        table = QTableWidget(matrix_dialog)
        table.setRowCount(len(self.last_set1))
        table.setColumnCount(len(self.last_set2))
        table.setHorizontalHeaderLabels(self.last_set2)
        table.setVerticalHeaderLabels(self.last_set1)

        for i, row in enumerate(self.last_matrix):
            for j, value in enumerate(row):
                table.setItem(i, j, QTableWidgetItem(f"{value:.2f}"))

        table.resizeColumnsToContents()
        table.resizeRowsToContents()
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        table.setParent(matrix_dialog)
        table.setGeometry(10, 10, 780, 580)

        matrix_dialog.exec_()

    def ratcliff_obershelp_coefficient(self, s1, s2):
        matches = sum(1 for a, b in zip(s1, s2) if a == b)
        return 2.0 * matches / (len(s1) + len(s2))

    def levenshtein_similarity(self, word1, word2):
        return ratio(word1, word2)

    def jaro_winkler_similarity(self, word1, word2):
        return JaroWinkler.similarity(word1, word2)

    def find_best_matches(self, matrix, set1_list, set2_list):
        matches = []
        visited_set2 = set()

        while len(matches) < min(len(set1_list), len(set2_list)):
            max_value = -1
            max_indices = (-1, -1)

            for i in range(len(set1_list)):
                for j in range(len(set2_list)):
                    if j in visited_set2:
                        continue
                    if matrix[i][j] > max_value:
                        max_value = matrix[i][j]
                        max_indices = (i, j)

            if max_indices == (-1, -1):
                break

            i, j = max_indices
            matches.append((set1_list[i], set2_list[j], matrix[i][j]))
            visited_set2.add(j)

        return matches

    def show_heatmap(self):
        if not hasattr(self, "last_matrix") or self.last_matrix is None:
            self.result_area.setText("Сначала обработайте слова для создания матрицы.")
            return

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            self.last_matrix,
            annot=False,
            xticklabels=self.last_set2,
            yticklabels=self.last_set1,
            cmap="coolwarm",
            cbar_kws={'label': 'Схожесть'}
        )
        plt.title("Тепловая карта схожести слов")
        plt.xlabel("Слова из файла 2")
        plt.ylabel("Слова из файла 1")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()

    def export_matrix_to_excel(self):
        if not hasattr(self, "last_matrix") or self.last_matrix is None:
            self.result_area.setText("Сначала обработайте слова для создания матрицы.")
            return

        file_path, _ = QFileDialog.getSaveFileName(self, "Сохранить файл", "", "Excel Files (*.xlsx)")

        if file_path:
            df = pd.DataFrame(
                self.last_matrix,
                index=self.last_set1,
                columns=self.last_set2
            )

            try:
                df.to_excel(file_path, sheet_name="Similarity Matrix")
                self.result_area.setText(f"Матрица успешно экспортирована в файл:\n{file_path}")
            except Exception as e:
                self.result_area.setText(f"Ошибка при экспорте: {str(e)}")

    def clear_results(self):
        self.words_area1.clear()
        self.words_area2.clear()
        self.result_area.clear()
        self.matrix_area.clear()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = WordProcessingApp()
    ex.show()
    sys.exit(app.exec_())
