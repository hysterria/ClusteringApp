from data_generator_functions import generate_dataset
from anonymizer_functions import *
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox
from pyclustering.cluster.cure import cure
from pyclustering.utils import distance_metric, type_metric
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from joblib import Parallel, delayed
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
from queue import Queue


class ClusteringApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Кластеризация данных")
        self.root.geometry("1400x900")

        self.df = None
        self.anonymized_df = None
        self.results = {}
        self.figures = []
        self.task_queue = Queue()
        self.running = False

        self.create_widgets()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def create_widgets(self):
        control_frame = ttk.Frame(self.root, padding=10)
        control_frame.pack(side=tk.LEFT, fill=tk.Y)

        ttk.Button(control_frame, text="1. Сгенерировать данные",
                   command=self.generate_data).pack(pady=5, fill=tk.X)
        ttk.Button(control_frame, text="2. Обезличить данные",
                   command=self.anonymize_data).pack(pady=5, fill=tk.X)
        ttk.Button(control_frame, text="3. Выполнить кластеризацию",
                   command=self.start_clustering).pack(pady=5, fill=tk.X)

        self.settings_frame = ttk.LabelFrame(control_frame, text="Настройки", padding=10)
        self.settings_frame.pack(pady=10, fill=tk.X)

        ttk.Label(self.settings_frame, text="Число кластеров:").grid(row=0, column=0)
        self.n_clusters = ttk.Entry(self.settings_frame)
        self.n_clusters.insert(0, "5")
        self.n_clusters.grid(row=0, column=1, padx=5)

        ttk.Label(self.settings_frame, text="Число признаков:").grid(row=1, column=0)
        self.n_features = ttk.Entry(self.settings_frame)
        self.n_features.insert(0, "5")
        self.n_features.grid(row=1, column=1, padx=5)

        self.results_frame = ttk.LabelFrame(control_frame, text="Результаты", padding=10)
        self.results_frame.pack(pady=10, fill=tk.X)

        self.results_text = tk.Text(self.results_frame, height=8, width=30,
                                    font=('Courier New', 10), wrap=tk.WORD)
        self.results_text.pack(fill=tk.BOTH)
        self.results_text.configure(state='disabled')

        self.viz_frame = ttk.Frame(self.root)
        self.viz_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.plot_frames = [[ttk.Frame(self.viz_frame) for _ in range(2)] for _ in range(2)]
        for i in range(2):
            for j in range(2):
                self.plot_frames[i][j].grid(row=i, column=j, padx=10, pady=10, sticky='nsew')

        self.viz_frame.grid_rowconfigure(0, weight=1)
        self.viz_frame.grid_rowconfigure(1, weight=1)
        self.viz_frame.grid_columnconfigure(0, weight=1)
        self.viz_frame.grid_columnconfigure(1, weight=1)

    def start_clustering(self):
        if self.running:
            messagebox.showwarning("Внимание", "Кластеризация уже выполняется!")
            return

        self.running = True
        threading.Thread(target=self.run_clustering, daemon=True).start()

    def run_clustering(self):
        try:
            self.clear_visualizations()
            self.results = {}
            distance_data = []

            n_clusters = int(self.n_clusters.get())
            n_features = int(self.n_features.get())

            X_original = self.prepare_data(self.df)
            X_anonymized = self.prepare_data(self.anonymized_df, is_anonymized=True)

            phases = [
                ('Исходные данные', X_original, False, 0),
                ('Отбор признаков (исходные)', X_original, True, 1),
                ('Обезличенные данные', X_anonymized, False, 2),
                ('Отбор признаков (обезличенные)', X_anonymized, True, 3)
            ]

            for name, data, use_feature_selection, pos in phases:
                if not self.running:
                    break

                if use_feature_selection:
                    selected_cols = self.select_features(data, n_features)
                    data = data[:, selected_cols]

                clusters = self.run_cure(data, n_clusters)
                compactness, distances = self.calculate_compactness(data, clusters)
                self.results[name] = compactness
                distance_data.append((distances, name))

                self.root.after(0, self.update_visualization, data, clusters, name, pos)

            if self.running:
                self.root.after(0, self.finalize_results, distance_data)

        except Exception as e:
            self.root.after(0, messagebox.showerror, "Ошибка", f"Ошибка кластеризации: {str(e)}")
        finally:
            self.running = False

    def select_features(self, data, n_features):
        def process_feature(i):
            clusters = self.run_cure(data[:, [i]], 3)
            return (i, self.calculate_compactness(data[:, [i]], clusters)[0])

        scores = Parallel(n_jobs=-1, prefer="threads")(delayed(process_feature)(i) for i in range(data.shape[1]))
        scores.sort(key=lambda x: x[1])
        return [x[0] for x in scores[:n_features]]

    def update_visualization(self, data, clusters, title, position):
        fig = plt.figure(figsize=(6, 4), dpi=100)
        ax = fig.add_subplot(111)

        if hasattr(data, 'toarray'):
            data = data.toarray()

        pca = PCA(n_components=2)
        data_2d = pca.fit_transform(data)

        colors = ['r', 'g', 'b', 'c', 'm', 'y']
        for i, cluster in enumerate(clusters):
            points = data_2d[cluster]
            ax.scatter(points[:, 0], points[:, 1],
                       s=30, color=colors[i % len(colors)],
                       label=f'Cluster {i + 1}', alpha=0.7)

        ax.set_title(f'{title}\nVariance: {sum(pca.explained_variance_ratio_):.2f}', fontsize=10)
        ax.set_xlabel('PC1', fontsize=8)
        ax.set_ylabel('PC2', fontsize=8)
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)

        canvas = FigureCanvasTkAgg(fig, master=self.plot_frames[position // 2][position % 2])
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.figures.append(fig)

    def finalize_results(self, distance_data):
        self.plot_distance_histograms(
            [d[0] for d in distance_data],
            [d[1] for d in distance_data]
        )
        self.update_results_display()
        messagebox.showinfo("Успех", "Кластеризация успешно завершена!")


    def clear_visualizations(self):
        for row in self.plot_frames:
            for frame in row:
                for widget in frame.winfo_children():
                    widget.destroy()
        self.figures.clear()

    def update_results_display(self):
        self.results_text.configure(state='normal')
        self.results_text.delete(1.0, tk.END)

        if not self.results:
            self.results_text.insert(tk.END, "Результаты отсутствуют\nВыполните кластеризацию")
            self.results_text.configure(state='disabled')
            return

        text = "Компактность:\n"
        text += "-" * 30 + "\n"
        for name, value in self.results.items():
            text += f"{name[:15]:<15} {value:>8.2f}\n"

        self.results_text.insert(tk.END, text)
        self.results_text.configure(state='disabled')

    def calculate_compactness(self, data, clusters):
        total = 0.0
        all_distances = []
        for cluster in clusters:
            if len(cluster) == 0:
                continue
            points = data[cluster]
            centroid = np.mean(points, axis=0)
            distances = pairwise_distances(points, [centroid], metric='sqeuclidean')
            total += np.sum(distances)
            all_distances.extend(distances.flatten())
        return (total / len(clusters)) if clusters else 0.0, all_distances

    def plot_distance_histograms(self, distances_list, titles):
        plt.figure(figsize=(12, 8))
        for i, (distances, title) in enumerate(zip(distances_list, titles)):
            plt.subplot(2, 2, i + 1)
            plt.hist(distances, bins=20, alpha=0.7, color=f'C{i}')
            plt.title(f'{title}\nСреднее расстояние: {np.mean(distances):.2f}')
            plt.xlabel('Расстояние до центроида')
            plt.ylabel('Количество точек')
            plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def visualize_clusters(self, data, clusters, title, position):
        fig = plt.figure(figsize=(6, 4), dpi=100)
        ax = fig.add_subplot(111)

        if hasattr(data, 'toarray'):
            data = data.toarray()

        pca = PCA(n_components=2)
        data_2d = pca.fit_transform(data)

        colors = ['r', 'g', 'b', 'c', 'm', 'y']
        for i, cluster in enumerate(clusters):
            points = data_2d[cluster]
            ax.scatter(points[:, 0], points[:, 1],
                       s=30, color=colors[i % len(colors)],
                       label=f'Cluster {i + 1}', alpha=0.7)

        ax.set_title(f'{title}\nVariance: {sum(pca.explained_variance_ratio_):.2f}', fontsize=10)
        ax.set_xlabel('PC1', fontsize=8)
        ax.set_ylabel('PC2', fontsize=8)
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)

        canvas = FigureCanvasTkAgg(fig, master=self.plot_frames[position // 2][position % 2])
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.figures.append(fig)

    def generate_data(self):
        try:
            generate_dataset(banks_prob=[40, 30, 20, 10],
                             payment_systems_prob=[50, 30, 20],
                             n_rows=1000)
            self.df = pd.read_csv('original_dataset.csv')
            messagebox.showinfo("Успех", "Данные успешно сгенерированы!")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка генерации данных: {str(e)}")

    def anonymize_data(self):
        if self.df is None:
            messagebox.showerror("Ошибка", "Сначала сгенерируйте данные!")
            return

        try:
            self.anonymized_df = self.anonymize_data_func(self.df)
            self.anonymized_df.to_csv('anonymized_dataset.csv', index=False)
            messagebox.showinfo("Успех", "Данные успешно обезличены!")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка обезличивания: {str(e)}")

    def anonymize_data_func(self, df):
        anonymized = []
        for _, row in df.iterrows():
            row_dict = row.to_dict()
            de_name(row_dict)
            de_age(row_dict)
            de_email(row_dict)
            de_phone(row_dict)
            de_passport(row_dict)
            de_marsh(row_dict)
            de_train_type(row_dict)
            de_wagon(row_dict)
            de_price(row_dict)
            de_card(row_dict)
            de_date(row_dict)
            anonymized.append(row_dict)
        return pd.DataFrame(anonymized)

    def prepare_data(self, df, is_anonymized=False):
        df_processed = df.copy()
        columns_to_drop = ['ФИО', 'Email', 'Телефон', 'Паспортные данные', 'Карта оплаты']
        df_processed = df_processed.drop(columns=[col for col in columns_to_drop if col in df_processed.columns])

        if is_anonymized:
            age_mapping = {'<18': 16, '18-24': 21, '25-34': 29, '35-49': 42, '50-64': 57, '65+': 70}
            if 'Возраст' in df_processed.columns:
                df_processed['Возраст'] = df_processed['Возраст'].map(age_mapping)

            price_mapping = {'< 1000': 500, '1000 - 2000': 1500, '2000 - 4000': 3000,
                             '4000 - 5000': 4500, '>=5000': 5500}
            if 'Стоимость' in df_processed.columns:
                df_processed['Стоимость'] = df_processed['Стоимость'].map(price_mapping)

            df_processed = df_processed.drop(columns=['Дата отъезда', 'Дата приезда'], errors='ignore')
        else:
            for date_col in ['Дата отъезда', 'Дата приезда']:
                if date_col in df_processed.columns:
                    df_processed[date_col] = pd.to_datetime(
                        df_processed[date_col], format='%Y-%m-%dT%H:%M', errors='coerce'
                    ).astype('int64') // 10 ** 9

        numeric_features = df_processed.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = df_processed.select_dtypes(include=['object']).columns.tolist()

        preprocessor = ColumnTransformer([
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

        processed_data = preprocessor.fit_transform(df_processed)
        if hasattr(processed_data, 'toarray'):
            processed_data = processed_data.toarray()
        return processed_data

    def run_cure(self, data, n_clusters):
        metric = distance_metric(type_metric.EUCLIDEAN_SQUARE)
        cure_instance = cure(
            data=data.tolist(),
            number_cluster=n_clusters,
            number_represent_points=5,
            compression=0.2
        )
        cure_instance.process()
        return cure_instance.get_clusters()


    def on_close(self):
        self.running = False
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = ClusteringApp(root)
    root.mainloop()