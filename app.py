import os
import tkinter as tk
from tkinter import messagebox, ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

# ==========================================================
# 1. MOTORES ESTATÍSTICOS (OMS + TANNER)
# ==========================================================
class WHOReference:
    def __init__(self):
        self.months = np.arange(0, 25, 1)
        self._lms_data = {
            "weight": np.array([
                [0.3487, 3.2322, 0.14171], [0.2598, 4.1873, 0.13783],
                [0.1982, 5.1275, 0.13110], [0.1554, 5.8458, 0.12563],
                [0.1245, 6.4237, 0.12163], [0.1008, 6.8985, 0.11867],
                [0.0818, 7.2970, 0.11646], [0.0661, 7.6416, 0.11477],
                [0.0526, 7.9489, 0.11346], [0.0408, 8.2304, 0.11244],
                [0.0302, 8.4947, 0.11164], [0.0206, 8.7468, 0.11102],
                [0.0117, 8.9897, 0.11054], [0.0034, 9.2259, 0.11019],
                [-0.0044, 9.4568, 0.10994], [-0.0117, 9.6834, 0.10978],
                [-0.0186, 9.9066, 0.10970], [-0.0252, 10.1271, 0.10969],
                [-0.0315, 10.3453, 0.10974], [-0.0375, 10.5617, 0.10985],
                [-0.0433, 10.7766, 0.11001], [-0.0489, 10.9902, 0.11021],
                [-0.0543, 11.2028, 0.11046], [-0.0596, 11.4146, 0.11074],
                [-0.0647, 11.6257, 0.11106]
            ]),
            "length": np.array([
                [1.0, 49.1477, 0.03790], [1.0, 53.6872, 0.03639],
                [1.0, 57.0673, 0.03548], [1.0, 59.8029, 0.03494],
                [1.0, 62.0899, 0.03463], [1.0, 64.0497, 0.03447],
                [1.0, 65.7311, 0.03441], [1.0, 67.2873, 0.03442],
                [1.0, 68.7498, 0.03448], [1.0, 70.1382, 0.03457],
                [1.0, 71.4646, 0.03469], [1.0, 72.7380, 0.03482],
                [1.0, 73.9654, 0.03497], [1.0, 75.1528, 0.03513],
                [1.0, 76.3049, 0.03530], [1.0, 77.4255, 0.03547],
                [1.0, 78.5178, 0.03565], [1.0, 79.5846, 0.03584],
                [1.0, 80.6282, 0.03603], [1.0, 81.6507, 0.03623],
                [1.0, 82.6537, 0.03643], [1.0, 83.6387, 0.03663],
                [1.0, 84.6072, 0.03684], [1.0, 85.5603, 0.03706],
                [1.0, 86.4988, 0.03727]
            ]),
            "head_circumference": np.array([
                [1.0, 33.8814, 0.03527], [1.0, 36.5369, 0.03264],
                [1.0, 38.2562, 0.03138], [1.0, 39.5284, 0.03067],
                [1.0, 40.5401, 0.03023], [1.0, 41.3854, 0.02996],
                [1.0, 42.1129, 0.02980], [1.0, 42.7509, 0.02972],
                [1.0, 43.3188, 0.02970], [1.0, 43.8302, 0.02972],
                [1.0, 44.2952, 0.02977], [1.0, 44.7214, 0.02984],
                [1.0, 45.1147, 0.02993], [1.0, 45.4800, 0.03003],
                [1.0, 45.8211, 0.03014], [1.0, 46.1411, 0.03025],
                [1.0, 46.4423, 0.03037], [1.0, 46.7269, 0.03050],
                [1.0, 46.9964, 0.03062], [1.0, 47.2522, 0.03075],
                [1.0, 47.4955, 0.03088], [1.0, 47.7275, 0.03101],
                [1.0, 47.9490, 0.03114], [1.0, 48.1607, 0.03127],
                [1.0, 48.3634, 0.03140]
            ]),
            "bmi": np.array([
                [0.6085, 13.3444, 0.09117], [0.3957, 14.8690, 0.08985],
                [0.1793, 16.0378, 0.08865], [0.0360, 16.5841, 0.08779],
                [-0.0526, 16.7865, 0.08722], [-0.1062, 16.8229, 0.08688],
                [-0.1384, 16.7628, 0.08671], [-0.1583, 16.6433, 0.08667],
                [-0.1705, 16.4883, 0.08674], [-0.1779, 16.3142, 0.08687],
                [-0.1824, 16.1321, 0.08705], [-0.1852, 15.9500, 0.08728],
                [-0.1868, 15.7733, 0.08754], [-0.1877, 15.6053, 0.08781],
                [-0.1882, 15.4479, 0.08810], [-0.1883, 15.3023, 0.08839],
                [-0.1884, 15.1689, 0.08868], [-0.1883, 15.0478, 0.08898],
                [-0.1882, 14.9388, 0.08927], [-0.1881, 14.8415, 0.08955],
                [-0.1881, 14.7551, 0.08983], [-0.1881, 14.6788, 0.09010],
                [-0.1882, 14.6119, 0.09036], [-0.1883, 14.5535, 0.09062],
                [-0.1885, 14.5029, 0.09087]
            ])
        }
        self.interpolators = {}
        for metric, data in self._lms_data.items():
            self.interpolators[metric] = {
                "L": interp1d(self.months, data[:, 0], bounds_error=False, fill_value="extrapolate"),
                "M": interp1d(self.months, data[:, 1], bounds_error=False, fill_value="extrapolate"),
                "S": interp1d(self.months, data[:, 2], bounds_error=False, fill_value="extrapolate")
            }

    def calculate_value(self, metric, age_months, z_score):
        interp = self.interpolators[metric]
        L = float(interp["L"](age_months))
        M = float(interp["M"](age_months))
        S = float(interp["S"](age_months))
        if abs(L) < 1e-4:
            return M * np.exp(S * z_score)
        return M * ((1.0 + L * S * z_score) ** (1.0 / L))

    def calculate_z_score(self, metric, age_months, measurement):
        if measurement <= 0:
            return 0.0
        interp = self.interpolators[metric]
        L = float(interp["L"](age_months))
        M = float(interp["M"](age_months))
        S = float(interp["S"](age_months))
        if abs(L) < 1e-4:
            return np.log(measurement / M) / S
        return (((measurement / M) ** L) - 1.0) / (L * S)

    def generate_curve(self, metric, z_score, ages_vec):
        return np.array([self.calculate_value(metric, a, z_score) for a in ages_vec])


class TargetHeightTanner:
    def __init__(self, alt_mae_cm, alt_pai_cm):
        self.alt_mae = float(alt_mae_cm)
        self.alt_pai = float(alt_pai_cm)
        self.target_height = (self.alt_mae + (self.alt_pai - 13.0)) / 2.0
        self.z_target = (self.target_height - 163.2) / 6.5
        self.z_target_inf = ((self.target_height - 5.0) - 163.2) / 6.5
        self.z_target_sup = ((self.target_height + 5.0) - 163.2) / 6.5

    def get_genetic_weight(self, age_months):
        return float(np.clip(0.15 + (0.60 * (age_months / 24.0)), 0.15, 0.75))


# ==========================================================
# 2. MODELO DE DADOS
# ==========================================================
class GrowthTracker:
    def __init__(self, filepath="consultas_bebe.csv"):
        self.filepath = filepath
        self.columns = ["Id", "Data", "Idade (meses)", "Peso (kg)", "Comprimento (cm)", "Perímetro Cefálico (cm)", "IMC (kg/m²)"]
        self.df = self._load()

    def _load(self):
        if os.path.exists(self.filepath):
            try:
                return pd.read_csv(self.filepath)
            except Exception:
                return pd.DataFrame(columns=self.columns)
        return pd.DataFrame(columns=self.columns)

    def save(self):
        self.df.to_csv(self.filepath, index=False)

    def add_record(self, data_str, idade, peso, comp, pc):
        imc = round(peso / ((comp / 100) ** 2), 2)
        novo_id = 1 if self.df.empty else int(self.df["Id"].max()) + 1
        registro = {
            "Id": novo_id,
            "Data": data_str,
            "Idade (meses)": float(idade),
            "Peso (kg)": float(peso),
            "Comprimento (cm)": float(comp),
            "Perímetro Cefálico (cm)": float(pc),
            "IMC (kg/m²)": imc
        }
        self.df = pd.concat([self.df, pd.DataFrame([registro])], ignore_index=True).sort_values("Idade (meses)").reset_index(drop=True)
        self.save()

    def delete_record(self, record_id):
        self.df = self.df[self.df["Id"] != int(record_id)].reset_index(drop=True)
        self.save()

    def load_seed_data(self):
        dados = [
            {"Id": 1, "Data": "2025-08-10", "Idade (meses)": 0.0, "Peso (kg)": 3.30, "Comprimento (cm)": 49.5, "Perímetro Cefálico (cm)": 34.2, "IMC (kg/m²)": 13.47},
            {"Id": 2, "Data": "2025-09-12", "Idade (meses)": 1.1, "Peso (kg)": 4.35, "Comprimento (cm)": 54.0, "Perímetro Cefálico (cm)": 36.8, "IMC (kg/m²)": 14.92},
            {"Id": 3, "Data": "2025-10-15", "Idade (meses)": 2.2, "Peso (kg)": 5.30, "Comprimento (cm)": 57.5, "Perímetro Cefálico (cm)": 38.5, "IMC (kg/m²)": 16.03},
            {"Id": 4, "Data": "2025-12-14", "Idade (meses)": 4.1, "Peso (kg)": 6.55, "Comprimento (cm)": 62.8, "Perímetro Cefálico (cm)": 40.7, "IMC (kg/m²)": 16.61},
            {"Id": 5, "Data": "2026-02-16", "Idade (meses)": 6.2, "Peso (kg)": 7.45, "Comprimento (cm)": 66.2, "Perímetro Cefálico (cm)": 42.4, "IMC (kg/m²)": 17.00}
        ]
        self.df = pd.DataFrame(dados)
        self.save()


# ==========================================================
# 3. INTERFACE GRÁFICA DESKTOP (TKINTER)
# ==========================================================
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Acompanhamento do Crescimento Pediátrico (OMS + Tanner)")
        self.geometry("1280x820")
        self.minsize(1050, 700)

        self.who = WHOReference()
        self.tracker = GrowthTracker()

        self._build_ui()
        self._update_table()
        self._render_charts()

    def _build_ui(self):
        # Container Principal Dividido (Esquerda: Controles/Tabela, Direita: Gráficos 2x2)
        main_paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        left_frame = ttk.Frame(main_paned, width=420)
        right_frame = ttk.Frame(main_paned)
        main_paned.add(left_frame, weight=0)
        main_paned.add(right_frame, weight=1)

        # 1. Bloco Pais (Tanner)
        frame_tanner = ttk.LabelFrame(left_frame, text="Potencial Genético Familiar (Tanner)", padding=8)
        frame_tanner.pack(fill=tk.X, pady=(0, 6))

        ttk.Label(frame_tanner, text="Alt. Mãe (cm):").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.ent_mae = ttk.Entry(frame_tanner, width=8)
        self.ent_mae.insert(0, "162.0")
        self.ent_mae.grid(row=0, column=1, padx=4, pady=2)

        ttk.Label(frame_tanner, text="Alt. Pai (cm):").grid(row=0, column=2, sticky=tk.W, pady=2)
        self.ent_pai = ttk.Entry(frame_tanner, width=8)
        self.ent_pai.insert(0, "178.0")
        self.ent_pai.grid(row=0, column=3, padx=4, pady=2)

        btn_update_tanner = ttk.Button(frame_tanner, text="Atualizar Alvo", command=self._render_charts)
        btn_update_tanner.grid(row=0, column=4, padx=6, pady=2)

        # 2. Bloco Formulário de Consulta
        frame_form = ttk.LabelFrame(left_frame, text="Adicionar Medição", padding=8)
        frame_form.pack(fill=tk.X, pady=(0, 6))

        ttk.Label(frame_form, text="Data (AAAA-MM-DD):").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.ent_data = ttk.Entry(frame_form, width=12)
        self.ent_data.insert(0, pd.to_datetime("today").strftime("%Y-%m-%d"))
        self.ent_data.grid(row=0, column=1, sticky=tk.W, pady=2)

        ttk.Label(frame_form, text="Idade (meses):").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.ent_idade = ttk.Entry(frame_form, width=12)
        self.ent_idade.insert(0, "7.5")
        self.ent_idade.grid(row=1, column=1, sticky=tk.W, pady=2)

        ttk.Label(frame_form, text="Peso (kg):").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.ent_peso = ttk.Entry(frame_form, width=12)
        self.ent_peso.insert(0, "8.0")
        self.ent_peso.grid(row=2, column=1, sticky=tk.W, pady=2)

        ttk.Label(frame_form, text="Altura (cm):").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.ent_comp = ttk.Entry(frame_form, width=12)
        self.ent_comp.insert(0, "68.0")
        self.ent_comp.grid(row=3, column=1, sticky=tk.W, pady=2)

        ttk.Label(frame_form, text="Perím. Cefálico (cm):").grid(row=4, column=0, sticky=tk.W, pady=2)
        self.ent_pc = ttk.Entry(frame_form, width=12)
        self.ent_pc.insert(0, "43.0")
        self.ent_pc.grid(row=4, column=1, sticky=tk.W, pady=2)

        btn_box = ttk.Frame(frame_form)
        btn_box.grid(row=5, column=0, columnspan=2, pady=(6, 0), sticky=tk.EW)
        ttk.Button(btn_box, text="Salvar Consulta", command=self._add_record).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        ttk.Button(btn_box, text="Dados de Teste", command=self._load_seed).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)

        # 3. Bloco Tabela de Consultas
        frame_table = ttk.LabelFrame(left_frame, text="Histórico de Consultas", padding=8)
        frame_table.pack(fill=tk.BOTH, expand=True, pady=(0, 6))

        self.tree = ttk.Treeview(frame_table, columns=("Id", "Data", "Meses", "Peso", "Comp", "PC", "IMC"), show="headings", height=8)
        col_widths = {"Id": 32, "Data": 80, "Meses": 52, "Peso": 52, "Comp": 52, "PC": 48, "IMC": 48}
        for col, width in col_widths.items():
            self.tree.heading(col, text=col)
            self.tree.column(col, width=width, anchor=tk.CENTER)

        scrollbar = ttk.Scrollbar(frame_table, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        btn_del = ttk.Button(left_frame, text="Excluir Consulta Selecionada", command=self._delete_selected)
        btn_del.pack(fill=tk.X, pady=(0, 4))

        # 4. Painel Gráfico 2x2
        self.fig, self.axes = plt.subplots(2, 2, figsize=(8, 6), dpi=96)
        self.fig.subplots_adjust(hspace=0.38, wspace=0.28, left=0.08, right=0.96, bottom=0.08, top=0.94)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _get_tanner_model(self):
        try:
            mae = float(self.ent_mae.get())
            pai = float(self.ent_pai.get())
            return TargetHeightTanner(mae, pai)
        except ValueError:
            return TargetHeightTanner(162.0, 178.0)

    def _add_record(self):
        try:
            data_str = self.ent_data.get().strip()
            idade = float(self.ent_idade.get())
            peso = float(self.ent_peso.get())
            comp = float(self.ent_comp.get())
            pc = float(self.ent_pc.get())

            if peso <= 0 or comp <= 0 or pc <= 0 or idade < 0:
                messagebox.showerror("Erro de Validação", "Valores devem ser positivos.")
                return

            self.tracker.add_record(data_str, idade, peso, comp, pc)
            self._update_table()
            self._render_charts()
        except ValueError:
            messagebox.showerror("Erro", "Insira valores numéricos válidos.")

    def _delete_selected(self):
        selected_item = self.tree.selection()
        if not selected_item:
            messagebox.showwarning("Aviso", "Selecione uma linha na tabela para excluir.")
            return
        item_vals = self.tree.item(selected_item[0], "values")
        rec_id = int(item_vals[0])
        self.tracker.delete_record(rec_id)
        self._update_table()
        self._render_charts()

    def _load_seed(self):
        self.tracker.load_seed_data()
        self._update_table()
        self._render_charts()

    def _update_table(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        if not self.tracker.df.empty:
            for _, row in self.tracker.df.iterrows():
                self.tree.insert("", tk.END, values=(
                    int(row["Id"]),
                    row["Data"],
                    f"{row['Idade (meses)']:.1f}",
                    f"{row['Peso (kg)']:.2f}",
                    f"{row['Comprimento (cm)']:.1f}",
                    f"{row['Perímetro Cefálico (cm)']:.1f}",
                    f"{row['IMC (kg/m²)']:.2f}"
                ))

    def _render_charts(self):
        tanner = self._get_tanner_model()
        tracker_df = self.tracker.df
        ages_grid = np.linspace(0, 24, 180)

        plots_config = [
            ("Peso (kg)", "weight", self.axes[0, 0]),
            ("Comprimento (cm)", "length", self.axes[0, 1]),
            ("Perímetro Cefálico (cm)", "head_circumference", self.axes[1, 0]),
            ("IMC (kg/m²)", "bmi", self.axes[1, 1])
        ]

        z_curves = [
            (-2.0, "#E06666", "--", 1.0, "-2 DP"),
            (-1.0, "#F6B26B", "--", 1.0, "-1 DP"),
            (0.0, "#93C47D", "-", 1.6, "Mediana"),
            (1.0, "#F6B26B", "--", 1.0, "+1 DP"),
            (2.0, "#E06666", "--", 1.0, "+2 DP")
        ]

        for label, metric_key, ax in plots_config:
            ax.clear()
            ax.grid(True, linestyle=":", alpha=0.6)

            # 1. Curvas OMS
            for z, color, style, width, z_lbl in z_curves:
                y = self.who.generate_curve(metric_key, z, ages_grid)
                ax.plot(ages_grid, y, color=color, linestyle=style, linewidth=width, alpha=0.55, label=z_lbl)

            # 2. Canal de Tanner no Comprimento
            if metric_key == "length":
                c_mid = self.who.generate_curve("length", tanner.z_target, ages_grid)
                c_inf = self.who.generate_curve("length", tanner.z_target_inf, ages_grid)
                c_sup = self.who.generate_curve("length", tanner.z_target_sup, ages_grid)
                ax.plot(ages_grid, c_mid, color="#8E44AD", linestyle="-.", linewidth=1.6, label=f"Alvo ({tanner.target_height:.1f}cm)")
                ax.fill_between(ages_grid, c_inf, c_sup, color="#8E44AD", alpha=0.10)

            # 3. Dados e Predições
            if not tracker_df.empty:
                idades = tracker_df["Idade (meses)"].values
                valores = tracker_df[label].values
                ax.plot(idades, valores, color="#0B5394", linewidth=2.2, marker="o", markersize=5, label="Bebê", zorder=6)

                ult_idade = idades[-1]
                if ult_idade < 24:
                    futuro_x = np.linspace(ult_idade, 24, 50)
                    if metric_key == "length":
                        z_atual = self.who.calculate_z_score("length", ult_idade, valores[-1])
                        y_pred = []
                        for t in futuro_x:
                            w = tanner.get_genetic_weight(t)
                            z_p = (1.0 - w) * z_atual + (w * tanner.z_target)
                            y_pred.append(self.who.calculate_value("length", t, z_p))
                        y_pred = np.array(y_pred)
                        y_sup, y_inf = y_pred + 1.2, y_pred - 1.2
                    else:
                        z_hist = [self.who.calculate_z_score(metric_key, idades[i], valores[i]) for i in range(len(idades))]
                        pesos = np.exp(np.linspace(-1, 0, len(z_hist)))
                        z_proj = np.average(z_hist, weights=pesos)
                        y_pred = np.array([self.who.calculate_value(metric_key, t, z_proj) for t in futuro_x])
                        y_sup = np.array([self.who.calculate_value(metric_key, t, z_proj + 0.5) for t in futuro_x])
                        y_inf = np.array([self.who.calculate_value(metric_key, t, z_proj - 0.5) for t in futuro_x])

                    ax.plot(futuro_x, y_pred, color="#0052CC", linestyle=":", linewidth=2.0, label="Previsão", zorder=5)
                    ax.fill_between(futuro_x, y_inf, y_sup, color="#0052CC", alpha=0.15)

            ax.set_title(label, fontsize=10, fontweight="bold")
            ax.set_xlabel("Meses", fontsize=8)
            ax.set_xlim(0, 24)
            ax.legend(loc="upper left", fontsize=6.5, frameon=True)

        self.canvas.draw()


if __name__ == "__main__":
    app = App()
    app.mainloop()