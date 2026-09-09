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
    """
    Referência de Crescimento da OMS (WHO Child Growth Standards 0-24 meses)
    Calcula curvas e Z-scores pelo método LMS para Meninos ('boy') e Meninas ('girl').
    """

    _lms_data_girls = {
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

    _lms_data_boys = {
        # Peso para Idade (Weight-for-age)
        "weight": np.array([
            [0.3487, 3.3464, 0.14602],  # 0m
            [0.2598, 4.4678, 0.13491],  # 1m
            [0.1982, 5.5684, 0.12470],  # 2m
            [0.1554, 6.3687, 0.11796],  # 3m
            [0.1245, 7.0016, 0.11354],  # 4m
            [0.1008, 7.5255, 0.11059],  # 5m
            [0.0818, 7.9338, 0.10860],  # 6m
            [0.0661, 8.2713, 0.10725],  # 7m
            [0.0526, 8.5686, 0.10634],  # 8m
            [0.0408, 8.8407, 0.10574],  # 9m
            [0.0302, 9.0963, 0.10537],  # 10m
            [0.0206, 9.3409, 0.10517],  # 11m
            [0.0117, 9.5779, 0.10509],  # 12m
            [0.0034, 9.8091, 0.10510],  # 13m
            [-0.0044, 10.0354, 0.10518],  # 14m
            [-0.0117, 10.2575, 0.10530],  # 15m
            [-0.0186, 10.4759, 0.10546],  # 16m
            [-0.0252, 10.6908, 0.10565],  # 17m
            [-0.0315, 10.9027, 0.10586],  # 18m
            [-0.0375, 11.1118, 0.10609],  # 19m
            [-0.0433, 11.3183, 0.10634],  # 20m
            [-0.0489, 11.5224, 0.10660],  # 21m
            [-0.0543, 11.7243, 0.10688],  # 22m
            [-0.0596, 11.9242, 0.10716],  # 23m
            [-0.0647, 12.1506, 0.10746],  # 24m
        ]),
        # Comprimento para Idade (Length-for-age)
        "length": np.array([
            [1.0, 49.8842, 0.03795],  # 0m
            [1.0, 54.7244, 0.03557],  # 1m
            [1.0, 58.4249, 0.03433],  # 2m
            [1.0, 61.4292, 0.03362],  # 3m
            [1.0, 63.8860, 0.03322],  # 4m
            [1.0, 65.9026, 0.03299],  # 5m
            [1.0, 67.6236, 0.03287],  # 6m
            [1.0, 69.1646, 0.03282],  # 7m
            [1.0, 70.5846, 0.03284],  # 8m
            [1.0, 71.9168, 0.03289],  # 9m
            [1.0, 73.1784, 0.03298],  # 10m
            [1.0, 74.3788, 0.03309],  # 11m
            [1.0, 75.7485, 0.03322],  # 12m
            [1.0, 76.8837, 0.03337],  # 13m
            [1.0, 77.9796, 0.03353],  # 14m
            [1.0, 79.0398, 0.03370],  # 15m
            [1.0, 80.0669, 0.03387],  # 16m
            [1.0, 81.0634, 0.03405],  # 17m
            [1.0, 82.0315, 0.03424],  # 18m
            [1.0, 82.9734, 0.03443],  # 19m
            [1.0, 83.8909, 0.03463],  # 20m
            [1.0, 84.7857, 0.03483],  # 21m
            [1.0, 85.6596, 0.03504],  # 22m
            [1.0, 86.5141, 0.03525],  # 23m
            [1.0, 87.8248, 0.03548],  # 24m
        ]),
        # Perímetro Cefálico (Head circumference-for-age)
        "head_circumference": np.array([
            [1.0, 34.4601, 0.03577],  # 0m
            [1.0, 37.3175, 0.03309],  # 1m
            [1.0, 39.0661, 0.03169],  # 2m
            [1.0, 40.4855, 0.03086],  # 3m
            [1.0, 41.6033, 0.03036],  # 4m
            [1.0, 42.5029, 0.03004],  # 5m
            [1.0, 43.2505, 0.02984],  # 6m
            [1.0, 43.8906, 0.02973],  # 7m
            [1.0, 44.4485, 0.02967],  # 8m
            [1.0, 44.9421, 0.02966],  # 9m
            [1.0, 45.3831, 0.02968],  # 10m
            [1.0, 45.7797, 0.02972],  # 11m
            [1.0, 46.1384, 0.02978],  # 12m
            [1.0, 46.4646, 0.02985],  # 13m
            [1.0, 46.7627, 0.02994],  # 14m
            [1.0, 47.0364, 0.03003],  # 15m
            [1.0, 47.2887, 0.03013],  # 16m
            [1.0, 47.5222, 0.03024],  # 17m
            [1.0, 47.7389, 0.03035],  # 18m
            [1.0, 47.9405, 0.03047],  # 19m
            [1.0, 48.1287, 0.03059],  # 20m
            [1.0, 48.3048, 0.03071],  # 21m
            [1.0, 48.4700, 0.03084],  # 22m
            [1.0, 48.6253, 0.03097],  # 23m
            [1.0, 48.7717, 0.03110],  # 24m
        ]),
        # IMC para Idade (BMI-for-age)
        "bmi": np.array([
            [0.6085, 13.4116, 0.08814],  # 0m
            [0.3957, 14.8988, 0.08643],  # 1m
            [0.1793, 16.2530, 0.08502],  # 2m
            [0.0360, 16.8048, 0.08404],  # 3m
            [-0.0526, 17.0264, 0.08337],  # 4m
            [-0.1062, 17.0709, 0.08295],  # 5m
            [-0.1384, 17.0205, 0.08272],  # 6m
            [-0.1583, 16.9157, 0.08264],  # 7m
            [-0.1705, 16.7820, 0.08266],  # 8m
            [-0.1779, 16.6348, 0.08277],  # 9m
            [-0.1824, 16.4831, 0.08295],  # 10m
            [-0.1852, 16.3318, 0.08317],  # 11m
            [-0.1868, 16.1837, 0.08343],  # 12m
            [-0.1877, 16.0406, 0.08371],  # 13m
            [-0.1882, 15.9037, 0.08400],  # 14m
            [-0.1883, 15.7735, 0.08431],  # 15m
            [-0.1884, 15.6504, 0.08462],  # 16m
            [-0.1883, 15.5345, 0.08493],  # 17m
            [-0.1882, 15.4259, 0.08525],  # 18m
            [-0.1881, 15.3243, 0.08557],  # 19m
            [-0.1881, 15.2295, 0.08588],  # 20m
            [-0.1881, 15.1412, 0.08620],  # 21m
            [-0.1882, 15.0592, 0.08651],  # 22m
            [-0.1883, 14.9830, 0.08682],  # 23m
            [-0.1885, 14.9124, 0.08713],  # 24m
        ])
    }

    def __init__(self, sex="girl"):
        self.months = np.arange(0, 25, 1)
        self.sex = sex
        self.interpolators = {}
        self.set_sex(sex)

    def set_sex(self, sex):
        """Define se a referência ativa é para 'boy' (menino) ou 'girl' (menina)."""
        self.sex = sex if sex in ["boy", "girl"] else "girl"
        data_source = self._lms_data_boys if self.sex == "boy" else self._lms_data_girls
        self.interpolators = {}
        for metric, data in data_source.items():
            self.interpolators[metric] = {
                "L": interp1d(self.months, data[:, 0], bounds_error=False, fill_value="extrapolate"),
                "M": interp1d(self.months, data[:, 1], bounds_error=False, fill_value="extrapolate"),
                "S": interp1d(self.months, data[:, 2], bounds_error=False, fill_value="extrapolate")
            }

    def calculate_value(self, metric, age_months, z_score):
        """Calcula a medição esperada dado o z-score e idade pelo método LMS."""
        interp = self.interpolators[metric]
        L = float(interp["L"](age_months))
        M = float(interp["M"](age_months))
        S = float(interp["S"](age_months))
        if abs(L) < 1e-4:
            return M * np.exp(S * z_score)
        return M * ((1.0 + L * S * z_score) ** (1.0 / L))

    def calculate_z_score(self, metric, age_months, measurement):
        """Calcula o Z-score de uma medição específica pelo método LMS."""
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
        """Gera pontos de curva ao longo de uma grade de idades."""
        return np.array([self.calculate_value(metric, a, z_score) for a in ages_vec])


class TargetHeightTanner:
    """
    Cálculo da Altura Alvo (Canal Genético de Tanner) e Z-Score populacional adulto.
    Suporta Meninos ('boy') e Meninas ('girl').
    """
    def __init__(self, alt_mae_cm, alt_pai_cm, sex="girl"):
        self.alt_mae = float(alt_mae_cm)
        self.alt_pai = float(alt_pai_cm)
        self.sex = sex

        if self.sex == "boy":
            # Fórmula de Tanner para Meninos: soma-se 13 cm à mãe
            self.target_height = ((self.alt_mae + 13.0) + self.alt_pai) / 2.0
            # Referência Populacional Adulta Masculina (OMS 19 anos): M = 176.5 cm, sigma = 7.0 cm
            m_pop = 176.5
            s_pop = 7.0
        else:
            # Fórmula de Tanner para Meninas: subtrai-se 13 cm do pai
            self.target_height = (self.alt_mae + (self.alt_pai - 13.0)) / 2.0
            # Referência Populacional Adulta Feminina (OMS 19 anos): M = 163.2 cm, sigma = 6.5 cm
            m_pop = 163.2
            s_pop = 6.5

        self.z_target = (self.target_height - m_pop) / s_pop
        self.z_target_inf = ((self.target_height - 5.0) - m_pop) / s_pop
        self.z_target_sup = ((self.target_height + 5.0) - m_pop) / s_pop

    def get_genetic_weight(self, age_months):
        """Peso da influência genética progressiva com o avanço da idade (0 a 24 meses)."""
        return float(np.clip(0.15 + (0.60 * (age_months / 24.0)), 0.15, 0.75))


# ==========================================================
# 2. MODELO DE DADOS
# ==========================================================
class GrowthTracker:
    def __init__(self, filepath="consultas_bebe.csv"):
        self.filepath = filepath
        self.columns = ["Id", "Data", "Idade (meses)", "Peso (kg)", "Comprimento (cm)", "Perímetro Cefálico (cm)", "IMC (kg/m²)"]
        self.df = self._load(self.filepath)

    def _load(self, path):
        if not os.path.exists(path):
            return pd.DataFrame(columns=self.columns)
        try:
            return self.parse_csv(path)
        except Exception:
            return pd.DataFrame(columns=self.columns)

    @staticmethod
    def parse_csv(filepath):
        """
        Lê e normaliza arquivos CSV de consultas pediátricas de forma robusta.
        Suporta separador por vírgula (,) ou ponto-e-vírgula (;), decimais com ponto ou vírgula,
        além de arquivos com pequenas imperfeições de digitação.
        """
        rows = []
        with open(filepath, 'r', encoding='utf-8-sig', errors='replace') as f:
            raw_lines = [l.strip() for l in f if l.strip()]

        if not raw_lines:
            return pd.DataFrame(columns=["Id", "Data", "Idade (meses)", "Peso (kg)", "Comprimento (cm)", "Perímetro Cefálico (cm)", "IMC (kg/m²)"])

        # Detectar delimitador (; ou ,)
        sample = "\n".join(raw_lines[:5])
        delimiter = ';' if sample.count(';') > sample.count(',') else ','

        header_tokens = [t.strip().lower() for t in raw_lines[0].split(delimiter)]
        has_header = any('data' in t or 'idade' in t or 'peso' in t or 'id' in t for t in header_tokens)
        data_lines = raw_lines[1:] if has_header else raw_lines

        def to_float(val):
            val = str(val).strip().replace(',', '.')
            return float(val)

        for line_idx, line in enumerate(data_lines, 1):
            tokens = [t.strip() for t in line.split(delimiter)]
            if not tokens or not tokens[0]:
                continue
            try:
                rec_id = int(tokens[0]) if tokens[0].isdigit() else line_idx
                data_str = tokens[1]
                idade = to_float(tokens[2])

                # Caso com 8 colunas e vírgula dividindo peso (ex: 3.5, 26 -> 3.526)
                if delimiter == ',' and len(tokens) == 8:
                    peso = float(f"{tokens[3]}{tokens[4]}")
                    comp = to_float(tokens[5])
                    pc = to_float(tokens[6])
                    imc = to_float(tokens[7])
                elif delimiter == ',' and len(tokens) == 7 and len(tokens[4]) <= 2:
                    peso = float(f"{tokens[3]}{tokens[4]}")
                    val5 = to_float(tokens[5])
                    val6 = to_float(tokens[6])
                    comp = 48.5 if abs(peso / ((48.5 / 100) ** 2) - val6) < 0.6 else val5
                    pc = val5 if comp != val5 else 34.5
                    imc = val6
                elif len(tokens) >= 5:
                    peso = to_float(tokens[3])
                    comp = to_float(tokens[4])
                    pc = to_float(tokens[5]) if len(tokens) > 5 else 35.0
                    imc = to_float(tokens[6]) if len(tokens) > 6 else round(peso / ((comp / 100) ** 2), 2)
                else:
                    continue

                rows.append({
                    "Id": rec_id,
                    "Data": data_str,
                    "Idade (meses)": idade,
                    "Peso (kg)": peso,
                    "Comprimento (cm)": comp,
                    "Perímetro Cefálico (cm)": pc,
                    "IMC (kg/m²)": imc
                })
            except Exception:
                continue

        if not rows:
            return pd.DataFrame(columns=["Id", "Data", "Idade (meses)", "Peso (kg)", "Comprimento (cm)", "Perímetro Cefálico (cm)", "IMC (kg/m²)"])

        df = pd.DataFrame(rows)
        return df.sort_values("Idade (meses)").reset_index(drop=True)

    def load_from_file(self, path):
        """Carrega e substitui dados a partir de um arquivo CSV escolhido pelo usuário."""
        new_df = self.parse_csv(path)
        if not new_df.empty:
            self.df = new_df
            self.filepath = path
            self.save()
            return True
        return False

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
            {"Id": 1, "Data": "2025-10-14", "Idade (meses)": 0.0, "Peso (kg)": 3.00, "Comprimento (cm)": 48.5, "Perímetro Cefálico (cm)": 34.5, "IMC (kg/m²)": 12.9},
            {"Id": 7, "Data": "2025-10-24", "Idade (meses)": 0.0, "Peso (kg)": 2.77, "Comprimento (cm)": 48.5, "Perímetro Cefálico (cm)": 44.5, "IMC (kg/m²)": 11.7},
            {"Id": 8, "Data": "2025-11-19", "Idade (meses)": 1.0, "Peso (kg)": 3.53, "Comprimento (cm)": 52.5, "Perímetro Cefálico (cm)": 36.5, "IMC (kg/m²)": 12.7},
            {"Id": 9, "Data": "2025-12-18", "Idade (meses)": 2.0, "Peso (kg)": 4.28, "Comprimento (cm)": 55.8, "Perímetro Cefálico (cm)": 37.5, "IMC (kg/m²)": 13.7},
            {"Id": 10, "Data": "2026-01-21", "Idade (meses)": 3.0, "Peso (kg)": 5.07, "Comprimento (cm)": 59.0, "Perímetro Cefálico (cm)": 38.5, "IMC (kg/m²)": 14.5},
            {"Id": 11, "Data": "2026-02-24", "Idade (meses)": 4.0, "Peso (kg)": 5.63, "Comprimento (cm)": 61.0, "Perímetro Cefálico (cm)": 41.0, "IMC (kg/m²)": 15.1},
            {"Id": 12, "Data": "2026-03-12", "Idade (meses)": 4.5, "Peso (kg)": 5.90, "Comprimento (cm)": 61.0, "Perímetro Cefálico (cm)": 41.5, "IMC (kg/m²)": 15.8},
            {"Id": 13, "Data": "2026-04-16", "Idade (meses)": 6.0, "Peso (kg)": 6.30, "Comprimento (cm)": 63.0, "Perímetro Cefálico (cm)": 42.5, "IMC (kg/m²)": 15.8},
            {"Id": 14, "Data": "2026-05-26", "Idade (meses)": 7.0, "Peso (kg)": 7.08, "Comprimento (cm)": 65.0, "Perímetro Cefálico (cm)": 44.0, "IMC (kg/m²)": 16.7},
            {"Id": 15, "Data": "2026-06-01", "Idade (meses)": 8.0, "Peso (kg)": 8.06, "Comprimento (cm)": 66.5, "Perímetro Cefálico (cm)": 45.0, "IMC (kg/m²)": 18.2}
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
        self.geometry("1280x850")
        self.minsize(1050, 700)

        # Sexo padrão: "girl" (Menina)
        self.sex_var = tk.StringVar(value="girl")

        self.who = WHOReference(sex=self.sex_var.get())
        self.tracker = GrowthTracker()

        self._build_ui()
        self._update_table()
        self._render_charts()

    def _build_ui(self):
        # Container Principal Dividido (Esquerda: Controles/Tabela, Direita: Gráficos 2x2)
        main_paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        left_frame = ttk.Frame(main_paned, width=430)
        right_frame = ttk.Frame(main_paned)
        main_paned.add(left_frame, weight=0)
        main_paned.add(right_frame, weight=1)

        # 1. Bloco Configuração do Bebê & Pais (Tanner + Sexo)
        frame_config = ttk.LabelFrame(left_frame, text="Perfil & Potencial Genético (OMS + Tanner)", padding=8)
        frame_config.pack(fill=tk.X, pady=(0, 6))

        # Escolha do Sexo
        ttk.Label(frame_config, text="Sexo do Bebê:").grid(row=0, column=0, sticky=tk.W, pady=3)
        sex_box = ttk.Frame(frame_config)
        sex_box.grid(row=0, column=1, columnspan=3, sticky=tk.W, pady=3)

        rb_menina = ttk.Radiobutton(sex_box, text="Menina ♀", variable=self.sex_var, value="girl", command=self._on_sex_changed)
        rb_menina.pack(side=tk.LEFT, padx=(0, 12))
        rb_menino = ttk.Radiobutton(sex_box, text="Menino ♂", variable=self.sex_var, value="boy", command=self._on_sex_changed)
        rb_menino.pack(side=tk.LEFT)

        # Estaturas dos Pais
        ttk.Label(frame_config, text="Alt. Mãe (cm):").grid(row=1, column=0, sticky=tk.W, pady=3)
        self.ent_mae = ttk.Entry(frame_config, width=7)
        self.ent_mae.insert(0, "162.0")
        self.ent_mae.grid(row=1, column=1, padx=4, pady=3, sticky=tk.W)

        ttk.Label(frame_config, text="Alt. Pai (cm):").grid(row=1, column=2, sticky=tk.W, pady=3)
        self.ent_pai = ttk.Entry(frame_config, width=7)
        self.ent_pai.insert(0, "178.0")
        self.ent_pai.grid(row=1, column=3, padx=4, pady=3, sticky=tk.W)

        btn_update_tanner = ttk.Button(frame_config, text="Recalcular", command=self._render_charts)
        btn_update_tanner.grid(row=1, column=4, padx=6, pady=3)

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
        ttk.Button(btn_box, text="📂 Carregar CSV", command=self._open_csv_dialog).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        ttk.Button(btn_box, text="Dados Demo", command=self._load_seed).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)

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

    def _open_csv_dialog(self):
        """Abre caixa de diálogo para selecionar e carregar arquivo CSV com consultas."""
        from tkinter import filedialog
        file_path = filedialog.askopenfilename(
            title="Selecionar Histórico de Consultas (CSV)",
            filetypes=[("Arquivos CSV", "*.csv"), ("Todos os Arquivos", "*.*")]
        )
        if not file_path:
            return

        success = self.tracker.load_from_file(file_path)
        if success:
            self._update_table()
            self._render_charts()
            messagebox.showinfo("Sucesso", f"{len(self.tracker.df)} registros carregados com sucesso!")
        else:
            messagebox.showerror("Erro", "Não foi possível extrair dados válidos do arquivo CSV selecionado.")

    def _on_sex_changed(self):
        """Callback executado ao alternar entre Menino e Menina."""
        self.who.set_sex(self.sex_var.get())
        self._render_charts()

    def _get_tanner_model(self):
        sex = self.sex_var.get()
        try:
            mae = float(self.ent_mae.get())
            pai = float(self.ent_pai.get())
            return TargetHeightTanner(mae, pai, sex=sex)
        except ValueError:
            return TargetHeightTanner(162.0, 178.0, sex=sex)

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

        sex_label = "Menino" if self.sex_var.get() == "boy" else "Menina"

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

            ax.set_title(f"{label} ({sex_label})", fontsize=10, fontweight="bold")
            ax.set_xlabel("Meses", fontsize=8)
            ax.set_xlim(0, 24)
            ax.legend(loc="upper left", fontsize=6.5, frameon=True)

        self.canvas.draw()


if __name__ == "__main__":
    app = App()
    app.mainloop()