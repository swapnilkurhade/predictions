from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, conlist
from typing import List, Optional, Dict, Any, Literal
from datetime import datetime, date, timedelta
import numpy as np
import math
import itertools
import joblib

# ----------------------------
# Model loading
# ----------------------------
MODEL_PATH = "soil_npk_model.joblib"
try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError as e:
    # We don't crash the server; endpoints will 400 if called without model
    model = None


# ----------------------------
# Data Schemas
# ----------------------------
class SoilFeatures(BaseModel):
    """Order-sensitive features for the NPK prediction model."""
    Ph: float = Field(..., description="Soil pH")
    Electrical_Conductivity: float
    Organic_Carbon: float
    Available_Nitrogen: float
    Available_Phosphorus: float
    Available_Potassium: float
    Exchangeable_Sodium: float
    Free_Lime: float
    Iron: float
    Manganese: float
    Zinc: float
    Copper: float
    Sulphar: float
    Boron: float

    def as_feature_array(self) -> List[float]:
        return [
            self.Ph,
            self.Electrical_Conductivity,
            self.Organic_Carbon,
            self.Available_Nitrogen,
            self.Available_Phosphorus,
            self.Available_Potassium,
            self.Exchangeable_Sodium,
            self.Free_Lime,
            self.Iron,
            self.Manganese,
            self.Zinc,
            self.Copper,
            self.Sulphar,
            self.Boron,
        ]


class WeatherData(BaseModel):
    # Weather (current)
    temperature: float
    rainfall: float  # daily
    evapotranspiration: float
    humidity: float
    wind_speed: float
    # Satellite
    ndvi: float
    evi: float
    soil_moisture: float
    # 7-day forecast aggregates
    forecast_temp_avg: float
    forecast_rainfall_total: float
    # Additional
    lux_intensity: float
    ph: Optional[float] = None  # If omitted, we'll copy from soil features pH


class AdjustmentMode(BaseModel):
    mode: Literal["current_week", "current_and_next", "next_two", "range", "specific"] = "current_week"
    start_week: Optional[int] = None
    end_week: Optional[int] = None
    weeks: Optional[List[int]] = None


class ScheduleRequest(BaseModel):
    planting_date: date
    plantation_type: Literal['Setling (nursery plant)', 'Seed set']
    seedling_weeks: Optional[int] = 0
    cane_variety: str

    soil: SoilFeatures

    # Split percentages (must be 0..100)
    basal_n_pct: float = Field(..., ge=0, le=100)
    basal_p_pct: float = Field(..., ge=0, le=100)
    basal_k_pct: float = Field(..., ge=0, le=100)

    include_weather: bool = False
    weather: Optional[WeatherData] = None
    adjustment: Optional[AdjustmentMode] = None


class PredictNPKRequest(BaseModel):
    soil: SoilFeatures


# ----------------------------
# Core Logic (adapted from your script)
# ----------------------------
class WeatherSatelliteLogic:
    def __init__(self):
        self.thresholds = {
            'high_temp': 32.0,
            'low_temp': 15.0,
            'heavy_rainfall': 50.0,
            'low_rainfall': 5.0,
            'high_et': 6.0,
            'low_et': 2.0,
            'low_ndvi': 0.3,
            'high_ndvi': 0.7,
            'low_evi': 0.2,
            'high_evi': 0.5,
            'low_soil_moisture': 0.2,
            'high_soil_moisture': 0.4,
            'low_lux': 20000,
            'high_lux': 60000,
            'low_ph': 5.5,
            'high_ph': 7.5,
        }

    def analyze_conditions(self, w: WeatherData, ph_fallback: float) -> Dict[str, Any]:
        adj = {
            'nitrogen_factor': 1.0,
            'phosphorus_factor': 1.0,
            'potassium_factor': 1.0,
            'frequency_factor': 1.0,
            'water_factor': 1.0,
            'recommendations': []
        }

        # Temperature
        if w.temperature > self.thresholds['high_temp']:
            adj['nitrogen_factor'] *= 1.15
            adj['potassium_factor'] *= 1.20
            adj['frequency_factor'] *= 1.3
            adj['water_factor'] *= 1.4
            adj['recommendations'].append("HIGH TEMPERATURE: Increase N & K, more frequent fertigation")
        elif w.temperature < self.thresholds['low_temp']:
            adj['nitrogen_factor'] *= 0.85
            adj['phosphorus_factor'] *= 1.10
            adj['frequency_factor'] *= 0.8
            adj['recommendations'].append("LOW TEMPERATURE: Reduce N, increase P, reduce frequency")

        # Rainfall (current)
        if w.rainfall > self.thresholds['heavy_rainfall']:
            adj['nitrogen_factor'] *= 1.25
            adj['potassium_factor'] *= 1.15
            adj['frequency_factor'] *= 1.2
            adj['recommendations'].append("HEAVY RAINFALL: Increase N & K for leaching compensation")
        elif w.rainfall < self.thresholds['low_rainfall']:
            adj['nitrogen_factor'] *= 0.90
            adj['water_factor'] *= 1.3
            adj['recommendations'].append("LOW RAINFALL: Reduce concentration, increase water")

        # Evapotranspiration
        if w.evapotranspiration > self.thresholds['high_et']:
            adj['potassium_factor'] *= 1.25
            adj['frequency_factor'] *= 1.4
            adj['water_factor'] *= 1.5
            adj['recommendations'].append("HIGH ET: Increase K and water, more frequent application")
        elif w.evapotranspiration < self.thresholds['low_et']:
            adj['frequency_factor'] *= 0.8
            adj['water_factor'] *= 0.9
            adj['recommendations'].append("LOW ET: Reduce application frequency")

        # NDVI
        if w.ndvi < self.thresholds['low_ndvi']:
            adj['nitrogen_factor'] *= 1.30
            adj['phosphorus_factor'] *= 1.20
            adj['recommendations'].append("LOW NDVI: Plant stress – boost N and P")
        elif w.ndvi > self.thresholds['high_ndvi']:
            adj['nitrogen_factor'] *= 0.90
            adj['potassium_factor'] *= 1.10
            adj['recommendations'].append("HIGH NDVI: Healthy – reduce N slightly, improve quality with K")

        # EVI
        if w.evi < self.thresholds['low_evi']:
            adj['nitrogen_factor'] *= 1.20
            adj['recommendations'].append("LOW EVI: Increase N")
        elif w.evi > self.thresholds['high_evi']:
            adj['phosphorus_factor'] *= 1.15
            adj['potassium_factor'] *= 1.10
            adj['recommendations'].append("HIGH EVI: Focus on P & K for quality")

        # Soil moisture
        if w.soil_moisture < self.thresholds['low_soil_moisture']:
            adj['nitrogen_factor'] *= 0.85
            adj['water_factor'] *= 1.6
            adj['frequency_factor'] *= 1.3
            adj['recommendations'].append("LOW SOIL MOISTURE: Reduce fert conc., add more water")
        elif w.soil_moisture > self.thresholds['high_soil_moisture']:
            adj['nitrogen_factor'] *= 1.10
            adj['frequency_factor'] *= 0.9
            adj['recommendations'].append("HIGH SOIL MOISTURE: Adjust for potential leaching")

        # Forecast
        if w.forecast_rainfall_total > 100:
            adj['nitrogen_factor'] *= 1.20
            adj['potassium_factor'] *= 1.15
            adj['recommendations'].append("FORECAST: Heavy rain expected – pre-load N & K")
        elif w.forecast_rainfall_total < 10:
            adj['nitrogen_factor'] *= 0.85
            adj['water_factor'] *= 1.4
            adj['recommendations'].append("FORECAST: Dry spell – conservative fertilization, more water")

        # Lux
        if w.lux_intensity < self.thresholds['low_lux']:
            adj['nitrogen_factor'] *= 0.9
            adj['recommendations'].append(f"LOW LUX: Reduce N due to low sunlight ({w.lux_intensity} lux)")
            adj['special_fertilizer'] = {'name': 'Urea', 'amount': 5, 'reason': 'Low Lux'}
        elif w.lux_intensity > self.thresholds['high_lux']:
            adj['nitrogen_factor'] *= 1.1
            adj['recommendations'].append(f"HIGH LUX: Increase N due to high sunlight ({w.lux_intensity} lux)")
            adj['special_fertilizer'] = {'name': 'Urea', 'amount': 10, 'reason': 'High Lux'}

        # pH
        ph_val = w.ph if w.ph is not None else ph_fallback
        if ph_val < self.thresholds['low_ph']:
            adj['phosphorus_factor'] *= 0.85
            adj['recommendations'].append(f"LOW SOIL PH: Reduce P, consider liming (pH={ph_val})")
            adj['special_fertilizer_ph'] = {'name': 'Single Super Phosphate', 'amount': 8, 'reason': 'Low pH'}
        elif ph_val > self.thresholds['high_ph']:
            adj['potassium_factor'] *= 0.85
            adj['recommendations'].append(f"HIGH SOIL PH: Reduce K, consider acidifying amendments (pH={ph_val})")
            adj['special_fertilizer_ph'] = {'name': 'Muriate of Potash', 'amount': 6, 'reason': 'High pH'}

        return adj


class FertilizerCalculator:
    def __init__(self):
        self.npk_schedule = {
            1: {'N': 0, 'P': 0, 'K': 0},
            2: {'N': 0.98, 'P': 0.99, 'K': 0.59},
            3: {'N': 0.98, 'P': 0.99, 'K': 0.59},
            4: {'N': 0.98, 'P': 0.99, 'K': 0.59},
            5: {'N': 0.98, 'P': 0.99, 'K': 0.59},
            6: {'N': 3.01, 'P': 2.99, 'K': 1.04},
            7: {'N': 3.01, 'P': 2.99, 'K': 1.04},
            8: {'N': 3.01, 'P': 2.99, 'K': 1.04},
            9: {'N': 3.01, 'P': 2.99, 'K': 1.04},
            10: {'N': 3.01, 'P': 2.99, 'K': 1.04},
            11: {'N': 3.01, 'P': 2.99, 'K': 1.04},
            12: {'N': 5, 'P': 5.08, 'K': 0.98},
            13: {'N': 5, 'P': 5.08, 'K': 0.98},
            14: {'N': 5, 'P': 5.08, 'K': 0.98},
            15: {'N': 5, 'P': 5.08, 'K': 0.98},
            16: {'N': 3.99, 'P': 4.04, 'K': 1.95},
            17: {'N': 3.99, 'P': 4.04, 'K': 1.95},
            18: {'N': 3.99, 'P': 4.04, 'K': 1.95},
            19: {'N': 3.99, 'P': 4.04, 'K': 1.95},
            20: {'N': 3.99, 'P': 4.04, 'K': 1.95},
            21: {'N': 3.99, 'P': 4.04, 'K': 1.95},
            22: {'N': 3.01, 'P': 2.99, 'K': 1.95},
            23: {'N': 3.01, 'P': 2.99, 'K': 1.95},
            24: {'N': 3.01, 'P': 2.99, 'K': 1.95},
            25: {'N': 3.01, 'P': 2.99, 'K': 1.95},
            26: {'N': 3.01, 'P': 2.99, 'K': 1.95},
            27: {'N': 3.01, 'P': 2.99, 'K': 1.95},
            28: {'N': 1.96, 'P': 1.95, 'K': 2.93},
            29: {'N': 1.96, 'P': 1.95, 'K': 2.93},
            30: {'N': 1.96, 'P': 1.95, 'K': 2.93},
            31: {'N': 1.96, 'P': 1.95, 'K': 2.93},
            32: {'N': 1.37, 'P': 1.3, 'K': 4.04},
            33: {'N': 1.37, 'P': 1.3, 'K': 4.04},
            34: {'N': 1.37, 'P': 1.3, 'K': 4.04},
            35: {'N': 1.37, 'P': 1.3, 'K': 4.04},
            36: {'N': 1.37, 'P': 1.3, 'K': 4.04},
            37: {'N': 1.37, 'P': 1.3, 'K': 4.04},
            38: {'N': 0, 'P': 0, 'K': 4.1},
            39: {'N': 0, 'P': 0, 'K': 4.1},
            40: {'N': 0, 'P': 0, 'K': 4.1},
            41: {'N': 0, 'P': 0, 'K': 4.1},
            42: {'N': 0, 'P': 0, 'K': 2.93},
            43: {'N': 0, 'P': 0, 'K': 2.93},
            44: {'N': 0, 'P': 0, 'K': 2.93},
            45: {'N': 0, 'P': 0, 'K': 2.93},
        }

        self.fertilizers = {
            '19:19:19': {'N': 0.19, 'P': 0.19, 'K': 0.19, 'price': 144},
            '12:61:00': {'N': 0.12, 'P': 0.61, 'K': 0.00, 'price': 148},
            '00:52:34': {'N': 0.00, 'P': 0.52, 'K': 0.34, 'price': 192},
            '13:00:45': {'N': 0.13, 'P': 0.00, 'K': 0.45, 'price': 152},
            '00:00:50': {'N': 0.00, 'P': 0.00, 'K': 0.50, 'price': 104},
            '13:40:13': {'N': 0.13, 'P': 0.40, 'K': 0.13, 'price': 140},
            'Urea': {'N': 0.46, 'P': 0.00, 'K': 0.00, 'price': 5.4},
            'ammonium sulphate': {'N': 0.21, 'P': 0.00, 'K': 0.00, 'price': 20},
        }

        self.other_basal_components = [
            'DAP :100 Kg',
            'MoP :50 Kg',
            '14:35:14: 25Kg',
            'Nimboli pend: 100 kg',
            'Compost FYM: 2 ton',
            'Baggase Ash: 600 Kg',
            'Azotobacter: 2 Kg',
            'PSB: 2 Kg',
            'Poly sulphate: 25 kg',
            'Zink sulphate: 8.8 kg',
            'ferous sulphate: 10 kg',
            'Calcium sillicate: 333 kg',
            'Magnesium sulphate: 11 Kg',
            'Borox: 1 Kg',
            'Calcium Nitrate: 5.5 Kg',
        ]

        self.drenching_schedule = {
            1: {'water': '200 litters water', 'treatment': 'Humic -1 kg \n 19:19:19 -1 kg \n micronutrient -200gram \n IBA -2 gm \n IBA Solvent 100 ml'},
            5: {'water': '200 litters water', 'treatment': 'Carbendazim -200gm \n chlorantraniliprole -30 ml'},
            9: {'water': '200 litters water', 'treatment': 'VAM -100gm \n NPK Culture -2 litter \n Humic -1kg \n Fulvic -1 kg'},
            60: {'water': '200 litters water', 'treatment': 'NPK bacteria -2 litter '},
            120: {'water': '200 litters water', 'treatment': 'NPK bacteria -2 litter '},
            180: {'water': '200 litters water', 'treatment': 'NPK bacteria -2 litter '},
            240: {'water': '200 litters water', 'treatment': 'NPK bacteria -2 litter '},
            300: {'water': '200 litters water', 'treatment': 'NPK bacteria -2 litter '},
            360: {'water': '200 litters water', 'treatment': 'NPK bacteria -2 litter '},
            420: {'water': '200 litters water', 'treatment': 'NPK bacteria -2 litter '},
        }

        self.foliar_spray_schedule = {
            45: {'water': '60 litter', 'treatment': '6 BA -2.5 gm \n 6 BA Solvent -50 ml \n GA -2.5 gm \n GA solvent -50 ml \n 19:19:19 -600 gm \n micronutrient -150 gm \n silicon -600 gm'},
            65: {'water': '90 litter', 'treatment': '6 BA -3.5 gm \n 6 BA Solvent -50 ml \n GA -3.5 gm \n GA solvent -50 ml \n 19:19:19 -900 gm \n micronutrient -225 gm \n silicon -450 gm'},
            85: {'water': '135 litter', 'treatment': '6 BA -5.5 gm \n 6 BA Solvent -100 ml \n GA -5.5 gm \n GA solvent -100 ml \n 19:19:19 -1.5 Kg \n micronutrient -350 gm \n silicon -750 gm'},
            105: {'water': '180 litter', 'treatment': '6 BA -7 gm \n 6 BA Solvent -100 ml \n GA -7 gm \n GA solvent -100 ml \n 19:19:19 -2 Kg \n micronutrient -500 gm \n silicon -1Kg'},
            125: {'water': '195 litter', 'treatment': '6 BA -8 gm \n 6 BA Solvent -100 ml \n GA -8 gm \n GA solvent -100 ml \n 19:19:19 -2 Kg \n micronutrient -500 gm \n silicon -1Kg'},
        }

    @staticmethod
    def kg_ha_to_kg_acre(kg_per_ha: float) -> float:
        return kg_per_ha * 0.404686

    def calculate_weekly_npk(self, total_n: float, total_p: float, total_k: float,
                             adjustments: Optional[Dict[str, Any]] = None,
                             target_weeks: Optional[List[int]] = None) -> Dict[int, Dict[str, Any]]:
        out: Dict[int, Dict[str, Any]] = {}
        for week in range(1, 46):
            if week in self.npk_schedule:
                base_n = (total_n * self.npk_schedule[week]['N']) / 100
                base_p = (total_p * self.npk_schedule[week]['P']) / 100
                base_k = (total_k * self.npk_schedule[week]['K']) / 100
                if adjustments and target_weeks and week in target_weeks:
                    out[week] = {
                        'N': base_n * adjustments['nitrogen_factor'],
                        'P': base_p * adjustments['phosphorus_factor'],
                        'K': base_k * adjustments['potassium_factor'],
                        'adjusted': True,
                        'adjustment_reason': f"Environmental conditions adjusted for week {week}",
                        'frequency_factor': adjustments['frequency_factor'],
                    }
                else:
                    out[week] = {
                        'N': base_n,
                        'P': base_p,
                        'K': base_k,
                        'adjusted': False,
                        'frequency_factor': 1.0,
                    }
        return out

    # ---------- Fertilizer optimization ----------
    def calculate_optimized_fertigation(self, n_needed: float, p_needed: float, k_needed: float,
                                        tolerance: float = 0.05, max_fertilizers: int = 3) -> Dict[str, Any]:
        if n_needed <= 1e-6 and p_needed <= 1e-6 and k_needed <= 1e-6:
            return {"fertilizers": {}, "total_cost": 0.0, "nutrients_supplied": {"nitrogen": 0.0, "phosphorus": 0.0, "potassium": 0.0}}

        best: List[Dict[str, Any]] = []
        keys = list(self.fertilizers.keys())
        for r in range(1, min(max_fertilizers + 1, len(keys) + 1)):
            for combo in itertools.combinations(keys, r):
                result = self._solve_combo(combo, n_needed, p_needed, k_needed, tolerance)
                if result:
                    best.append(result)
        if not best:
            return self._fallback(n_needed, p_needed, k_needed)
        best.sort(key=lambda x: x['total_cost'])
        return best[0]

    def _solve_combo(self, names: tuple[str, ...], n_needed: float, p_needed: float, k_needed: float, tol: float) -> Optional[Dict[str, Any]]:
        try:
            A = []
            prices = []
            for nm in names:
                f = self.fertilizers[nm]
                A.append([f['N'], f['P'], f['K']])
                prices.append(f['price'])
            A = np.array(A).T  # 3 x n
            prices = np.array(prices)
            req = np.array([n_needed, p_needed, k_needed])

            sol = self._find_feasible(A, req, tol)
            if sol is not None and np.all(sol >= 0):
                supplied = A @ sol
                if np.all(supplied >= req * (1 - tol)) and np.all(supplied <= req * (1 + tol * 3)):
                    total_cost = float(np.sum(sol * prices))
                    combo: Dict[str, float] = {}
                    for i, nm in enumerate(names):
                        if sol[i] > 0.01:
                            combo[nm] = round(float(sol[i]), 2)
                    return {
                        'fertilizers': combo,
                        'total_cost': round(total_cost, 2),
                        'nutrients_supplied': {
                            'nitrogen': round(float(supplied[0]), 2),
                            'phosphorus': round(float(supplied[1]), 2),
                            'potassium': round(float(supplied[2]), 2),
                        },
                        'cost_per_kg': round(total_cost / float(np.sum(sol)), 2) if float(np.sum(sol)) > 0 else 0,
                    }
        except Exception:
            pass
        return None

    def _find_feasible(self, A: np.ndarray, req: np.ndarray, tol: float) -> Optional[np.ndarray]:
        n_f = A.shape[1]
        best_sol = None
        min_cost = float('inf')
        max_amounts = []
        for j in range(n_f):
            max_amount = 0.0
            for i in range(3):
                if A[i, j] > 1e-9:
                    max_amount = max(max_amount, req[i] / A[i, j] * 2)
            max_amounts.append(max_amount if max_amount > 1e-9 else 100)

        steps = 20
        for _ in range(100):
            sol = np.zeros(n_f)
            for j in range(n_f):
                sol[j] = np.random.uniform(0, max_amounts[j] / steps) * np.random.randint(0, steps + 1)
            supplied = A @ sol
            prices = np.array([self.fertilizers[name]['price'] for name in list(self.fertilizers.keys())[:n_f]])
            cost = float(np.sum(sol * prices))
            meets_min = np.all(supplied >= req * (1 - tol))
            not_too_much = np.all(supplied <= req * (1 + tol * 3))
            if meets_min and not_too_much and cost < min_cost:
                min_cost = cost
                best_sol = sol.copy()
        return best_sol

    def _fallback(self, n_needed: float, p_needed: float, k_needed: float) -> Dict[str, Any]:
        combo: Dict[str, float] = {}
        total_cost = 0.0
        supplied_n = supplied_p = supplied_k = 0.0
        if n_needed > 0:
            urea_needed = n_needed / self.fertilizers['Urea']['N']
            combo['Urea'] = round(float(urea_needed), 2)
            total_cost += float(urea_needed * self.fertilizers['Urea']['price'])
            supplied_n += round(float(urea_needed * self.fertilizers['Urea']['N']), 2)
        if p_needed > 0:
            p_f = p_needed / self.fertilizers['12:61:00']['P']
            combo['12:61:00'] = round(float(p_f), 2)
            total_cost += float(p_f * self.fertilizers['12:61:00']['price'])
            supplied_p += round(float(p_f * self.fertilizers['12:61:00']['P']), 2)
        if k_needed > 0:
            k_f = k_needed / self.fertilizers['00:00:50']['K']
            combo['00:00:50'] = round(float(k_f), 2)
            total_cost += float(k_f * self.fertilizers['00:00:50']['price'])
            supplied_k += round(float(k_f * self.fertilizers['00:00:50']['K']), 2)
        return {
            'fertilizers': combo,
            'total_cost': round(total_cost, 2),
            'nutrients_supplied': {
                'nitrogen': supplied_n,
                'phosphorus': supplied_p,
                'potassium': supplied_k,
            },
            'cost_per_kg': round(total_cost / max(1e-9, sum(combo.values())), 2) if combo else 0,
        }

    def calculate_fertilizer_combinations(self, n_needed: float, p_needed: float, k_needed: float) -> List[Dict[str, Any]]:
        if n_needed <= 1e-6 and p_needed <= 1e-6 and k_needed <= 1e-6:
            return []
        optimal = self.calculate_optimized_fertigation(n_needed, p_needed, k_needed)
        combos = [
            {
                'fertilizers': optimal['fertilizers'],
                'total_cost': optimal['total_cost'],
                'cost_per_kg': optimal.get('cost_per_kg', 0),
                'nutrients_supplied': optimal['nutrients_supplied'],
            }
        ]
        fallback = self._fallback(n_needed, p_needed, k_needed)
        if abs(fallback['total_cost'] - optimal['total_cost']) > 0.01:
            combos.append({
                'fertilizers': fallback['fertilizers'],
                'total_cost': fallback['total_cost'],
                'cost_per_kg': fallback.get('cost_per_kg', 0),
                'nutrients_supplied': fallback['nutrients_supplied'],
            })
        return combos

    # ---------- High-level schedule ----------
    def generate_schedule(self, *,
                          planting_date: date,
                          season: str,
                          plantation_type: str,
                          seedling_weeks: int,
                          cane_variety: str,
                          predicted_npk_total_kg_ha: List[float],
                          basal_n_pct: float,
                          basal_p_pct: float,
                          basal_k_pct: float,
                          include_weather: bool,
                          weather: Optional[WeatherData],
                          adjustments: Optional[Dict[str, Any]],
                          target_weeks: Optional[List[int]]) -> Dict[str, Any]:
        total_n_kg_ha, total_p_kg_ha, total_k_kg_ha = predicted_npk_total_kg_ha
        fertigation_n_pct = 100 - basal_n_pct
        fertigation_p_pct = 100 - basal_p_pct
        fertigation_k_pct = 100 - basal_k_pct

        # 25% increase for Co/CO/C0 86032
        up = cane_variety.upper()
        if '86032' in up and ('CO' in up or 'C0' in up):
            total_n_kg_ha *= 1.25
            total_p_kg_ha *= 1.25
            total_k_kg_ha *= 1.25

        basal_n_kg_ha = total_n_kg_ha * (basal_n_pct / 100.0)
        basal_p_kg_ha = total_p_kg_ha * (basal_p_pct / 100.0)
        basal_k_kg_ha = total_k_kg_ha * (basal_k_pct / 100.0)

        fert_n_kg_ha = total_n_kg_ha * (fertigation_n_pct / 100.0)
        fert_p_kg_ha = total_p_kg_ha * (fertigation_p_pct / 100.0)
        fert_k_kg_ha = total_k_kg_ha * (fertigation_k_pct / 100.0)

        # per acre
        fert_n_ac = self.kg_ha_to_kg_acre(fert_n_kg_ha)
        fert_p_ac = self.kg_ha_to_kg_acre(fert_p_kg_ha)
        fert_k_ac = self.kg_ha_to_kg_acre(fert_k_kg_ha)
        basal_n_ac = self.kg_ha_to_kg_acre(basal_n_kg_ha)
        basal_p_ac = self.kg_ha_to_kg_acre(basal_p_kg_ha)
        basal_k_ac = self.kg_ha_to_kg_acre(basal_k_kg_ha)

        weekly_npk = self.calculate_weekly_npk(fert_n_ac, fert_p_ac, fert_k_ac, adjustments, target_weeks)

        carried_over = {'N': 0.0, 'P': 0.0, 'K': 0.0}
        total_cost = 0.0
        adjusted_weeks_cost = 0.0
        normal_weeks_cost = 0.0
        basal_cost = 0.0
        events: List[Dict[str, Any]] = []

        # Basal event (Day 0)
        basal_combos = self.calculate_fertilizer_combinations(basal_n_ac, basal_p_ac, basal_k_ac)
        if basal_combos:
            basal_cost = float(basal_combos[0]['total_cost'])
        events.append({
            'day': 0,
            'date': planting_date.isoformat(),
            'type': 'BASAL DOSE',
            'week': 0,
            'data': {
                'npk_required_acre': {'N': round(basal_n_ac, 2), 'P': round(basal_p_ac, 2), 'K': round(basal_k_ac, 2)},
                'fertilizer_combinations': basal_combos,
                'other_components': self.other_basal_components,
            },
        })
        total_cost += basal_cost

        # Drenching
        for day_num, details in self.drenching_schedule.items():
            event_week = (day_num // 7) + 1
            water_amount = details['water']
            if adjustments and target_weeks and event_week in target_weeks and adjustments['water_factor'] != 1.0:
                import re
                m = re.search(r'(\d+)\s*litters?', water_amount)
                if m:
                    original = int(m.group(1))
                    water_amount = f"{int(original * adjustments['water_factor'])} litters water (ADJUSTED)"
            events.append({
                'day': day_num,
                'date': (planting_date + timedelta(days=day_num)).isoformat(),
                'type': 'DRENCHING',
                'week': event_week,
                'data': {'water': water_amount, 'treatment': details['treatment']},
            })

        # Foliar
        for day_num, details in self.foliar_spray_schedule.items():
            event_week = (day_num // 7) + 1
            water_amount = details['water']
            if adjustments and target_weeks and event_week in target_weeks and adjustments['water_factor'] != 1.0:
                import re
                m = re.search(r'(\d+)\s*litter', water_amount)
                if m:
                    original = int(m.group(1))
                    water_amount = f"{int(original * adjustments['water_factor'])} litter (ADJUSTED)"
            events.append({
                'day': day_num,
                'date': (planting_date + timedelta(days=day_num)).isoformat(),
                'type': 'FOLIAR SPRAY',
                'week': event_week,
                'data': {'water': water_amount, 'treatment': details['treatment']},
            })

        # Weekly NPK applications
        for week in range(1, 46):
            wk = weekly_npk[week]
            day_num = (week - 1) * 7 + 7
            n_req = wk['N'] + carried_over['N']
            p_req = wk['P'] + carried_over['P']
            k_req = wk['K'] + carried_over['K']
            is_adj = wk['adjusted']
            freq = wk['frequency_factor']

            # reset carry, then compute new carry if reduced frequency
            carried_over = {'N': 0.0, 'P': 0.0, 'K': 0.0}
            if is_adj and freq < 1.0:
                carried_over['N'] += n_req * (1 - freq)
                carried_over['P'] += p_req * (1 - freq)
                carried_over['K'] += k_req * (1 - freq)
                n_req *= freq
                p_req *= freq
                k_req *= freq

            combos = self.calculate_fertilizer_combinations(n_req, p_req, k_req)
            event_data: Dict[str, Any] = {
                'npk_required': {'N': round(n_req, 2), 'P': round(p_req, 2), 'K': round(k_req, 2)},
                'fertilizer_combinations': combos,
                'is_adjusted': is_adj,
                'carried_over_npk': {k: round(v, 2) for k, v in carried_over.items()},
            }
            if is_adj:
                base_n = (fert_n_ac * self.npk_schedule[week]['N']) / 100
                base_p = (fert_p_ac * self.npk_schedule[week]['P']) / 100
                base_k = (fert_k_ac * self.npk_schedule[week]['K']) / 100
                event_data['original_npk'] = {'N': round(base_n, 2), 'P': round(base_p, 2), 'K': round(base_k, 2)}
                event_data['adjustment_note'] = wk['adjustment_reason']
                if freq != 1.0:
                    event_data['frequency_note'] = (
                        f"INCREASED FREQUENCY: Apply {freq:.1f}x more often" if freq > 1.0 else f"REDUCED FREQUENCY: {1/freq:.1f}x less often"
                    )

            events.append({
                'day': day_num,
                'date': (planting_date + timedelta(days=day_num)).isoformat(),
                'type': 'WEEKLY NPK APPLICATION',
                'week': week,
                'data': event_data,
            })

            if combos:
                if is_adj:
                    adjusted_weeks_cost += float(combos[0]['total_cost'])
                else:
                    normal_weeks_cost += float(combos[0]['total_cost'])
                total_cost += float(combos[0]['total_cost'])

            # Add special fert suggestions onto adjusted weeks only
            if is_adj and adjustments:
                if 'special_fertilizer' in adjustments:
                    sf = adjustments['special_fertilizer']
                    event_data.setdefault('special_conditions', []).append({'reason': sf['reason'], 'name': sf['name'], 'amount_kg_acre': sf['amount']})
                if 'special_fertilizer_ph' in adjustments:
                    sfp = adjustments['special_fertilizer_ph']
                    event_data.setdefault('special_conditions', []).append({'reason': sfp['reason'], 'name': sfp['name'], 'amount_kg_acre': sfp['amount']})

        events.sort(key=lambda x: x['day'])

        # Cost impact estimate for adjusted weeks (fertigation part only)
        cost_impact = None
        if target_weeks and adjusted_weeks_cost > 0 and adjustments:
            avg_factor = (adjustments['nitrogen_factor'] + adjustments['phosphorus_factor'] + adjustments['potassium_factor']) / 3
            estimated_original = adjusted_weeks_cost / max(1e-9, avg_factor)
            cost_impact = round(adjusted_weeks_cost - estimated_original, 2)

        report = {
            'meta': {
                'planting_date': planting_date.isoformat(),
                'season': season,
                'plantation_type': plantation_type,
                'seedling_weeks': seedling_weeks,
                'cane_variety': cane_variety,
                'target_weeks': target_weeks or [],
            },
            'predicted_total_npk_kg_ha': {
                'N': round(total_n_kg_ha, 2),
                'P': round(total_p_kg_ha, 2),
                'K': round(total_k_kg_ha, 2),
            },
            'split_kg_ha': {
                'basal': {
                    'N': round(basal_n_kg_ha, 2), 'P': round(basal_p_kg_ha, 2), 'K': round(basal_k_kg_ha, 2)
                },
                'fertigation': {
                    'N': round(fert_n_kg_ha, 2), 'P': round(fert_p_kg_ha, 2), 'K': round(fert_k_kg_ha, 2)
                },
            },
            'split_kg_acre': {
                'basal': {
                    'N': round(basal_n_ac, 2), 'P': round(basal_p_ac, 2), 'K': round(basal_k_ac, 2)
                },
                'fertigation': {
                    'N': round(fert_n_ac, 2), 'P': round(fert_p_ac, 2), 'K': round(fert_k_ac, 2)
                },
            },
            'environment': None,
            'events': events,
            'cost_summary': {
                'total_estimated_cost': round(total_cost, 2),
                'estimated_basal_cost': round(basal_cost, 2),
                'estimated_fertigation_cost_adjusted_weeks': round(adjusted_weeks_cost, 2),
                'estimated_fertigation_cost_normal_weeks': round(normal_weeks_cost, 2),
                'estimated_cost_impact_of_adjustments': cost_impact,
            },
        }

        if include_weather and weather and adjustments is not None:
            report['environment'] = {
                'current': {
                    'temperature': weather.temperature,
                    'rainfall': weather.rainfall,
                    'evapotranspiration': weather.evapotranspiration,
                    'ndvi': weather.ndvi,
                    'evi': weather.evi,
                    'soil_moisture': weather.soil_moisture,
                },
                'adjustment_factors_for_selected_weeks': {
                    'nitrogen_factor': adjustments['nitrogen_factor'],
                    'phosphorus_factor': adjustments['phosphorus_factor'],
                    'potassium_factor': adjustments['potassium_factor'],
                    'frequency_factor': adjustments['frequency_factor'],
                    'water_factor': adjustments['water_factor'],
                    'recommendations': adjustments['recommendations'],
                },
            }
        return report


# ----------------------------
# Helpers
# ----------------------------

def predict_npk_from_soil(soil: SoilFeatures) -> List[float]:
    if model is None:
        raise HTTPException(status_code=400, detail=f"Model not loaded at '{MODEL_PATH}'. Place the file and restart.")
    arr = np.array(soil.as_feature_array()).reshape(1, -1)
    pred = model.predict(arr)[0]
    return [float(pred[0]), float(pred[1]), float(pred[2])]


def compute_target_weeks(planting_date: date, adj: Optional[AdjustmentMode]) -> List[int]:
    # current week based on today (server time)
    today = date.today()
    days = (today - planting_date).days
    current_week = max(1, (days // 7) + 1)

    if adj is None:
        return [current_week] if current_week <= 45 else []

    mode = adj.mode
    if mode == 'current_week':
        return [current_week] if current_week <= 45 else []
    if mode == 'current_and_next':
        return [w for w in [current_week, current_week + 1] if w <= 45]
    if mode == 'next_two':
        return [w for w in [current_week + 1, current_week + 2] if w <= 45]
    if mode == 'range' and adj.start_week and adj.end_week:
        s, e = adj.start_week, adj.end_week
        if 1 <= s <= e <= 45:
            return list(range(s, e + 1))
        else:
            raise HTTPException(status_code=422, detail="Invalid week range; must satisfy 1 ≤ start ≤ end ≤ 45")
    if mode == 'specific' and adj.weeks:
        if all(1 <= w <= 45 for w in adj.weeks):
            return sorted(list(set(adj.weeks)))
        else:
            raise HTTPException(status_code=422, detail="All weeks must be between 1 and 45")
    return []


# ----------------------------
# FastAPI App & Routes
# ----------------------------
app = FastAPI(title="Fertigation API", version="1.0.0", description="Sugarcane fertigation schedule & fertilizer optimization API")

@app.get("/")
def hello():
    return {"status": "ok", "model_loaded": "loaded"}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict_npk")
def predict_npk_endpoint(payload: PredictNPKRequest):
    pred = predict_npk_from_soil(payload.soil)
    return {"predicted_npk_total_kg_ha": {"N": pred[0], "P": pred[1], "K": pred[2]}}


@app.post("/schedule")
def schedule(payload: ScheduleRequest):
    # 1) Predict NPK totals (kg/ha)
    predicted_npk = predict_npk_from_soil(payload.soil)

    # 2) Weather adjustments (optional)
    adjustments = None
    weather_logic = WeatherSatelliteLogic()
    if payload.include_weather:
        if not payload.weather:
            raise HTTPException(status_code=422, detail="include_weather=true requires 'weather' object")
        ph_fb = payload.weather.ph if payload.weather.ph is not None else payload.soil.Ph
        adjustments = weather_logic.analyze_conditions(payload.weather, ph_fallback=ph_fb)

    # 3) Determine target weeks
    target_weeks = compute_target_weeks(payload.planting_date, payload.adjustment)

    # 4) Build schedule
    calc = FertilizerCalculator()
    report = calc.generate_schedule(
        planting_date=payload.planting_date,
        season="N/A",
        plantation_type=payload.plantation_type,
        seedling_weeks=payload.seedling_weeks or 0,
        cane_variety=payload.cane_variety,
        predicted_npk_total_kg_ha=predicted_npk,
        basal_n_pct=payload.basal_n_pct,
        basal_p_pct=payload.basal_p_pct,
        basal_k_pct=payload.basal_k_pct,
        include_weather=payload.include_weather,
        weather=payload.weather,
        adjustments=adjustments,
        target_weeks=target_weeks,
    )
    return report


# If you want to run directly: `uvicorn main:app --reload`
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
