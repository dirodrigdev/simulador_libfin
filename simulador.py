import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import re
import json

# --- 1. UTILIDADES ---
def fmt(v): return f"{int(v):,}".replace(",", ".")

def clean_input(label, val, key):
    # Usamos session_state para persistencia
    if key not in st.session_state:
        st.session_state[key] = fmt(val)
    
    val_str = st.text_input(label, value=st.session_state[key], key=key)
    clean_val = re.sub(r'\.', '', val_str)
    clean_val = re.sub(r'\D', '', clean_val)
    return int(clean_val) if clean_val else 0

def fmt_pct(v): return f"{v*100:.1f}%"

# --- 2. CONFIGURACIÓN ---
@dataclass
class AssetBucket:
    name: str; weight: float = 0.0; is_bond: bool = False

@dataclass
class WithdrawalTramo:
    from_year: int; to_year: int; amount_nominal_monthly_start: float

@dataclass
class ExtraCashflow:
    year: int; amount: float; name: str

@dataclass
class SimulationConfig:
    horizon_years: int = 40; steps_per_year: int = 12; initial_capital: float = 1800000000; n_sims: int = 2000
    mu_normal_rv: float = 0.11; mu_normal_rf: float = 0.06; inflation_mean: float = 0.035; inflation_vol: float = 0.012
    is_active_managed: bool = True; use_guardrails: bool = True; guardrail_trigger: float = 0.20; guardrail_cut: float = 0.10
    enable_prop: bool = True; net_inmo_value: float = 500000000; new_rent_cost: float = 1500000
    emergency_months_trigger: int = 24; extra_cashflows: List[ExtraCashflow] = field(default_factory=list)
    mu_local_rv: float = -0.15; mu_local_rf: float = 0.08; corr_local: float = -0.25  
    mu_global_rv: float = -0.22; mu_global_rf: float = -0.02; corr_global: float = 0.75  
    prob_enter_local: float = 0.005; prob_enter_global: float = 0.004; prob_exit_crisis: float = 0.085  
    corr_normal: float = 0.35
    t_df: int = 8
    random_seed: Optional[int] = None

# --- 3. MOTOR SOVEREIGN ---
class InstitutionalSimulator:
    def __init__(self, config, assets, withdrawals):
        self.cfg = config; self.assets = assets; self.withdrawals = withdrawals
        self.dt = 1/config.steps_per_year; self.total_steps = int(config.horizon_years * config.steps_per_year)
        self.mu_regimes = np.array([[self.cfg.mu_normal_rv, self.cfg.mu_normal_rf],[self.cfg.mu_local_rv, self.cfg.mu_local_rf],[self.cfg.mu_global_rv, self.cfg.mu_global_rf]])
        vol_f = 0.80 if self.cfg.is_active_managed else 1.0
        self.sigma_regimes = np.array([[0.15, 0.05], [0.22, 0.12], [0.30, 0.14]]) * vol_f
        self.L_mats = [
            np.linalg.cholesky(np.array([[1.0, float(self.cfg.corr_normal)], [float(self.cfg.corr_normal), 1.0]])),
            np.linalg.cholesky(np.array([[1.0, float(self.cfg.corr_local)], [float(self.cfg.corr_local), 1.0]])),
            np.linalg.cholesky(np.array([[1.0, float(self.cfg.corr_global)], [float(self.cfg.corr_global), 1.0]])),
        ]
        self.p_norm_l = self.cfg.prob_enter_local; self.p_norm_g = self.cfg.prob_enter_global; self.p_exit = self.cfg.prob_exit_crisis
        self.rng = np.random.default_rng(self.cfg.random_seed)

    def run(self):
        n_sims, n_steps = self.cfg.n_sims, self.total_steps
        cap_paths = np.zeros((n_sims, n_steps + 1)); cap_paths[:, 0] = self.cfg.initial_capital
        cpi_paths = np.ones((n_sims, n_steps + 1)); is_alive = np.ones(n_sims, dtype=bool)
        ruin_idx = np.full(n_sims, -1); has_h = np.full(n_sims, self.cfg.enable_prop, dtype=bool)
        asset_vals = np.zeros((n_sims, 2))
        rv_w = next((a.weight for a in self.assets if not a.is_bond), 0.6)
        asset_vals[:, 0] = self.cfg.initial_capital * rv_w
        asset_vals[:, 1] = self.cfg.initial_capital * (1 - rv_w)
        
        regime = np.zeros(n_sims, dtype=int)
        df = int(self.cfg.t_df)
        # Multivariate T-Student via shared Chi-square factor per (sim, t) to induce tail dependence.
        g = self.rng.standard_normal((n_sims, n_steps, 2))
        w = self.rng.chisquare(df, (n_sims, n_steps, 1))
        Z_raw = (g / np.sqrt(w / df)) / np.sqrt(df / (df - 2))  # variance-normalized
        inf_sh = self.rng.normal(self.cfg.inflation_mean * self.dt, self.cfg.inflation_vol * np.sqrt(self.dt), (n_sims, n_steps))

        for t in range(n_steps):
            alive = is_alive
            if not np.any(alive): break
            # Markov regimes: ensure a crisis can't enter and exit in the same month.
            reg_prev = regime.copy()
            m0 = (reg_prev == 0) & alive
            if np.any(m0):
                r = self.rng.random(np.sum(m0))
                idx = np.where(m0)[0]
                regime[idx[r < self.p_norm_l]] = 1
                regime[idx[(r >= self.p_norm_l) & (r < (self.p_norm_l + self.p_norm_g))]] = 2
            mc = (reg_prev > 0) & alive
            if np.any(mc):
                idxc = np.where(mc)[0]
                regime[idxc[self.rng.random(np.sum(mc)) < self.p_exit]] = 0

            z_t = Z_raw[:, t, :]; z_f = np.zeros_like(z_t)
            for r_idx, L in enumerate(self.L_mats):
                m = (regime == r_idx) & alive
                if np.any(m): z_f[m] = np.dot(z_t[m], L.T)
            
            p_def = np.ones(n_sims); p_def[regime > 0] = 0.85
            mus, sigs = self.mu_regimes[regime], self.sigma_regimes[regime]
            asset_vals[alive] *= np.exp((mus[alive]-0.5*sigs[alive]**2)*self.dt + sigs[alive]*np.sqrt(self.dt)*z_f[alive]*p_def[alive, None])
            cpi_paths[:, t+1] = cpi_paths[:, t] * (1 + (inf_sh[:, t] + (regime == 1)*0.003))

            if (t+1)%12 == 0:
                y = (t+1)//12
                for e in self.cfg.extra_cashflows:
                    if e.year == y: asset_vals[alive, 1] += e.amount * cpi_paths[alive, t+1]

            cur_y = (t+1)/12
            m_spend = np.zeros(n_sims)
            for w in self.withdrawals:
                if w.from_year <= cur_y < w.to_year:
                    m_spend = w.amount_nominal_monthly_start * cpi_paths[:, t+1]
                    break

            # Guardrails: reduce spending if real patrimonio cae por debajo del umbral.
            if self.cfg.use_guardrails and np.any(alive) and np.any(m_spend > 0):
                cur_real = np.sum(asset_vals, 1) / cpi_paths[:, t+1]
                trig_gr = alive & (cur_real < (self.cfg.initial_capital * (1 - self.cfg.guardrail_trigger)))
                if np.any(trig_gr):
                    m_spend[trig_gr] *= (1 - self.cfg.guardrail_cut)
            
            trig = alive & has_h & (np.sum(asset_vals, 1) < m_spend * self.cfg.emergency_months_trigger)
            if np.any(trig):
                asset_vals[trig, 1] += self.cfg.net_inmo_value * cpi_paths[trig, t+1]; has_h[trig] = False

            # Si se vendió la propiedad, se asume costo adicional de arriendo (nominal) indexado por CPI.
            out = m_spend + (self.cfg.enable_prop & (~has_h)) * (self.cfg.new_rent_cost * cpi_paths[:, t+1])
            wd = np.minimum(out, np.sum(asset_vals, 1))
            rf_b = np.maximum(asset_vals[:, 1], 0); t_rf = np.minimum(wd, rf_b)
            asset_vals[:, 1] -= t_rf; asset_vals[:, 0] -= (wd - t_rf)

            asset_vals = np.maximum(asset_vals, 0); cap_paths[:, t+1] = np.sum(asset_vals, 1)
            dead = (cap_paths[:, t+1] <= 1000) & alive
            if np.any(dead): is_alive[dead]=False; ruin_idx[dead]=t+1; cap_paths[dead, t+1:]=0; asset_vals[dead]=0

        return cap_paths, cpi_paths, ruin_idx


# --- 3B. PORTAFOLIO POR INSTRUMENTOS (Modo "Cartera Real") ---

@dataclass
class InstrumentPosition:
    instrument_id: str
    name: str
    value_clp: float
    rv_share: float
    # Mandato / banda del fondo (aprox). Se usa para modelar "risk-off" del gestor en crisis.
    rv_min: float = 0.0
    rv_max: float = 1.0
    # Liquidez operativa aproximada (dias habiles) para rescate / liquidacion.
    liquidity_days: int = 3

    bucket: str = "BAL"  # RF_PURA | BAL | RV | AFP | PASIVO
    priority: int = 10   # menor = se vende antes
    include_withdrawals: bool = True


@dataclass
class PortfolioRulesConfig:
    rf_reserve_years: float = 3.5
    rebalance_every_months: int = 12
    rebalance_only_when_normal: bool = True
    # Si >0, los balanceados "se ponen defensivos" en crisis (reduce RV efectiva). Es una aproximación.
    manager_riskoff_in_crisis: float = 0.0  # 0..1
    # Bucket ordering for withdrawals
    bucket_order: Dict[str, int] = field(default_factory=lambda: {"RF_PURA": 0, "BAL": 1, "RV": 2, "AFP": 3, "PASIVO": 99})


def _normalize_portfolio_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza distintas fuentes (dashboard vs JSON) a un esquema común."""
    d = df.copy()
    # Common fields
    if "id_instrumento" in d.columns and "instrument_id" not in d.columns:
        d["instrument_id"] = d["id_instrumento"].astype(str)
    if "nombre" in d.columns and "name" not in d.columns:
        d["name"] = d["nombre"].astype(str)
    if "saldo_clp" in d.columns and "value_clp" not in d.columns:
        d["value_clp"] = pd.to_numeric(d["saldo_clp"], errors="coerce").fillna(0.0)

    keep_cols = [c for c in ["instrument_id", "name", "value_clp", "tipo", "subtipo", "moneda"] if c in d.columns]
    d = d[keep_cols].copy()
    d["instrument_id"] = d["instrument_id"].astype(str)
    d["name"] = d["name"].astype(str)
    d["value_clp"] = pd.to_numeric(d["value_clp"], errors="coerce").fillna(0.0)
    return d


def parse_portfolio_json(json_str: str) -> pd.DataFrame:
    data = json.loads(json_str)
    regs = data.get("registros", [])
    if not regs:
        raise ValueError("JSON sin 'registros'")
    inst = regs[0].get("instrumentos", [])
    if not inst:
        raise ValueError("JSON sin 'instrumentos'")
    df = pd.DataFrame(inst)
    return _normalize_portfolio_df(df)


def default_instrument_meta() -> Dict[str, Dict[str, Any]]:
    """Defaults por instrumento.

    - rv_share: % RV objetivo aproximado (0..1)
    - rv_min / rv_max: banda plausible del mandato (para modelar "risk-off" en crisis)
    - liquidity_days: dias habiles aproximados de rescate (para orden y UX)

    OJO: esto es *config por defecto*; en la UI se puede editar.
    """
    return {
        # --- SURA (Seguro de Vida) ---
        # Mandato oficial: instrumentos de capitalización min 60% / max 100% (ver ficha).
        "SURA_SEGURO_MULTIACTIVO_AGRESIVO_SERIE_F": {"rv_share": 0.8913, "rv_min": 0.60, "rv_max": 1.00, "liquidity_days": 4, "bucket": "RV", "priority": 40},
        "SURA_SEGURO_MULTIACTIVO_MODERADO_SERIE_F": {"rv_share": 0.5452, "rv_min": 0.30, "rv_max": 0.60, "liquidity_days": 4, "bucket": "BAL", "priority": 35},
        "SURA_SEGURO_RENTA_LOCAL_UF_SERIE_F": {"rv_share": 0.0, "rv_min": 0.0, "rv_max": 0.0, "liquidity_days": 3, "bucket": "RF_PURA", "priority": 0},
        "SURA_SEGURO_RENTA_BONOS_CHILE_SF": {"rv_share": 0.0, "rv_min": 0.0, "rv_max": 0.0, "liquidity_days": 3, "bucket": "RF_PURA", "priority": 0},

        # --- BTG (Fondos Mutuos) ---
        # RV estimada desde la "Distribución por Clase de Activo" de cada ficha.
        # - Gestión Agresiva: RV Int + RV Nac ≈ 68.54%. Mandato: capitalización 0..90%.
        "BTG_GESTION_AGRESIVA": {"rv_share": 0.6854, "rv_min": 0.00, "rv_max": 0.90, "liquidity_days": 2, "bucket": "BAL", "priority": 45},
        # - Gestión Activa: RV Int + RV Nac ≈ 38.65% (según ficha). Banda exacta del mandato no aparece en el extracto, se deja amplia.
        "BTG_GESTION_ACTIVA": {"rv_share": 0.3865, "rv_min": 0.00, "rv_max": 0.90, "liquidity_days": 2, "bucket": "BAL", "priority": 30},
        # - Gestión Conservadora: por mandato deuda >=80% => RV <=20%. RV actual ≈ 17.73%.
        "BTG_GESTION_CONSERVADORA": {"rv_share": 0.1773, "rv_min": 0.00, "rv_max": 0.20, "liquidity_days": 1, "bucket": "BAL", "priority": 15},

        # --- Fondo de inversión / crédito privado ---
        # Moneda Renta CLP (fondo de inversión): acciones ~2.9% (resto deuda/alternativos)
        "MONEDA_RENTA_CLP": {"rv_share": 0.029, "rv_min": 0.0, "rv_max": 0.10, "liquidity_days": 3, "bucket": "RF_PURA", "priority": 5},

        # --- DAP y caja ---
        "DAP_CLP_10122025": {"rv_share": 0.0, "rv_min": 0.0, "rv_max": 0.0, "liquidity_days": 0, "bucket": "RF_PURA", "priority": 0},
        "WISE_USD": {"rv_share": 0.0, "rv_min": 0.0, "rv_max": 0.0, "liquidity_days": 0, "bucket": "RF_PURA", "priority": 0},
        "GLOBAL66_USD": {"rv_share": 0.0, "rv_min": 0.0, "rv_max": 0.0, "liquidity_days": 0, "bucket": "RF_PURA", "priority": 0},

        # --- Offshore (SURA) ---
        "SURA_USD_SHORT_DURATION": {"rv_share": 0.0, "rv_min": 0.0, "rv_max": 0.0, "liquidity_days": 5, "bucket": "RF_PURA", "priority": 2},
        "SURA_USD_MONEY_MARKET": {"rv_share": 0.0, "rv_min": 0.0, "rv_max": 0.0, "liquidity_days": 5, "bucket": "RF_PURA", "priority": 1},

        # --- Previsional / AFP / APV ---
        # Por defecto queda incluido como parte del patrimonio y *retirable*, pero en orden de venta queda al final.
        "SURA_APV_MULTIACTIVO_AGRESIVO": {"rv_share": 0.8913, "rv_min": 0.80, "rv_max": 1.00, "liquidity_days": 10, "bucket": "RV", "priority": 80},
        "SURA_DC_MULTIACTIVO_AGRESIVO": {"rv_share": 0.8913, "rv_min": 0.80, "rv_max": 1.00, "liquidity_days": 10, "bucket": "RV", "priority": 80},
        "AFP_PLANVITAL_OBLIGATORIA": {"rv_share": 0.75, "rv_min": 0.60, "rv_max": 0.85, "liquidity_days": 10, "bucket": "RV", "priority": 85},
        "AFP_PLANVITAL_AV_TRANSITORIO": {"rv_share": 0.75, "rv_min": 0.60, "rv_max": 0.85, "liquidity_days": 10, "bucket": "RV", "priority": 85},
        "AFP_PLANVITAL_AV_OPCIONAL": {"rv_share": 0.75, "rv_min": 0.60, "rv_max": 0.85, "liquidity_days": 10, "bucket": "RV", "priority": 85},
        "AFP_PLANVITAL_AV_RETIRO10": {"rv_share": 0.75, "rv_min": 0.60, "rv_max": 0.85, "liquidity_days": 10, "bucket": "RV", "priority": 85},
        "AFP_PLANVITAL_AV_GENERAL": {"rv_share": 0.75, "rv_min": 0.60, "rv_max": 0.85, "liquidity_days": 10, "bucket": "RV", "priority": 85},
        "AFP_PLANVITAL_AV_54BIS": {"rv_share": 0.75, "rv_min": 0.60, "rv_max": 0.85, "liquidity_days": 10, "bucket": "RV", "priority": 85},
    }



def enrich_with_meta(df: pd.DataFrame) -> pd.DataFrame:
    meta = default_instrument_meta()
    out = df.copy()
    out["rv_share"] = 0.0
    out["rv_min"] = 0.0
    out["rv_max"] = 1.0
    out["liquidity_days"] = 3
    out["bucket"] = "BAL"
    out["priority"] = 30
    out["include_withdrawals"] = True

    for i, r in out.iterrows():
        iid = str(r.get("instrument_id", ""))
        tip = str(r.get("tipo", ""))
        if "Pasivo" in tip or "Deuda" in tip:
            out.loc[i, "bucket"] = "PASIVO"
            out.loc[i, "include_withdrawals"] = False
            out.loc[i, "priority"] = 99
            continue

        if iid in meta:
            out.loc[i, "rv_share"] = float(meta[iid]["rv_share"])
            out.loc[i, "rv_min"] = float(meta[iid].get("rv_min", 0.0))
            out.loc[i, "rv_max"] = float(meta[iid].get("rv_max", 1.0))
            out.loc[i, "liquidity_days"] = int(meta[iid].get("liquidity_days", 3))
            out.loc[i, "bucket"] = meta[iid]["bucket"]
            out.loc[i, "priority"] = int(meta[iid]["priority"])
        else:
            # Heurística suave: cash / money market / bonos = RF
            name = str(r.get("name", "")).lower()
            if any(k in iid for k in ["USD_MONEY_MARKET", "SHORT_DURATION", "WISE", "GLOBAL66", "DAP"]):
                out.loc[i, "rv_share"] = 0.0
                out.loc[i, "rv_min"] = 0.0
                out.loc[i, "rv_max"] = 0.0
                out.loc[i, "liquidity_days"] = 0
                out.loc[i, "bucket"] = "RF_PURA"
                out.loc[i, "priority"] = 0
            elif "renta" in name and "bonos" in name:
                out.loc[i, "rv_share"] = 0.0
                out.loc[i, "rv_min"] = 0.0
                out.loc[i, "rv_max"] = 0.0
                out.loc[i, "liquidity_days"] = 3
                out.loc[i, "bucket"] = "RF_PURA"
                out.loc[i, "priority"] = 0
            else:
                out.loc[i, "rv_share"] = 0.30
                out.loc[i, "rv_min"] = 0.0
                out.loc[i, "rv_max"] = 0.6
                out.loc[i, "liquidity_days"] = 3
                out.loc[i, "bucket"] = "BAL"
                out.loc[i, "priority"] = 30

    # clamp + consistencia
    out["rv_min"] = pd.to_numeric(out["rv_min"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    out["rv_max"] = pd.to_numeric(out["rv_max"], errors="coerce").fillna(1.0).clip(0.0, 1.0)
    # Asegura rv_min <= rv_max
    out["rv_max"] = out[["rv_min", "rv_max"]].max(axis=1)
    out["rv_share"] = pd.to_numeric(out["rv_share"], errors="coerce").fillna(0.0)
    out["rv_share"] = out.apply(lambda r: min(max(float(r["rv_share"]), float(r["rv_min"])), float(r["rv_max"])), axis=1)
    out["liquidity_days"] = pd.to_numeric(out["liquidity_days"], errors="coerce").fillna(3).astype(int)
    out["value_clp"] = pd.to_numeric(out["value_clp"], errors="coerce").fillna(0.0)
    return out


class PortfolioSimulator:
    """Simula una cartera de instrumentos como combinaciones RV/RF con rebalanceo interno mensual.

    Nota: esto aproxima "fondos balanceados" como mix objetivo (rv_share) que se rebalancea cada mes.
    """

    def __init__(self, cfg: SimulationConfig, positions: List[InstrumentPosition], withdrawals: List[WithdrawalTramo], rules: PortfolioRulesConfig):
        self.cfg = cfg
        self.positions = positions
        self.withdrawals = withdrawals
        self.rules = rules
        self.dt = 1 / cfg.steps_per_year
        self.total_steps = int(cfg.horizon_years * cfg.steps_per_year)
        self.rng = np.random.default_rng(cfg.random_seed)

        # Market regimes (same as InstitutionalSimulator)
        self.mu_regimes = np.array([
            [cfg.mu_normal_rv, cfg.mu_normal_rf],
            [cfg.mu_local_rv, cfg.mu_local_rf],
            [cfg.mu_global_rv, cfg.mu_global_rf],
        ])
        vol_f = 0.80 if cfg.is_active_managed else 1.0
        self.sigma_regimes = np.array([
            [0.15, 0.05],
            [0.22, 0.12],
            [0.30, 0.14],
        ]) * vol_f
        self.L_mats = [
            np.linalg.cholesky(np.array([[1.0, float(cfg.corr_normal)], [float(cfg.corr_normal), 1.0]])),
            np.linalg.cholesky(np.array([[1.0, float(cfg.corr_local)], [float(cfg.corr_local), 1.0]])),
            np.linalg.cholesky(np.array([[1.0, float(cfg.corr_global)], [float(cfg.corr_global), 1.0]])),
        ]

    def _withdraw_order(self) -> List[int]:
        # Order by bucket order, then by priority, then by rv_share
        def k(p: InstrumentPosition):
            return (
                self.rules.bucket_order.get(p.bucket, 50),
                int(p.priority),
                int(getattr(p, 'liquidity_days', 3)),
                float(p.rv_share),
                p.instrument_id,
            )
        idxs = [i for i, p in enumerate(self.positions) if p.include_withdrawals and p.bucket != "PASIVO" and p.value_clp > 0]
        return sorted(idxs, key=lambda i: k(self.positions[i]))

    def _pick_rf_sink(self) -> Optional[int]:
        # A donde "cae" el rebalanceo anual hacia RF pura.
        # Preferimos el instrumento RF mas liquido (menos dias) y con menor prioridad.
        candidates = [i for i, p in enumerate(self.positions) if p.bucket == "RF_PURA" and p.include_withdrawals]
        if not candidates:
            return None
        return sorted(
            candidates,
            key=lambda i: (
                int(getattr(self.positions[i], 'liquidity_days', 3)),
                int(self.positions[i].priority),
                self.positions[i].instrument_id,
            ),
        )[0]

    def run(self):
        n_sims, n_steps = self.cfg.n_sims, self.total_steps
        n_instr = len(self.positions)
        if n_instr == 0:
            raise ValueError("Cartera vacía")

        values = np.zeros((n_sims, n_instr), dtype=float)
        for j, p in enumerate(self.positions):
            values[:, j] = float(p.value_clp)

        cap_paths = np.zeros((n_sims, n_steps + 1), dtype=float)
        cap_paths[:, 0] = np.sum(values, axis=1)
        cpi_paths = np.ones((n_sims, n_steps + 1), dtype=float)
        is_alive = np.ones(n_sims, dtype=bool)
        ruin_idx = np.full(n_sims, -1)
        has_h = np.full(n_sims, self.cfg.enable_prop, dtype=bool)
        regime = np.zeros(n_sims, dtype=int)

        df = int(self.cfg.t_df)
        g = self.rng.standard_normal((n_sims, n_steps, 2))
        w = self.rng.chisquare(df, (n_sims, n_steps, 1))
        Z_raw = (g / np.sqrt(w / df)) / np.sqrt(df / (df - 2))
        inf_sh = self.rng.normal(self.cfg.inflation_mean * self.dt, self.cfg.inflation_vol * np.sqrt(self.dt), (n_sims, n_steps))

        wd_order = self._withdraw_order()
        rf_sink = self._pick_rf_sink()
        rf_idxs = [j for j, p in enumerate(self.positions) if p.bucket == "RF_PURA" and p.include_withdrawals]

        initial_real = cap_paths[:, 0].copy()  # CPI starts at 1

        for t in range(n_steps):
            alive = is_alive
            if not np.any(alive):
                break

            # Markov (no enter+exit same month)
            reg_prev = regime.copy()
            m0 = (reg_prev == 0) & alive
            if np.any(m0):
                r = self.rng.random(np.sum(m0))
                idx = np.where(m0)[0]
                regime[idx[r < self.cfg.prob_enter_local]] = 1
                regime[idx[(r >= self.cfg.prob_enter_local) & (r < (self.cfg.prob_enter_local + self.cfg.prob_enter_global))]] = 2
            mc = (reg_prev > 0) & alive
            if np.any(mc):
                idxc = np.where(mc)[0]
                regime[idxc[self.rng.random(np.sum(mc)) < self.cfg.prob_exit_crisis]] = 0

            # Market step
            z_t = Z_raw[:, t, :]
            z_f = np.zeros_like(z_t)
            for r_idx, L in enumerate(self.L_mats):
                m = (regime == r_idx) & alive
                if np.any(m):
                    z_f[m] = np.dot(z_t[m], L.T)

            p_def = np.ones(n_sims)
            p_def[regime > 0] = 0.85
            mus = self.mu_regimes[regime]
            sigs = self.sigma_regimes[regime]
            # factor returns (continuous)
            r_rv = (mus[:, 0] - 0.5 * sigs[:, 0] ** 2) * self.dt + sigs[:, 0] * np.sqrt(self.dt) * z_f[:, 0] * p_def
            r_rf = (mus[:, 1] - 0.5 * sigs[:, 1] ** 2) * self.dt + sigs[:, 1] * np.sqrt(self.dt) * z_f[:, 1] * p_def

            # Update instrument values as monthly rebalanced mix
            for j, p in enumerate(self.positions):
                if p.bucket == "PASIVO":
                    continue

                # Exposición objetivo
                rv_base = float(p.rv_share)
                rv_min = float(getattr(p, "rv_min", 0.0))
                rv_max = float(getattr(p, "rv_max", 1.0))

                # En crisis, el gestor puede bajar la exposición a RV dentro de su banda (aprox).
                rv_eff = np.full(n_sims, rv_base, dtype=float)
                if self.rules.manager_riskoff_in_crisis > 0 and p.bucket in ("BAL", "RV", "AFP"):
                    crisis = (regime > 0) & alive
                    if np.any(crisis):
                        target = rv_base - float(self.rules.manager_riskoff_in_crisis) * (rv_base - rv_min)
                        target = min(max(target, rv_min), rv_max)
                        rv_eff[crisis] = target

                v = values[:, j]
                v_rv = v * rv_eff
                v_rf = v * (1 - rv_eff)
                v_new = v_rv * np.exp(r_rv) + v_rf * np.exp(r_rf)
                values[:, j] = np.where(alive, v_new, v)

            # Inflation
            cpi_paths[:, t + 1] = cpi_paths[:, t] * (1 + (inf_sh[:, t] + (regime == 1) * 0.003))

            # Extra cashflows -> into RF sink (or first instrument)
            if (t + 1) % 12 == 0:
                y = (t + 1) // 12
                for e in self.cfg.extra_cashflows:
                    if e.year == y:
                        sink = rf_sink if rf_sink is not None else 0
                        values[alive, sink] += e.amount * cpi_paths[alive, t + 1]

            # Monthly spend (nominal)
            cur_y = (t + 1) / 12
            m_spend = np.zeros(n_sims)
            for wtr in self.withdrawals:
                if wtr.from_year <= cur_y < wtr.to_year:
                    m_spend = wtr.amount_nominal_monthly_start * cpi_paths[:, t + 1]
                    break

            # Guardrails
            if self.cfg.use_guardrails and np.any(m_spend > 0):
                cur_real = np.sum(values, axis=1) / cpi_paths[:, t + 1]
                trig_gr = alive & (cur_real < (initial_real * (1 - self.cfg.guardrail_trigger)))
                if np.any(trig_gr):
                    m_spend[trig_gr] *= (1 - self.cfg.guardrail_cut)

            # Property emergency sale -> inject into RF sink
            if self.cfg.enable_prop:
                tot = np.sum(values, axis=1)
                trig = alive & has_h & (tot < (m_spend * self.cfg.emergency_months_trigger))
                if np.any(trig):
                    sink = rf_sink if rf_sink is not None else 0
                    values[trig, sink] += self.cfg.net_inmo_value * cpi_paths[trig, t + 1]
                    has_h[trig] = False

            # Withdrawal waterfall
            out = m_spend + (self.cfg.enable_prop & (~has_h)) * (self.cfg.new_rent_cost * cpi_paths[:, t + 1])
            need = out.copy()
            for j in wd_order:
                if not np.any(alive & (need > 0)):
                    break
                take = np.minimum(values[:, j], need)
                take = np.where(alive, take, 0.0)
                values[:, j] -= take
                need -= take

            # After withdrawals, if still need > 0 -> ruin
            dead = alive & (need > 1e-6)
            if np.any(dead):
                is_alive[dead] = False
                ruin_idx[dead] = t + 1
                values[dead, :] = 0.0

            cap_paths[:, t + 1] = np.sum(values, axis=1)

            # Annual refill of RF_PURA reserve (manual rebalance approximation)
            if self.rules.rebalance_every_months > 0 and ((t + 1) % self.rules.rebalance_every_months == 0) and (self.rules.rf_reserve_years > 0):
                if rf_sink is not None and len(rf_idxs) > 0:
                    target = (self.rules.rf_reserve_years * 12) * np.maximum(m_spend, 0)  # nominal reserve target
                    cur_rf = np.sum(values[:, rf_idxs], axis=1)
                    need_rf = np.maximum(0.0, target - cur_rf)

                    # Only rebalance when normal, if requested
                    allow = alive.copy()
                    if self.rules.rebalance_only_when_normal:
                        allow &= (regime == 0)
                    if np.any(allow & (need_rf > 0)):
                        # pull pro-rata from sources: BAL -> RV -> AFP
                        sources_by_bucket = ["BAL", "RV", "AFP"]
                        for b in sources_by_bucket:
                            src = [j for j, p in enumerate(self.positions) if p.bucket == b and p.include_withdrawals]
                            if not src:
                                continue
                            pool = np.sum(values[:, src], axis=1)
                            can = allow & (need_rf > 0) & (pool > 0)
                            if not np.any(can):
                                continue
                            # distribute need_rf across src pro-rata
                            weights = np.zeros((n_sims, len(src)))
                            weights[can, :] = values[can][:, src] / pool[can][:, None]
                            transfer = weights * need_rf[:, None]
                            # cap transfers by available value
                            for k, j in enumerate(src):
                                t_k = np.minimum(values[:, j], transfer[:, k])
                                t_k = np.where(can, t_k, 0.0)
                                values[:, j] -= t_k
                                values[:, rf_sink] += t_k
                                need_rf -= t_k

        return cap_paths, cpi_paths, ruin_idx

# --- 4. INTERFAZ ---
def app(
    default_rf: int = 720000000,
    default_rv: int = 1080000000,
    default_inmo_neto: int = 500000000,
    portfolio_df: Optional[pd.DataFrame] = None,
    macro_data: Optional[Dict[str, Any]] = None,
    portfolio_json: Optional[str] = None,
    **_ignored_kwargs,
):
    st.markdown("## 🦅 Panel de Decisión Patrimonial")
    
    SC_RET = {"Conservador": [0.08, 0.045], "Histórico (11%)": [0.11, 0.06], "Crecimiento (13%)": [0.13, 0.07]}
    SC_GLO = {"Crash Financiero": [-0.22, -0.02, 0.75], "Colapso Sistémico": [-0.30, -0.06, 0.92], "Recesión Estándar": [-0.15, 0.01, 0.55]}

    with st.sidebar:
        st.header("⚙️ Configuración")
        use_portfolio = st.toggle(
            "Modo Cartera Real (Instrumentos)",
            value=st.session_state.get("use_portfolio", False),
            help="Simula instrumento por instrumento (RF pura + balanceados + AFP/APV), y aplica la regla de retiro por prioridad.",
        )
        st.session_state["use_portfolio"] = use_portfolio
        is_active = st.toggle("Gestión Activa / Balanceados", value=True)
        sel_glo = st.selectbox("Crisis Global", list(SC_GLO.keys()), index=0)
        st.caption(f"Stress RV: {SC_GLO[sel_glo][0]*100}%")
        
        st.divider()
        sel_ret = st.selectbox("Rentabilidad Normal", list(SC_RET.keys()) + ["Personalizado"], index=1)
        if sel_ret == "Personalizado":
            c_rv = st.number_input("RV % Nom.", 0.0, 25.0, 11.0)/100
            c_rf = st.number_input("RF % Nom.", 0.0, 15.0, 6.0)/100
        else:
            c_rv, c_rf = SC_RET[sel_ret]
            st.info(f"RV: {c_rv*100}% | RF: {c_rf*100}%")
        
        n_sims = st.slider("Simulaciones", 500, 3000, 1000)
        horiz = st.slider("Horizonte", 10, 50, 40)

    tab_sim, tab_opt = st.tabs(["📊 Simulador", "🎯 Optimizador"])

    # Valores forzados para Diego
    tot_ini = default_rf + default_rv
    pct_rv_ini = 60

    with tab_sim:
        # --- 1) Selección de modo ---
        positions: List[InstrumentPosition] = []
        rules = PortfolioRulesConfig()
        portfolio_ready = False

        if use_portfolio:
            st.subheader("📦 Cartera Real (Instrumentos)")
            st.caption("Edita el % RV de cada instrumento y su prioridad de retiro. Por defecto: RF pura primero → luego balanceados → lo más RV al final.")

            src = st.radio(
                "Fuente de cartera",
                ["Usar datos del Dashboard", "Pegar JSON"],
                horizontal=True,
            )

            df_src = None
            if src == "Usar datos del Dashboard":
                df_src = portfolio_df
                if df_src is None and "portfolio_df" in st.session_state:
                    df_src = st.session_state["portfolio_df"]
            else:
                default_txt = portfolio_json or st.session_state.get("portfolio_json", "")
                txt = st.text_area("JSON", value=default_txt, height=180, placeholder="Pega acá el JSON completo (registros → instrumentos)")
                if txt.strip():
                    st.session_state["portfolio_json"] = txt
                    try:
                        df_src = parse_portfolio_json(txt)
                    except Exception as e:
                        st.error(f"No pude parsear el JSON: {e}")

            if df_src is None or (hasattr(df_src, "empty") and df_src.empty):
                st.warning("No hay datos de cartera para simular en modo instrumentos.")
            else:
                df_base = _normalize_portfolio_df(df_src)
                df_base = enrich_with_meta(df_base)

                edited = st.data_editor(
                    df_base,
                    use_container_width=True,
                    num_rows="fixed",
                    column_config={
                        "rv_share": st.column_config.NumberColumn("% RV (0-1)", min_value=0.0, max_value=1.0, step=0.01),
                        "priority": st.column_config.NumberColumn("Prioridad retiro", min_value=0, max_value=99, step=1),
                        "include_withdrawals": st.column_config.CheckboxColumn("Retirable"),
                        "bucket": st.column_config.SelectboxColumn("Bucket", options=["RF_PURA", "BAL", "RV", "AFP", "PASIVO"]),
                    },
                    disabled=["instrument_id", "name", "rv_min", "rv_max", "liquidity_days"],
                    key="portfolio_editor",
                )

                tot = float(edited["value_clp"].sum())
                rv_amt = float((edited["value_clp"] * edited["rv_share"]).sum())
                rf_pura_amt = float(edited.loc[edited["bucket"] == "RF_PURA", "value_clp"].sum())
                rv_pct = 0.0 if tot <= 0 else 100.0 * rv_amt / tot
                # Para compatibilidad con la UI "simple" (p.ej. optimizer)
                rv_sl = int(round(rv_pct))

                m1, m2, m3 = st.columns(3)
                with m1: st.metric("Patrimonio (CLP)", f"${fmt(tot)}")
                with m2: st.metric("Motor (RV) estimado", f"{rv_pct:.1f}%")
                with m3: st.metric("RF pura hoy", f"${fmt(rf_pura_amt)}")

                rr1, rr2, rr3, rr4 = st.columns(4)
                with rr1:
                    rules.rf_reserve_years = st.slider("Reserva RF pura (años)", 0.0, 6.0, 3.5, 0.5)
                with rr2:
                    reb = st.checkbox("Rebalance anual RF pura", value=True)
                    rules.rebalance_every_months = 12 if reb else 0
                with rr3:
                    rules.rebalance_only_when_normal = st.checkbox("Rebalance solo si NO hay crisis", value=True)
                with rr4:
                    rules.manager_riskoff_in_crisis = st.slider(
                        "Gestor defensivo en crisis (0-1)",
                        0.0,
                        1.0,
                        float(rules.manager_riskoff_in_crisis),
                        0.05,
                        help="Si subes esto, los balanceados reducen su %RV efectivo en crisis dentro de su banda RV min/max. Útil si crees que el gestor rota a bonos cuando se pone feo.",
                    )

                positions = []
                for _, r in edited.iterrows():
                    positions.append(
                        InstrumentPosition(
                            instrument_id=str(r["instrument_id"]),
                            name=str(r["name"]),
                            value_clp=float(r["value_clp"]),
                            rv_share=float(r["rv_share"]),
                            rv_min=float(r.get("rv_min", 0.0)),
                            rv_max=float(r.get("rv_max", 1.0)),
                            liquidity_days=int(r.get("liquidity_days", 3)),
                            bucket=str(r["bucket"]),
                            priority=int(r["priority"]),
                            include_withdrawals=bool(r["include_withdrawals"]),
                        )
                    )

                cap_val = int(round(tot))
                rv_sl = rv_pct
                portfolio_ready = cap_val > 0 and any(p.include_withdrawals for p in positions)

        if not use_portfolio:
            c1, c2, c3 = st.columns(3)
            with c1: cap_val = clean_input("Capital Total ($)", tot_ini, "cap")
            with c2: rv_sl = st.slider("Motor (RV %)", 0, 100, pct_rv_ini)
            with c3: st.metric("Reserva RF", f"${fmt(cap_val * (1-rv_sl/100))}")
        
        g1, g2, g3 = st.columns(3)
        with g1: r1 = clean_input("Gasto F1", 6000000, "r1"); d1 = st.number_input("Años F1", 0, 40, 7)
        with g2: r2 = clean_input("Gasto F2", 5500000, "r2"); d2 = st.number_input("Años F2", 0, 40, 13)
        with g3: r3 = clean_input("Gasto F3", 5000000, "r3")

        with st.expander("💸 Inyecciones o Salidas"):
            if 'extra_events' not in st.session_state: st.session_state.extra_events = []
            ce1, ce2, ce3, ce4 = st.columns([1,2,2,1])
            with ce1: ev_y = st.number_input("Año", 1, 40, 5, key="evy")
            with ce2: ev_a = clean_input("Monto ($)", 0, "eva")
            with ce3: ev_t = st.selectbox("Tipo", ["Entrada", "Salida"], key="evt")
            if ce4.button("Add"): st.session_state.extra_events.append(ExtraCashflow(ev_y, ev_a if ev_t=="Entrada" else -ev_a, "Hito"))
            for e in st.session_state.extra_events: st.text(f"Año {e.year}: ${fmt(e.amount)}")
            if st.button("Limpiar"): st.session_state.extra_events = []

        enable_p = st.checkbox("Venta Casa Emergencia", value=True)
        val_h = clean_input("Valor Neto Casa ($)", default_inmo_neto, "vi") if enable_p else 0

        if st.button("🚀 INICIAR SIMULACIÓN", type="primary"):
            wds = [WithdrawalTramo(0, d1, r1), WithdrawalTramo(d1, d1+d2, r2), WithdrawalTramo(d1+d2, horiz, r3)]
            cfg = SimulationConfig(
                horizon_years=horiz,
                initial_capital=cap_val,
                n_sims=n_sims,
                is_active_managed=is_active,
                enable_prop=enable_p,
                net_inmo_value=val_h,
                mu_normal_rv=c_rv,
                mu_normal_rf=c_rf,
                extra_cashflows=st.session_state.extra_events,
                mu_global_rv=SC_GLO[sel_glo][0],
                mu_global_rf=SC_GLO[sel_glo][1],
                corr_global=SC_GLO[sel_glo][2],
            )

            if use_portfolio:
                if not portfolio_ready:
                    st.error("Modo instrumentos activo, pero no hay cartera válida (o no hay nada marcado como retirable).")
                    st.stop()
                sim = PortfolioSimulator(cfg, positions, wds, rules)
                paths, cpi, r_i = sim.run()
            else:
                assets = [AssetBucket("RV", float(rv_sl)/100), AssetBucket("RF", (100-float(rv_sl))/100, True)]
                sim = InstitutionalSimulator(cfg, assets, wds)
                paths, cpi, r_i = sim.run()
            prob = (1 - (np.sum(r_i > -1)/n_sims))*100
            st.markdown(f"<div style='text-align:center; padding:20px; border-radius:10px; background:rgba(30,30,30,0.5);'><h1>Éxito: {prob:.1f}%</h1><h3>Legado Real: ${fmt(np.median(paths[:,-1]/cpi[:,-1]))}</h3></div>", unsafe_allow_html=True)
            y_ax = np.arange(paths.shape[1])/12; fig = go.Figure()
            fig.add_trace(go.Scatter(x=np.concatenate([y_ax, y_ax[::-1]]), y=np.concatenate([np.percentile(paths, 90, 0), np.percentile(paths, 10, 0)[::-1]]), fill='toself', fillcolor='rgba(59,130,246,0.2)', line=dict(color='rgba(0,0,0,0)'), name='Rango 80%'))
            fig.add_trace(go.Scatter(x=y_ax, y=np.percentile(paths, 50, 0), line=dict(color='#3b82f6', width=3), name='Mediana'))
            fig.update_layout(title="Proyección Patrimonio (Nominal)", template="plotly_dark"); st.plotly_chart(fig, use_container_width=True)

    with tab_opt:
        st.subheader("🎯 Buscador de Soluciones Estratégicas")
        target = st.slider("Meta Éxito %", 70, 100, 95)
        opt_var = st.selectbox("Optimizar:", ["Mix RV %", "Gasto Fase 1", "Gasto Fase 2", "Gasto Fase 3"])
        if st.button("🔍 CALCULAR ÓPTIMO"):
            res = []
            with st.spinner("Ejecutando..."):
                if "Mix" in opt_var:
                    vals = np.linspace(0, 100, 11)
                else:
                    base = r1 if "1" in opt_var else (r2 if "2" in opt_var else r3)
                    vals = np.linspace(base*0.4, base*1.6, 11)
                seed = 12345  # Common Random Numbers para comparar manzanas con manzanas
                for v in vals:
                    cur_rv = v/100 if "Mix" in opt_var else rv_sl/100
                    a_t = [AssetBucket("RV", cur_rv), AssetBucket("RF", (1-cur_rv), True)]
                    tw1 = (v if opt_var == "Gasto Fase 1" else r1)
                    tw2 = (v if opt_var == "Gasto Fase 2" else r2)
                    tw3 = (v if opt_var == "Gasto Fase 3" else r3)
                    w_t = [WithdrawalTramo(0, d1, tw1), WithdrawalTramo(d1, d1+d2, tw2), WithdrawalTramo(d1+d2, horiz, tw3)]
                    c_t = SimulationConfig(
                        horizon_years=horiz,
                        initial_capital=cap_val,
                        n_sims=400,
                        random_seed=seed,
                        is_active_managed=is_active,
                        enable_prop=enable_p,
                        net_inmo_value=val_h,
                        mu_normal_rv=c_rv,
                        mu_normal_rf=c_rf,
                        mu_global_rv=SC_GLO[sel_glo][0],
                        mu_global_rf=SC_GLO[sel_glo][1],
                        corr_global=SC_GLO[sel_glo][2],
                        extra_cashflows=st.session_state.extra_events,
                    )
                    _, _, ri = InstitutionalSimulator(c_t, a_t, w_t).run()
                    res.append({"v": v, "p": (1-(np.sum(ri>-1)/400))*100})
            df_o = pd.DataFrame(res); best = df_o.iloc[(df_o['p']-target).abs().argsort()[:1]].iloc[0]
            st.success(f"✅ Valor ideal: **{fmt(best['v']) if 'Gasto' in opt_var else str(int(best['v']))+'%'}**")
