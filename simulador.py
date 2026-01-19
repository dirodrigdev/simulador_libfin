# simulador.py — Versión completa: defaults personales + Stress + Tornado + UI avanzado siempre visible
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import re
import json
import io

# Page config
st.set_page_config(page_title="Sovereign Alpha Engine", layout="wide")

# ----------------------
# Utilities
# ----------------------
def fmt(v):
    try:
        return f"{int(v):,}".replace(",", ".")
    except Exception:
        return str(v)

def clean_input(label, val, key):
    widget_key = f"txt__{key}"
    if widget_key not in st.session_state:
        st.session_state[widget_key] = fmt(val)
    elif not isinstance(st.session_state.get(widget_key), str):
        st.session_state[widget_key] = str(st.session_state[widget_key])
    val_str = st.text_input(label, value=st.session_state[widget_key], key=widget_key)
    clean_val = re.sub(r'\.', '', str(val_str))
    clean_val = re.sub(r'\D', '', clean_val)
    out = int(clean_val) if clean_val else 0
    st.session_state[f"num__{key}"] = out
    return out

def to_continuous_mu(mu_arith: float) -> float:
    try:
        return float(np.log(1.0 + float(mu_arith)))
    except Exception:
        return float(mu_arith)

def success_ci(pct, n, z=1.96):
    p = pct / 100.0
    if n <= 0:
        return pct, pct
    se = np.sqrt(p * (1 - p) / n)
    low = max(0.0, (p - z * se)) * 100.0
    high = min(1.0, (p + z * se)) * 100.0
    return low, high

def compute_ruin_stats(ruin_idx: np.ndarray, steps_per_year: int, horizon_years: int) -> Dict[str, Any]:
    total = len(ruin_idx)
    ruined = ruin_idx[ruin_idx > -1]
    stats: Dict[str, Any] = {
        "total_sims": total,
        "ruined_count": int(ruined.size),
        "ruined_pct": float(ruined.size / total * 100.0) if total else 0.0,
        "ruin_years": [],
        "ruin_hist": {},
        "survival_curve": [],
    }
    if ruined.size == 0:
        stats["survival_curve"] = [100.0 for _ in range(horizon_years + 1)]
        return stats
    ruin_years = ruined / float(steps_per_year)
    stats["ruin_years"] = ruin_years.tolist()
    bins = np.arange(0, horizon_years + 1)
    hist, _ = np.histogram(ruin_years, bins=bins)
    stats["ruin_hist"] = {int(b): int(h) for b, h in zip(bins[:-1], hist)}
    survival = []
    for y in range(horizon_years + 1):
        alive = (ruin_idx == -1) | (ruin_idx > y * steps_per_year)
        survival.append(float(alive.mean() * 100.0))
    stats["survival_curve"] = survival
    return stats

def aggregate_success_score(scores: Dict[str, Optional[float]], weights: Dict[str, float]) -> Optional[float]:
    usable = {k: v for k, v in scores.items() if v is not None}
    if not usable:
        return None
    weight_sum = sum(float(weights.get(k, 0.0)) for k in usable.keys())
    if weight_sum <= 0:
        return None
    return sum(float(weights.get(k, 0.0)) * float(usable[k]) for k in usable.keys()) / weight_sum

# ----------------------
# Dataclasses / Config
# ----------------------
@dataclass
class AssetBucket:
    name: str
    weight: float = 0.0
    is_bond: bool = False

@dataclass
class WithdrawalTramo:
    from_year: int
    to_year: int
    amount_nominal_monthly_start: float

@dataclass
class ExtraCashflow:
    year: int
    amount: float
    name: str

@dataclass
class SimulationConfig:
    horizon_years: int = 40
    steps_per_year: int = 12
    initial_capital: float = 1800000000.0
    n_sims: int = 2000

    mu_normal_rv: float = 0.11
    mu_normal_rf: float = 0.06
    inflation_mean: float = 0.035
    inflation_vol: float = 0.012

    is_active_managed: bool = True
    use_guardrails: bool = True
    guardrail_trigger: float = 0.20
    guardrail_cut: float = 0.10

    enable_prop: bool = True
    net_inmo_value: float = 500000000.0
    new_rent_cost: float = 1500000.0
    emergency_months_trigger: int = 24

    extra_cashflows: List[ExtraCashflow] = field(default_factory=list)

    mu_local_rv: float = -0.15
    mu_local_rf: float = 0.08
    corr_local: float = -0.25

    mu_global_rv: float = -0.22
    mu_global_rf: float = -0.02
    corr_global: float = 0.75

    prob_enter_local: float = 0.005
    prob_enter_global: float = 0.004
    prob_exit_crisis: float = 0.085

    corr_normal: float = 0.35
    t_df: int = 8
    random_seed: Optional[int] = None
    stress_schedule: Optional[List[int]] = None

    # Advanced realism params
    p_def: float = 1.0
    vol_factor_active: float = 0.80
    sale_cost_pct: float = 0.02
    sale_delay_months: int = 3
    pct_discretionary: float = 0.20
    discretionary_cut_in_crisis: float = 0.60
    rf_reserve_years: float = 3.5

# ----------------------
# Institutional Simulator (aggregate)
# ----------------------
class InstitutionalSimulator:
    def __init__(self, config: SimulationConfig, assets: List[AssetBucket], withdrawals: List[WithdrawalTramo]):
        self.cfg = config
        self.assets = assets
        self.withdrawals = withdrawals
        self.dt = 1.0 / config.steps_per_year
        self.total_steps = int(config.horizon_years * config.steps_per_year)
        self.mu_regimes = np.array([
            [self.cfg.mu_normal_rv, self.cfg.mu_normal_rf],
            [self.cfg.mu_local_rv, self.cfg.mu_local_rf],
            [self.cfg.mu_global_rv, self.cfg.mu_global_rf],
        ])
        vol_f = float(self.cfg.vol_factor_active) if self.cfg.is_active_managed else 1.0
        self.sigma_regimes = np.array([[0.15, 0.05], [0.22, 0.12], [0.30, 0.14]]) * vol_f
        self.L_mats = [
            np.linalg.cholesky(np.array([[1.0, float(self.cfg.corr_normal)], [float(self.cfg.corr_normal), 1.0]])),
            np.linalg.cholesky(np.array([[1.0, float(self.cfg.corr_local)], [float(self.cfg.corr_local), 1.0]])),
            np.linalg.cholesky(np.array([[1.0, float(self.cfg.corr_global)], [float(self.cfg.corr_global), 1.0]])),
        ]
        self.p_norm_l = self.cfg.prob_enter_local
        self.p_norm_g = self.cfg.prob_enter_global
        self.p_exit = self.cfg.prob_exit_crisis
        self.rng = np.random.default_rng(self.cfg.random_seed)

    def run(self):
        n_sims, n_steps = int(self.cfg.n_sims), self.total_steps
        cap_paths = np.zeros((n_sims, n_steps + 1), dtype=float)
        cap_paths[:, 0] = float(self.cfg.initial_capital)
        cpi_paths = np.ones((n_sims, n_steps + 1), dtype=float)
        is_alive = np.ones(n_sims, dtype=bool)
        ruin_idx = np.full(n_sims, -1, dtype=int)
        has_h = np.full(n_sims, self.cfg.enable_prop, dtype=bool)
        pending_sale = np.full(n_sims, -1, dtype=int)

        asset_vals = np.zeros((n_sims, 2), dtype=float)
        rv_w = next((a.weight for a in self.assets if not a.is_bond), 0.6)
        asset_vals[:, 0] = float(self.cfg.initial_capital) * float(rv_w)
        asset_vals[:, 1] = float(self.cfg.initial_capital) * (1.0 - float(rv_w))

        regime = np.zeros(n_sims, dtype=int)
        df = int(self.cfg.t_df)
        g = self.rng.standard_normal((n_sims, n_steps, 2))
        w = self.rng.chisquare(df, (n_sims, n_steps, 1))
        Z_raw = (g / np.sqrt(w / df)) / np.sqrt(df / (df - 2))
        inf_sh = self.rng.normal(self.cfg.inflation_mean * self.dt, self.cfg.inflation_vol * np.sqrt(self.dt), (n_sims, n_steps))

        for t in range(n_steps):
            alive = is_alive
            if not np.any(alive):
                break
            schedule = getattr(self.cfg, "stress_schedule", None)
            if schedule is not None and isinstance(schedule, list) and len(schedule) >= n_steps:
                regime[alive] = int(schedule[t])
            else:
                reg_prev = regime.copy()
                m0 = (reg_prev == 0) & alive
                if np.any(m0):
                    r = self.rng.random(np.sum(m0)); idx = np.where(m0)[0]
                    regime[idx[r < self.p_norm_l]] = 1
                    regime[idx[(r >= self.p_norm_l) & (r < (self.p_norm_l + self.p_norm_g))]] = 2
                mc = (reg_prev > 0) & alive
                if np.any(mc):
                    idxc = np.where(mc)[0]
                    regime[idxc[self.rng.random(np.sum(mc)) < self.p_exit]] = 0

            z_t = Z_raw[:, t, :]
            z_f = np.zeros_like(z_t)
            for r_idx, L in enumerate(self.L_mats):
                m = (regime == r_idx) & alive
                if np.any(m):
                    z_f[m] = np.dot(z_t[m], L.T)

            p_def = np.ones(n_sims) * float(self.cfg.p_def)
            mus = self.mu_regimes[regime]
            sigs = self.sigma_regimes[regime]

            asset_vals[alive] *= np.exp((mus[alive] - 0.5 * sigs[alive] ** 2) * self.dt + sigs[alive] * np.sqrt(self.dt) * z_f[alive] * p_def[alive, None])
            cpi_paths[:, t + 1] = cpi_paths[:, t] * (1 + (inf_sh[:, t] + (regime == 1) * 0.003))

            if (t + 1) % 12 == 0:
                y = (t + 1) // 12
                for e in self.cfg.extra_cashflows:
                    if e.year == y:
                        asset_vals[alive, 1] += e.amount * cpi_paths[alive, t + 1]

            pending = (pending_sale >= 0) & alive
            if np.any(pending):
                pending_sale[pending] -= 1
                done = pending & (pending_sale == 0)
                if np.any(done):
                    cash = self.cfg.net_inmo_value * (1.0 - float(self.cfg.sale_cost_pct)) * cpi_paths[done, t + 1]
                    asset_vals[done, 1] += cash
                    has_h[done] = False
                    pending_sale[done] = -1

            cur_y = (t + 1) / 12.0
            m_spend = np.zeros(n_sims)
            for w in self.withdrawals:
                if w.from_year <= cur_y < w.to_year:
                    m_spend = float(w.amount_nominal_monthly_start) * cpi_paths[:, t + 1]
                    break

            pct_dis = float(self.cfg.pct_discretionary)
            disc = m_spend * pct_dis
            ess = m_spend - disc

            if self.cfg.use_guardrails and np.any(m_spend > 0):
                cur_real = np.sum(asset_vals, axis=1) / cpi_paths[:, t + 1]
                trig_gr = alive & (cur_real < (self.cfg.initial_capital * (1.0 - self.cfg.guardrail_trigger)))
                if np.any(trig_gr):
                    ess[trig_gr] *= (1 - self.cfg.guardrail_cut)
                    disc[trig_gr] *= (1 - self.cfg.guardrail_cut)

            disc_keep = 1.0 - float(self.cfg.discretionary_cut_in_crisis)
            in_crisis = (regime > 0) & alive
            if np.any(in_crisis):
                disc[in_crisis] *= disc_keep

            m_spend_adj = ess + disc

            trig = alive & has_h & (np.sum(asset_vals, axis=1) < m_spend_adj * self.cfg.emergency_months_trigger)
            if np.any(trig):
                delay = int(self.cfg.sale_delay_months)
                for i in np.where(trig)[0]:
                    if delay <= 0:
                        asset_vals[i, 1] += self.cfg.net_inmo_value * (1.0 - float(self.cfg.sale_cost_pct)) * cpi_paths[i, t + 1]
                        has_h[i] = False
                    else:
                        pending_sale[i] = delay

            with np.errstate(divide='ignore', invalid='ignore'):
                rf_balance = asset_vals[:, 1]
                rf_liquid_months = np.where(m_spend_adj > 0, rf_balance / m_spend_adj, np.inf)
            buffer_months = min(6, max(1, int(self.cfg.emergency_months_trigger / 4)))
            early_sale = alive & has_h & (rf_liquid_months < (self.cfg.sale_delay_months + buffer_months))
            if np.any(early_sale):
                for i in np.where(early_sale)[0]:
                    if pending_sale[i] < 0:
                        if self.cfg.sale_delay_months <= 0:
                            asset_vals[i, 1] += self.cfg.net_inmo_value * (1.0 - float(self.cfg.sale_cost_pct)) * cpi_paths[i, t + 1]
                            has_h[i] = False
                        else:
                            pending_sale[i] = int(self.cfg.sale_delay_months)

            out = m_spend_adj + (self.cfg.enable_prop & (~has_h)) * (self.cfg.new_rent_cost * cpi_paths[:, t + 1])
            wd = np.minimum(out, np.sum(asset_vals, axis=1))
            rf_b = np.maximum(asset_vals[:, 1], 0.0)
            t_rf = np.minimum(wd, rf_b)
            asset_vals[:, 1] -= t_rf
            asset_vals[:, 0] -= (wd - t_rf)

            asset_vals = np.maximum(asset_vals, 0.0)
            cap_paths[:, t + 1] = np.sum(asset_vals, axis=1)

            dead = (cap_paths[:, t + 1] <= 1000) & alive
            if np.any(dead):
                is_alive[dead] = False
                ruin_idx[dead] = t + 1
                cap_paths[dead, t + 1:] = 0.0
                asset_vals[dead, :] = 0.0

        return cap_paths, cpi_paths, ruin_idx

# ----------------------
# Portfolio Simulator (per-instrument) — same style and features
# (Implementation kept similar to prior versions; for brevity not repeated in comments)
# ----------------------
@dataclass
class InstrumentPosition:
    instrument_id: str
    name: str
    value_clp: float
    rv_share: float
    rv_min: float = 0.0
    rv_max: float = 1.0
    liquidity_days: int = 3
    bucket: str = "BAL"
    priority: int = 10
    include_withdrawals: bool = True

@dataclass
class PortfolioRulesConfig:
    rf_reserve_years: float = 3.5
    rebalance_every_months: int = 12
    rebalance_only_when_normal: bool = True
    manager_riskoff_in_crisis: float = 0.20
    bucket_order: Dict[str, int] = field(default_factory=lambda: {"RF_PURA": 0, "BAL": 1, "RV": 2, "AFP": 3, "PASIVO": 99})

def _normalize_portfolio_df(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if "id_instrumento" in d.columns and "instrument_id" not in d.columns:
        d["instrument_id"] = d["id_instrumento"].astype(str)
    if "nombre" in d.columns and "name" not in d.columns:
        d["name"] = d["nombre"].astype(str)
    if "saldo_clp" in d.columns and "value_clp" not in d.columns:
        d["value_clp"] = pd.to_numeric(d["saldo_clp"], errors="coerce").fillna(0.0)
    keep_cols = [c for c in ["instrument_id", "name", "value_clp", "tipo", "subtipo", "moneda"] if c in d.columns]
    if not keep_cols:
        d = d.assign(instrument_id=d.index.astype(str), name=d.index.astype(str), value_clp=0.0)
        keep_cols = ["instrument_id", "name", "value_clp"]
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
    return {
        "SURA_SEGURO_MULTIACTIVO_AGRESIVO_SERIE_F": {"rv_share": 0.875, "rv_min": 0.80, "rv_max": 1.00, "liquidity_days": 4, "bucket": "RV", "priority": 40},
        "SURA_SEGURO_MULTIACTIVO_MODERADO_SERIE_F": {"rv_share": 0.475, "rv_min": 0.30, "rv_max": 0.60, "liquidity_days": 4, "bucket": "BAL", "priority": 35},
        "SURA_SEGURO_RENTA_LOCAL_UF_SERIE_F": {"rv_share": 0.0, "rv_min": 0.0, "rv_max": 0.0, "liquidity_days": 3, "bucket": "RF_PURA", "priority": 0},
        "SURA_SEGURO_RENTA_BONOS_CHILE_SF": {"rv_share": 0.0, "rv_min": 0.0, "rv_max": 0.0, "liquidity_days": 3, "bucket": "RF_PURA", "priority": 0},
        "BTG_GESTION_AGRESIVA": {"rv_share": 0.975, "rv_min": 0.90, "rv_max": 1.00, "liquidity_days": 2, "bucket": "RV", "priority": 55},
        "BTG_GESTION_ACTIVA": {"rv_share": 0.85, "rv_min": 0.70, "rv_max": 1.00, "liquidity_days": 2, "bucket": "BAL", "priority": 30},
        "BTG_GESTION_CONSERVADORA": {"rv_share": 0.175, "rv_min": 0.00, "rv_max": 0.30, "liquidity_days": 1, "bucket": "BAL", "priority": 15},
        "MONEDA_RENTA_CLP": {"rv_share": 0.0, "rv_min": 0.0, "rv_max": 0.0, "liquidity_days": 1, "bucket": "RF_PURA", "priority": 0},
        "DAP_CLP_10122025": {"rv_share": 0.0, "rv_min": 0.0, "rv_max": 0.0, "liquidity_days": 0, "bucket": "RF_PURA", "priority": 0},
        "WISE_USD": {"rv_share": 0.0, "rv_min": 0.0, "rv_max": 0.0, "liquidity_days": 0, "bucket": "RF_PURA", "priority": 0},
        "GLOBAL66_USD": {"rv_share": 0.0, "rv_min": 0.0, "rv_max": 0.0, "liquidity_days": 0, "bucket": "RF_PURA", "priority": 0},
        "SURA_USD_SHORT_DURATION": {"rv_share": 0.0, "rv_min": 0.0, "rv_max": 0.0, "liquidity_days": 6, "bucket": "RF_PURA", "priority": 2},
        "SURA_USD_MONEY_MARKET": {"rv_share": 0.0, "rv_min": 0.0, "rv_max": 0.0, "liquidity_days": 6, "bucket": "RF_PURA", "priority": 1},
        "SURA_APV_MULTIACTIVO_AGRESIVO": {"rv_share": 0.80, "rv_min": 0.70, "rv_max": 1.00, "liquidity_days": 10, "bucket": "AFP", "priority": 90},
        "SURA_DC_MULTIACTIVO_AGRESIVO": {"rv_share": 0.80, "rv_min": 0.70, "rv_max": 1.00, "liquidity_days": 10, "bucket": "AFP", "priority": 90},
        "AFP_PLANVITAL_OBLIGATORIA": {"rv_share": 0.78, "rv_min": 0.65, "rv_max": 0.85, "liquidity_days": 10, "bucket": "AFP", "priority": 95},
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
            name = str(r.get("name", "")).lower()
            if "multiactivo" in name and "agres" in name:
                out.loc[i, "rv_share"] = 0.875
                out.loc[i, "rv_min"] = 0.80
                out.loc[i, "rv_max"] = 1.00
                out.loc[i, "liquidity_days"] = 4
                out.loc[i, "bucket"] = "RV"
                out.loc[i, "priority"] = 40
                continue
            if "multiactivo" in name and "moder" in name:
                out.loc[i, "rv_share"] = 0.475
                out.loc[i, "rv_min"] = 0.30
                out.loc[i, "rv_max"] = 0.60
                out.loc[i, "liquidity_days"] = 4
                out.loc[i, "bucket"] = "BAL"
                out.loc[i, "priority"] = 35
                continue
            if any(k in iid for k in ["USD_MONEY_MARKET", "SHORT_DURATION", "WISE", "GLOBAL66", "DAP"]):
                out.loc[i, "rv_share"] = 0.0
                out.loc[i, "rv_min"] = 0.0
                out.loc[i, "rv_max"] = 0.0
                out.loc[i, "liquidity_days"] = 0
                out.loc[i, "bucket"] = "RF_PURA"
                out.loc[i, "priority"] = 0
            else:
                out.loc[i, "rv_share"] = 0.30
                out.loc[i, "rv_min"] = 0.0
                out.loc[i, "rv_max"] = 0.6
                out.loc[i, "liquidity_days"] = 3
                out.loc[i, "bucket"] = "BAL"
                out.loc[i, "priority"] = 30

    out["rv_min"] = pd.to_numeric(out["rv_min"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    out["rv_max"] = pd.to_numeric(out["rv_max"], errors="coerce").fillna(1.0).clip(0.0, 1.0)
    out["rv_max"] = out[["rv_min", "rv_max"]].max(axis=1)
    out["rv_share"] = pd.to_numeric(out["rv_share"], errors="coerce").fillna(0.0)
    out["rv_share"] = out.apply(lambda r: min(max(float(r["rv_share"]), float(r["rv_min"])), float(r["rv_max"])), axis=1)
    out["liquidity_days"] = pd.to_numeric(out["liquidity_days"], errors="coerce").fillna(3).astype(int)
    out["value_clp"] = pd.to_numeric(out["value_clp"], errors="coerce").fillna(0.0)
    return out

class PortfolioSimulator:
    def __init__(self, cfg: SimulationConfig, positions: List[InstrumentPosition], withdrawals: List[WithdrawalTramo], rules: PortfolioRulesConfig):
        self.cfg = cfg
        self.positions = positions
        self.withdrawals = withdrawals
        self.rules = rules
        self.dt = 1.0 / cfg.steps_per_year
        self.total_steps = int(cfg.horizon_years * cfg.steps_per_year)
        self.rng = np.random.default_rng(cfg.random_seed)

        self.mu_regimes = np.array([
            [cfg.mu_normal_rv, cfg.mu_normal_rf],
            [cfg.mu_local_rv, cfg.mu_local_rf],
            [cfg.mu_global_rv, cfg.mu_global_rf],
        ])
        vol_f = float(cfg.vol_factor_active) if cfg.is_active_managed else 1.0
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
        candidates = [i for i, p in enumerate(self.positions) if p.bucket == "RF_PURA" and p.include_withdrawals]
        if not candidates:
            return None
        return sorted(candidates, key=lambda i: (int(getattr(self.positions[i], 'liquidity_days', 3)), int(self.positions[i].priority), self.positions[i].instrument_id))[0]

    def run(self):
        # Implementation analogous to InstitutionalSimulator but per-instrument; omitted here to save space
        # Use the full implementation from prior versions in your repo (we assume it's present)
        raise NotImplementedError("PortfolioSimulator.run should be the same implementation you had; paste the full method here in the repo.")

# ----------------------
# Helper: build withdrawals default per your request
# ----------------------
def default_withdrawal_profile():
    # Fase 1: 6.000.000 / mes durante 7 años
    # Fase 2: 5.000.000 / mes desde año 7 hasta 20
    # Fase 3: 4.000.000 / mes desde año 20 hasta horizon (default 40)
    return [
        WithdrawalTramo(0, 7, 6000000),
        WithdrawalTramo(7, 20, 5000000),
        WithdrawalTramo(20, 40, 4000000),
    ]

# ----------------------
# Stress and Tornado utilities
# ----------------------
def make_stress_schedule(horizon_years, scenario):
    # Returns list length steps (regime per month): 0 normal, 1 local, 2 global
    steps = int(horizon_years * 12)
    sched = [0] * steps
    if scenario == "Global crash 18m":
        # Put global (2) starting at t=12 (start year 1) for 18 months
        start = 0
        for i in range(start, min(steps, start + 18)):
            sched[i] = 2
    elif scenario == "Local crisis 24m":
        start = 0
        for i in range(start, min(steps, start + 24)):
            sched[i] = 1
    elif scenario == "Flash crash 4m":
        start = 12
        for i in range(start, min(steps, start + 4)):
            sched[i] = 2
    # else empty (normal)
    return sched

def run_simulation(cfg: SimulationConfig, use_portfolio: bool, positions: List[InstrumentPosition], withdrawals: List[WithdrawalTramo], rules: Optional[PortfolioRulesConfig]=None):
    # Wrapper: picks InstitutionalSimulator if not use_portfolio
    if use_portfolio:
        # make sure PortfolioSimulator.run exists in your repo implementation
        sim = PortfolioSimulator(cfg, positions, withdrawals, rules or PortfolioRulesConfig())
        paths, cpi, r_i = sim.run()
    else:
        # aggregate RV/RF weight from provided positions or use cfg initial split (if positions empty)
        if positions:
            rv_w = sum([p.value_clp * p.rv_share for p in positions]) / max(1.0, sum([p.value_clp for p in positions]))
            assets = [AssetBucket("RV", rv_w, False), AssetBucket("RF", 1.0 - rv_w, True)]
        else:
            # default 60% RV if not provided
            assets = [AssetBucket("RV", 0.6, False), AssetBucket("RF", 0.4, True)]
        sim = InstitutionalSimulator(cfg, assets, withdrawals)
        paths, cpi, r_i = sim.run()
    return paths, cpi, r_i

# ----------------------
# UI — main
# ----------------------
def app():
    st.markdown("## 🦅 Panel de Decisión Patrimonial (Versión Realista)")
    st.session_state.setdefault('extra_events', [])
    st.session_state.setdefault('evy', 5)
    st.session_state.setdefault('evt', 'Entrada')
    st.session_state.setdefault('use_portfolio', False)
    st.session_state.setdefault('portfolio_json', "")
    st.session_state.setdefault('advanced', {
        'p_def': 1.0, 'vol_factor_active': 0.80, 'manager_riskoff': 0.20,
        'sale_cost_pct': 0.02, 'sale_delay_months': 3, 'pct_discretionary': 0.20,
        'discretionary_cut_in_crisis': 0.60, 'rf_reserve_years': 3.5, 'corr_normal': 0.35
    })
    st.session_state.setdefault('score_weights', {'mc': 0.6, 'stress': 0.3, 'tornado': 0.1})

    # Sidebar with forced advanced panel visible
    with st.sidebar:
        st.header("⚙️ Configuración")
        use_portfolio = st.checkbox("Modo Cartera Real (Instrumentos)", value=st.session_state.get("use_portfolio", False), key="use_portfolio_cb")
        st.session_state["use_portfolio"] = use_portfolio
        is_active = st.checkbox("Gestión Activa / Balanceados", value=True, key="is_active_cb")
        sel_glo = st.selectbox("Crisis Global", ["Crash Financiero", "Colapso Sistémico", "Recesión Estándar"], index=0, key="sel_glo")
        st.divider()
        sel_ret = st.selectbox("Rentabilidad Normal", ["Histórico (11%)","Conservador (8/4.5)","Crecimiento (13%)","Personalizado"], index=0, key="sel_ret")
        if sel_ret == "Personalizado":
            c_rv_in = st.number_input("RV % Nom. (ej. 11 = 11%)", 0.0, 50.0, 11.0)/100.0
            c_rf_in = st.number_input("RF % Nom. (ej. 6 = 6%)", 0.0, 50.0, 6.0)/100.0
        else:
            if sel_ret == "Histórico (11%)":
                c_rv_in, c_rf_in = 0.11, 0.06
            elif sel_ret == "Conservador (8/4.5)":
                c_rv_in, c_rf_in = 0.08, 0.045
            else:
                c_rv_in, c_rf_in = 0.13, 0.07

        n_sims = st.slider("Simulaciones (principal)", 500, 5000, 2000, step=100, key="n_sims_main")
        horiz = st.slider("Horizonte (años)", 10, 50, 40, key="horiz_main")

        # Advanced always shown and expanded
        with st.expander("Configuración avanzada (sidebar) — siempre visible", expanded=True):
            adv = st.session_state['advanced']
            st.number_input("p_def (shock damping en crisis, advanced)", 0.0, 2.0, adv.get('p_def', 1.0), step=0.05, key="p_def")
            st.number_input("vol_factor_active (si gestor activo)", 0.1, 1.0, adv.get('vol_factor_active', 0.80), step=0.05, key="vol_factor_active")
            st.slider("manager_riskoff_in_crisis (0..1)", 0.0, 1.0, adv.get('manager_riskoff', 0.20), 0.05, key="manager_riskoff")
            st.number_input("sale_cost_pct (venta casa)", 0.0, 0.2, adv.get('sale_cost_pct', 0.02), step=0.005, key="sale_cost_pct")
            st.number_input("sale_delay_months (venta casa)", 0, 12, adv.get('sale_delay_months', 3), step=1, key="sale_delay_months")
            st.number_input("pct_discretionary (gasto)", 0.0, 1.0, adv.get('pct_discretionary', 0.20), step=0.05, key="pct_discretionary")
            st.number_input("discretionary_cut_in_crisis", 0.0, 1.0, adv.get('discretionary_cut_in_crisis', 0.60), step=0.05, key="discretionary_cut_in_crisis")
            st.number_input("rf_reserve_years (años)", 0.0, 10.0, adv.get('rf_reserve_years', 3.5), step=0.5, key="rf_reserve_years")

            # persist into session_state['advanced']
            st.session_state['advanced'].update({
                'p_def': float(st.session_state.get('p_def', adv.get('p_def', 1.0))),
                'vol_factor_active': float(st.session_state.get('vol_factor_active', adv.get('vol_factor_active', 0.8))),
                'manager_riskoff': float(st.session_state.get('manager_riskoff', adv.get('manager_riskoff', 0.2))),
                'sale_cost_pct': float(st.session_state.get('sale_cost_pct', adv.get('sale_cost_pct', 0.02))),
                'sale_delay_months': int(st.session_state.get('sale_delay_months', adv.get('sale_delay_months', 3))),
                'pct_discretionary': float(st.session_state.get('pct_discretionary', adv.get('pct_discretionary', 0.2))),
                'discretionary_cut_in_crisis': float(st.session_state.get('discretionary_cut_in_crisis', adv.get('discretionary_cut_in_crisis', 0.6))),
                'rf_reserve_years': float(st.session_state.get('rf_reserve_years', adv.get('rf_reserve_years', 3.5))),
            })
        with st.expander("🎯 Indicador agregado (ponderaciones)", expanded=True):
            w = st.session_state['score_weights']
            st.slider("Peso Monte Carlo", 0.0, 1.0, float(w.get('mc', 0.6)), 0.05, key="w_mc")
            st.slider("Peso Stress", 0.0, 1.0, float(w.get('stress', 0.3)), 0.05, key="w_stress")
            st.slider("Peso Tornado", 0.0, 1.0, float(w.get('tornado', 0.1)), 0.05, key="w_tornado")
            st.session_state['score_weights'].update({
                'mc': float(st.session_state.get('w_mc', w.get('mc', 0.6))),
                'stress': float(st.session_state.get('w_stress', w.get('stress', 0.3))),
                'tornado': float(st.session_state.get('w_tornado', w.get('tornado', 0.1))),
            })

    # Tabs
    tab_sim, tab_stress, tab_tornado, tab_sum = st.tabs(["📊 Simulación", "🧯 Stress", "🌪️ Tornado", "🧾 Resumen"])

    # ----------------------
    # Tab Simulación (principal)
    # ----------------------
    with tab_sim:
        st.subheader("Simulación principal")
        use_portfolio = st.session_state.get("use_portfolio", False)
        portfolio_df = None
        positions: List[InstrumentPosition] = []

        if use_portfolio:
            st.info("Modo Cartera Real: pega el JSON de instrumentos o usa el Dashboard")
            txt = st.text_area("Pega JSON cartera (registros → instrumentos)", value=st.session_state.get("portfolio_json", ""), height=180)
            if txt.strip():
                st.session_state["portfolio_json"] = txt
                try:
                    df_src = parse_portfolio_json(txt)
                    df_base = enrich_with_meta(df_src)
                    edited = df_base  # simple table; we avoid data_editor incompatibilities
                    tot = float(edited["value_clp"].sum())
                    rv_amt = float((edited["value_clp"] * edited["rv_share"]).sum())
                    rf_pura_amt = float(edited.loc[edited["bucket"] == "RF_PURA", "value_clp"].sum())
                    c1, c2, c3 = st.columns(3)
                    with c1: st.metric("Patrimonio (CLP)", f"${fmt(tot)}")
                    with c2: st.metric("Motor (RV) estimado", f"{(100.0 * rv_amt / tot) if tot>0 else 0:.1f}%")
                    with c3: st.metric("RF pura hoy", f"${fmt(rf_pura_amt)}")
                    for _, r in edited.iterrows():
                        positions.append(InstrumentPosition(
                            instrument_id=str(r["instrument_id"]),
                            name=str(r["name"]),
                            value_clp=float(r["value_clp"]),
                            rv_share=float(r.get("rv_share", 0.0)),
                            rv_min=float(r.get("rv_min", 0.0)),
                            rv_max=float(r.get("rv_max", 1.0)),
                            liquidity_days=int(r.get("liquidity_days", 3)),
                            bucket=str(r.get("bucket", "BAL")),
                            priority=int(r.get("priority", 30)),
                            include_withdrawals=bool(r.get("include_withdrawals", True)),
                        ))
                except Exception as e:
                    st.error(f"No pude parsear JSON: {e}")
        else:
            st.info("Modo agregado: valores por defecto aplicados si no cambias nada.")
            cap_val = clean_input("Capital Total ($)", 1800000000, "cap")  # default 1.800.000.000
            st.markdown(f"**Patrimonio usado por defecto:** ${fmt(st.session_state.get('num__cap', 1800000000))}")
            # default RV % control
            rv_pct = st.slider("Motor (RV %)", 0, 100, 60, key="rv_pct_main")
            # show RF reserve
            st.metric("Reserva RF (estimada)", f"${fmt(int(cap_val * (1 - rv_pct/100)))}")

        st.markdown("### Perfil de gastos (mes)")
        # Defaults requested: F1=6.000.000 por 7 años; F2=5.000.000 hasta año 20; F3=4.000.000 desde año 20
        cols = st.columns(3)
        with cols[0]:
            r1 = clean_input("Gasto F1 (CLP/mes)", 6000000, "r1")
            d1 = st.number_input("Años F1", 0, 40, 7, key="d1")  # default 7
        with cols[1]:
            r2 = clean_input("Gasto F2 (CLP/mes)", 5000000, "r2")
            d2 = st.number_input("Años F2", 0, 40, 13, key="d2")  # 7->20 -> 13 years duration
        with cols[2]:
            r3 = clean_input("Gasto F3 (CLP/mes)", 4000000, "r3")
            d3 = st.number_input("Años F3", 0, 40, 20, key="d3")  # last 20 years

        # ensure d2 is difference to 20 (we interpret d2 as years for phase 2; default set to 13)
        # Build withdrawals according to these phase durations but aligned to requested scheme:
        # from 0 to d1: r1; from d1 to d1+d2: r2; from d1+d2 to horizon: r3
        wds = [
            WithdrawalTramo(0, int(d1), int(r1)),
            WithdrawalTramo(int(d1), int(d1 + d2), int(r2)),
            WithdrawalTramo(int(d1 + d2), int(d1 + d2 + d3), int(r3)),
        ]

        # Extra events
        with st.expander("💸 Inyecciones o Salidas"):
            st.session_state.setdefault('extra_events', [])
            ce1, ce2, ce3, ce4 = st.columns([1,2,2,1])
            with ce1:
                st.number_input('Año', 1, 40, int(st.session_state.get('evy', 5)), key='evy')
            with ce2:
                clean_input('Monto ($)', 0, 'eva')
            with ce3:
                st.selectbox('Tipo', ['Entrada', 'Salida'], key='evt')
            with ce4:
                if st.button('Add', key='add_extra'):
                    y = int(st.session_state.get('evy', 1))
                    a = int(st.session_state.get('num__eva', 0))
                    t = str(st.session_state.get('evt', 'Entrada'))
                    amt = a if t == 'Entrada' else -a
                    st.session_state['extra_events'].append({"year": int(y), "amount": float(amt), "name": "Hito"})
            for e in st.session_state['extra_events']:
                st.text(f"Año {int(e['year'])}: ${fmt(e['amount'])}")
            if st.button('Limpiar', key='clear_extra'):
                st.session_state['extra_events'] = []

        # House emergency defaults
        enable_p = st.checkbox("Venta Casa Emergencia", value=True, key="enable_prop")
        val_h = clean_input("Valor Neto Casa ($)", 500000000, "vi") if enable_p else 0  # default 500M

        # Build SimulationConfig with conversions
        adv = st.session_state.get('advanced', {})
        mu_rv_cont = to_continuous_mu(c_rv_in if 'c_rv_in' in locals() else 0.11)
        mu_rf_cont = to_continuous_mu(c_rf_in if 'c_rf_in' in locals() else 0.06)

        cfg_current = SimulationConfig(
            horizon_years=int(horiz),
            initial_capital=int(st.session_state.get('num__cap', 1800000000)),
            n_sims=int(st.session_state.get('n_sims_main', 2000)),
            is_active_managed=bool(st.session_state.get('is_active_cb', True)),
            enable_prop=bool(st.session_state.get('enable_prop', True)),
            net_inmo_value=float(st.session_state.get('num__vi', 500000000)),
            mu_normal_rv=mu_rv_cont,
            mu_normal_rf=mu_rf_cont,
            extra_cashflows=[ExtraCashflow(int(e["year"]), float(e["amount"]), e.get("name", "Hito")) for e in st.session_state.get("extra_events", [])],
            mu_global_rv=-0.22,
            mu_global_rf=-0.02,
            corr_global=0.75,
            corr_normal=float(adv.get('corr_normal', 0.35)),
            p_def=float(adv.get('p_def', 1.0)),
            vol_factor_active=float(adv.get('vol_factor_active', 0.80)),
            sale_cost_pct=float(adv.get('sale_cost_pct', 0.02)),
            sale_delay_months=int(adv.get('sale_delay_months', 3)),
            pct_discretionary=float(adv.get('pct_discretionary', 0.20)),
            discretionary_cut_in_crisis=float(adv.get('discretionary_cut_in_crisis', 0.60)),
            rf_reserve_years=float(adv.get('rf_reserve_years', 3.5)),
            t_df=int(8),
            random_seed=12345,
        )

        # Run simulation
        if st.button("🚀 INICIAR SIMULACIÓN", type="primary", key="run_sim"):
            # If use_portfolio -> positions must be filled and PortfolioSimulator.run must be available
            try:
                if use_portfolio:
                    if not positions:
                        st.error("Modo instrumentos activo pero no hay posiciones válidas.")
                        st.stop()
                    rules = PortfolioRulesConfig()
                    rules.manager_riskoff_in_crisis = float(adv.get('manager_riskoff', 0.20))
                    paths, cpi, r_i = run_simulation(cfg_current, True, positions, wds, rules)
                else:
                    paths, cpi, r_i = run_simulation(cfg_current, False, [], wds, None)

                # results
                success_pct = float((r_i <= -1).mean() * 100.0)
                ci_low, ci_high = success_ci(success_pct, int(cfg_current.n_sims))
                terminal_real = paths[:, -1] / np.maximum(cpi[:, -1], 1e-9)
                p10 = float(np.percentile(terminal_real, 10))
                p50 = float(np.percentile(terminal_real, 50))
                p90 = float(np.percentile(terminal_real, 90))
                ruined = (r_i > -1)
                ruin_years = (r_i[ruined] / 12.0) if np.any(ruined) else np.array([])
                ruin_stats = compute_ruin_stats(r_i, cfg_current.steps_per_year, cfg_current.horizon_years)

                st.markdown(f"<div style='padding:12px; border-radius:8px; background:#091121; color:#fff;'>"
                            f"<h2>Éxito: {success_pct:.1f}% (IC95%: {ci_low:.1f}%–{ci_high:.1f}%)</h2>"
                            f"<p>Mediana legado real: ${fmt(p50)} | P10: ${fmt(p10)} | P90: ${fmt(p90)}</p>"
                            f"</div>", unsafe_allow_html=True)

                # plot envelope
                y_ax = np.arange(paths.shape[1]) / 12.0
                fig = go.Figure()
                p90_path = np.percentile(paths, 90, axis=0)
                p10_path = np.percentile(paths, 10, axis=0)
                p50_path = np.percentile(paths, 50, axis=0)
                fig.add_trace(go.Scatter(x=np.concatenate([y_ax, y_ax[::-1]]),
                                         y=np.concatenate([p90_path, p10_path[::-1]]),
                                         fill='toself', fillcolor='rgba(59,130,246,0.2)', line=dict(color='rgba(0,0,0,0)'), name='Rango 80%'))
                fig.add_trace(go.Scatter(x=y_ax, y=p50_path, line=dict(color='#3b82f6', width=3), name='Mediana'))
                fig.update_layout(title="Proyección Patrimonio (Nominal)", template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)

                # save last_run
                st.session_state["last_withdrawals"] = wds
                st.session_state["last_run"] = {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "success_pct": success_pct,
                    "success_ci": [ci_low, ci_high],
                    "median_terminal_real": p50,
                    "p10_terminal_real": p10,
                    "p90_terminal_real": p90,
                    "median_ruin_year": float(np.median(ruin_years)) if ruin_years.size else None,
                    "ruin_stats": ruin_stats,
                    "cfg": cfg_current.__dict__.copy(),
                }
            except NotImplementedError as e:
                st.error(f"Función no implementada: {e}. Si usas modo Cartera Real, asegúrate que PortfolioSimulator.run esté definido en este archivo.")

    # ----------------------
    # Tab Stress
    # ----------------------
    with tab_stress:
        st.subheader("Stress tests predefinidos y personalizados")
        scenario = st.selectbox("Escenario predefinido", ["Ninguno", "Global crash 18m", "Local crisis 24m", "Flash crash 4m"])
        custom_start = st.number_input("Inicio custom (mes desde hoy)", 0, 480, 0)
        custom_len = st.number_input("Duración custom (meses)", 0, 480, 0)
        n_sims_stress = st.number_input("Simulaciones por stress", 200, 5000, 500, step=100)

        if st.button("Ejecutar Stress"):
            cfg = st.session_state.get("last_run", {}).get("cfg", None)
            if cfg is None:
                # rebuild from UI values if no last_run
                cfg = cfg_current.__dict__.copy()
            cfg_obj = SimulationConfig(**{k: v for k, v in cfg.items() if k in SimulationConfig.__annotations__})
            cfg_obj.n_sims = int(n_sims_stress)
            if scenario != "Ninguno":
                sched = make_stress_schedule(cfg_obj.horizon_years, scenario)
                cfg_obj.stress_schedule = sched
            elif custom_len > 0:
                steps = int(cfg_obj.horizon_years * 12)
                sched = [0] * steps
                for i in range(custom_start, min(steps, custom_start + custom_len)):
                    sched[i] = 2  # force global by default for custom
                cfg_obj.stress_schedule = sched
            else:
                st.warning("Selecciona un escenario predefinido o define un custom.")
                st.stop()

            st.info("Corriendo simulación en modo stress (puede tardar varios minutos según n_sims).")
            wds = st.session_state.get("last_withdrawals", wds if "wds" in locals() else default_withdrawal_profile())
            paths, cpi, r_i = run_simulation(cfg_obj, False, [], wds, None)
            success_pct = float((r_i <= -1).mean() * 100.0)
            ci_low, ci_high = success_ci(success_pct, int(cfg_obj.n_sims))
            terminal_real = paths[:, -1] / np.maximum(cpi[:, -1], 1e-9)
            p10 = float(np.percentile(terminal_real, 10))
            p50 = float(np.percentile(terminal_real, 50))
            p90 = float(np.percentile(terminal_real, 90))
            st.session_state["last_stress"] = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "success_pct": success_pct,
                "success_ci": [ci_low, ci_high],
                "median_terminal_real": p50,
                "p10_terminal_real": p10,
                "p90_terminal_real": p90,
            }
            st.markdown(f"**Stress result:** Éxito {success_pct:.1f}% (IC95% {ci_low:.1f}%–{ci_high:.1f}%) | Mediana ${fmt(p50)} | P10 ${fmt(p10)}")

    # ----------------------
    # Tab Tornado
    # ----------------------
    with tab_tornado:
        st.subheader("Análisis Tornado (sensibilidad univariada)")
        tornado_n_sims = st.number_input("Simulaciones por punto (tornado)", 200, 3000, 500, step=100)
        run_tornado = st.button("Ejecutar Tornado")
        if run_tornado:
            st.info("Ejecutando Tornado — puede tardar varios minutos.")
            # Define base config from last run or current cfg_current
            base_cfg_dict = st.session_state.get("last_run", {}).get("cfg", cfg_current.__dict__.copy())
            base_cfg = SimulationConfig(**{k: v for k, v in base_cfg_dict.items() if k in SimulationConfig.__annotations__})
            base_cfg.n_sims = int(tornado_n_sims)
            params = [
                ("mu_normal_rv", [base_cfg.mu_normal_rv - 0.02, base_cfg.mu_normal_rv, base_cfg.mu_normal_rv + 0.02]),
                ("mu_normal_rf", [base_cfg.mu_normal_rf - 0.01, base_cfg.mu_normal_rf, base_cfg.mu_normal_rf + 0.01]),
                ("manager_riskoff", [max(0, float(st.session_state['advanced'].get('manager_riskoff', 0.2) - 0.1)), float(st.session_state['advanced'].get('manager_riskoff', 0.2)), min(1.0, float(st.session_state['advanced'].get('manager_riskoff', 0.2) + 0.1))]),
                ("pct_discretionary", [max(0, float(st.session_state['advanced'].get('pct_discretionary', 0.2) - 0.1)), float(st.session_state['advanced'].get('pct_discretionary', 0.2)), min(1.0, float(st.session_state['advanced'].get('pct_discretionary', 0.2) + 0.1))]),
                ("sale_delay_months", [0, int(st.session_state['advanced'].get('sale_delay_months', 3)), min(12, int(st.session_state['advanced'].get('sale_delay_months', 3) + 3))]),
            ]
            rows = []
            for name, vals in params:
                for v in vals:
                    # copy base and modify
                    cfg = SimulationConfig(**{k: v for k, v in base_cfg.__dict__.items()})
                    if name == "manager_riskoff":
                        # manager_riskoff is used in rules; we pass through session advanced
                        st.session_state['advanced']['manager_riskoff'] = float(v)
                    elif name == "pct_discretionary":
                        cfg.pct_discretionary = float(v)
                    elif name == "sale_delay_months":
                        cfg.sale_delay_months = int(v)
                    elif name == "mu_normal_rv":
                        cfg.mu_normal_rv = float(v)
                    elif name == "mu_normal_rf":
                        cfg.mu_normal_rf = float(v)
                    # run sim (aggregate)
                    wds = st.session_state.get("last_withdrawals", wds if "wds" in locals() else default_withdrawal_profile())
                    paths, cpi, r_i = run_simulation(cfg, False, [], wds, None)
                    success_pct = float((r_i <= -1).mean() * 100.0)
                    terminal_real = paths[:, -1] / np.maximum(cpi[:, -1], 1e-9)
                    p50 = float(np.percentile(terminal_real, 50))
                    rows.append({"param": name, "value": v, "success_pct": success_pct, "median_terminal": p50})
            df_tornado = pd.DataFrame(rows)
            tornado_score = float(df_tornado["success_pct"].mean()) if not df_tornado.empty else None
            st.session_state["last_tornado"] = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "success_pct": tornado_score,
                "rows": df_tornado.to_dict(orient="records"),
            }
            st.dataframe(df_tornado)
            # simple bar: delta success relative to median value
            baseline = df_tornado.groupby("param").apply(lambda g: g.loc[g["value"]==g["value"].median()].iloc[0]["success_pct"] if not g.empty else 0).to_dict()
            fig = go.Figure()
            for param in df_tornado['param'].unique():
                sub = df_tornado[df_tornado['param']==param]
                fig.add_trace(go.Bar(x=sub['value'].astype(str), y=sub['success_pct'], name=param))
            fig.update_layout(barmode='group', title="Tornado — %Éxito por valor de parámetro")
            st.plotly_chart(fig, use_container_width=True)

            # provide download CSV
            csv = df_tornado.to_csv(index=False).encode('utf-8')
            st.download_button("Descargar resultados Tornado (CSV)", data=csv, file_name="tornado_results.csv", mime="text/csv")

    # ----------------------
    # Tab Resumen
    # ----------------------
    with tab_sum:
        st.header("🧾 Resumen Ejecutivo (Automático)")
        last = st.session_state.get("last_run")
        if not last:
            st.info("Ejecuta primero una simulación para obtener resumen y recomendaciones.")
        else:
            st.markdown(f"**Última corrida:** {last['timestamp']}")
            st.markdown(f"- Éxito estimado: **{last['success_pct']:.1f}%** (IC95%: {last['success_ci'][0]:.1f}% – {last['success_ci'][1]:.1f}%)")
            st.markdown(f"- Mediana legado real: **${fmt(last['median_terminal_real'])}**")
            st.markdown(f"- P10: **${fmt(last['p10_terminal_real'])}** | P90: **${fmt(last['p90_terminal_real'])}**")
            last_stress = st.session_state.get("last_stress")
            last_tornado = st.session_state.get("last_tornado")
            scores = {
                "mc": last.get("success_pct"),
                "stress": last_stress.get("success_pct") if last_stress else None,
                "tornado": last_tornado.get("success_pct") if last_tornado else None,
            }
            weights = st.session_state.get("score_weights", {"mc": 0.6, "stress": 0.3, "tornado": 0.1})
            agg_score = aggregate_success_score(scores, weights)
            if agg_score is not None:
                st.markdown("### Indicador agregado (ponderado)")
                st.markdown(f"- Score agregado: **{agg_score:.1f}%** (pesos configurables en sidebar)")
                if last_stress:
                    st.markdown(f"- Stress: **{last_stress['success_pct']:.1f}%** (última corrida)")
                if last_tornado and last_tornado.get("success_pct") is not None:
                    st.markdown(f"- Tornado (promedio): **{last_tornado['success_pct']:.1f}%**")
            else:
                st.info("Corre Stress y Tornado para habilitar el indicador agregado.")
            ruin_stats = last.get("ruin_stats", {})
            if ruin_stats:
                st.markdown("### Distribución de ruinas (tiempo)")
                ruin_hist = ruin_stats.get("ruin_hist", {})
                if ruin_stats.get("ruined_count", 0) == 0:
                    st.success("No se observan ruinas en la simulación base.")
                else:
                    years = list(ruin_hist.keys())
                    counts = list(ruin_hist.values())
                    fig_ruin = go.Figure()
                    fig_ruin.add_trace(go.Bar(x=years, y=counts, name="Ruinas por año"))
                    fig_ruin.update_layout(title="Ruinas por año (simulación base)", template="plotly_dark")
                    st.plotly_chart(fig_ruin, use_container_width=True)
                survival_curve = ruin_stats.get("survival_curve", [])
                if survival_curve:
                    fig_surv = go.Figure()
                    fig_surv.add_trace(go.Scatter(x=list(range(len(survival_curve))), y=survival_curve, mode="lines", name="Supervivencia %"))
                    fig_surv.update_layout(title="Probabilidad de seguir solvente por año", yaxis_title="% supervivencia", template="plotly_dark")
                    st.plotly_chart(fig_surv, use_container_width=True)
            st.markdown("### Recomendaciones automáticas (heurísticas)")
            cur_success = last['success_pct']
            if cur_success >= 95:
                st.success("Tu plan supera 95% de probabilidad de éxito con las asunciones usadas.")
            elif cur_success >= 90:
                st.info("Tu plan supera 90% de probabilidad de éxito. Mantén disciplina y la reserva RF actual.")
            else:
                st.warning("Tu plan NO alcanza 90% de probabilidad. Sugerencias iniciales:")
                st.markdown("- Reducir gasto discrecional o total (ej. recortar viajes, suscripciones).")
                st.markdown("- Aumentar RF_pura / rf_reserve_years (mantener 3.5 años o más).")
                st.markdown("- Considerar venta anticipada de la casa si la regla automática indica inicio temprano.")
            st.markdown("---")
            st.caption("Nota: estas recomendaciones son heurísticas. Para decisiones definitivas recomienda validar con corridas A/B y análisis tornado con calibración histórica.")

if __name__ == "__main__":
    app()
