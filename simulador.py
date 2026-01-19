import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import re
import json

# -----------------------------------------------------------------------
# Simulador patrimonial realista
# - Incluye InstitutionalSimulator y PortfolioSimulator (modo "Cartera Real")
# - Parámetros avanzados expuestos en sidebar y en fallback en la página principal
# - Maneja venta de inmueble con fricciones y delay, gasto discrecional y reglas de guardrail
# -----------------------------------------------------------------------

# --- 0. UTILIDADES ---
def fmt(v):
    try:
        return f"{int(v):,}".replace(",", ".")
    except Exception:
        return str(v)

def clean_input(label, val, key):
    """Input numérico robusto con namespaced key para evitar colisiones de Streamlit."""
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
    """Convierte retorno aritmético anual (ej 0.11) a drift continuo (log-return)."""
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

# --- 1. DATACLASSES / CONFIG ---
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

    # Nominal arith inputs will be converted to log-drift where required
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

    # Advanced params (realism)
    p_def: float = 1.0  # advanced: shock damping in crisis (default neutral)
    vol_factor_active: float = 0.80
    sale_cost_pct: float = 0.02
    sale_delay_months: int = 3
    pct_discretionary: float = 0.20
    discretionary_cut_in_crisis: float = 0.60
    rf_reserve_years: float = 3.5

# --- 2. INSTITUTIONAL SIMULATOR (aggregate assets) ---
class InstitutionalSimulator:
    def __init__(self, config: SimulationConfig, assets: List[AssetBucket], withdrawals: List[WithdrawalTramo]):
        self.cfg = config
        self.assets = assets
        self.withdrawals = withdrawals
        self.dt = 1.0 / config.steps_per_year
        self.total_steps = int(config.horizon_years * config.steps_per_year)

        # expect mu_* are already continuous drifts when passed here
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
        pending_sale = np.full(n_sims, -1, dtype=int)  # -1 means none

        # aggregate assets: two buckets RV & RF
        asset_vals = np.zeros((n_sims, 2), dtype=float)
        rv_w = next((a.weight for a in self.assets if not a.is_bond), 0.6)
        asset_vals[:, 0] = float(self.cfg.initial_capital) * float(rv_w)
        asset_vals[:, 1] = float(self.cfg.initial_capital) * (1.0 - float(rv_w))

        regime = np.zeros(n_sims, dtype=int)
        df = int(self.cfg.t_df)

        # Build multivariate T innovations with shared chi-square for tail dependence
        g = self.rng.standard_normal((n_sims, n_steps, 2))
        w = self.rng.chisquare(df, (n_sims, n_steps, 1))
        Z_raw = (g / np.sqrt(w / df)) / np.sqrt(df / (df - 2))
        inf_sh = self.rng.normal(self.cfg.inflation_mean * self.dt, self.cfg.inflation_vol * np.sqrt(self.dt), (n_sims, n_steps))

        for t in range(n_steps):
            alive = is_alive
            if not np.any(alive):
                break

            # regimes (Markov) or stress schedule
            schedule = getattr(self.cfg, "stress_schedule", None)
            if schedule is not None and isinstance(schedule, list) and len(schedule) >= n_steps:
                regime[alive] = int(schedule[t])
            else:
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

            z_t = Z_raw[:, t, :]
            z_f = np.zeros_like(z_t)
            for r_idx, L in enumerate(self.L_mats):
                m = (regime == r_idx) & alive
                if np.any(m):
                    z_f[m] = np.dot(z_t[m], L.T)

            p_def = np.ones(n_sims) * float(self.cfg.p_def)  # advanced param (default 1.0)
            mus = self.mu_regimes[regime]
            sigs = self.sigma_regimes[regime]
            # apply log-normal step to each bucket (RV, RF)
            asset_vals[alive] *= np.exp((mus[alive] - 0.5 * sigs[alive] ** 2) * self.dt + sigs[alive] * np.sqrt(self.dt) * z_f[alive] * p_def[alive, None])

            # inflation
            cpi_paths[:, t + 1] = cpi_paths[:, t] * (1 + (inf_sh[:, t] + (regime == 1) * 0.003))

            # extra cashflows (annual)
            if (t + 1) % 12 == 0:
                y = (t + 1) // 12
                for e in self.cfg.extra_cashflows:
                    if e.year == y:
                        asset_vals[alive, 1] += e.amount * cpi_paths[alive, t + 1]

            # pending sale processing
            pending = (pending_sale >= 0) & alive
            if np.any(pending):
                pending_sale[pending] -= 1
                done = pending & (pending_sale == 0)
                if np.any(done):
                    cash = self.cfg.net_inmo_value * (1.0 - float(self.cfg.sale_cost_pct)) * cpi_paths[done, t + 1]
                    asset_vals[done, 1] += cash
                    has_h[done] = False
                    pending_sale[done] = -1

            # monthly withdrawal calculation (discretionary + essential)
            cur_y = (t + 1) / 12.0
            m_spend = np.zeros(n_sims)
            for w in self.withdrawals:
                if w.from_year <= cur_y < w.to_year:
                    m_spend = float(w.amount_nominal_monthly_start) * cpi_paths[:, t + 1]
                    break

            pct_dis = float(self.cfg.pct_discretionary)
            disc = m_spend * pct_dis
            ess = m_spend - disc

            # guardrails
            if self.cfg.use_guardrails and np.any(m_spend > 0):
                cur_real = np.sum(asset_vals, axis=1) / cpi_paths[:, t + 1]
                trig_gr = alive & (cur_real < (self.cfg.initial_capital * (1.0 - self.cfg.guardrail_trigger)))
                if np.any(trig_gr):
                    ess[trig_gr] *= (1 - self.cfg.guardrail_cut)
                    disc[trig_gr] *= (1 - self.cfg.guardrail_cut)

            # discretionary cut in crisis (user behavioral response)
            disc_keep = 1.0 - float(self.cfg.discretionary_cut_in_crisis)
            in_crisis = (regime > 0) & alive
            if np.any(in_crisis):
                disc[in_crisis] *= disc_keep

            m_spend_adj = ess + disc

            # property emergency sale trigger
            trig = alive & has_h & (np.sum(asset_vals, axis=1) < m_spend_adj * self.cfg.emergency_months_trigger)
            if np.any(trig):
                delay = int(self.cfg.sale_delay_months)
                for i in np.where(trig)[0]:
                    if delay <= 0:
                        asset_vals[i, 1] += self.cfg.net_inmo_value * (1.0 - float(self.cfg.sale_cost_pct)) * cpi_paths[i, t + 1]
                        has_h[i] = False
                    else:
                        pending_sale[i] = delay

            # automatic early sale rule based on RF liquidity months
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

            # withdrawals: use RF first, then RV
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

# --- 3. PORTFOLIO SIMULATOR (detalle por instrumento) ---
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
        # minimal fallback
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
        return sorted(
            candidates,
            key=lambda i: (
                int(getattr(self.positions[i], 'liquidity_days', 3)),
                int(self.positions[i].priority),
                self.positions[i].instrument_id,
            ),
        )[0]

    def run(self):
        n_sims, n_steps = int(self.cfg.n_sims), self.total_steps
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
        ruin_idx = np.full(n_sims, -1, dtype=int)
        has_h = np.full(n_sims, self.cfg.enable_prop, dtype=bool)
        pending_sale = np.full(n_sims, -1, dtype=int)
        regime = np.zeros(n_sims, dtype=int)

        df = int(self.cfg.t_df)
        g = self.rng.standard_normal((n_sims, n_steps, 2))
        w = self.rng.chisquare(df, (n_sims, n_steps, 1))
        Z_raw = (g / np.sqrt(w / df)) / np.sqrt(df / (df - 2))
        inf_sh = self.rng.normal(self.cfg.inflation_mean * self.dt, self.cfg.inflation_vol * np.sqrt(self.dt), (n_sims, n_steps))

        wd_order = self._withdraw_order()
        rf_sink = self._pick_rf_sink()
        rf_idxs = [j for j, p in enumerate(self.positions) if p.bucket == "RF_PURA" and p.include_withdrawals]

        initial_real = cap_paths[:, 0].copy()

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
                    regime[idx[r < self.cfg.prob_enter_local]] = 1
                    regime[idx[(r >= self.cfg.prob_enter_local) & (r < (self.cfg.prob_enter_local + self.cfg.prob_enter_global))]] = 2
                mc = (reg_prev > 0) & alive
                if np.any(mc):
                    idxc = np.where(mc)[0]
                    regime[idxc[self.rng.random(np.sum(mc)) < self.cfg.prob_exit_crisis]] = 0

            z_t = Z_raw[:, t, :]; z_f = np.zeros_like(z_t)
            for r_idx, L in enumerate(self.L_mats):
                m = (regime == r_idx) & alive
                if np.any(m):
                    z_f[m] = np.dot(z_t[m], L.T)

            p_def = np.ones(n_sims) * float(self.cfg.p_def)
            mus = self.mu_regimes[regime]
            sigs = self.sigma_regimes[regime]
            r_rv = (mus[:, 0] - 0.5 * sigs[:, 0] ** 2) * self.dt + sigs[:, 0] * np.sqrt(self.dt) * z_f[:, 0] * p_def
            r_rf = (mus[:, 1] - 0.5 * sigs[:, 1] ** 2) * self.dt + sigs[:, 1] * np.sqrt(self.dt) * z_f[:, 1] * p_def

            for j, p in enumerate(self.positions):
                if p.bucket == "PASIVO":
                    continue

                rv_base = float(p.rv_share)
                rv_min = float(getattr(p, "rv_min", 0.0))
                rv_max = float(getattr(p, "rv_max", 1.0))

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

            cpi_paths[:, t + 1] = cpi_paths[:, t] * (1 + (inf_sh[:, t] + (regime == 1) * 0.003))

            # pending sales
            pending = (pending_sale >= 0) & alive
            if np.any(pending):
                pending_sale[pending] -= 1
                done = pending & (pending_sale == 0)
                if np.any(done):
                    cash = self.cfg.net_inmo_value * (1.0 - float(self.cfg.sale_cost_pct)) * cpi_paths[done, t + 1]
                    sink = rf_sink if rf_sink is not None else 0
                    values[done, sink] += cash
                    has_h[done] = False
                    pending_sale[done] = -1

            if (t + 1) % 12 == 0:
                y = (t + 1) // 12
                for e in self.cfg.extra_cashflows:
                    if e.year == y:
                        sink = rf_sink if rf_sink is not None else 0
                        values[alive, sink] += e.amount * cpi_paths[alive, t + 1]

            cur_y = (t + 1) / 12.0
            m_spend = np.zeros(n_sims)
            for wtr in self.withdrawals:
                if wtr.from_year <= cur_y < wtr.to_year:
                    m_spend = wtr.amount_nominal_monthly_start * cpi_paths[:, t + 1]
                    break

            pct_dis = float(self.cfg.pct_discretionary)
            disc = m_spend * pct_dis
            ess = m_spend - disc

            if self.cfg.use_guardrails and np.any(m_spend > 0):
                cur_real = np.sum(values, axis=1) / cpi_paths[:, t + 1]
                trig_gr = alive & (cur_real < (initial_real * (1 - self.cfg.guardrail_trigger)))
                if np.any(trig_gr):
                    ess[trig_gr] *= (1 - self.cfg.guardrail_cut)
                    disc[trig_gr] *= (1 - self.cfg.guardrail_cut)

            disc_keep = 1.0 - float(self.cfg.discretionary_cut_in_crisis)
            in_crisis = (regime > 0) & alive
            if np.any(in_crrisis := in_crisis):
                disc[in_crrisis] *= disc_keep

            m_spend_adj = ess + disc

            tot = np.sum(values, axis=1)
            trig = alive & has_h & (tot < (m_spend_adj * self.cfg.emergency_months_trigger))
            if np.any(trig):
                delay = int(self.cfg.sale_delay_months)
                for i in np.where(trig)[0]:
                    if delay <= 0:
                        sink = rf_sink if rf_sink is not None else 0
                        values[i, sink] += self.cfg.net_inmo_value * (1.0 - float(self.cfg.sale_cost_pct)) * cpi_paths[i, t + 1]
                        has_h[i] = False
                    else:
                        pending_sale[i] = delay

            rf_balance = np.sum(values[:, rf_idxs], axis=1) if rf_idxs else np.zeros(n_sims)
            with np.errstate(divide='ignore', invalid='ignore'):
                rf_liquid_months = np.where(m_spend_adj > 0, rf_balance / m_spend_adj, np.inf)
            buffer_months = min(6, max(1, int(self.cfg.emergency_months_trigger / 4)))
            early_sale = alive & has_h & (rf_liquid_months < (self.cfg.sale_delay_months + buffer_months))
            if np.any(early_sale):
                for i in np.where(early_sale)[0]:
                    if pending_sale[i] < 0:
                        if self.cfg.sale_delay_months <= 0:
                            sink = rf_sink if rf_sink is not None else 0
                            values[i, sink] += self.cfg.net_inmo_value * (1.0 - float(self.cfg.sale_cost_pct)) * cpi_paths[i, t + 1]
                            has_h[i] = False
                        else:
                            pending_sale[i] = int(self.cfg.sale_delay_months)

            out = m_spend_adj + (self.cfg.enable_prop & (~has_h)) * (self.cfg.new_rent_cost * cpi_paths[:, t + 1])
            need = out.copy()
            for j in wd_order:
                if not np.any(alive & (need > 0)):
                    break
                take = np.minimum(values[:, j], need)
                take = np.where(alive, take, 0.0)
                values[:, j] -= take
                need -= take

            dead = alive & (need > 1e-6)
            if np.any(dead):
                is_alive[dead] = False
                ruin_idx[dead] = t + 1
                values[dead, :] = 0.0

            cap_paths[:, t + 1] = np.sum(values, axis=1)

            # refill RF logic at rebalancing
            if self.rules.rebalance_every_months > 0 and ((t + 1) % self.rules.rebalance_every_months == 0) and (self.rules.rf_reserve_years > 0):
                if rf_sink is not None and len(rf_idxs) > 0:
                    target = (self.rules.rf_reserve_years * 12) * np.maximum(m_spend_adj, 0)
                    cur_rf = np.sum(values[:, rf_idxs], axis=1)
                    need_rf = np.maximum(0.0, target - cur_rf)
                    allow = alive.copy()
                    if self.rules.rebalance_only_when_normal:
                        allow &= (regime == 0)
                    if np.any(allow & (need_rf > 0)):
                        sources_by_bucket = ["BAL", "RV", "AFP"]
                        for b in sources_by_bucket:
                            src = [j for j, p in enumerate(self.positions) if p.bucket == b and p.include_withdrawals]
                            if not src:
                                continue
                            pool = np.sum(values[:, src], axis=1)
                            can = allow & (need_rf > 0) & (pool > 0)
                            if not np.any(can):
                                continue
                            weights = np.zeros((n_sims, len(src)))
                            weights[can, :] = values[can][:, src] / pool[can][:, None]
                            transfer = weights * need_rf[:, None]
                            for k, j in enumerate(src):
                                t_k = np.minimum(values[:, j], transfer[:, k])
                                t_k = np.where(can, t_k, 0.0)
                                values[:, j] -= t_k
                                values[:, rf_sink] += t_k
                                need_rf -= t_k

        return cap_paths, cpi_paths, ruin_idx

# -----------------------------------------------------------------------
#  UI ENTRYPOINT
# -----------------------------------------------------------------------
def app(
    default_rf: int = 720000000,
    default_rv: int = 1080000000,
    default_inmo_neto: int = 500000000,
    portfolio_df: Optional[pd.DataFrame] = None,
    macro_data: Optional[Dict[str, Any]] = None,
    portfolio_json: Optional[str] = None,
):
    # Session defaults & safety
    st.session_state.setdefault('extra_events', [])
    st.session_state.setdefault('evy', 5)
    st.session_state.setdefault('evt', 'Entrada')
    st.session_state.setdefault('use_portfolio', False)
    st.session_state.setdefault('advanced', {
        'p_def': 1.0,
        'vol_factor_active': 0.80,
        'manager_riskoff': 0.20,
        'sale_cost_pct': 0.02,
        'sale_delay_months': 3,
        'pct_discretionary': 0.20,
        'discretionary_cut_in_crisis': 0.60,
        'rf_reserve_years': 3.5,
        'corr_normal': 0.35,
    })

    st.set_page_config(page_title="Sovereign Alpha Engine", layout="wide")
    st.markdown("## 🦅 Panel de Decisión Patrimonial (Versión Realista)")

    SC_RET = {"Conservador": [0.08, 0.045], "Histórico (11%)": [0.11, 0.06], "Crecimiento (13%)": [0.13, 0.07]}
    SC_GLO = {"Crash Financiero": [-0.22, -0.02, 0.75], "Colapso Sistémico": [-0.30, -0.06, 0.92], "Recesión Estándar": [-0.15, 0.01, 0.55]}

    # Fallback advanced controls on main page (in case sidebar widgets are not rendered)
    show_adv_main = st.checkbox("Mostrar configuración avanzada (fallback)", value=False, key="show_adv_main")
    if show_adv_main:
        st.caption("Panel avanzado (fallback): los mismos controles que en la barra lateral.")
        adv = st.session_state.get('advanced', {})
        p_def_adv = st.number_input("p_def (shock damping en crisis, advanced)", 0.0, 2.0, adv.get('p_def', 1.0), step=0.05)
        vol_factor_active = st.number_input("vol_factor_active (si gestor activo)", 0.1, 1.0, adv.get('vol_factor_active', 0.80), step=0.05)
        manager_riskoff = st.slider("manager_riskoff_in_crisis (0..1)", 0.0, 1.0, adv.get('manager_riskoff', 0.20), 0.05)
        sale_cost = st.number_input("sale_cost_pct (venta casa)", 0.0, 0.2, adv.get('sale_cost_pct', 0.02), step=0.005)
        sale_delay = st.number_input("sale_delay_months (venta casa)", 0, 12, adv.get('sale_delay_months', 3), step=1)
        pct_discr = st.number_input("pct_discretionary (gasto)", 0.0, 1.0, adv.get('pct_discretionary', 0.20), step=0.05)
        disc_cut = st.number_input("discretionary_cut_in_crisis", 0.0, 1.0, adv.get('discretionary_cut_in_crisis', 0.60), step=0.05)
        rf_reserve_years = st.number_input("rf_reserve_years (años)", 0.0, 10.0, adv.get('rf_reserve_years', 3.5), step=0.5)

        st.session_state['advanced'] = {
            'p_def': float(p_def_adv),
            'vol_factor_active': float(vol_factor_active),
            'manager_riskoff': float(manager_riskoff),
            'sale_cost_pct': float(sale_cost),
            'sale_delay_months': int(sale_delay),
            'pct_discretionary': float(pct_discr),
            'discretionary_cut_in_crisis': float(disc_cut),
            'rf_reserve_years': float(rf_reserve_years),
            'corr_normal': float(st.session_state['advanced'].get('corr_normal', 0.35)),
        }

    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Configuración")
        use_portfolio = st.checkbox("Modo Cartera Real (Instrumentos)", value=st.session_state.get("use_portfolio", False))
        st.session_state["use_portfolio"] = use_portfolio
        is_active = st.checkbox("Gestión Activa / Balanceados", value=True)
        sel_glo = st.selectbox("Crisis Global", list(SC_GLO.keys()), index=0)
        st.caption(f"Stress RV: {SC_GLO[sel_glo][0]*100:.1f}%")
        st.divider()
        sel_ret = st.selectbox("Rentabilidad Normal", list(SC_RET.keys()) + ["Personalizado"], index=1)
        if sel_ret == "Personalizado":
            c_rv_in = st.number_input("RV % Nom. (ej. 11 = 11%)", 0.0, 50.0, 11.0) / 100.0
            c_rf_in = st.number_input("RF % Nom. (ej. 6 = 6%)", 0.0, 50.0, 6.0) / 100.0
        else:
            c_rv_in, c_rf_in = SC_RET[sel_ret]
            st.info(f"RV: {c_rv_in*100:.1f}% | RF: {c_rf_in*100:.1f}%")
        n_sims = st.slider("Simulaciones", 500, 5000, 2000, step=100)
        horiz = st.slider("Horizonte (años)", 10, 50, 40)

        with st.expander("Configuración avanzada (sidebar)", expanded=False):
            st.subheader("Parámetros avanzados (default prudente)")
            adv = st.session_state.get('advanced', {})
            p_def_adv = st.number_input("p_def (shock damping en crisis, advanced)", 0.0, 2.0, adv.get('p_def', 1.0), step=0.05)
            vol_factor_active = st.number_input("vol_factor_active (si gestor activo)", 0.1, 1.0, adv.get('vol_factor_active', 0.80), step=0.05)
            manager_riskoff = st.slider("manager_riskoff_in_crisis (0..1)", 0.0, 1.0, adv.get('manager_riskoff', 0.20), 0.05)
            sale_cost = st.number_input("sale_cost_pct (venta casa)", 0.0, 0.2, adv.get('sale_cost_pct', 0.02), step=0.005)
            sale_delay = st.number_input("sale_delay_months (venta casa)", 0, 12, adv.get('sale_delay_months', 3), step=1)
            pct_discr = st.number_input("pct_discretionary (gasto)", 0.0, 1.0, adv.get('pct_discretionary', 0.20), step=0.05)
            disc_cut = st.number_input("discretionary_cut_in_crisis", 0.0, 1.0, adv.get('discretionary_cut_in_crisis', 0.60), step=0.05)
            rf_reserve_years = st.number_input("rf_reserve_years (años)", 0.0, 10.0, adv.get('rf_reserve_years', 3.5), step=0.5)

            st.session_state['advanced'] = {
                'p_def': float(p_def_adv),
                'vol_factor_active': float(vol_factor_active),
                'manager_riskoff': float(manager_riskoff),
                'sale_cost_pct': float(sale_cost),
                'sale_delay_months': int(sale_delay),
                'pct_discretionary': float(pct_discr),
                'discretionary_cut_in_crisis': float(disc_cut),
                'rf_reserve_years': float(rf_reserve_years),
                'corr_normal': float(st.session_state['advanced'].get('corr_normal', 0.35)),
            }

    # Tabs
    tab_sim, tab_stress, tab_diag, tab_sum, tab_opt = st.tabs(["📊 Simulación", "🧯 Stress", "🩻 Diagnóstico", "🧾 Resumen", "🎯 Optimizador"])

    # --- Simulación Tab (principal) ---
    with tab_sim:
        # Inputs: either portfolio real or aggregated simple inputs
        positions: List[InstrumentPosition] = []
        rules = PortfolioRulesConfig()
        portfolio_ready = False

        if use_portfolio:
            st.subheader("📦 Cartera Real (Instrumentos)")
            src = st.radio("Fuente de cartera", ["Pegar JSON", "Usar datos del Dashboard"], horizontal=True)
            df_src = None
            if src == "Pegar JSON":
                default_txt = portfolio_json or st.session_state.get("portfolio_json", "")
                txt = st.text_area("JSON", value=default_txt, height=200, placeholder="Pega acá el JSON completo (registros → instrumentos)")
                if txt.strip():
                    st.session_state["portfolio_json"] = txt
                    try:
                        df_src = parse_portfolio_json(txt)
                    except Exception as e:
                        st.error(f"No pude parsear el JSON: {e}")
            else:
                df_src = portfolio_df if portfolio_df is not None else None

            if df_src is None or (hasattr(df_src, "empty") and df_src.empty):
                st.warning("No hay datos de cartera para simular en modo instrumentos.")
            else:
                df_base = enrich_with_meta(_normalize_portfolio_df(df_src))
                # data_editor can fail on older streamlit; wrap in try
                try:
                    edited = st.data_editor(
                        df_base,
                        use_container_width=True,
                        column_config={
                            "rv_share": st.column_config.NumberColumn("% RV (0-1)", min_value=0.0, max_value=1.0, step=0.01),
                            "rv_min": st.column_config.NumberColumn("RV min (0-1)", min_value=0.0, max_value=1.0, step=0.01),
                            "rv_max": st.column_config.NumberColumn("RV max (0-1)", min_value=0.0, max_value=1.0, step=0.01),
                            "liquidity_days": st.column_config.NumberColumn("Liquidez (dias habiles)", min_value=0, max_value=30, step=1),
                            "priority": st.column_config.NumberColumn("Prioridad retiro", min_value=0, max_value=99, step=1),
                            "include_withdrawals": st.column_config.CheckboxColumn("Retirable"),
                            "bucket": st.column_config.SelectboxColumn("Bucket", options=["RF_PURA", "BAL", "RV", "AFP", "PASIVO"]),
                        },
                        disabled=["instrument_id", "name"],
                        key="portfolio_editor",
                    )
                except Exception:
                    edited = df_base.copy()

                tot = float(edited["value_clp"].sum())
                rv_amt = float((edited["value_clp"] * edited["rv_share"]).sum())
                rf_pura_amt = float(edited.loc[edited["bucket"] == "RF_PURA", "value_clp"].sum())
                rv_pct = 0.0 if tot <= 0 else 100.0 * rv_amt / tot

                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("Patrimonio (CLP)", f"${fmt(tot)}")
                with c2:
                    st.metric("Motor (RV) estimado", f"{rv_pct:.1f}%")
                with c3:
                    st.metric("RF pura hoy", f"${fmt(rf_pura_amt)}")

                # portfolio rules
                rr1, rr2, rr3, rr4 = st.columns(4)
                with rr1:
                    rules.rf_reserve_years = st.slider("Reserva RF pura (años)", 0.0, 10.0, float(st.session_state.get('advanced', {}).get('rf_reserve_years', 3.5)), 0.5)
                with rr2:
                    reb = st.checkbox("Rebalance anual RF pura", value=True)
                    rules.rebalance_every_months = 12 if reb else 0
                with rr3:
                    rules.rebalance_only_when_normal = st.checkbox("Rebalance solo si NO hay crisis", value=True)
                with rr4:
                    rules.manager_riskoff_in_crisis = st.slider("Gestor defensivo en crisis (0-1)", 0.0, 1.0, float(st.session_state.get('advanced', {}).get('manager_riskoff', 0.20)), 0.05)

                positions = []
                for _, r in edited.iterrows():
                    positions.append(
                        InstrumentPosition(
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
                        )
                    )
                portfolio_ready = tot > 0 and any(p.include_withdrawals for p in positions)

        # Simple aggregated mode (no instruments)
        if not use_portfolio:
            st.subheader("📈 Entrada rápida (modo agregado)")
            tot_default = default_rf + default_rv
            cap_val = clean_input("Capital Total ($)", tot_default, "cap")
            rv_pct = st.slider("Motor (RV %)", 0, 100, 60)
            st.metric("Reserva RF", f"${fmt(cap_val * (1 - rv_pct/100))}")

        # Spending profile inputs
        st.markdown("### 💸 Perfil de gasto")
        g1, g2, g3 = st.columns(3)
        with g1:
            r1 = clean_input("Gasto F1 (CLP/mes)", 6000000, "r1")
            d1 = st.number_input("Años F1", 0, 40, 20)
        with g2:
            r2 = clean_input("Gasto F2 (CLP/mes)", 4000000, "r2")
            d2 = st.number_input("Años F2", 0, 40, 20)
        with g3:
            r3 = clean_input("Gasto F3 (CLP/mes)", 4000000, "r3")

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

        enable_p = st.checkbox("Venta Casa Emergencia", value=True)
        val_h = clean_input("Valor Neto Casa ($)", default_inmo_neto, "vi") if enable_p else 0

        # Build extra cashflows objects
        extra_events_objs = [ExtraCashflow(int(e["year"]), float(e["amount"]), e.get("name", "Hito")) for e in st.session_state.get("extra_events", [])]

        # Build cfg_current and convert arithmetic mu to continuous drift
        adv = st.session_state.get('advanced', {})
        mu_rv_cont = to_continuous_mu(c_rv_in) if 'c_rv_in' in locals() else to_continuous_mu(0.11)
        mu_rf_cont = to_continuous_mu(c_rf_in) if 'c_rf_in' in locals() else to_continuous_mu(0.06)

        cfg_current = SimulationConfig(
            horizon_years=horiz,
            initial_capital=int(cap_val) if not use_portfolio else (int(sum([p.value_clp for p in positions])) if positions else default_rf + default_rv),
            n_sims=int(n_sims),
            is_active_managed=bool(is_active),
            enable_prop=bool(enable_p),
            net_inmo_value=float(val_h),
            mu_normal_rv=float(mu_rv_cont),
            mu_normal_rf=float(mu_rf_cont),
            extra_cashflows=extra_events_objs,
            mu_global_rv=float(SC_GLO[sel_glo][0]),
            mu_global_rf=float(SC_GLO[sel_glo][1]),
            corr_global=float(SC_GLO[sel_glo][2]),
            corr_normal=float(st.session_state.get('advanced', {}).get('corr_normal', 0.35)),
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

        # Run simulation button
        if st.button("🚀 INICIAR SIMULACIÓN", type="primary", key="run_sim"):
            wds = [
                WithdrawalTramo(0, int(d1), int(r1)),
                WithdrawalTramo(int(d1), int(d1 + d2), int(r2)),
                WithdrawalTramo(int(d1 + d2), int(horiz), int(r3))
            ]

            if use_portfolio:
                if not portfolio_ready:
                    st.error("Modo instrumentos activo, pero la cartera no está lista o no hay activos retibles.")
                    st.stop()
                rules.manager_riskoff_in_crisis = float(st.session_state.get('advanced', {}).get('manager_riskoff', rules.manager_riskoff_in_crisis))
                sim = PortfolioSimulator(cfg_current, positions, wds, rules)
                paths, cpi, r_i = sim.run()
            else:
                a_t = [AssetBucket("RV", float(rv_pct) / 100.0), AssetBucket("RF", (100.0 - float(rv_pct)) / 100.0, True)]
                sim = InstitutionalSimulator(cfg_current, a_t, wds)
                paths, cpi, r_i = sim.run()

            prob = (r_i <= -1).mean()
            success_pct = float((1.0 - ((r_i > -1).mean())) * 100.0)
            ci_low, ci_high = success_ci(success_pct, int(cfg_current.n_sims))
            terminal_real = paths[:, -1] / np.maximum(cpi[:, -1], 1e-9)
            p10 = float(np.percentile(terminal_real, 10))
            p50 = float(np.percentile(terminal_real, 50))
            p90 = float(np.percentile(terminal_real, 90))
            ruined = (r_i > -1)
            ruin_years = (r_i[ruined] / 12.0) if np.any(ruined) else np.array([])

            st.markdown(
                f"<div style='padding:12px; border-radius:8px; background:#091121; color:#fff;'>"
                f"<h2>Éxito: {success_pct:.1f}% (IC95%: {ci_low:.1f}%–{ci_high:.1f}%)</h2>"
                f"<p>Mediana legado real: ${fmt(p50)} | P10: ${fmt(p10)} | P90: ${fmt(p90)}</p>"
                f"</div>", unsafe_allow_html=True
            )

            # Plot envelope
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

            st.session_state["last_run"] = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "success_pct": success_pct,
                "success_ci": [ci_low, ci_high],
                "median_terminal_real": p50,
                "p10_terminal_real": p10,
                "p90_terminal_real": p90,
                "median_ruin_year": float(np.median(ruin_years)) if ruin_years.size else None,
                "cfg": cfg_current.__dict__.copy(),
            }

    # Other tabs (Stress, Diagnóstico, Resumen, Optimizador) can reuse last_run to show deeper analysis.
    # For brevity these tabs are placeholders but can be expanded; they will read st.session_state["last_run"].

    with tab_sum:
        st.header("🧾 Resumen Ejecutivo (Beta)")
        last = st.session_state.get("last_run")
        if not last:
            st.info("Ejecuta primero una simulación para obtener un resumen ejecutivo y recomendaciones.")
        else:
            st.markdown(f"**Última corrida:** {last['timestamp']}")
            st.markdown(f"- Éxito estimado: **{last['success_pct']:.1f}%** (IC95%: {last['success_ci'][0]:.1f}% – {last['success_ci'][1]:.1f}%)")
            st.markdown(f"- Mediana legado real: **${fmt(last['median_terminal_real'])}**")
            st.markdown(f"- P10: **${fmt(last['p10_terminal_real'])}** | P90: **${fmt(last['p90_terminal_real'])}**")
            st.markdown("### Recomendaciones iniciales (automáticas)")
            # Simple heuristic recommendations for 90/95% targets
            cur_success = last['success_pct']
            if cur_success >= 95:
                st.success("Tu plan supera 95% de probabilidad de éxito con las asunciones actuales.")
            elif cur_success >= 90:
                st.info("Tu plan supera 90% de probabilidad de éxito. Considera mantener RF_reserve y disciplina de gasto.")
            else:
                st.warning("Tu plan NO alcanza 90% de probabilidad. Sugerencias automáticas:")
                st.markdown("- Reducir gasto discrecional o total.")
                st.markdown("- Incrementar RF_pura o reservar más años en RF (rf_reserve_years).")
                st.markdown("- Considerar venta anticipada de inmueble según regla automática (si aplica).")
            st.markdown("---")
            st.caption("Nota: este resumen es un primer diagnóstico. Para recomendaciones a medida se requiere calibración con series históricas y análisis tornado detallado.")

# If run as main, start Streamlit app
if __name__ == "__main__":
    app()
