# app.py
# ------------------------------------------------------------
# Projeto 6 – Dashboard Estratégico de Supply Chain (Streamlit)
# Disciplina: Fundamentos em ciência de dados | Período: 2025.2
#
# Como executar:
# 1) pip install streamlit pandas numpy plotly
# 2) Coloque estes arquivos NA MESMA PASTA do app.py:
#    - FCD_clientes.csv
#    - FCD_estoque.csv
#    - FCD_produtos.csv
#    - FCD_compras.csv
#    - FCD_logistica (1).csv
#    - FCD_vendas.csv
# 3) streamlit run app.py
# ------------------------------------------------------------

from __future__ import annotations

import os
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# =========================
# Config
# =========================
st.set_page_config(
    page_title="Dashboard Estratégico de Supply Chain",
    page_icon="📦",
    layout="wide",
)

warnings.filterwarnings("ignore")


# =========================
# Arquivos (nomes EXATOS)
# =========================
FILES = {
    "clientes": "FCD_clientes.csv",
    "estoque": "FCD_estoque.csv",
    "produtos": "FCD_produtos.csv",
    "compras": "FCD_compras.csv",
    "logistica": "FCD_logistica.csv",
    "vendas": "FCD_vendas.csv",
}


# =========================
# Utilitários
# =========================
def _find_file(filename: str) -> Path:
    """Procura o arquivo no diretório do app, no cwd e (opcional) em /mnt/data."""
    candidates = []

    # Pasta do app.py (quando disponível)
    try:
        candidates.append(Path(__file__).resolve().parent / filename)
    except Exception:
        pass

    # Diretório atual
    candidates.append(Path.cwd() / filename)

    # Ambiente sandbox (se existir)
    candidates.append(Path("/mnt/data") / filename)

    for p in candidates:
        if p.exists():
            return p

    raise FileNotFoundError(
        f"Não encontrei o arquivo '{filename}'.\n"
        f"Coloque-o na mesma pasta do app.py (ou no diretório atual)."
    )


def _read_csv_smart(path: Path) -> pd.DataFrame:
    """
    Lê CSV tentando detectar separador.
    (Esses dados normalmente vêm com ';', mas pode variar.)
    """
    # 1ª tentativa (padrão)
    df = pd.read_csv(path)

    # Se veio “tudo em 1 coluna”, tenta ';'
    if df.shape[1] == 1:
        df2 = pd.read_csv(path, sep=";")
        if df2.shape[1] > 1:
            return df2

    # Se ainda 1 coluna, tenta autodetecção mais agressiva
    if df.shape[1] == 1:
        df2 = pd.read_csv(path, sep=None, engine="python")
        if df2.shape[1] > 1:
            return df2

    return df


def _to_datetime(series: pd.Series) -> pd.Series:
    """Converte para datetime com tolerância a formatos mistos (dd/mm/yyyy e yyyy-mm-dd)."""
    s = series.astype(str).str.strip()
    # Primeiro tenta dayfirst (bom para dd/mm/yyyy)
    dt1 = pd.to_datetime(s, errors="coerce", dayfirst=True, infer_datetime_format=True)
    # Se ficou muito NaT, tenta sem dayfirst
    if dt1.isna().mean() > 0.25:
        dt2 = pd.to_datetime(s, errors="coerce", dayfirst=False, infer_datetime_format=True)
        # escolhe o que converteu mais
        return dt2 if dt2.isna().mean() < dt1.isna().mean() else dt1
    return dt1


def _to_numeric(series: pd.Series) -> pd.Series:
    """
    Converte para número de forma robusta para pt-BR e também para colunas já numéricas.

    Problema comum: se a coluna já vem como float (ex.: 1234.56), remover "." transforma em 123456.
    Aqui só removemos separador de milhar quando detectamos vírgula como separador decimal.
    """
    # Se já for numérico, não mexe em separadores
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    s = series.astype(str).str.strip()
    s = s.replace({"": np.nan, "nan": np.nan, "None": np.nan})
    s = s.str.replace("\u00a0", "", regex=False).str.replace(" ", "", regex=False)

    # Heurística:
    # - Se tem vírgula, assume decimal "," e milhar "."  -> remove "." e troca "," por "."
    # - Se não tem vírgula, assume decimal "." e remove possíveis milhares ","
    has_comma = s.str.contains(",", na=False)
    s2 = s.copy()
    s2.loc[has_comma] = s2.loc[has_comma].str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    s2.loc[~has_comma] = s2.loc[~has_comma].str.replace(",", "", regex=False)

    return pd.to_numeric(s2, errors="coerce")


def _month_start(d: pd.Series) -> pd.Series:
    return pd.to_datetime(d.dt.to_period("M").dt.to_timestamp())


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b not in (0, 0.0, None) else 0.0


def _brl(x: float) -> str:
    try:
        return f"R$ {x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return f"R$ {x}"


def _brl_compact(x: float) -> str:
    """Formato curto para KPIs (evita truncamento visual)."""
    try:
        x = float(x)
    except Exception:
        return _brl(0.0)

    ax = abs(x)
    if ax >= 1_000_000_000:
        return f"R$ {x/1_000_000_000:.2f} bi".replace(".", ",")
    if ax >= 1_000_000:
        return f"R$ {x/1_000_000:.2f} mi".replace(".", ",")
    if ax >= 1_000:
        return f"R$ {x/1_000:.1f} mil".replace(".", ",")
    return _brl(x)


# =========================
# Carregamento (cache)
# =========================
@st.cache_data(show_spinner=False)
def load_data() -> dict[str, pd.DataFrame]:
    data = {}
    for key, fname in FILES.items():
        path = _find_file(fname)
        df = _read_csv_smart(path)
        data[key] = df
    return data


def prepare_data(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    clientes = data["clientes"].copy()
    estoque = data["estoque"].copy()
    produtos = data["produtos"].copy()
    compras = data["compras"].copy()
    logistica = data["logistica"].copy()
    vendas = data["vendas"].copy()

    # Padroniza colunas (tira espaços)
    for df in (clientes, estoque, produtos, compras, logistica, vendas):
        df.columns = [c.strip() for c in df.columns]

    # Datas
    if "data_referencia" in estoque.columns:
        estoque["data_referencia"] = _to_datetime(estoque["data_referencia"])
    if "data_compra" in compras.columns:
        compras["data_compra"] = _to_datetime(compras["data_compra"])
    if "data_pedido" in logistica.columns:
        logistica["data_pedido"] = _to_datetime(logistica["data_pedido"])
    if "data_entrega" in logistica.columns:
        logistica["data_entrega"] = _to_datetime(logistica["data_entrega"])
    if "data_venda" in vendas.columns:
        vendas["data_venda"] = _to_datetime(vendas["data_venda"])

    # Numéricos principais
    for col in ["quantidade_estoque", "estoque_minimo"]:
        if col in estoque.columns:
            estoque[col] = _to_numeric(estoque[col])
    for col in ["preco_unitario", "custo_unitario", "estoque_inicial", "peso_kg"]:
        if col in produtos.columns:
            produtos[col] = _to_numeric(produtos[col])
    for col in ["quantidade_comprada", "valor_unitario", "valor_total", "prazo_entrega_dias"]:
        if col in compras.columns:
            compras[col] = _to_numeric(compras[col])
    for col in ["prazo_estimado_dias", "prazo_real_dias", "custo_transporte"]:
        if col in logistica.columns:
            logistica[col] = _to_numeric(logistica[col])
    for col in ["quantidade_vendida", "valor_unitario", "valor_total"]:
        if col in vendas.columns:
            vendas[col] = _to_numeric(vendas[col])

    # Chaves como int (quando possível)
    for df, cols in [
        (clientes, ["cliente_id"]),
        (produtos, ["produto_id"]),
        (estoque, ["produto_id"]),
        (compras, ["produto_id"]),
        (vendas, ["produto_id", "cliente_id"]),
    ]:
        for c in cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

    # Remove linhas sem data (evita bugs nos filtros)
    if "data_referencia" in estoque.columns:
        estoque = estoque.dropna(subset=["data_referencia"])
    if "data_compra" in compras.columns:
        compras = compras.dropna(subset=["data_compra"])
    if "data_pedido" in logistica.columns:
        logistica = logistica.dropna(subset=["data_pedido"])
    if "data_venda" in vendas.columns:
        vendas = vendas.dropna(subset=["data_venda"])

    return {
        "clientes": clientes,
        "estoque": estoque,
        "produtos": produtos,
        "compras": compras,
        "logistica": logistica,
        "vendas": vendas,
    }


# =========================
# KPI e Métricas
# =========================
def compute_kpis(
    clientes: pd.DataFrame,
    estoque: pd.DataFrame,
    produtos: pd.DataFrame,
    compras: pd.DataFrame,
    logistica: pd.DataFrame,
    vendas: pd.DataFrame,
    date_start: pd.Timestamp,
    date_end: pd.Timestamp,
    categoria: str | None,
    produto_id: int | None,
    fornecedor: str | None,
    transportadora: str | None,
    canal_venda: str | None,
    loja_id: int | None,
) -> dict:
    # ---- Filtros base ----
    vendas_f = vendas[(vendas["data_venda"] >= date_start) & (vendas["data_venda"] <= date_end)].copy()
    compras_f = compras[(compras["data_compra"] >= date_start) & (compras["data_compra"] <= date_end)].copy()
    log_f = logistica[(logistica["data_pedido"] >= date_start) & (logistica["data_pedido"] <= date_end)].copy()
    estoque_f = estoque[(estoque["data_referencia"] >= date_start) & (estoque["data_referencia"] <= date_end)].copy()

    # Garante valor_total (alguns grupos calculam diferente; aqui padroniza)
    if "valor_total" not in vendas_f.columns and {"quantidade_vendida", "valor_unitario"}.issubset(vendas_f.columns):
        vendas_f["valor_total"] = vendas_f["quantidade_vendida"].fillna(0) * vendas_f["valor_unitario"].fillna(0)
    elif "valor_total" in vendas_f.columns and {"quantidade_vendida", "valor_unitario"}.issubset(vendas_f.columns):
        # completa valor_total ausente
        mask = vendas_f["valor_total"].isna()
        if mask.any():
            vendas_f.loc[mask, "valor_total"] = vendas_f.loc[mask, "quantidade_vendida"].fillna(0) * vendas_f.loc[mask, "valor_unitario"].fillna(0)

    if "valor_total" not in compras_f.columns and {"quantidade_comprada", "valor_unitario"}.issubset(compras_f.columns):
        compras_f["valor_total"] = compras_f["quantidade_comprada"].fillna(0) * compras_f["valor_unitario"].fillna(0)
    elif "valor_total" in compras_f.columns and {"quantidade_comprada", "valor_unitario"}.issubset(compras_f.columns):
        mask = compras_f["valor_total"].isna()
        if mask.any():
            compras_f.loc[mask, "valor_total"] = compras_f.loc[mask, "quantidade_comprada"].fillna(0) * compras_f.loc[mask, "valor_unitario"].fillna(0)

    # Aplica dimensões
    if canal_venda and "canal_venda" in vendas_f.columns:
        vendas_f = vendas_f[vendas_f["canal_venda"] == canal_venda]
    if loja_id is not None and "loja_id" in vendas_f.columns:
        vendas_f = vendas_f[vendas_f["loja_id"].astype("Int64") == loja_id]

    if fornecedor and "fornecedor" in compras_f.columns:
        compras_f = compras_f[compras_f["fornecedor"] == fornecedor]
    if transportadora and "transportadora" in log_f.columns:
        log_f = log_f[log_f["transportadora"] == transportadora]

    # Produto/Categoria (via tabela produtos)
    if categoria or (produto_id is not None):
        prod_f = produtos.copy()
        if categoria and "categoria" in prod_f.columns:
            prod_f = prod_f[prod_f["categoria"] == categoria]
        if produto_id is not None and "produto_id" in prod_f.columns:
            prod_f = prod_f[prod_f["produto_id"].astype("Int64") == produto_id]
        allowed_ids = set(prod_f["produto_id"].dropna().astype(int).tolist()) if "produto_id" in prod_f.columns else set()

        if "produto_id" in vendas_f.columns:
            vendas_f = vendas_f[vendas_f["produto_id"].dropna().astype(int).isin(allowed_ids)]
        if "produto_id" in compras_f.columns:
            compras_f = compras_f[compras_f["produto_id"].dropna().astype(int).isin(allowed_ids)]
        if "produto_id" in estoque_f.columns:
            estoque_f = estoque_f[estoque_f["produto_id"].dropna().astype(int).isin(allowed_ids)]

    # ---- KPI 1: Estoque Crítico (última data do período) ----
    estoque_critico_qtd = 0
    estoque_critico_df = pd.DataFrame()

    if not estoque_f.empty and {"produto_id", "quantidade_estoque", "estoque_minimo"}.issubset(estoque_f.columns):
        last_day = estoque_f["data_referencia"].max()
        snap = estoque_f[estoque_f["data_referencia"] == last_day].copy()
        snap["critico"] = snap["quantidade_estoque"] < snap["estoque_minimo"]
        estoque_critico_df = snap[snap["critico"]].copy()
        # conta produtos (únicos) em situação crítica
        estoque_critico_qtd = estoque_critico_df["produto_id"].nunique()

    # ---- KPI 2: Receita no período e Receita Mensal ----
    receita_periodo = float(vendas_f["valor_total"].sum()) if "valor_total" in vendas_f.columns else 0.0
    receita_mensal = pd.DataFrame()
    if not vendas_f.empty and {"data_venda", "valor_total"}.issubset(vendas_f.columns):
        vendas_f["mes"] = _month_start(vendas_f["data_venda"])
        receita_mensal = vendas_f.groupby("mes", as_index=False)["valor_total"].sum().rename(columns={"valor_total": "receita"})

    # KPI auxiliar: Receita mensal (último mês disponível no período filtrado)
    receita_ultimo_mes = 0.0
    mes_ultimo = None
    if not receita_mensal.empty and {"mes", "receita"}.issubset(receita_mensal.columns):
        mes_ultimo = pd.to_datetime(receita_mensal["mes"]).max()
        try:
            receita_ultimo_mes = float(receita_mensal.loc[receita_mensal["mes"] == mes_ultimo, "receita"].sum())
        except Exception:
            receita_ultimo_mes = 0.0

    # ---- KPI 3: Fornecedores mais usados (por frequência e por gasto) ----
    fornecedores_freq = pd.DataFrame()
    fornecedores_gasto = pd.DataFrame()
    if not compras_f.empty and "fornecedor" in compras_f.columns:
        fornecedores_freq = (
            compras_f.groupby("fornecedor", as_index=False)
            .size()
            .rename(columns={"size": "num_compras"})
            .sort_values("num_compras", ascending=False)
        )
        if "valor_total" in compras_f.columns:
            fornecedores_gasto = (
                compras_f.groupby("fornecedor", as_index=False)["valor_total"]
                .sum()
                .rename(columns={"valor_total": "gasto_total"})
                .sort_values("gasto_total", ascending=False)
            )

    # KPI auxiliar: fornecedor mais usado / maior gasto
    fornecedor_top_freq = None
    fornecedor_top_num_compras = 0
    if not fornecedores_freq.empty and {"fornecedor", "num_compras"}.issubset(fornecedores_freq.columns):
        fornecedor_top_freq = str(fornecedores_freq.iloc[0]["fornecedor"])
        try:
            fornecedor_top_num_compras = int(fornecedores_freq.iloc[0]["num_compras"])
        except Exception:
            fornecedor_top_num_compras = 0

    fornecedor_top_gasto = None
    fornecedor_top_gasto_valor = 0.0
    if not fornecedores_gasto.empty and {"fornecedor", "gasto_total"}.issubset(fornecedores_gasto.columns):
        fornecedor_top_gasto = str(fornecedores_gasto.iloc[0]["fornecedor"])
        try:
            fornecedor_top_gasto_valor = float(fornecedores_gasto.iloc[0]["gasto_total"])
        except Exception:
            fornecedor_top_gasto_valor = 0.0

    # ---- KPI 4: Taxa de Entrega no Prazo ----
    taxa_prazo = 0.0
    entregas_stats = {}
    if not log_f.empty and {"prazo_estimado_dias", "prazo_real_dias"}.issubset(log_f.columns):
        valid = log_f.dropna(subset=["prazo_estimado_dias", "prazo_real_dias"]).copy()
        if not valid.empty:
            valid["no_prazo"] = valid["prazo_real_dias"] <= valid["prazo_estimado_dias"]
            total = int(len(valid))
            ontime = int(valid["no_prazo"].sum())
            taxa_prazo = _safe_div(ontime, total) * 100.0
            entregas_stats = {"total": total, "no_prazo": ontime, "atraso": total - ontime}

    # ---- Indicadores Financeiros: Custo Total de Compras ----
    custo_total_compras = float(compras_f["valor_total"].sum()) if "valor_total" in compras_f.columns else 0.0

    # ---- Margem Bruta Estimada ----
    # Estratégia:
    # 1) custo médio por produto: média ponderada das compras (valor_unitario por quantidade)
    # 2) CMV estimado do período: quantidade_vendida * custo_medio
    # 3) Margem: Receita - CMV
    custo_medio = pd.DataFrame(columns=["produto_id", "custo_medio"])
    # Prioriza custo médio NO PERÍODO FILTRADO (mais consistente com o que colegas normalmente fazem).
    base_source = compras_f if (not compras_f.empty) else compras
    if not base_source.empty and {"produto_id", "quantidade_comprada", "valor_unitario"}.issubset(base_source.columns):
        base = base_source.dropna(subset=["produto_id", "quantidade_comprada", "valor_unitario"]).copy()
        base = base[base["quantidade_comprada"] > 0]
        if not base.empty:
            # média ponderada do valor_unitario por quantidade_comprada
            # (forma vetorizada para garantir a coluna "custo_medio" e evitar KeyError)
            base["valor_x_qtd"] = base["valor_unitario"] * base["quantidade_comprada"]
            agg = (
                base.groupby("produto_id", as_index=False)
                .agg(qtd_total=("quantidade_comprada", "sum"), valor_total=("valor_x_qtd", "sum"))
            )
            agg = agg[agg["qtd_total"] > 0].copy()
            agg["custo_medio"] = agg["valor_total"] / agg["qtd_total"]
            custo_medio = agg[["produto_id", "custo_medio"]].copy()
            # normaliza tipo de chave para casar com vendas
            custo_medio["produto_id"] = pd.to_numeric(custo_medio["produto_id"], errors="coerce").astype("Int64")

    # fallback: se não tiver compras suficientes, usa custo_unitario da tabela produtos
    if custo_medio.empty and {"produto_id", "custo_unitario"}.issubset(produtos.columns):
        custo_medio = (
            produtos[["produto_id", "custo_unitario"]]
            .dropna()
            .rename(columns={"custo_unitario": "custo_medio"})
            .copy()
        )
        custo_medio["produto_id"] = pd.to_numeric(custo_medio["produto_id"], errors="coerce").astype("Int64")

    cmv_estimado = 0.0
    margem_bruta = 0.0
    margem_pct = 0.0

    if not vendas_f.empty and {"produto_id", "quantidade_vendida"}.issubset(vendas_f.columns):
        tmp = vendas_f.copy()
        # normaliza chave
        if "produto_id" in tmp.columns:
            tmp["produto_id"] = pd.to_numeric(tmp["produto_id"], errors="coerce").astype("Int64")

        # junta custo médio e fallback de custo_unitario do cadastro de produtos
        if not custo_medio.empty:
            tmp = tmp.merge(custo_medio, on="produto_id", how="left")
        if {"produto_id", "custo_unitario"}.issubset(produtos.columns):
            prod_cost = produtos[["produto_id", "custo_unitario"]].copy()
            prod_cost["produto_id"] = pd.to_numeric(prod_cost["produto_id"], errors="coerce").astype("Int64")
            tmp = tmp.merge(prod_cost, on="produto_id", how="left")
        if "custo_medio" not in tmp.columns:
            tmp["custo_medio"] = np.nan
        if "custo_unitario" not in tmp.columns:
            tmp["custo_unitario"] = np.nan

        custo_final = tmp["custo_medio"].fillna(tmp["custo_unitario"]).fillna(0)
        tmp["cmv"] = tmp["quantidade_vendida"].fillna(0) * custo_final
        cmv_estimado = float(tmp["cmv"].sum())
        margem_bruta = float(receita_periodo - cmv_estimado)
        margem_pct = _safe_div(margem_bruta, receita_periodo) * 100.0

    # ---- Séries para gráficos combinados ----
    # Estoque total ao longo do tempo (somatório)
    estoque_series = pd.DataFrame()
    if not estoque_f.empty and {"data_referencia", "quantidade_estoque"}.issubset(estoque_f.columns):
        estoque_series = (
            estoque_f.groupby("data_referencia", as_index=False)["quantidade_estoque"]
            .sum()
            .rename(columns={"data_referencia": "data", "quantidade_estoque": "estoque_total"})
        )
        estoque_series["mes"] = _month_start(pd.to_datetime(estoque_series["data"]))

        estoque_mensal = (
            estoque_series.groupby("mes", as_index=False)["estoque_total"]
            .mean()
            .rename(columns={"estoque_total": "estoque_medio"})
        )
    else:
        estoque_mensal = pd.DataFrame(columns=["mes", "estoque_medio"])

    # Compras vs Vendas (mensal)
    compras_mensal = pd.DataFrame()
    if not compras_f.empty and {"data_compra", "valor_total"}.issubset(compras_f.columns):
        compras_f["mes"] = _month_start(compras_f["data_compra"])
        compras_mensal = compras_f.groupby("mes", as_index=False)["valor_total"].sum().rename(columns={"valor_total": "compras"})

    # Logística (mensal) - entregas e taxa no prazo
    log_mensal = pd.DataFrame()
    if not log_f.empty and "data_pedido" in log_f.columns:
        log_f["mes"] = _month_start(log_f["data_pedido"])
        if {"prazo_estimado_dias", "prazo_real_dias"}.issubset(log_f.columns):
            vv = log_f.dropna(subset=["prazo_estimado_dias", "prazo_real_dias"]).copy()
            if not vv.empty:
                vv["no_prazo"] = vv["prazo_real_dias"] <= vv["prazo_estimado_dias"]
                log_mensal = (
                    vv.groupby("mes", as_index=False)
                    .agg(entregas=("no_prazo", "size"), no_prazo=("no_prazo", "sum"))
                )
                log_mensal["taxa_no_prazo"] = (log_mensal["no_prazo"] / log_mensal["entregas"]) * 100.0
        if log_mensal.empty:
            log_mensal = log_f.groupby("mes", as_index=False).size().rename(columns={"size": "entregas"})
            log_mensal["taxa_no_prazo"] = np.nan

    return {
        "estoque_critico_qtd": estoque_critico_qtd,
        "estoque_critico_df": estoque_critico_df,
        "receita_periodo": receita_periodo,
        "receita_mensal": receita_mensal,
        "receita_ultimo_mes": receita_ultimo_mes,
        "mes_ultimo": mes_ultimo,
        "fornecedores_freq": fornecedores_freq,
        "fornecedores_gasto": fornecedores_gasto,
        "fornecedor_top_freq": fornecedor_top_freq,
        "fornecedor_top_num_compras": fornecedor_top_num_compras,
        "fornecedor_top_gasto": fornecedor_top_gasto,
        "fornecedor_top_gasto_valor": fornecedor_top_gasto_valor,
        "taxa_prazo": taxa_prazo,
        "entregas_stats": entregas_stats,
        "custo_total_compras": custo_total_compras,
        "cmv_estimado": cmv_estimado,
        "margem_bruta": margem_bruta,
        "margem_pct": margem_pct,
        "estoque_mensal": estoque_mensal,
        "compras_mensal": compras_mensal,
        "log_mensal": log_mensal,
        "vendas_f": vendas_f,
        "compras_f": compras_f,
        "log_f": log_f,
        "estoque_f": estoque_f,
    }


# =========================
# UI
# =========================
st.title("📦 Dashboard Estratégico de Supply Chain")
st.caption(
    "Visão consolidada de **Estoque**, **Vendas**, **Compras** e **Logística**, com KPIs estratégicos e indicadores financeiros."
)

# CSS: melhora legibilidade e evita truncamento nos KPIs
st.markdown(
    """
<style>
  /* reduz o “aperto” do topo e padroniza respiros */
  .block-container { padding-top: 1.6rem; padding-bottom: 1.6rem; }

  .kpi-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 14px;
    padding: 14px 16px;
    min-height: 92px;
    margin-bottom: 12px; /* espaço entre linhas de KPIs */
  }
  .kpi-label {
    font-size: 0.90rem;
    opacity: 0.88;
    margin-bottom: 6px;
  }
  .kpi-value {
    font-size: 1.65rem;
    font-weight: 700;
    line-height: 1.1;
    white-space: normal;
    overflow-wrap: anywhere;
  }
  .kpi-sub {
    margin-top: 6px;
    font-size: 0.85rem;
    opacity: 0.85;
  }
  .periodo {
    margin-top: 6px;
    margin-bottom: 14px;
    opacity: 0.85;
    font-size: 0.95rem;
  }
</style>
""",
    unsafe_allow_html=True,
)


def _kpi_card(label: str, value: str, sub: str | None = None) -> None:
    sub_html = f'<div class="kpi-sub">{sub}</div>' if sub else ""
    st.markdown(
        f"""
<div class="kpi-card">
  <div class="kpi-label">{label}</div>
  <div class="kpi-value">{value}</div>
  {sub_html}
</div>
""",
        unsafe_allow_html=True,
    )

with st.spinner("Carregando dados..."):
    raw = load_data()
    data = prepare_data(raw)

clientes = data["clientes"]
estoque = data["estoque"]
produtos = data["produtos"]
compras = data["compras"]
logistica = data["logistica"]
vendas = data["vendas"]

# Datas globais para range padrão (baseado em vendas, com fallback)
min_date = None
max_date = None
if "data_venda" in vendas.columns and not vendas["data_venda"].dropna().empty:
    min_date = vendas["data_venda"].min()
    max_date = vendas["data_venda"].max()
else:
    # fallback geral
    all_dates = []
    for df, c in [(estoque, "data_referencia"), (compras, "data_compra"), (logistica, "data_pedido")]:
        if c in df.columns and not df[c].dropna().empty:
            all_dates.append(df[c].min())
            all_dates.append(df[c].max())
    if all_dates:
        min_date = min(all_dates)
        max_date = max(all_dates)
    else:
        min_date = pd.Timestamp("2024-01-01")
        max_date = pd.Timestamp("2024-12-31")

# Sidebar filtros
st.sidebar.header("🎛️ Filtros")

date_start, date_end = st.sidebar.date_input(
    "Período (início / fim)",
    value=(min_date.date(), max_date.date()),
    min_value=min_date.date(),
    max_value=max_date.date(),
)

date_start = pd.Timestamp(date_start)
# inclui o dia final inteiro (evita excluir vendas/compras se houver horário no datetime)
date_end = pd.Timestamp(date_end) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)

# Categoria
categorias = ["(Todas)"]
if "categoria" in produtos.columns:
    categorias += sorted([c for c in produtos["categoria"].dropna().unique().tolist()])
categoria_sel = st.sidebar.selectbox("Categoria (Produtos)", categorias, index=0)
categoria = None if categoria_sel == "(Todas)" else categoria_sel

# Produto
produto_sel = None
produto_map = {}
if {"produto_id", "produto_nome"}.issubset(produtos.columns):
    prod_filtered = produtos.copy()
    if categoria and "categoria" in prod_filtered.columns:
        prod_filtered = prod_filtered[prod_filtered["categoria"] == categoria]
    # cria rótulo
    prod_filtered = prod_filtered.dropna(subset=["produto_id", "produto_nome"])
    prod_filtered["label"] = prod_filtered["produto_id"].astype(int).astype(str) + " - " + prod_filtered["produto_nome"].astype(str)
    labels = ["(Todos)"] + sorted(prod_filtered["label"].unique().tolist())
    produto_label = st.sidebar.selectbox("Produto (opcional)", labels, index=0)
    if produto_label != "(Todos)":
        pid = int(produto_label.split(" - ")[0].strip())
        produto_sel = pid

# Fornecedor
fornecedor = None
if "fornecedor" in compras.columns:
    forn = ["(Todos)"] + sorted(compras["fornecedor"].dropna().unique().tolist())
    fornecedor_sel = st.sidebar.selectbox("Fornecedor (Compras)", forn, index=0)
    fornecedor = None if fornecedor_sel == "(Todos)" else fornecedor_sel

# Transportadora
transportadora = None
if "transportadora" in logistica.columns:
    trans = ["(Todas)"] + sorted(logistica["transportadora"].dropna().unique().tolist())
    transportadora_sel = st.sidebar.selectbox("Transportadora (Logística)", trans, index=0)
    transportadora = None if transportadora_sel == "(Todas)" else transportadora_sel

# Canal de venda
canal_venda = None
if "canal_venda" in vendas.columns:
    canais = ["(Todos)"] + sorted(vendas["canal_venda"].dropna().unique().tolist())
    canal_sel = st.sidebar.selectbox("Canal de venda", canais, index=0)
    canal_venda = None if canal_sel == "(Todos)" else canal_sel

# Loja
loja_id = None
if "loja_id" in vendas.columns:
    lojas = ["(Todas)"] + sorted([int(x) for x in vendas["loja_id"].dropna().unique().tolist()])
    loja_sel = st.sidebar.selectbox("Loja (ID)", lojas, index=0)
    loja_id = None if loja_sel == "(Todas)" else int(loja_sel)

st.sidebar.divider()
st.sidebar.caption("Dica: selecione um **produto** para ver gráficos mais direcionados.")


# Computa KPIs/Métricas
k = compute_kpis(
    clientes=clientes,
    estoque=estoque,
    produtos=produtos,
    compras=compras,
    logistica=logistica,
    vendas=vendas,
    date_start=date_start,
    date_end=date_end,
    categoria=categoria,
    produto_id=produto_sel,
    fornecedor=fornecedor,
    transportadora=transportadora,
    canal_venda=canal_venda,
    loja_id=loja_id,
)

# =========================
# KPIs topo
# =========================
st.markdown(
    f"<div class='periodo'>Período filtrado: <b>{date_start.date()}</b> a <b>{date_end.date()}</b></div>",
    unsafe_allow_html=True,
)

# KPIs em 2 linhas (3 colunas) para não truncar valores
receita_mes_label = "📅 Receita mensal (últ. mês)"
if k.get("mes_ultimo") is not None:
    try:
        receita_mes_label = f"📅 Receita mensal ({pd.to_datetime(k['mes_ultimo']).strftime('%m/%Y')})"
    except Exception:
        pass

ft = k.get("fornecedor_top_freq")
ft_sub = f"{k.get('fornecedor_top_num_compras', 0)} compras" if ft else None

r1c1, r1c2, r1c3 = st.columns(3)
with r1c1:
    _kpi_card("🚨 Estoque crítico (produtos)", f"{k['estoque_critico_qtd']}")
with r1c2:
    _kpi_card(receita_mes_label, _brl_compact(k.get("receita_ultimo_mes", 0.0)), f"Total: {_brl(k.get('receita_ultimo_mes', 0.0))}")
with r1c3:
    _kpi_card("💰 Receita (período)", _brl_compact(k["receita_periodo"]), f"Total: {_brl(k['receita_periodo'])}")

# respiro extra entre as duas linhas (além do margin-bottom do card)
st.markdown("<div style='height: 6px'></div>", unsafe_allow_html=True)

r2c1, r2c2, r2c3 = st.columns(3)
with r2c1:
    _kpi_card("📦 Custo total de compras", _brl_compact(k["custo_total_compras"]), f"Total: {_brl(k['custo_total_compras'])}")
with r2c2:
    _kpi_card("🏭 Fornecedor mais usado", ft if ft else "—", ft_sub)
with r2c3:
    _kpi_card("⏱️ Entregas no prazo", f"{k['taxa_prazo']:.1f}%", None)

# =========================
# Abas principais
# =========================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["📌 Visão Geral", "📦 Estoque", "🧾 Vendas", "🧾 Compras", "🚚 Logística", "📊 Financeiro"]
)

# -------------------------
# Tab 1: Visão Geral
# -------------------------
with tab1:
    st.subheader("Visão consolidada da cadeia de suprimentos")

    # Gráfico combinado: Vendas (receita) vs Estoque (nível médio)
    left, right = st.columns([2, 1])

    with left:
        st.markdown("### 📈 Vendas vs Nível de Estoque (mensal)")

        # Junta receita_mensal e estoque_mensal
        receita_m = k["receita_mensal"].copy()
        estoque_m = k["estoque_mensal"].copy()

        if not receita_m.empty:
            receita_m["mes"] = pd.to_datetime(receita_m["mes"])
        if not estoque_m.empty:
            estoque_m["mes"] = pd.to_datetime(estoque_m["mes"])

        comb = pd.merge(receita_m, estoque_m, on="mes", how="outer").sort_values("mes")
        if comb.empty:
            st.info("Sem dados suficientes no período selecionado para montar o gráfico.")
        else:
            fig = px.line(
                comb,
                x="mes",
                y=["receita", "estoque_medio"],
                markers=True,
                labels={"value": "Valor", "mes": "Mês", "variable": "Métrica"},
            )
            st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown("### 🎯 Alertas rápidos")
        if k["estoque_critico_qtd"] > 0:
            st.warning(f"Existem **{k['estoque_critico_qtd']}** produtos em estoque crítico (no snapshot mais recente do período).")
        else:
            st.success("Nenhum produto em estoque crítico no período (snapshot mais recente).")

        if k["taxa_prazo"] > 0:
            st.info(f"Taxa de entregas no prazo: **{k['taxa_prazo']:.1f}%**")
        else:
            st.info("Sem dados suficientes para calcular entregas no prazo no período.")

        st.markdown("### 🧠 Decisão para gestores")
        st.write(
            "- Identificar itens críticos e priorizar reposição.\n"
            "- Ver relação entre vendas e nível de estoque para evitar ruptura/excesso.\n"
            "- Avaliar fornecedores e logística para reduzir custos e atrasos.\n"
            "- Acompanhar margem estimada para decisões financeiras."
        )

    st.divider()

    # Compras vs Vendas (mensal)
    st.markdown("### 🔄 Compras vs Vendas (mensal)")
    compras_m = k["compras_mensal"].copy()
    receita_m = k["receita_mensal"].copy()
    comb2 = pd.merge(receita_m, compras_m, on="mes", how="outer").sort_values("mes")

    if comb2.empty:
        st.info("Sem dados suficientes no período para compras vs vendas.")
    else:
        fig2 = px.bar(
            comb2,
            x="mes",
            y=["receita", "compras"],
            barmode="group",
            labels={"value": "Valor", "mes": "Mês", "variable": "Métrica"},
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Fornecedores mais usados
    st.markdown("### 🏭 Fornecedores (frequência e gasto)")
    colA, colB = st.columns(2)

    with colA:
        ff = k["fornecedores_freq"].head(10).copy()
        if ff.empty:
            st.info("Sem dados de fornecedores (compras) no período.")
        else:
            figf = px.bar(ff, x="num_compras", y="fornecedor", orientation="h", title="Top 10 por número de compras")
            st.plotly_chart(figf, use_container_width=True)

    with colB:
        fg = k["fornecedores_gasto"].head(10).copy()
        if fg.empty:
            st.info("Sem dados de gasto por fornecedor no período.")
        else:
            figg = px.bar(fg, x="gasto_total", y="fornecedor", orientation="h", title="Top 10 por gasto total")
            st.plotly_chart(figg, use_container_width=True)

# -------------------------
# Tab 2: Estoque
# -------------------------
with tab2:
    st.subheader("📦 Estoque")

    st.markdown("### 🚨 Produtos abaixo do estoque mínimo (snapshot mais recente no período)")
    crit = k["estoque_critico_df"].copy()

    if crit.empty:
        st.success("Nenhum produto crítico no snapshot mais recente do período.")
    else:
        # Enriquecer com nome/categoria
        if {"produto_id", "produto_nome"}.issubset(produtos.columns):
            crit = crit.merge(produtos[["produto_id", "produto_nome", "categoria", "marca"]], on="produto_id", how="left")

        # Ordena por maior “buraco” (min - atual)
        crit["gap"] = (crit["estoque_minimo"] - crit["quantidade_estoque"]).fillna(0)
        crit = crit.sort_values(["gap", "quantidade_estoque"], ascending=[False, True])

        st.dataframe(
            crit[
                [c for c in ["produto_id", "produto_nome", "categoria", "marca", "localizacao", "quantidade_estoque", "estoque_minimo", "gap"]
                 if c in crit.columns]
            ],
            use_container_width=True,
        )

    st.divider()
    st.markdown("### 📉 Evolução do estoque total (diário)")
    est = k["estoque_f"].copy()
    if est.empty or not {"data_referencia", "quantidade_estoque"}.issubset(est.columns):
        st.info("Sem dados de estoque no período selecionado.")
    else:
        est_series = est.groupby("data_referencia", as_index=False)["quantidade_estoque"].sum()
        fig = px.line(est_series, x="data_referencia", y="quantidade_estoque", markers=False, labels={"quantidade_estoque": "Estoque total"})
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.markdown("### 💸 Valor estimado do estoque (quantidade × custo_unitario)")
    if {"produto_id"}.issubset(est.columns) and {"produto_id", "custo_unitario"}.issubset(produtos.columns):
        est2 = est.merge(produtos[["produto_id", "custo_unitario", "categoria", "marca"]], on="produto_id", how="left")
        est2["valor_estoque_estimado"] = est2["quantidade_estoque"].fillna(0) * est2["custo_unitario"].fillna(0)
        # por categoria
        if "categoria" in est2.columns:
            cat_val = est2.groupby("categoria", as_index=False)["valor_estoque_estimado"].sum().sort_values("valor_estoque_estimado", ascending=False).head(15)
            figv = px.bar(cat_val, x="valor_estoque_estimado", y="categoria", orientation="h", title="Top categorias por valor estimado em estoque")
            st.plotly_chart(figv, use_container_width=True)
        else:
            st.info("Coluna 'categoria' não encontrada em produtos.")
    else:
        st.info("Não foi possível calcular valor do estoque (faltam colunas de custo/produto).")

# -------------------------
# Tab 3: Vendas
# -------------------------
with tab3:
    st.subheader("🧾 Vendas")

    vdf = k["vendas_f"].copy()

    if vdf.empty:
        st.info("Sem vendas no período selecionado.")
    else:
        st.markdown("### 📆 Receita mensal")
        rm = k["receita_mensal"].copy()
        if not rm.empty:
            fig = px.bar(rm, x="mes", y="receita", labels={"mes": "Mês", "receita": "Receita"})
            st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.markdown("### 🏷️ Top produtos por receita (período)")
        topn = 10
        if {"produto_id", "valor_total"}.issubset(vdf.columns) and {"produto_id", "produto_nome"}.issubset(produtos.columns):
            vp = vdf.groupby("produto_id", as_index=False)["valor_total"].sum().sort_values("valor_total", ascending=False).head(topn)
            vp = vp.merge(produtos[["produto_id", "produto_nome", "categoria", "marca"]], on="produto_id", how="left")
            figp = px.bar(vp, x="valor_total", y="produto_nome", orientation="h", title=f"Top {topn} produtos por receita")
            st.plotly_chart(figp, use_container_width=True)
        else:
            st.info("Sem colunas suficientes para Top produtos por receita.")

        st.divider()
        st.markdown("### 📄 Amostra das vendas filtradas")
        st.dataframe(vdf.head(200), use_container_width=True)

# -------------------------
# Tab 4: Compras
# -------------------------
with tab4:
    st.subheader("🧾 Compras")

    cdf = k["compras_f"].copy()

    if cdf.empty:
        st.info("Sem compras no período selecionado.")
    else:
        st.markdown("### 💵 Compras mensais (valor total)")
        cm = k["compras_mensal"].copy()
        if not cm.empty:
            fig = px.bar(cm, x="mes", y="compras", labels={"mes": "Mês", "compras": "Valor de compras"})
            st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.markdown("### 🏭 Fornecedores mais usados (frequência)")
        ff = k["fornecedores_freq"].head(15).copy()
        if not ff.empty:
            figf = px.bar(ff, x="num_compras", y="fornecedor", orientation="h")
            st.plotly_chart(figf, use_container_width=True)
        else:
            st.info("Sem dados de fornecedores no período.")

        st.divider()
        st.markdown("### 💰 Fornecedores por gasto total")
        fg = k["fornecedores_gasto"].head(15).copy()
        if not fg.empty:
            figg = px.bar(fg, x="gasto_total", y="fornecedor", orientation="h")
            st.plotly_chart(figg, use_container_width=True)
        else:
            st.info("Sem dados de gasto por fornecedor no período.")

        st.divider()
        st.markdown("### 📄 Amostra das compras filtradas")
        st.dataframe(cdf.head(200), use_container_width=True)

# -------------------------
# Tab 5: Logística
# -------------------------
with tab5:
    st.subheader("🚚 Logística")

    ldf = k["log_f"].copy()

    if ldf.empty:
        st.info("Sem registros logísticos no período selecionado.")
    else:
        st.markdown("### ⏱️ Taxa de entrega no prazo")
        stats = k.get("entregas_stats", {})
        if stats:
            colx, coly, colz = st.columns(3)
            colx.metric("Total de entregas", f"{stats.get('total', 0)}")
            coly.metric("No prazo", f"{stats.get('no_prazo', 0)}")
            colz.metric("Atrasadas", f"{stats.get('atraso', 0)}")
        st.progress(min(max(k["taxa_prazo"] / 100.0, 0.0), 1.0))

        st.divider()
        st.markdown("### 📅 Entregas por mês e taxa no prazo")
        lm = k["log_mensal"].copy()
        if not lm.empty:
            fig = px.bar(lm, x="mes", y="entregas", labels={"entregas": "Qtde entregas", "mes": "Mês"})
            st.plotly_chart(fig, use_container_width=True)

            if "taxa_no_prazo" in lm.columns and lm["taxa_no_prazo"].notna().any():
                fig2 = px.line(lm, x="mes", y="taxa_no_prazo", markers=True, labels={"taxa_no_prazo": "% no prazo", "mes": "Mês"})
                st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Sem dados suficientes para agregação mensal de logística.")

        st.divider()
        st.markdown("### 🧾 Custo de transporte (distribuição)")
        if "custo_transporte" in ldf.columns and ldf["custo_transporte"].notna().any():
            figc = px.histogram(ldf, x="custo_transporte", nbins=40, labels={"custo_transporte": "Custo de transporte"})
            st.plotly_chart(figc, use_container_width=True)
        else:
            st.info("Sem coluna 'custo_transporte' válida no período.")

        st.divider()
        st.markdown("### 📄 Amostra dos registros filtrados")
        st.dataframe(ldf.head(200), use_container_width=True)

# -------------------------
# Tab 6: Financeiro
# -------------------------
with tab6:
    st.subheader("📊 Indicadores Financeiros (estimados)")

    col1, col2, col3 = st.columns(3)
    col1.metric("Receita (período)", _brl(k["receita_periodo"]))
    col2.metric("CMV estimado (período)", _brl(k["cmv_estimado"]))
    col3.metric("Margem bruta estimada", f"{_brl(k['margem_bruta'])}  |  {k['margem_pct']:.1f}%")

    st.caption(
        "⚠️ *CMV estimado*: calculado por **custo médio de compras por produto** (média ponderada) e aplicado sobre a quantidade vendida no período. "
        "Se faltar histórico de compras, usa o **custo_unitario** da tabela de produtos como fallback."
    )

    st.divider()
    st.markdown("### 📈 Evolução mensal: Receita vs Compras")
    comb2 = pd.merge(k["receita_mensal"], k["compras_mensal"], on="mes", how="outer").sort_values("mes")
    if comb2.empty:
        st.info("Sem dados suficientes para série mensal financeira.")
    else:
        fig = px.line(comb2, x="mes", y=["receita", "compras"], markers=True, labels={"value": "Valor", "mes": "Mês", "variable": "Métrica"})
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.markdown("### 🧮 Resumo (no período filtrado)")
    resumo = pd.DataFrame(
        [
            {"Indicador": "Receita", "Valor": k["receita_periodo"]},
            {"Indicador": "Custo total de compras", "Valor": k["custo_total_compras"]},
            {"Indicador": "CMV estimado", "Valor": k["cmv_estimado"]},
            {"Indicador": "Margem bruta estimada", "Valor": k["margem_bruta"]},
        ]
    )
    resumo["Valor (R$)"] = resumo["Valor"].apply(_brl)
    st.dataframe(resumo[["Indicador", "Valor (R$)"]], use_container_width=True)

# Rodapé
st.divider()
with st.expander("📌 Verificação rápida: arquivos carregados"):
    st.write("Os arquivos foram carregados com estes nomes (exatos):")
    st.code("\n".join([f"- {v}" for v in FILES.values()]))
    st.write("Linhas por tabela:")
    st.json({k: int(v.shape[0]) for k, v in data.items()})