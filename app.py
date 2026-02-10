import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import folium
from streamlit_folium import st_folium


DATA_PATH_DEFAULT = "FCD_logistica.csv"


@dataclass(frozen=True)
class BrazilBBox:
    # Bounding box aproximado do Brasil (para coordenadas sintéticas determinísticas)
    lat_min: float = -34.0
    lat_max: float = 6.0
    lon_min: float = -74.0
    lon_max: float = -34.0


def _stable_unit_interval(text: str) -> float:
    """
    Mapeia texto -> número em [0,1) de forma determinística.
    """
    h = hashlib.md5(text.strip().lower().encode("utf-8")).hexdigest()
    # usa 10 hex chars (40 bits) para evitar floats com pouca variância
    x = int(h[:10], 16)
    return (x % 10_000_000) / 10_000_000.0


def city_to_coords(city: str, bbox: BrazilBBox = BrazilBBox()) -> tuple[float, float]:
    """
    Gera (lat, lon) determinísticos para uma cidade.
    Útil quando o dataset não tem lat/lon reais.
    """
    u = _stable_unit_interval(city + "|lat")
    v = _stable_unit_interval(city + "|lon")
    lat = bbox.lat_min + u * (bbox.lat_max - bbox.lat_min)
    lon = bbox.lon_min + v * (bbox.lon_max - bbox.lon_min)
    return float(lat), float(lon)


@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";")

    # padronizações simples
    for col in ["transportadora", "cidade_origem", "cidade_destino", "status_entrega"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # datas (dd/mm/aaaa)
    df["data_pedido"] = pd.to_datetime(df["data_pedido"], dayfirst=True, errors="coerce")
    df["data_entrega"] = pd.to_datetime(df["data_entrega"], dayfirst=True, errors="coerce")

    # numéricos
    df["prazo_estimado_dias"] = pd.to_numeric(df["prazo_estimado_dias"], errors="coerce")
    df["prazo_real_dias"] = pd.to_numeric(df["prazo_real_dias"], errors="coerce")
    df["custo_transporte"] = pd.to_numeric(df["custo_transporte"], errors="coerce")

    # features
    df["atraso_dias"] = df["prazo_real_dias"] - df["prazo_estimado_dias"]
    df["no_prazo"] = df["atraso_dias"] <= 0
    df["tempo_calc_dias"] = (df["data_entrega"] - df["data_pedido"]).dt.days

    return df


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Filtros")

    date_base = st.sidebar.selectbox(
        "Data base para filtro",
        options=["data_pedido", "data_entrega"],
        index=1,
        help="Filtra as entregas considerando a coluna escolhida.",
    )

    min_date = df[date_base].min()
    max_date = df[date_base].max()

    if pd.isna(min_date) or pd.isna(max_date):
        st.sidebar.warning("Não foi possível detectar datas válidas no dataset.")
        date_range = None
    else:
        date_range = st.sidebar.date_input(
            "Intervalo de datas",
            value=(min_date.date(), max_date.date()),
            min_value=min_date.date(),
            max_value=max_date.date(),
        )

    transportadoras = sorted(df["transportadora"].dropna().unique().tolist())
    status_opts = sorted(df["status_entrega"].dropna().unique().tolist())
    origens = sorted(df["cidade_origem"].dropna().unique().tolist())
    destinos = sorted(df["cidade_destino"].dropna().unique().tolist())

    sel_transp = st.sidebar.multiselect("Transportadora", transportadoras, default=transportadoras)
    sel_status = st.sidebar.multiselect("Status da entrega", status_opts, default=status_opts)

    with st.sidebar.expander("Origem/Destino", expanded=False):
        sel_origem = st.multiselect("Cidade de origem", origens, default=[])
        sel_destino = st.multiselect("Cidade de destino", destinos, default=[])

    with st.sidebar.expander("Faixas numéricas", expanded=False):
        prazo_min, prazo_max = float(df["prazo_real_dias"].min()), float(df["prazo_real_dias"].max())
        custo_min, custo_max = float(df["custo_transporte"].min()), float(df["custo_transporte"].max())

        sel_prazo = st.slider(
            "Prazo real (dias)",
            min_value=int(np.floor(prazo_min)),
            max_value=int(np.ceil(prazo_max)),
            value=(int(np.floor(prazo_min)), int(np.ceil(prazo_max))),
        )
        sel_custo = st.slider(
            "Custo de transporte",
            min_value=float(np.floor(custo_min)),
            max_value=float(np.ceil(custo_max)),
            value=(float(np.floor(custo_min)), float(np.ceil(custo_max))),
        )

    out = df.copy()

    if date_range and len(date_range) == 2:
        start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        out = out[(out[date_base] >= start) & (out[date_base] <= end)]

    out = out[out["transportadora"].isin(sel_transp)]
    out = out[out["status_entrega"].isin(sel_status)]

    if sel_origem:
        out = out[out["cidade_origem"].isin(sel_origem)]
    if sel_destino:
        out = out[out["cidade_destino"].isin(sel_destino)]

    out = out[out["prazo_real_dias"].between(sel_prazo[0], sel_prazo[1], inclusive="both")]
    out = out[out["custo_transporte"].between(sel_custo[0], sel_custo[1], inclusive="both")]

    return out


def kpis_section(df: pd.DataFrame) -> None:
    total = int(len(df))
    pct_no_prazo = float(df["no_prazo"].mean() * 100) if total else 0.0
    atraso_medio = float(df["atraso_dias"].mean()) if total else 0.0
    custo_total = float(df["custo_transporte"].sum()) if total else 0.0

    c1, c2, c3, c4 = st.columns([1, 1, 1, 1.2])
    c1.metric("Total de entregas", f"{total:,}".replace(",", "."))
    c2.metric("Entregas no prazo", f"{pct_no_prazo:.1f}%")
    c3.metric("Atraso médio (dias)", f"{atraso_medio:.2f}")
    c4.metric("Custo total (R$)", f"{custo_total:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=pct_no_prazo,
            number={"suffix": "%"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#16a34a" if pct_no_prazo >= 85 else "#f59e0b" if pct_no_prazo >= 70 else "#dc2626"},
                "steps": [
                    {"range": [0, 70], "color": "#fee2e2"},
                    {"range": [70, 85], "color": "#fef3c7"},
                    {"range": [85, 100], "color": "#dcfce7"},
                ],
            },
            title={"text": "KPI — % de entregas no prazo"},
        )
    )
    fig.update_layout(height=240, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)


def tempo_por_transportadora(df: pd.DataFrame) -> None:
    st.subheader("Tempo médio de entrega por transportadora")
    if df.empty:
        st.info("Sem dados para exibir com os filtros atuais.")
        return

    agg = (
        df.groupby("transportadora", as_index=False)
        .agg(tempo_medio=("prazo_real_dias", "mean"), entregas=("pedido_id", "count"), atraso_medio=("atraso_dias", "mean"))
        .sort_values("tempo_medio", ascending=True)
    )

    fig = px.bar(
        agg,
        x="transportadora",
        y="tempo_medio",
        color="atraso_medio",
        color_continuous_scale=["#16a34a", "#f59e0b", "#dc2626"],
        hover_data={"entregas": True, "atraso_medio": ":.2f", "tempo_medio": ":.2f"},
        labels={"transportadora": "Transportadora", "tempo_medio": "Tempo médio (dias)", "atraso_medio": "Atraso médio (dias)"},
        title="Média de prazo_real_dias por transportadora (cor = atraso médio)",
    )
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=60, b=10))
    st.plotly_chart(fig, use_container_width=True)


def build_flow_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["cidade_origem", "cidade_destino", "volume", "custo_total", "atraso_medio"])

    flows = (
        df.groupby(["cidade_origem", "cidade_destino"], as_index=False)
        .agg(
            volume=("pedido_id", "count"),
            custo_total=("custo_transporte", "sum"),
            atraso_medio=("atraso_dias", "mean"),
        )
        .sort_values(["volume", "custo_total"], ascending=False)
    )
    return flows


def map_section(df: pd.DataFrame) -> None:
    st.subheader("Mapa interativo — fluxos origem → destino")
    st.caption(
        "Como o dataset não possui lat/lon, o mapa usa coordenadas **determinísticas** por cidade (posição sintética e reprodutível)."
    )

    flows = build_flow_table(df)
    if flows.empty:
        st.info("Sem dados para exibir no mapa com os filtros atuais.")
        return

    top_n = st.slider("Quantidade de rotas (Top N por volume)", min_value=10, max_value=200, value=60, step=10)
    flows = flows.head(top_n).copy()

    # normalização para espessura
    vmin, vmax = flows["volume"].min(), flows["volume"].max()
    if vmin == vmax:
        flows["w"] = 4.0
    else:
        flows["w"] = 2.0 + 8.0 * (flows["volume"] - vmin) / (vmax - vmin)

    # mapa base
    m = folium.Map(location=[-14.2, -51.9], zoom_start=4, tiles="cartodbpositron")

    # volume por cidade (para markers)
    city_vol = pd.concat(
        [
            flows[["cidade_origem", "volume"]].rename(columns={"cidade_origem": "cidade"}),
            flows[["cidade_destino", "volume"]].rename(columns={"cidade_destino": "cidade"}),
        ],
        ignore_index=True,
    )
    city_vol = city_vol.groupby("cidade", as_index=False)["volume"].sum()

    # markers
    for _, row in city_vol.iterrows():
        lat, lon = city_to_coords(row["cidade"])
        folium.CircleMarker(
            location=(lat, lon),
            radius=float(3 + 7 * (row["volume"] - city_vol["volume"].min()) / max(1, (city_vol["volume"].max() - city_vol["volume"].min()))),
            color="#2563eb",
            fill=True,
            fill_opacity=0.65,
            tooltip=f"{row['cidade']} — volume (rotas top): {int(row['volume'])}",
        ).add_to(m)

    # polylines
    for _, r in flows.iterrows():
        o_lat, o_lon = city_to_coords(r["cidade_origem"])
        d_lat, d_lon = city_to_coords(r["cidade_destino"])

        atraso = float(r["atraso_medio"])
        if atraso <= 0:
            color = "#16a34a"
        elif atraso <= 2:
            color = "#f59e0b"
        else:
            color = "#dc2626"

        tooltip = (
            f"{r['cidade_origem']} → {r['cidade_destino']}<br>"
            f"Volume: {int(r['volume'])}<br>"
            f"Custo total: R$ {float(r['custo_total']):,.2f}<br>"
            f"Atraso médio: {atraso:.2f} dias"
        )

        folium.PolyLine(
            locations=[(o_lat, o_lon), (d_lat, d_lon)],
            weight=float(r["w"]),
            opacity=0.75,
            color=color,
            tooltip=folium.Tooltip(tooltip, sticky=True),
        ).add_to(m)

    st_folium(m, use_container_width=True, height=560)

    st.markdown("**Rotas exibidas (Top N)**")
    st.dataframe(
        flows[["cidade_origem", "cidade_destino", "volume", "custo_total", "atraso_medio"]].reset_index(drop=True),
        use_container_width=True,
    )


def custos_section(df: pd.DataFrame) -> None:
    st.subheader("Custos logísticos por região")
    if df.empty:
        st.info("Sem dados para exibir com os filtros atuais.")
        return

    group_mode = st.selectbox(
        "Agrupar custos por",
        options=[
            ("cidade_origem", "Cidade de origem"),
            ("cidade_destino", "Cidade de destino"),
            ("transportadora", "Transportadora"),
        ],
        format_func=lambda x: x[1],
    )[0]

    top_n = st.slider("Top N (maiores custos)", min_value=5, max_value=40, value=15, step=1)

    agg = (
        df.groupby(group_mode, as_index=False)
        .agg(custo_total=("custo_transporte", "sum"), entregas=("pedido_id", "count"), atraso_medio=("atraso_dias", "mean"))
        .sort_values("custo_total", ascending=False)
        .head(top_n)
    )

    fig = px.bar(
        agg,
        x=group_mode,
        y="custo_total",
        color="atraso_medio",
        color_continuous_scale=["#16a34a", "#f59e0b", "#dc2626"],
        hover_data={"entregas": True, "atraso_medio": ":.2f", "custo_total": ":.2f"},
        labels={group_mode: "Grupo", "custo_total": "Custo total (R$)", "atraso_medio": "Atraso médio (dias)"},
        title="Custo total (cor = atraso médio)",
    )
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=60, b=10))
    st.plotly_chart(fig, use_container_width=True)

    if group_mode in ("cidade_origem", "cidade_destino"):
        # mapa de bolhas (scatter_geo) para visualizar “hotspots” de custo
        geo = agg.copy()
        geo["lat"] = geo[group_mode].apply(lambda c: city_to_coords(c)[0])
        geo["lon"] = geo[group_mode].apply(lambda c: city_to_coords(c)[1])

        fig2 = px.scatter_geo(
            geo,
            lat="lat",
            lon="lon",
            size="custo_total",
            color="atraso_medio",
            color_continuous_scale=["#16a34a", "#f59e0b", "#dc2626"],
            hover_name=group_mode,
            hover_data={"custo_total": ":.2f", "entregas": True, "atraso_medio": ":.2f"},
            title="Mapa de custos (posições sintéticas por cidade)",
            scope="south america",
        )
        fig2.update_layout(height=520, margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("**Tabela (Top N)**")
    st.dataframe(agg.reset_index(drop=True), use_container_width=True)


def dados_section(df: pd.DataFrame) -> None:
    st.subheader("Dados filtrados")
    st.caption("Use esta aba para inspecionar os registros após aplicar filtros.")
    st.dataframe(df.reset_index(drop=True), use_container_width=True, height=520)


def main() -> None:
    st.set_page_config(page_title="Dashboard de Performance Logística", layout="wide")
    st.title("Dashboard de Performance Logística")
    st.caption("Projeto 5 — Fundamentos em Ciência de Dados (Streamlit + Pandas + Plotly + Folium)")

    with st.sidebar:
        st.markdown("**Arquivo de dados**")
        data_path = st.text_input("Caminho do CSV", value=DATA_PATH_DEFAULT)
        st.markdown("---")

    try:
        df = load_data(data_path)
    except FileNotFoundError:
        st.error(f"Arquivo não encontrado: `{data_path}`")
        st.stop()
    except Exception as e:
        st.error(f"Erro ao carregar o dataset: {e}")
        st.stop()

    df_f = apply_filters(df)

    tab1, tab2, tab3, tab4 = st.tabs(["Visão Geral", "Mapa de Rotas", "Custos por Região", "Dados"])

    with tab1:
        kpis_section(df_f)
        tempo_por_transportadora(df_f)

        st.subheader("Gargalos (top atrasos)")
        if df_f.empty:
            st.info("Sem dados para exibir com os filtros atuais.")
        else:
            worst = (
                df_f.groupby(["transportadora", "cidade_origem", "cidade_destino"], as_index=False)
                .agg(
                    entregas=("pedido_id", "count"),
                    atraso_medio=("atraso_dias", "mean"),
                    custo_total=("custo_transporte", "sum"),
                )
                .sort_values(["atraso_medio", "entregas"], ascending=[False, False])
                .head(15)
            )
            st.dataframe(worst, use_container_width=True)

    with tab2:
        map_section(df_f)

    with tab3:
        custos_section(df_f)

    with tab4:
        dados_section(df_f)


if __name__ == "__main__":
    main()

