import hashlib
import json
import unicodedata
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import folium
from streamlit_folium import st_folium


DATA_PATH_DEFAULT = "FCD_logistica.csv"
COORDS_CAPITAIS_PATH = "coordenada_capitais.json"


@dataclass(frozen=True)
class BrazilBBox:
    # Bounding box aproximado do Brasil (para coordenadas sintéticas determinísticas)
    lat_min: float = -34.0
    lat_max: float = 6.0
    lon_min: float = -74.0
    lon_max: float = -34.0


def normalize_city(text: str) -> str:
    """
    Normaliza nomes de cidade para comparação (casefold + remove acentos + normaliza espaços).
    Ex.: "São  Paulo" -> "sao paulo"
    """
    s = str(text or "").strip()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = " ".join(s.split())
    return s.casefold()


@st.cache_data(show_spinner=False)
def load_capitais_coords(path: str = COORDS_CAPITAIS_PATH) -> dict[str, tuple[float, float]]:
    """
    Carrega coordenadas reais (lat/lon) de cidades conhecidas.
    Formato esperado:
      { "capitais": [ { "cidade": "...", "latitude": ..., "longitude": ... }, ... ] }
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return {}
    except Exception:
        # se o JSON estiver inválido, não quebra o app (apenas cai no fallback sintético)
        return {}

    out: dict[str, tuple[float, float]] = {}
    for item in data.get("capitais", []):
        cidade = item.get("cidade")
        lat = item.get("latitude")
        lon = item.get("longitude")
        if cidade is None or lat is None or lon is None:
            continue
        out[normalize_city(cidade)] = (float(lat), float(lon))
    return out


def _stable_unit_interval(text: str) -> float:
    """
    Mapeia texto -> número em [0,1) de forma determinística.
    """
    h = hashlib.md5(normalize_city(text).encode("utf-8")).hexdigest()
    # usa 10 hex chars (40 bits) para evitar floats com pouca variância
    x = int(h[:10], 16)
    return (x % 10_000_000) / 10_000_000.0


def city_to_coords(city: str, bbox: BrazilBBox = BrazilBBox()) -> tuple[float, float]:
    """
    Retorna (lat, lon) para uma cidade:
    - Se existir em `coordenada_capitais.json`, usa coordenadas reais.
    - Caso contrário, gera coordenadas determinísticas (sintéticas).
    """
    coords = load_capitais_coords().get(normalize_city(city))
    if coords is not None:
        return coords

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
        .agg(
            tempo_medio=("prazo_real_dias", "mean"),
            entregas=("pedido_id", "count"),
            atraso_medio=("atraso_dias", "mean"),
            pct_no_prazo=("no_prazo", "mean"),
        )
        .sort_values("tempo_medio", ascending=True)
    )
    agg["pct_no_prazo"] = agg["pct_no_prazo"] * 100.0

    # Dot plot (destaca diferenças mesmo quando os valores são próximos)
    xmin = float(agg["tempo_medio"].min())
    xmax = float(agg["tempo_medio"].max())
    spread = max(0.0001, xmax - xmin)
    pad = max(0.25, 0.08 * spread)
    x_range = [max(0.0, xmin - pad), xmax + pad]

    fig = px.scatter(
        agg,
        x="tempo_medio",
        y="transportadora",
        color="atraso_medio",
        size="entregas",
        size_max=28,
        color_continuous_scale=["#16a34a", "#f59e0b", "#dc2626"],
        hover_data={
            "entregas": True,
            "pct_no_prazo": ":.1f",
            "atraso_medio": ":.2f",
            "tempo_medio": ":.2f",
        },
        labels={
            "transportadora": "Transportadora",
            "tempo_medio": "Tempo médio (dias)",
            "atraso_medio": "Atraso médio (dias)",
            "pct_no_prazo": "% no prazo",
            "entregas": "Entregas",
        },
        title="Tempo médio por transportadora (tamanho = entregas, cor = atraso médio)",
    )
    fig.update_traces(marker=dict(line=dict(width=0.6, color="rgba(0,0,0,0.35)")))
    fig.update_xaxes(range=x_range, showgrid=True)
    fig.update_yaxes(categoryorder="array", categoryarray=agg["transportadora"].tolist())
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
        "O dataset não possui lat/lon. O mapa usa coordenadas **reais** quando a cidade existir em `coordenada_capitais.json` "
        "e usa coordenadas **determinísticas** (sintéticas/reprodutíveis) no restante."
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


def gargalos_section(df: pd.DataFrame) -> None:
    st.subheader("Gargalos (top atrasos)")
    st.caption(
        "Rotas/combinações com pior atraso médio. Dica: aumente o mínimo de entregas para priorizar gargalos recorrentes."
    )

    if df.empty:
        st.info("Sem dados para exibir com os filtros atuais.")
        return

    c1, c2, c3 = st.columns([1, 1, 1.3])
    top_n = c1.slider("Top N", min_value=5, max_value=50, value=15, step=5)
    min_entregas = c2.number_input("Mínimo de entregas", min_value=1, max_value=9999, value=2, step=1)
    sort_by = c3.selectbox(
        "Ordenar por",
        options=[
            ("atraso_medio", "Atraso médio (desc)"),
            ("pct_atraso", "% atrasadas (desc)"),
            ("custo_total", "Custo total (desc)"),
            ("entregas", "Entregas (desc)"),
        ],
        format_func=lambda x: x[1],
        index=0,
    )[0]

    agg = (
        df.groupby(["transportadora", "cidade_origem", "cidade_destino"], as_index=False)
        .agg(
            entregas=("pedido_id", "count"),
            atraso_medio=("atraso_dias", "mean"),
            pct_atraso=("atraso_dias", lambda s: float((s > 0).mean()) if len(s) else 0.0),
            custo_total=("custo_transporte", "sum"),
            custo_medio=("custo_transporte", "mean"),
        )
        .query("entregas >= @min_entregas")
    )

    if agg.empty:
        st.warning("Nenhum gargalo atende aos filtros (ex.: mínimo de entregas).")
        return

    ascending = False
    agg = agg.sort_values([sort_by, "entregas"], ascending=[ascending, False]).head(int(top_n))
    agg["pct_atraso"] = agg["pct_atraso"] * 100.0

    # tabela “bonita” para exibição
    show = agg.rename(
        columns={
            "transportadora": "Transportadora",
            "cidade_origem": "Origem",
            "cidade_destino": "Destino",
            "entregas": "Entregas",
            "atraso_medio": "Atraso médio (dias)",
            "pct_atraso": "% atrasadas",
            "custo_total": "Custo total (R$)",
            "custo_medio": "Custo médio (R$)",
        }
    )

    # Importante: evitar `background_gradient` porque exige matplotlib.
    # Em vez disso, aplicamos cores via CSS (sem dependências extras).
    def _lerp(a: float, b: float, t: float) -> float:
        return a + (b - a) * t

    def _clamp01(x: float) -> float:
        return float(max(0.0, min(1.0, x)))

    def _bg_from_scale(v: float, vmin: float, vmax: float, c0: tuple[int, int, int], c1: tuple[int, int, int]) -> str:
        if pd.isna(v):
            return ""
        if vmax <= vmin:
            t = 0.5
        else:
            t = _clamp01((float(v) - float(vmin)) / (float(vmax) - float(vmin)))
        r = int(_lerp(c0[0], c1[0], t))
        g = int(_lerp(c0[1], c1[1], t))
        b = int(_lerp(c0[2], c1[2], t))
        return f"background-color: rgba({r},{g},{b},0.35)"

    def _col_scale_css(col: pd.Series, c0: tuple[int, int, int], c1: tuple[int, int, int]) -> list[str]:
        vmin = float(pd.to_numeric(col, errors="coerce").min())
        vmax = float(pd.to_numeric(col, errors="coerce").max())
        return [_bg_from_scale(v, vmin, vmax, c0, c1) for v in col]

    styler = show.style.format(
        {
            "Entregas": "{:,.0f}",
            "Atraso médio (dias)": "{:.2f}",
            "% atrasadas": "{:.1f}%",
            "Custo total (R$)": "R$ {:,.2f}",
            "Custo médio (R$)": "R$ {:,.2f}",
        }
    )
    # vermelho (pior atraso), laranja (% atrasadas), azul (custo total)
    styler = styler.apply(_col_scale_css, subset=["Atraso médio (dias)"], c0=(255, 245, 245), c1=(220, 38, 38))
    styler = styler.apply(_col_scale_css, subset=["% atrasadas"], c0=(255, 251, 235), c1=(245, 158, 11))
    styler = styler.apply(_col_scale_css, subset=["Custo total (R$)"], c0=(239, 246, 255), c1=(37, 99, 235))

    st.dataframe(styler, use_container_width=True, hide_index=True)

    csv_bytes = agg.to_csv(index=False, sep=";").encode("utf-8")
    st.download_button(
        "Baixar gargalos (CSV)",
        data=csv_bytes,
        file_name="gargalos_top_atrasos.csv",
        mime="text/csv",
        use_container_width=False,
    )


def dados_section(df: pd.DataFrame) -> None:
    st.subheader("Dados filtrados")
    st.caption("Use esta aba para inspecionar os registros após aplicar filtros.")
    st.dataframe(df.reset_index(drop=True), use_container_width=True, height=520)


def main() -> None:
    st.set_page_config(page_title="Dashboard de Performance Logística", layout="wide")
    st.title("Dashboard de Performance Logística")
    st.caption("Projeto 5 — Fundamentos em Ciência de Dados (Streamlit + Pandas + Plotly + Folium)")

    data_path = DATA_PATH_DEFAULT

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
        gargalos_section(df_f)

    with tab2:
        map_section(df_f)

    with tab3:
        custos_section(df_f)

    with tab4:
        dados_section(df_f)


if __name__ == "__main__":
    main()

