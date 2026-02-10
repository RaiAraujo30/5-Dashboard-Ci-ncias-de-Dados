# Dashboard de Performance Logística (Projeto 5 – FCD)

Implementação do **Projeto 5 – Dashboard de Performance Logística** da disciplina **Fundamentos em Ciência de Dados** (2025.2), usando **Python + Streamlit**.

O dashboard entrega:
- **KPI de Entregas no Prazo (%)**
- **Tempo médio de entrega por transportadora**
- **Mapa interativo com fluxos origem → destino**
- **Custos logísticos por região** (por origem/destino/transportadora)

> Observação: o dataset fornecido (`FCD_logistica.csv`) não possui latitude/longitude. Para permitir o **mapa interativo offline**, o app gera **coordenadas determinísticas** (a partir do nome da cidade) dentro de um bounding box do Brasil. As rotas/volumes são reais; apenas a posição no mapa é “sintética”, mas consistente.

## Como executar


### Pré-requisito (importante): versão do Python

Para evitar instalação lenta/compilação de pacotes científicos no Windows, **recomenda-se Python 3.12**.
Se você estiver usando **Python 3.14**, é comum o `pip` tentar compilar `pandas/numpy` do zero.

### 1) Criar ambiente virtual (recomendado)

No Windows (PowerShell):

```bash
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 2) Instalar dependências

```bash
pip install -r requirements.txt
```

### 3) Rodar o dashboard

```bash
streamlit run app.py
```

## Dataset

Colunas esperadas no CSV (separador `;`):
- `pedido_id`
- `data_pedido` (dd/mm/aaaa)
- `data_entrega` (dd/mm/aaaa)
- `transportadora`
- `cidade_origem`
- `cidade_destino`
- `prazo_estimado_dias`
- `prazo_real_dias`
- `custo_transporte`
- `status_entrega`

## Tecnologias

- Python
- Streamlit
- Pandas / NumPy
- Plotly
- Folium + streamlit-folium

