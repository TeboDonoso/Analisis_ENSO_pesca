import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import numpy as np

# 1. CONFIGURACIÓN DE LA PÁGINA
st.set_page_config(layout="wide", page_title="Análisis ENOS y Desdembarques, Chile 2020-24")
st.title("Panel Interactivo: Desembarques vs. Eventos ENOS")

# 2. CARGA DE DATOS
@st.cache_data
def cargar_datos():
    df_pesca = pd.read_csv('BD_desembarque.csv', sep=';', low_memory=False, encoding='latin-1')
    df_pesca.rename(columns={'año': 'Año', 'mes': 'Mes'}, inplace=True)
    df_enos = pd.read_csv('enos_chile.csv')
    return df_pesca, df_enos

df_pesca, df_enos = cargar_datos()

# 3. FILTROS INTERACTIVOS (Barra Lateral)
st.sidebar.header("⚙️ Parámetros de Análisis")

min_year = int(df_pesca['Año'].min())
max_year = int(df_pesca['Año'].max())
año_rango = st.sidebar.slider("Rango de Años:", min_value=min_year, max_value=max_year, value=(min_year, max_year))

lista_meses = ['Todos los Meses'] + sorted(df_pesca['Mes'].dropna().astype(int).unique().tolist())
mes_sel = st.sidebar.selectbox("Mes:", lista_meses)

lista_regiones = ['Todas las Regiones'] + sorted(df_pesca['region'].dropna().astype(str).unique().tolist())
region_sel = st.sidebar.selectbox("Región de Desembarque:", lista_regiones)

lista_subsectores = ['Todos los Subsectores'] + sorted(df_pesca['tipo_agente'].dropna().astype(str).unique().tolist())
subsector_sel = st.sidebar.selectbox("Subsector (Artesanal/Industrial):", lista_subsectores)

lista_especies = ['Todas las Especies'] + sorted(df_pesca['especie'].dropna().astype(str).unique().tolist())
especie_sel = st.sidebar.selectbox("Recurso / Especie:", lista_especies)

# 4. APLICAR FILTROS
df_filtrado = df_pesca.copy()

if año_rango:
    df_filtrado = df_filtrado[(df_filtrado['Año'] >= año_rango[0]) & (df_filtrado['Año'] <= año_rango[1])]
if mes_sel != 'Todos los Meses':
    df_filtrado = df_filtrado[df_filtrado['Mes'] == mes_sel]
if region_sel != 'Todas las Regiones':
    df_filtrado = df_filtrado[df_filtrado['region'] == region_sel]
if subsector_sel != 'Todos los Subsectores':
    df_filtrado = df_filtrado[df_filtrado['tipo_agente'] == subsector_sel]
if especie_sel != 'Todas las Especies':
    df_filtrado = df_filtrado[df_filtrado['especie'] == especie_sel]

# 5. PROCESAMIENTO
df_mensual = df_filtrado.groupby(['Año', 'Mes'])['toneladas'].sum().reset_index()
df_merged = pd.merge(df_mensual, df_enos, on=['Año', 'Mes'], how='inner')
df_merged['Fecha'] = pd.to_datetime(df_merged['Año'].astype(str) + '-' + df_merged['Mes'].astype(str).str.zfill(2))
df_merged.sort_values('Fecha', inplace=True)

# BOTÓN DE DESCARGA EN LA BARRA LATERAL
st.sidebar.markdown("---")
st.sidebar.subheader("💾 Exportar Datos")

if not df_merged.empty:
    csv = df_merged.drop(columns=['Fecha']).to_csv(index=False, encoding='utf-8-sig')
    st.sidebar.download_button(
        label="📥 Descargar Datos Filtrados (CSV)",
        data=csv,
        file_name='desembarques_vs_enos_filtrado.csv',
        mime='text/csv',
    )
else:
    st.sidebar.warning("No hay datos para descargar con estos filtros.")

if df_merged.empty:
    st.warning("⚠️ No hay registros de desembarque para esta combinación de filtros.")
else:
    # 6. ANÁLISIS ESTADÍSTICO (R2)
    st.markdown("### 📊 Indicadores Estadísticos")
    col1, col2, col3 = st.columns(3)
    
    total_tons = df_merged['toneladas'].sum()
    col1.metric("Total Toneladas (Periodo Seleccionado)", f"{total_tons:,.0f}")
    
    if len(df_merged) > 2 and df_merged['Intensidad'].std() > 0 and df_merged['toneladas'].std() > 0:
        r = np.corrcoef(df_merged['Intensidad'], df_merged['toneladas'])[0, 1]
        r2 = r**2
        tendencia = "Positiva 📈" if r > 0 else "Negativa 📉"
        
        col2.metric("Coeficiente de Determinación (R²)", f"{r2:.4f}")
        col3.metric("Relación con Intensidad El Niño", tendencia)
        
        st.info(f"💡 **Interpretación del R²:** El índice ENOS explica matemáticamente el **{r2*100:.2f}%** de la variación en las capturas de esta selección. Una relación *{tendencia.split()[0].lower()}* indica que frente a eventos más cálidos (El Niño), las capturas tienden a {'aumentar' if r>0 else 'disminuir'}.")
    else:
        col2.metric("Coeficiente R²", "Insuficientes datos")
        col3.metric("Relación", "N/A")

    # 7. CREACIÓN DE LOS GRÁFICOS
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"colspan": 2}, None], [{}, {}]],
        subplot_titles=(
            f"Evolución Histórica de Desembarques",
            "Distribución de Capturas por Condición",
            "Regresión Lineal: Toneladas vs Intensidad ENOS"
        ),
        vertical_spacing=0.15
    )

    color_map = {'El Niño': '#ef553b', 'La Niña': '#636efa', 'Neutral': '#00cc96'}
    estados = ['El Niño', 'La Niña', 'Neutral']

    for estado in estados:
        df_temp = df_merged[df_merged['Estado_ENOS'] == estado]
        if not df_temp.empty:
            fig.add_trace(
                go.Bar(x=df_temp['Fecha'], y=df_temp['toneladas'], name=estado, marker_color=color_map[estado]),
                row=1, col=1
            )
            fig.add_trace(
                go.Box(y=df_temp['toneladas'], name=estado, marker_color=color_map[estado], showlegend=False),
                row=2, col=1
            )

    fig.add_trace(
        go.Scatter(
            x=df_merged['Intensidad'], y=df_merged['toneladas'],
            mode='markers', marker=dict(color=df_merged['Intensidad'], colorscale='RdBu_r', size=10, opacity=0.7),
            name="Meses individuales", showlegend=False
        ), row=2, col=2
    )
    
    if len(df_merged) > 2 and df_merged['Intensidad'].std() > 0:
        m, b = np.polyfit(df_merged['Intensidad'], df_merged['toneladas'], 1)
        linea_x = np.array([-3, 3])
        linea_y = m * linea_x + b
        fig.add_trace(
            go.Scatter(x=linea_x, y=linea_y, mode='lines', line=dict(color='black', width=3, dash='dash'), name="Tendencia Lineal"),
            row=2, col=2
        )

    fig.update_layout(height=800, barmode='stack', template='plotly_white', margin=dict(t=40, b=40))
    fig.update_yaxes(title_text="Toneladas", row=1, col=1)
    fig.update_yaxes(title_text="Toneladas", row=2, col=1)
    fig.update_xaxes(title_text="Intensidad (-3 Niña a +3 Niño)", tickmode='linear', dtick=1, range=[-3.5, 3.5], row=2, col=2)
    fig.update_yaxes(title_text="Toneladas", row=2, col=2)

    st.plotly_chart(fig, use_container_width=True)

# --- NUEVO: SECCIÓN DE REFERENCIAS Y METODOLOGÍA ---
st.markdown("---")
st.markdown("### 📚 Fuentes de Datos y Replicabilidad")
st.markdown("""
* **Datos Pesqueros:** Histórico de Desembarques Totales (2000-2024), extraídos del *Anuario Estadístico* del **Servicio Nacional de Pesca y Acuicultura (SERNAPESCA)** de Chile (http://anuario.sernapesca.dataobservatory.net/).
* **Datos Oceanográficos:** Clasificación del fenómeno ENOS (El Niño-Oscilación del Sur) basada en el **Índice del Niño Oceánico (ONI)**, calculado por el *Climate Prediction Center* de la **NOAA** (Región Niño 3.4).
* **Metodología de Intensidad:** La intensidad fue categorizada de -3 a +3 según las anomalías térmicas mensuales para modelar correlaciones lineales.
* **Desarrollo:** Panel analítico construido utilizando Python, Pandas, Plotly y Streamlit.
""")
