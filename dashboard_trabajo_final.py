import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# Configuraciones generales
st.set_page_config(page_title="Dashboard de Ventas", layout="wide", initial_sidebar_state = "expanded")

sns.set_style("whitegrid")
# Cargar datos
df = pd.read_csv("data.csv")
df['Date'] = pd.to_datetime(df['Date'])

# Título principal
st.title("Dashboard de Ventas y Comportamiento de Clientes")

st.markdown("Este dashboard interactivo permite analizar ventas, comportamiento de clientes y desempeño por sucursal en una cadena de tiendas de conveniencia.")

# Filtros globales
branches = st.sidebar.multiselect("Selecciona Sucursal(es):", options=df["Branch"].unique(), default=df["Branch"].unique())
date_range = st.sidebar.date_input("Rango de Fechas:", [df["Date"].min(), df["Date"].max()])

# Convertir fechas del filtro
start_date = pd.to_datetime(date_range[0])
end_date = pd.to_datetime(date_range[1])

# Aplicar filtros
df_filtered = df[(df["Branch"].isin(branches)) & (df["Date"].between(start_date, end_date))]

#Separación en dos columnas por fila
col11, col12 = st.columns(2)

with col11:
    # 1. Evolución de Ventas Totales
    st.subheader("1. Evolución de las Ventas Totales")
    ventas_diarias = df_filtered.groupby("Date")["Total"].sum().reset_index()
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    sns.lineplot(data=ventas_diarias, x="Date", y="Total", ax=ax1)
    ax1.set_title("Ventas Totales por Fecha")
    ax1.set_xlabel("Fecha")
    ax1.set_ylabel("Total")
    st.pyplot(fig1)
    st.markdown("> **Conclusión:** La evolución temporal de las ventas revela patrones estacionales y picos de actividad que pueden vincularse a días específicos o promociones. Identificar los días con mayor facturación permite planificar mejor la logística, inventario y campañas de marketing. Además, las caídas en ventas podrían indicar oportunidades para optimizar presencia o engagement en esos días.")

with col12:
    # 2. Ingresos por Línea de Producto
    st.subheader("2. Ingresos por Línea de Producto")
    fig2, ax2 = plt.subplots(figsize=(6, 3))
    sns.barplot(data=df_filtered, x="Product line", y="Total", estimator=sum, ci=None, ax=ax2)
    ax2.set_title("Ingresos por Línea de Producto")
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
    st.pyplot(fig2)
    st.markdown("> **Conclusión:** No todas las líneas de producto contribuyen por igual a los ingresos. Este análisis permite identificar los productos estrella (de alta facturación) y aquellos con bajo rendimiento. Esta información puede guiar decisiones como rediseñar promociones, reconsiderar el espacio en estanterías, o incluso retirar líneas poco rentables.")

col21, col22 = st.columns(2)

with col21:
    # 3. Distribución de la Calificación de Clientes
    st.subheader("3. Distribución de la Calificación de Clientes")
    fig3, ax3 = plt.subplots(figsize=(6, 4))
    sns.histplot(df_filtered["Rating"], bins=10, kde=True, ax=ax3)
    ax3.set_title("Distribución de Calificaciones")
    st.pyplot(fig3)
    st.markdown("> **Conclusión:** La distribución de las calificaciones muestra una tendencia hacia evaluaciones positivas, lo que indica una experiencia de cliente generalmente satisfactoria. Sin embargo, el análisis también puede destacar la necesidad de investigar casos con puntuaciones bajas para identificar fallas en servicio o calidad del producto.")

with col22:    
    # 4. Gasto por Tipo de Cliente
    st.subheader("4. Comparación del Gasto por Tipo de Cliente")
    fig4, ax4 = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=df_filtered, x="Customer type", y="Total", ax=ax4)
    ax4.set_title("Distribución del Gasto por Tipo de Cliente")
    ax4.set_xlabel("Tipo de Cliente")
    ax4.set_ylabel("Total Gastado")
    st.pyplot(fig4)
    st.markdown("> **Conclusión:** Los clientes Member muestran un patrón de gasto ligeramente superior al de los Normal, lo que sugiere que los programas de fidelidad podrían estar funcionando. Esto refuerza la idea de invertir en retención y beneficios exclusivos, ya que estos clientes aportan más valor a largo plazo.")

col31, col32 = st.columns(2)

with col31:
    # 5. Relación entre Costo y Ganancia Bruta
    st.subheader("5. Relación entre Costo y Ganancia Bruta")
    fig5, ax5 = plt.subplots(figsize=(6, 4))
    sns.scatterplot(data=df_filtered, x="cogs", y="gross income", ax=ax5)
    ax5.set_title("Costo vs Ingreso Bruto")
    ax5.set_xlabel("Costo de Bienes Vendidos (COGS)")
    ax5.set_ylabel("Ganancia Bruta")
    st.pyplot(fig5)
    st.markdown("> **Conclusión:** Existe una fuerte relación lineal entre el costo (cogs) y la ganancia bruta, lo que es esperable en un sistema con margen fijo. Este tipo de visualización permite confirmar que no hay grandes anomalías contables y que la rentabilidad por venta es consistente. Si se observaran desviaciones, podría señalar errores o productos con márgenes distintos.")


with col32:
    st.subheader("6. Métodos de Pago Preferidos")
    fig6, ax6 = plt.subplots(figsize=(6, 4))
    sns.countplot(data=df_filtered, x="Payment", order=df_filtered["Payment"].value_counts().index, ax=ax6)
    ax6.set_title("Frecuencia de Métodos de Pago")
    ax6.set_xlabel("Método de Pago")
    ax6.set_ylabel("Cantidad de Transacciones")
    st.pyplot(fig6)
    st.markdown("> **Conclusión:** La mayoría de los clientes prefieren ciertos métodos de pago, como tarjeta de crédito o efectivo. Esto puede informar decisiones operativas (como cantidad de caja disponible) y estratégicas (como alianzas con emisores de tarjetas o incentivos por método de pago). Además, conocer el método más común puede simplificar procesos de devolución o atención al cliente.")



st.subheader("7. Correlación entre Variables Numéricas")
numeric_cols = ["Unit price", "Quantity", "Tax 5%", "Total", "cogs", "gross income", "Rating"]
corr_matrix = df_filtered[numeric_cols].corr()
fig7, ax7 = plt.subplots(figsize=(4, 3))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax7, cbar=False)
ax7.set_title("Matriz de Correlación")
st.pyplot(fig7)
st.markdown("> **Conclusión:** Se observa una fuerte correlación entre variables como cogs, Total y gross income, lo cual es lógico ya que estas métricas están directamente relacionadas. Una correlación baja o nula con Rating sugiere que la percepción del cliente no está directamente vinculada con el precio o cantidad, lo cual es positivo en cuanto a experiencia de compra. Este análisis también ayuda a evitar variables redundantes en modelos predictivos.")



st.subheader("8. Ingreso Bruto por Sucursal y Línea de Producto (3D)")

tipo_vista = st.selectbox(
    "Selecciona la vista del gráfico:",
    ("Pastel por línea de producto", "Barras 3D por sucursal y línea")
)

if tipo_vista == "Pastel por línea de producto":
    ingreso_por_linea = df_filtered.groupby("Product line")["gross income"].sum()
    fig_pie, ax_pie = plt.subplots(figsize=(4, 4))
    ax_pie.pie(
        ingreso_por_linea,
        labels=ingreso_por_linea.index,
        autopct="%1.1f%%",
        startangle=140,
        colors=sns.color_palette("Set2")
    )
    ax_pie.set_title("Proporción de Ingreso Bruto por Línea de Producto")
    st.pyplot(fig_pie)


else:
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    import numpy as np

    ingreso_grouped = df_filtered.groupby(["Branch", "Product line"])["gross income"].sum().reset_index()

    branches = ingreso_grouped["Branch"].unique()
    products = ingreso_grouped["Product line"].unique()

    branch_dict = {branch: i for i, branch in enumerate(branches)}
    product_dict = {prod: i for i, prod in enumerate(products)}

    x = [branch_dict[b] for b in ingreso_grouped["Branch"]]
    y = [product_dict[p] for p in ingreso_grouped["Product line"]]
    z = np.zeros(len(x))
    dx = dy = 0.5
    dz = ingreso_grouped["gross income"].values

    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111, projection='3d')
    colors = cm.viridis(dz / max(dz))

    ax.bar3d(x, y, z, dx, dy, dz, color=colors)
    ax.set_xticks(list(branch_dict.values()))
    ax.set_xticklabels(list(branch_dict.keys()))
    ax.set_yticks(list(product_dict.values()))
    ax.set_yticklabels(list(product_dict.keys()))
    ax.set_zlabel("Ingreso Bruto")
    ax.set_title("Ingreso Bruto por Sucursal y Línea de Producto (3D)")

    st.pyplot(fig)

    st.markdown("> **Conclusión (3D):** El análisis combinado por sucursal y línea de producto permite identificar qué tiendas están capitalizando mejor ciertas categorías. Algunas sucursales sobresalen en líneas específicas, lo que puede responder a diferencias demográficas, ubicación o gestión. Esto permite personalizar estrategias por sucursal, como ajustar surtido o reforzar promociones de líneas que aún tienen potencial de crecimiento.")
