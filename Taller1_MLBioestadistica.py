import numpy as np
import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif, RFE, chi2
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, chi2_contingency
import seaborn as sns
from mca import MCA  # Asegúrate de tener instalada la librería: pip install mca
from sklearn.base import BaseEstimator, TransformerMixin
import prince
from sklearn.preprocessing import FunctionTransformer, LabelEncoder, StandardScaler
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA


st.set_page_config(page_title="Taller1_MLBioestadística", layout="wide")
st.title("Análisis, preprocesamiento y reducción de dimensionalidad Dry Eye Disease")

@st.cache_data
def load_data():
    url = "https://github.com/andrews06ml/Taller1_MLBioestad-stica/blob/main/Dry_Eye_Dataset.csv"
    df = pd.read_csv(url)
    df.columns = df.columns.str.strip()
    return df

df = load_data()

# Asignar condiciones
# def assign_condition(row):
#     if (row["BPXSY1"] >= 140 or row["BPXSY2"] >= 140 or 
#         row["BPXDI1"] >= 90 or row["BPXDI2"] >= 90):
#         return "hypertension"
#     elif row["BMXBMI"] >= 30:
#         return "diabetes"
#     elif ((row["RIAGENDR"] == 1 and row["BMXWAIST"] > 102) or 
#           (row["RIAGENDR"] == 2 and row["BMXWAIST"] > 88)):
#         return "high cholesterol"
#     else:
#         return "healthy"

# df["Condition"] = df.apply(assign_condition, axis=1)

# Diccionario de códigos por variable categórica
category_mappings = {
    "Gender": {
        M: "Masculino",
        F: "Femenino"
    },
    "Sleep disorder": {
        Y: "Si",
        N: "No",
    },
    "Caffeine consumption": {
        Y: "Si",
        N: "No",
    },
    "Alcohol consumption": {
        Y: "Si",
        N: "No",
    }
}

def apply_categorical_mappings(df, mappings):
    for col, mapping in mappings.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)
    return df

df = apply_categorical_mappings(df, category_mappings)

st.markdown("""
# 🏥 **Taller No. 1 - Machine learning II para bioestadística**
---

**Integrantes:**
- *Andres Felipe Montenegro*
- *Samuel Forero Martinez*

## **Base de datos utilizada para el desarrollor del taller: *DRY EYE DISEASE*** 👀

Se trata de un conjunto de datos amplio pensado para modelos predictivos y análisis diagnóstico de la Enfermedad del Ojo Seco (DED), usando variables como calidad y duración del sueño, enrojecimiento ocular, picazón, tiempo frente a pantallas, uso de filtro de luz azul y fatiga ocular.

Incluye datos estructurados de personas entre 18 y 45 años, permitiendo investigar la relación entre hábitos de vida y salud ocular. Puede aplicarse en machine learning, análisis estadístico y decisiones clínicas para mejorar la detección temprana y tratamientos personalizados, e incluso predecir enfermedades relacionadas con el sueño como el insomnio, que puede estar vinculado a enfermedades de la superficie ocular.

El archivo contiene información de unas 20.000 personas (adolescentes, adultos de mediana edad y mayores) de ambos sexos. Incluye columnas como pasos diarios, tiempo de sueño, pulso, presión arterial, hábitos de alimentación y bebida, niveles de estrés, problemas médicos (ansiedad, hipertensión, asma, etc.) y medicamentos utilizados, además de atributos oculares básicos para predecir la DED. Está en formato CSV y puede aportar a diversos tipos de investigación médica relacionada con la salud ocular y los hábitos de las personas.

 ***Link con mayor información: [Enfermedad del Ojo seco](https://www.kaggle.com/datasets/dakshnagra/dry-eye-disease)***

### Propósito del Análisis en esta App:

- Clasificar a los participantes en grupos de salud: **hipertensión**, **diabetes**, **alto colesterol** y **saludable**, basados en indicadores clínicos.
- Realizar un análisis exploratorio con técnicas de reducción de dimensionalidad como:
  - PCA (Análisis de Componentes Principales) para variables numéricas.
  - MCA (Análisis de Correspondencias Múltiples) para variables categóricas.
- Seleccionar las variables más relevantes para la clasificación usando técnicas estadísticas y de machine learning.

### Importancia:

NHANES es un recurso valioso para investigadores, médicos y políticas públicas que buscan entender factores de riesgo y prevalencia de enfermedades crónicas en la población estadounidense. Este análisis ayuda a identificar patrones clave en los datos que pueden guiar intervenciones de salud.

---

**Fuente:** [NHANES 2015-2016](https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?BeginYear=2015)
""")

st.markdown("---")
# Mostrar info y variables categóricas lado a lado
st.header("1. Cargue y exploración inicial de la base de datos")

st.subheader("Base de datos")
pd.set_option('display.max_columns', None)
df.head(10)

st.markdown("""
La base de datos utilizada cuenta con 20.000 registros y 26 variables, donde cada fila corresponde a un sujeto de estudio, hombre o mujer, con edades comprendidas entre los 18 y 45 años. La información recopilada es de gran relevancia para el desarrollo de investigaciones médicas orientadas a la salud ocular y los hábitos de vida de la población.

Entre las variables incluidas se encuentran datos sobre horas y calidad del sueño, niveles de estrés, frecuencia de actividad física, pulso, presión arterial, peso y estatura. También se registran aspectos relacionados con hábitos alimenticios y de consumo de bebidas, como la ingesta de cafeína y alcohol, así como el tabaquismo.

En cuanto a la salud visual, la base de datos contiene información sobre síntomas y signos oculares, tales como enrojecimiento, irritación, malestar o fatiga visual y la presencia de ojo seco. Asimismo, se incluyen datos sobre tiempo de exposición a pantallas y antecedentes de problemas médicos que puedan estar vinculados al estado de salud ocular.

Este conjunto de datos ofrece un panorama integral de factores fisiológicos, conductuales y ambientales, lo que lo convierte en una herramienta valiosa para identificar patrones, analizar relaciones y proponer estrategias de prevención y tratamiento en el ámbito de la salud visual.
""")


st.subheader("Resumen de datos")
# Crear columnas para mostrar info_df y category_df lado a lado
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Tipo de Dato y Nulos**")
    info_df = pd.DataFrame({
        "Column": df.columns,
        "Non-Null Count": df.notnull().sum().values,
        "Dtype": df.dtypes.values
    })
    st.dataframe(info_df, use_container_width=True)

# Detección automática de variables categóricas
categorical_vars = [col for col in df.columns 
                    if df[col].dtype == 'object' or 
                       df[col].dtype == 'string' or 
                       df[col].nunique() <= 10]

with col2:
    st.markdown("**Variables Categóricas Detectadas**")
    category_info = []
    for col in categorical_vars:
        unique_vals = df[col].dropna().unique()
        category_info.append({
            "Variable": col,
            "Unique Classes": ", ".join(map(str, sorted(unique_vals)))
        })

    category_df = pd.DataFrame(category_info)
    st.dataframe(category_df, use_container_width=True)


# Filtros
with st.sidebar:
    st.header("Filtros")
    gender_filter = st.multiselect("Genero", sorted(df["Gender"].dropna().unique()))
    sleep_filter = st.multiselect("Transtornos del sueño", sorted(df["Sleep disorder"].dropna().unique()))
    cafe_filter = st.multiselect("Consumo de cafe", sorted(df["Caffeine consumption"].dropna().unique()))
    alcohol_filter = st.multiselect("Consumo de alcohol", sorted(df["Alcohol consumption"].dropna().unique()))
    #st.markdown("---")
    #k_vars = st.slider("Number of variables to select", 2, 10, 5)

# Aplicar filtros
for col, values in {
    "Genero": gender_filter, "Transtornos del sueño": sleep_filter, "Consumo de cafe": cafe_filter, "Consumo de alcohol": alcohol_filter
}.items():
    if values:
        df = df[df[col].isin(values)]

if df.empty:
    st.warning("No data available after applying filters.")
    st.stop()

# Mostrar advertencias
problematic_cols = df.columns[df.dtypes == "object"].tolist()
nullable_ints = df.columns[df.dtypes.astype(str).str.contains("Int64")].tolist()

st.write("### ⚠️ Columnas potencialmente problemáticas para Arrow/Streamlit:")
if problematic_cols or nullable_ints:
    st.write("**Tipo 'object':**", problematic_cols)
    st.write("**Tipo 'Int64' (nullable):**", nullable_ints)
else:
    st.success("✅ No hay columnas problemáticas detectadas.")

st.markdown("---")
st.header("2. Análisis exploratorio de datos")

# Creación de nuevas variables según el valor de la presión arterial y conversión de genero
df = (df.assign(**df['Blood pressure'].str.split('/', expand=True)
                 .rename(columns={0:'Presion_Sistolica', 1:'Presion_Diastolica'})
                 .apply(pd.to_numeric, errors='coerce'))
        .assign(Presion_Pulso=lambda x: x['Presion_Sistolica'] - x['Presion_Diastolica'])
        .drop(columns=['Blood pressure'])
     )

# Convertir todas las columnas tipo object a category
df = df.astype({col: "category" for col in df.select_dtypes(include="object").columns})

# Variables numericas
df_numericas = df.select_dtypes(include=[np.number])
# nombres de columnas numericas
columnas_numericas = df_numericas.columns.tolist()

# Variables categoricas
df_categoricas = df.select_dtypes(include=["category"])

# nombres de columnas categoricas
columnas_categoricas = df_categoricas.columns.tolist()


st.subheader("Distribución de las variables")

# Para variables numericas
df_numericas.hist(figsize=(15,10), bins=30, edgecolor='black')
plt.show()


# Crear figura con 4 filas y 4 columnas
fig, axes = plt.subplots(4, 4, figsize=(25, 20))
axes = axes.flatten()  # Convertir la matriz de ejes a lista

# Recorrer las columnas y graficar
for i, col in enumerate(df_categoricas):
    sns.countplot(x=col, data=df, ax=axes[i])
    axes[i].set_title(col)
    axes[i].tick_params(axis='x', rotation=90)

# Ocultar ejes vacíos si hay menos de 16 gráficas
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

st.subheader("Validación de datos atípicos")

# Crear figura con 3 filas y 3 columnas
fig, axes = plt.subplots(4, 4, figsize=(20, 15))
axes = axes.flatten()  # Convertir la matriz de ejes a lista

for i, col in enumerate(df_numericas):
    sns.boxplot(x=df[col], ax=axes[i])
    axes[i].set_title(col)
    axes[i].tick_params(axis='x', rotation=45)

# Eliminar ejes vacíos si hay menos gráficos que subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()
st.write("Como se puede ver en los diagramas de cajas y bigotes, las 10 variables numéricas contenidas en la base de datos no cuentan con valores atípicos.")

st.subheader("Balance de la variable dependiente (dry eye disease)")

# Conteo absoluto
print(df["Dry Eye Disease"].value_counts())

# Porcentaje
print(df["Dry Eye Disease"].value_counts(normalize=True) * 100)

# Visualización
sns.countplot(x="Dry Eye Disease", data=df)
plt.title("Distribución de la variable objetivo (Dry Eye Disease)")
plt.show()
st.write("Para esta actividad vamos a tomar como variable objetivo (Dry Eye Disease) que significa que el sujeto tiene la enfermedad del ojo seco. donde Y es si y N es no. Se observa que existen más casos en la base donde el sujeto tiene la enfermedad por lo que podría ser de gran ayuda a la hora de realizar el modelo de clasificación.")

st.subheader("Correlaciones")

# Identificar tipos de variables
num_vars = df.select_dtypes(include=[np.number]).columns
cat_vars = df.select_dtypes(exclude=[np.number]).columns.drop("Dry Eye Disease", errors="ignore")

print("Variables numéricas:", list(num_vars))
print("Variables categóricas:", list(cat_vars))

# Correlación variables numéricas vs dry disease
correlations = {}
for col in num_vars:
    corr, pval = spearmanr(df[col], df['Dry Eye Disease'])
    correlations[col] = {"Spearman_corr": corr, "p-value": pval}

cor_num = pd.DataFrame(correlations).T
print("\nCorrelación con variables numéricas:")
print(cor_num.sort_values("Spearman_corr", ascending=False))

# Asociación variables categóricas vs dry disease
assoc_cat = {}
for col in cat_vars:
    table = pd.crosstab(df[col], df['Dry Eye Disease'])
    chi2, p, dof, expected = chi2_contingency(table)
    assoc_cat[col] = {"Chi2": chi2, "p-value": p}

assoc_cat_df = pd.DataFrame(assoc_cat).T
print("\nAsociación con variables categóricas:")
print(assoc_cat_df.sort_values("p-value"))

# Visualización correlaciones numéricas 
plt.figure(figsize=(8,5))
sns.heatmap(df[num_vars].corr(method="spearman"), annot=True, cmap="coolwarm")
plt.title("Matriz de correlación (Spearman)")
plt.show()

# Visualización correlaciones categóricas
def cramers_v(confusion_matrix):
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

# Calcular matriz de Cramer’s V para todas las categóricas
cat_vars = df.select_dtypes(exclude=[np.number]).columns
assoc_matrix = pd.DataFrame(index=cat_vars, columns=cat_vars)

for col1 in cat_vars:
    for col2 in cat_vars:
        table = pd.crosstab(df[col1], df[col2])
        assoc_matrix.loc[col1, col2] = cramers_v(table)

assoc_matrix = assoc_matrix.astype(float)

# Heatmap
plt.figure(figsize=(10,8))
sns.heatmap(assoc_matrix, annot=True, cmap="coolwarm", vmin=0, vmax=1)
plt.title("Mapa de calor de asociación (Cramer's V) entre variables categóricas")
plt.show()

st.header("Separación del df y la variable a predecir")

# Separar la variable dependiente de las demás
X = df.drop('Dry Eye Disease', axis=1)
y = df['Dry Eye Disease']

# Separar las variables numericas de las categoricas de X
X_num = X.select_dtypes(include=[np.number])
X_cat = X.select_dtypes(exclude=[np.number])

# Conjunto de entrenamiento y prueba numerico
from sklearn.model_selection import train_test_split
X_train_num, X_test_num, y_train_num, y_test_num = train_test_split(X_num, y, test_size=0.2, random_state=42)

# Conjunto de entrenamiento y prueba categórico
X_train_cat, X_test_cat, y_train_cat, y_test_cat = train_test_split(X_cat, y, test_size=0.2, random_state=42)

print(f"Base, categórica: {X_train_cat.shape}")
print(f"Base, numérica: {X_train_num.shape}")
