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
st.write("Para esta actividad vamos a tomar como variable objetivo "Dry Eye Disease" que significa que el sujeto tiene la enfermedad del ojo seco. donde Y es si y N es no. Se observa que existen más casos en la base donde el sujeto tiene la enfermedad por lo que podría ser de gran ayuda a la hora de realizar el modelo de clasificación.")

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





# Separar variables
#target_col = "Condition"

# Evitar que la variable objetivo quede en X
#X = df.drop(columns=[target_col])
y = df["Condition"]


numeric_features = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
if 'Condition' in numeric_features:
    numeric_features.remove('Condition')

X_num = df[numeric_features]
X_cat = df[categorical_vars]

st.markdown("""
## ¿Qué es el Análisis de Componentes Principales (PCA)?

El **Análisis de Componentes Principales (PCA)** es una técnica estadística usada para reducir la dimensionalidad de grandes conjuntos de datos con muchas variables numéricas, manteniendo la mayor cantidad posible de la variabilidad original.

### ¿Cómo funciona?

- PCA transforma las variables originales en un nuevo conjunto de variables no correlacionadas llamadas **componentes principales**.
- Cada componente principal es una combinación lineal de las variables originales.
- Los primeros componentes capturan la mayor parte de la variabilidad en los datos.
- Esto permite visualizar y analizar datos complejos en menos dimensiones, facilitando la interpretación.

### ¿Por qué usamos PCA en este análisis?

- Para explorar patrones y agrupamientos en las variables numéricas del dataset NHANES.
- Para visualizar relaciones entre individuos y variables de forma simplificada.
- Para identificar cuáles variables contribuyen más a la variabilidad total.

---
""")

# Pipeline con imputación, escalado y PCA
numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=10))
])
X_num_pca = numeric_pipeline.fit_transform(X_num)

# DataFrame con componentes
pca_df = pd.DataFrame(X_num_pca, columns=[f"PC{i+1}" for i in range(10)])
pca_df["Condition"] = y.values

# Extraer el objeto PCA del pipeline
pca = numeric_pipeline.named_steps["pca"]

# Obtener la varianza explicada de cada componente
explained_variance_ratio = pca.explained_variance_ratio_

# Calcular la varianza acumulada
cumulative_variance = np.cumsum(explained_variance_ratio)

# Graficar
fig, ax = plt.subplots(figsize=(8,5))
ax.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--')
ax.set_xlabel("Número de Componentes Principales")
ax.set_ylabel("Varianza Acumulada")
ax.set_title("Varianza Acumulada de las Componentes Principales")
ax.set_xticks(range(1, len(cumulative_variance) + 1))
ax.grid(True)
st.pyplot(fig)

# Gráfico PCA PC1 vs PC2
st.subheader("PCA - PC1 vs PC2")
fig, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="Condition", palette="Set2", alpha=0.8, ax=ax)
ax.set_title("PCA - PC1 vs PC2")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
st.pyplot(fig)

st.markdown("""
### ¿Qué son los Loadings en PCA?

En el Análisis de Componentes Principales (PCA), los **loadings** (o cargas factoriales) representan la relación entre las variables originales y las componentes principales.

---

### Cómo se calculan

PCA busca nuevas variables llamadas **componentes principales**, que son combinaciones lineales de las variables originales.

Cada componente principal $PC_j$ se define como:

$$
PC_j = a_{1j} x_1 + a_{2j} x_2 + \cdots + a_{pj} x_p
$$

donde:

- $x_1, x_2, \ldots, x_p$ son las variables originales (normalizadas si se aplica escalado),
- $a_{ij}$ son los coeficientes o **loadings** de la variable $i$ en la componente $j$.

---

### Interpretación

- El loading $a_{ij}$ indica **cuánto aporta** la variable $i$ a la componente principal $j$.
- Valores altos (en valor absoluto) significan que esa variable influye fuertemente en la componente.
- El signo indica la dirección de la relación (positiva o negativa).

---

### Cálculo práctico

- Los loadings corresponden a los **autovectores** (vectores propios) de la matriz de covarianza o correlación de los datos.
- En `scikit-learn`, el atributo `components_` del objeto PCA contiene estos loadings:  
  - Cada fila es una componente principal,  
  - Cada columna es una variable original.

---

### Usos

- Analizar los loadings ayuda a interpretar qué variables definen cada componente.
- También es útil para seleccionar variables importantes según su contribución.
""")

# Obtener los loadings del PCA (componentes * características)
loadings = numeric_pipeline.named_steps["pca"].components_

# Convertir a DataFrame con nombres de columnas
loadings_df = pd.DataFrame(
    loadings,
    columns=X_num.columns,
    index=[f"PC{i+1}" for i in range(loadings.shape[0])]
).T  # Transponer para que columnas sean PCs y filas las variables

# Ordenar las filas por la importancia de la variable en la suma de cuadrados de los componentes
# Esto agrupa por aquellas variables con mayor contribución total
loading_magnitude = (loadings_df**2).sum(axis=1)
loadings_df["Importance"] = loading_magnitude
loadings_df_sorted = loadings_df.sort_values(by="Importance", ascending=False).drop(columns="Importance")

# Graficar heatmap ordenado
st.subheader("🔍 Heatmap de Loadings del PCA (Componentes Principales)")

fig, ax = plt.subplots(figsize=(10, 12))
sns.heatmap(loadings_df_sorted, annot=True, cmap="coolwarm", center=0, ax=ax)
st.pyplot(fig)

st.markdown("""
### ¿Qué es el Análisis de Correspondencias Múltiples (MCA)?

El Análisis de Correspondencias Múltiples (MCA) es una técnica estadística exploratoria utilizada para analizar y visualizar datos categóricos. Es una extensión del Análisis de Correspondencias Simple (CA) cuando hay más de dos variables categóricas.

**Objetivos principales del MCA:**

- **Reducir la dimensionalidad** de datos categóricos complejos.
- **Identificar patrones** y relaciones entre categorías de variables.
- **Visualizar** asociaciones entre individuos y categorías en un espacio de menor dimensión.

**¿Cómo funciona?**

El MCA transforma las variables categóricas en un espacio numérico, similar al Análisis de Componentes Principales (PCA) para variables numéricas. Luego, representa las observaciones y categorías en un mapa factorial bidimensional o tridimensional, donde la proximidad entre puntos indica similitudes.

**Aplicaciones comunes:**

- Encuestas y estudios sociales con muchas variables categóricas.
- Análisis de perfiles de consumidores.
- Estudios epidemiológicos para agrupar características clínicas o sociodemográficas.

En resumen, MCA es una herramienta poderosa para explorar y resumir grandes conjuntos de datos categóricos y facilita la interpretación visual de relaciones complejas.
""")

# Pipeline para MCA con prince
categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse=False))
])

# Transformar datos categóricos
X_cat_encoded = categorical_pipeline.fit_transform(X_cat)

# Crear DataFrame con nombres de columnas después del one-hot encoding
encoded_columns = categorical_pipeline.named_steps["encoder"].get_feature_names_out(X_cat.columns)
X_cat_encoded_df = pd.DataFrame(X_cat_encoded, columns=encoded_columns, index=X_cat.index)

# Aplicar MCA con prince
mca = prince.MCA(n_components=2, random_state=42)
X_cat_mca = mca.fit_transform(X_cat_encoded_df)

# Agregar columna de condición para colorear
mca_df = X_cat_mca.copy()
mca_df["Condition"] = y.values
mca_df.columns = ["Dim1", "Dim2", "Condition"]

st.subheader("MCA - Dim1 vs Dim2")
fig, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(data=mca_df, x="Dim1", y="Dim2", hue="Condition", palette="Set1", alpha=0.8, ax=ax)
ax.set_title("MCA - Dim1 vs Dim2")
ax.set_xlabel("Dim1")
ax.set_ylabel("Dim2")
st.pyplot(fig)

# ======================
# HEATMAP DE CONTRIBUCIONES EN MCA
# ======================


st.subheader("🔍 Contribuciones de las Variables Categóricas al MCA")

# Obtener contribuciones a las dimensiones
contribs = mca.column_contributions_

#st.write("Column names in contributions DataFrame:", contribs.columns.tolist())

# Seleccionar contribuciones a Dim1 y Dim2
contribs_selected = contribs[[0, 1]]  # 0 = Dim1, 1 = Dim2
contribs_selected.columns = ["Dim1", "Dim2"]

# Ordenar por Dim1 para mejor visualización (opcional)
#contribs_sorted = contribs_selected.sort_values(by=0, ascending=False)
contribs_sorted = contribs_selected.sort_values("Dim1", ascending=False)

# Crear heatmap
fig, ax = plt.subplots(figsize=(10, max(6, 0.3 * len(contribs_sorted))))
sns.heatmap(contribs_sorted, cmap="YlGnBu", annot=True, fmt=".2f", ax=ax)
ax.set_title("Contribuciones de las Variables a las Dimensiones del MCA")
st.pyplot(fig)


# ======================
# 🔧 Construir X
# ======================

# Separar por tipo
x_num = df.select_dtypes(include=["float64", "int64"]).drop(columns=["SEQN"], errors="ignore")
x_cat = df.select_dtypes(include=["object", "category", "bool", "string"])

# Guardar nombres originales
num_features = x_num.columns.tolist()
cat_features = x_cat.columns.tolist()

# Preprocesadores
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

# ColumnTransformer combinado
preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, num_features),
    ("cat", categorical_transformer, cat_features)
])

# Ajustar y transformar
preprocessor.fit(df)

# Obtener nombres después del preprocesamiento
cat_encoded_columns = preprocessor.named_transformers_["cat"].named_steps["encoder"].get_feature_names_out(cat_features)
feature_names = np.concatenate([num_features, cat_encoded_columns])

# Transformar el dataset
X = preprocessor.transform(df)
X_df = pd.DataFrame(X, columns=feature_names)

# ======================
# 🔍 Selección de Variables
# ======================
st.header("🔍 Selección de Variables")

# Variable objetivo
y = df["Condition"]

# Train/Test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X_df, y, test_size=0.2, stratify=y, random_state=42
)

# ============================
# 1️⃣ Selección basada en modelos
# ============================
with st.expander("1️⃣ Selección basada en modelos (Random Forest)"):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    importances = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)

    st.subheader("Importancia de variables")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=importances.head(7), y=importances.head(7).index, ax=ax)
    st.pyplot(fig)

    st.write("Precisión en test:", np.round(model.score(X_test, y_test), 3))

# ============================
# 3️⃣ Selección por envoltura
# ============================
with st.expander("3️⃣ Selección por envoltura (RFE con Regresión Logística)"):
    logistic = LogisticRegression(max_iter=500, solver='liblinear')
    rfe = RFE(estimator=logistic, n_features_to_select=7)
    rfe.fit(X_train, y_train)
    selected_features = X_train.columns[rfe.support_]

    st.write("Variables seleccionadas:", selected_features.tolist())

    coefs = pd.Series(logistic.fit(X_train[selected_features], y_train).coef_[0], index=selected_features)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=coefs, y=coefs.index, ax=ax)
    st.pyplot(fig)
    
















