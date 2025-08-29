import numpy as np
import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif, RFE, chi2, RFECV
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
from imblearn.over_sampling import BorderlineSMOTE
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc, RocCurveDisplay
)
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    HistGradientBoostingClassifier, ExtraTreesClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from scipy.stats import randint, uniform
from sklearn.svm import SVC
from sklearn.preprocessing import label_binarize
import matplotlib
from xgboost import XGBClassifier


st.set_page_config(page_title="Taller1_MLBioestadística", layout="wide")
st.title("Análisis, preprocesamiento y reducción de dimensionalidad Dry Eye Disease")

@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/andrews06ml/Taller1_MLBioestadistica/refs/heads/main/Dry_Eye_Dataset.csv"
    df = pd.read_csv(url)
    return df

df = load_data()

st.text("""
El taller se encuentra organizado en las siguientes secciones:

Usa las pestañas de abajo para ver la exploración de datos, las técnicas de 
reducción de dimensionalidad y los selectores de variables para la base de datos Dry Eye Disease.

Haz clic en la pestaña que quieras revisar del taller.
""")

tab1, tab2, tab3, tab4 = st.tabs(["📊 Exploración de datos", " ✅**Tarea 1 - ACP y MCA**", "✅**Tarea 2 - Aplicación de selectores**"], "🔍**Modelos de clasificación**")

with tab1:
    st.dataframe(df, use_container_width=True)
    
    # Diccionario de códigos por variable categórica
    category_mappings = {
        "Gender": {
            "M": "Masculino",
            "F": "Femenino"
        },
        "Sleep disorder": {
            "Y": "Si",
            "N": "No"
        },
        "Caffeine consumption": {
            "Y": "Si",
            "N": "No"
        },
        "Alcohol consumption": {
            "Y": "Si",
            "N": "No"
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
    
    st.subheader("Tipos de variables")
    st.write(df.dtypes.value_counts())
    
    st.markdown("""
    Se observa que de las 26 variables con las que cuenta la base, 16 son categóricas y 10 son numéricas. Adicionalmente no se evidencian valores faltantes en ningun registro por lo que no hay necesidad de imputar ni eliminar variables.
    """)
    
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
    
    st.markdown("### Variables numéricas")
    # Para variables numericas
    fig, ax = plt.subplots(figsize=(15,10))
    df_numericas.hist(ax=ax, bins=30, edgecolor='black')  # si df_numericas es un DataFrame, esto funciona
    st.pyplot(fig)
    
    st.markdown("### Variables categóricas")
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
    st.pyplot(fig)
    
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
    st.pyplot(fig)
    st.text("Como se puede ver en los diagramas de cajas y bigotes, las 10 variables numéricas contenidas en la base de datos no cuentan con valores atípicos.")
    
    st.subheader("Balance de la variable dependiente (dry eye disease)")
    
    # Porcentaje
    st.text("Porcentaje de pacientes por tipo de respuesta")
    st.write(df["Dry Eye Disease"].value_counts(normalize=True) * 100)
    
    # Visualización
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.countplot(x="Dry Eye Disease", data=df, ax=ax)
    ax.set_title("Distribución de la variable objetivo (Dry Eye Disease)")
    st.pyplot(fig)
    st.text("Para esta actividad vamos a tomar como variable objetivo (Dry Eye Disease) que significa que el sujeto tiene la enfermedad del ojo seco. donde Y es si y N es no. Se observa que existen más casos en la base donde el sujeto tiene la enfermedad por lo que podría ser de gran ayuda a la hora de realizar el modelo de clasificación.")
    
    st.subheader("Correlaciones")
    
    # Identificar tipos de variables
    num_vars = df.select_dtypes(include=[np.number]).columns
    cat_vars = df.select_dtypes(exclude=[np.number]).columns.drop("Dry Eye Disease", errors="ignore")
    
    st.write("Variables numéricas:", list(num_vars))
    st.write("Variables categóricas:", list(cat_vars))
    
    # Correlación variables numéricas vs dry disease
    correlations = {}
    for col in num_vars:
        corr, pval = spearmanr(df[col], df['Dry Eye Disease'])
        correlations[col] = {"Spearman_corr": corr, "p-value": pval}
    
    cor_num = pd.DataFrame(correlations).T
    st.write("\nCorrelación con variables numéricas:")
    st.write(cor_num.sort_values("Spearman_corr", ascending=False))
    
    # Asociación variables categóricas vs dry disease
    assoc_cat = {}
    for col in cat_vars:
        table = pd.crosstab(df[col], df['Dry Eye Disease'])
        chi2_cont, p, dof, expected = chi2_contingency(table)
        assoc_cat[col] = {"Chi2_cont": chi2_cont, "p-value": p}
    
    assoc_cat_df = pd.DataFrame(assoc_cat).T
    st.write("\nAsociación con variables categóricas:")
    st.write(assoc_cat_df.sort_values("p-value"))
    
    # Visualización correlaciones numéricas 
    fig, ax = plt.subplots(figsize=(8,5))
    sns.heatmap(df[num_vars].corr(method="spearman"), annot=True, cmap="coolwarm", annot_kws={"size": 6})
    ax.set_title("Matriz de correlación (Spearman)")
    st.pyplot(fig)
    
    # Visualización correlaciones categóricas
    def cramers_v(confusion_matrix):
        chi2_cont = chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum().sum()
        phi2 = chi2_cont / n
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
    fig, ax = plt.subplots(figsize=(10,8))
    sns.heatmap(assoc_matrix, annot=True, cmap="coolwarm", vmin=0, vmax=1, annot_kws={"size": 7})
    ax.set_title("Mapa de calor de asociación (Cramer's V) entre variables categóricas")
    st.pyplot(fig)
    
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
    
    st.write(f"Base, categórica: {X_train_cat.shape}")
    st.write(f"Base, numérica: {X_train_num.shape}")
    st.text("Se realizan los conjuntos de entrenamiento para las variables numéricas y categóricas distribuidos de la siguiente forma: 80% entrenamiento y 20% prueba")

with tab2:
    st.subheader("Reducción de dimensionalidad con PCA para variables numéricas")
    st.markdown("### Gráfica de número de componentes que explican más del 80% de la varianza acumulada")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train_num)
    
    # PCA con todos los componentes
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)
    
    # Varianza acumulada
    explained_var = np.cumsum(pca.explained_variance_ratio_)

    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(range(1, len(explained_var) + 1), explained_var, marker='o', linestyle='--')
    ax.axhline(y=0.9, color='r', linestyle='-')
    ax.set_xlabel('Número de componentes principales')
    ax.set_ylabel('Varianza acumulada explicada')
    ax.set_title('Varianza acumulada explicada por PCA')
    ax.grid(True)
    st.pyplot(fig)

    st.markdown("### Gráfica de número de componentes que explican más del 80% de la varianza acumulada")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Crear el scatterplot sobre el eje `ax`
    sns.scatterplot(
        x=X_pca[:, 0],
        y=X_pca[:, 1],
        hue=y_train_num,
        palette='Set1',
        alpha=0.7,
        ax=ax)

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('Scatterplot PC1 vs PC2')
    ax.legend(title='Clase', labels=['Enfermo', 'No enfermo'])
    st.pyplot(fig)

    st.markdown("### Loadings")
    # Conocer los nombres de las columnas de X_train
    columnas_numericas = X_train_num.columns.tolist()
    
    # Loadings (cargas) de las variables en las PCs
    loadings = pd.DataFrame(pca.components_.T, columns=[f'PC{i+1}' for i in range(pca.n_components_)], index=columnas_numericas)
    
    fig, ax = plt.subplots(figsize=(12,8))
    sns.heatmap(loadings.iloc[:,:11], annot=True, cmap='coolwarm', center=0, ax=ax)
    ax.set_title('Heatmap de loadings (primeras 10 PCs)')
    st.pyplot(fig)
    
    # Aplicar PCA
    pca = PCA(n_components=0.90)  # Selecciona número mínimo de PCs que expliquen 90% de la varianza
    X_pca = pca.fit_transform(X_scaled)
    
    st.write(f"Número de componentes principales para explicar 90% varianza: {pca.n_components_}")
    st.write(f"Varianza explicada acumulada por estas componentes: {sum(pca.explained_variance_ratio_):.4f}")

    st.subheader("Reducción de dimensionalidad con MCA para variables categóricas")

    # Copiamos el DataFrame para no modificar el original
    X_train_cat_encoded = X_train_cat.copy()

    # Aplicamos onehotencoder a cada columna categórica
    X_train_cat_encoded = pd.get_dummies(X_train_cat, drop_first=False)

    # Aplicar MCA
    mca = prince.MCA(n_components=14,random_state=42)
    mca = mca.fit(X_train_cat_encoded)

    # Valores singulares y autovalores
    # Varianza acumulado
    cum_explained_var = mca.eigenvalues_summary['% of variance (cumulative)']
    cum_explained_var = cum_explained_var.str.rstrip("%").astype(float)
    cum_explained_var = cum_explained_var / 100

    # Graficar varianza acumulada
    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(range(1, len(cum_explained_var)+1), cum_explained_var, marker='o', linestyle='--')
    ax.axhline(y=0.9, color='r', linestyle='-')
    ax.set_xlabel('Dimensiones MCA')
    ax.set_ylabel('Varianza acumulada explicada')
    ax.set_title('Varianza acumulada explicada por MCA')
    ax.grid(True)
    st.pyplot(fig)

    # Coordenadas individuos (2 primeras dimensiones)
    coords = mca.row_coordinates(X_train_cat_encoded)

    fig, ax = plt.subplots(figsize=(8,6))
    sns.scatterplot(x=coords[0], y=coords[1], hue=y_train_cat, palette='Set1', alpha=0.7, ax=ax)
    ax.set_xlabel('Dimensión 1')
    ax.set_ylabel('Dimensión 2')
    ax.set_title('Scatterplot MCA Dim 1 vs Dim 2')
    ax.legend(title='Clase', labels=['Y', 'N']) # Ajustar etiquetas de leyenda
    st.pyplot(fig)

    # Coordenadas de las columnas (variables categóricas)
    loadings_cat = mca.column_coordinates(X_train_cat_encoded).iloc[:, :2]
    
    # Calcular contribución de cada variable (cuadrado / suma por dimensión)
    loadings_sq = loadings_cat ** 2
    contrib_cat = loadings_sq.div(loadings_sq.sum(axis=0), axis=1)
    
    # Sumar contribuciones por variable
    contrib_var = contrib_cat.sum(axis=1).sort_values(ascending=False)
    
    # Graficar contribuciones variables
    fig, ax = plt.subplots(figsize=(12, 6))
    contrib_var.plot(kind='bar', color='teal', ax=ax)
    ax.set_ylabel('Contribución total a Dim 1 y 2')
    ax.set_title('Contribución de variables a las primeras 2 dimensiones MCA')
    ax.tick_params(rotation=90, axis = "x")
    fig.tight_layout()
    st.pyplot(fig)

    st.markdown("### Conclusión")
    st.markdown(""" Despues de realizar técnicas de reducción de dimensionalidad tanto para variables numéricas (ACP), como para variables categóricas (MCA), se concluye que:

    - Para el caso del análisis de componentes principales (ACP), se requieren 11 componentes principales para explicar más del 90% de la varianza de las variables numéricas. Esto sugiere que la estructura de las variables numéricas es relativamente compleja, y que existe información relevante distribuida en múltiples dimensiones.

    - Con respecto al análisis de correspondencias múltiples (MCA) mostró que se requieren 13 dimensiones para explicar más del 90% de la varianza de las variables categóricas. Esto indica que las variables tienen información bastante diversa y no estan muy correlacionadas.

    Teniendo en cuenta el resultado propuesto para el ejercicio, se realiza la combinación de las 11 componentes y las 13 dimensiones en una sola base de datos con el objetivo de posteriormente hacer un modelo de clasificación para la variable "Dry Eye Disease" que cuente con lo más relevante de ambos tipos de variables.
    """)

    st.markdown("### Paso final - Concatenar los componentes con las dimensiones para crear la base de datos final")

    # Concatenar los componentes principales con las dimensiones
    X_reduced = np.hstack((X_pca, coords.iloc[:,0:13]))

    # convertir array en dataframe
    X_reduced_df = pd.DataFrame(X_reduced, columns=[f'PC{i+1}' for i in range(pca.n_components_)] + [f'MCA{i}' for i in range(1,14)])

    X_reduced_df

with tab3:
    st.subheader("Selector por filtrado (chi2), incrustada (Randomforest), envoltura (Regresion logistica)")

    # 0. escalar y convertir las variables de la base de datos de entrenamiento
    num_features = X_train_num.columns.tolist()
    cat_features = X_train_cat.columns.tolist()
    
    # Codificar Y/N a 0/1 para variables categóricas
    X_encoded = X.copy()
    for col in X_encoded.select_dtypes(include=['object', 'category']):
        if X_encoded[col].nunique() == 2:  # Solo codificar variables binarias
            X_encoded[col] = X_encoded[col].map({'N': 0, 'No': 0, 'Si': 1,'Y': 1, 'M': 0, 'Masculino': 0, 'F': 1, 'Femenino': 1 })
    
    # Separar la base en entrenamiento y prueba después de codificar
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
    
    # Definir las listas de variables numéricas y categóricas
    num_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_features = X_train.select_dtypes(exclude=[np.number]).columns.tolist()
    
    # --- 1. Selección por filtrado (SelectKBest con chi2) ---
    # Aplicar MinMaxScaler solo a las columnas numéricas de X_train
    X_train_num_scaled = MinMaxScaler().fit_transform(X_train[num_features])
    X_train_num_scaled_df = pd.DataFrame(X_train_num_scaled, columns=num_features, index=X_train.index)
    
    # Combinar con las variables categóricas codificadas (que ya son 0 o 1)
    X_train_processed_filter = pd.concat([X_train_num_scaled_df, X_train[cat_features]], axis=1)

    # Aplicar SelectKBest con chi2 sobre el dataframe preprocesado
    selector = SelectKBest(score_func=chi2, k="all")
    
    # Asegurarse que y_train es numérica para chi2
    selector.fit(X_train_processed_filter, y_train.map({'N': 0, 'Y': 1}))
    
    scores_filter = selector.scores_
    features = X_train_processed_filter.columns
    
    indices_filter = np.argsort(scores_filter)[::-1]
    sorted_scores_filter = scores_filter[indices_filter]
    sorted_features_filter = features[indices_filter]
    
    cumulative_filter = np.cumsum(sorted_scores_filter) / np.sum(sorted_scores_filter)
    cutoff_filter = np.searchsorted(cumulative_filter, 0.90) + 1
    selected_filter = sorted_features_filter[:cutoff_filter]
    
    # --- 2. Selección incrustada (RandomForest feature_importances_) ---
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_processed_filter, y_train)
    
    importances_embedded = rf.feature_importances_
    indices_embedded = np.argsort(importances_embedded)[::-1]
    sorted_importances_embedded = importances_embedded[indices_embedded]
    sorted_features_embedded = features[indices_embedded]
    
    cumulative_embedded = np.cumsum(sorted_importances_embedded) / np.sum(sorted_importances_embedded)
    cutoff_embedded = np.searchsorted(cumulative_embedded, 0.90) + 1
    selected_embedded = sorted_features_embedded[:cutoff_embedded]
    
    # --- 3. Selección por envoltura (RFECV con LogisticRegression) ---
    
    # Modelo base para RFECV
    model = LogisticRegression(max_iter=1000)
    
    rfecv = RFECV(estimator=model, step=1, cv=5, scoring='accuracy')
    pipeline = Pipeline([('scaler', scaler), ('feature_selection', rfecv)])
    pipeline.fit(X_train_processed_filter, y_train)
    
    # Obtener las features seleccionadas por RFECV
    selected_wrap = X_train_processed_filter.columns[rfecv.support_]
    
    # Obtener los coeficientes del modelo ajustado por RFECV para las variables seleccionadas
    coefs = rfecv.estimator_.coef_.flatten()
    
    # Ordenar las variables seleccionadas por el valor absoluto de sus coeficientes
    indices_wrap = np.argsort(np.abs(coefs))[::-1]
    abs_coefs_sorted = np.abs(coefs)[indices_wrap]
    selected_wrap_sorted_by_coefs = selected_wrap[indices_wrap]
    
    # Calcular la suma acumulada de los valores absolutos de los coeficientes
    cumulative_wrap = np.cumsum(abs_coefs_sorted) / np.sum(abs_coefs_sorted)
    
    # Seleccionar variables hasta que la suma acumulada alcance 0.9 (90%)
    cutoff_wrap = np.searchsorted(cumulative_wrap, 0.90) + 1
    selected_wrap_90 = selected_wrap_sorted_by_coefs[:cutoff_wrap]
    
    # --- Resultados ---
    st.write("Número de variables para 90% importancia:")
    st.write(f"Filtrado (chi2): {len(selected_filter)} variables")
    st.write(f"Incrustado (Random Forest): {len(selected_embedded)} variables")
    st.write(f"Envoltura (RFECV coef): {len(selected_wrap_90)} variables")
    
    st.write("\nVariables seleccionadas por filtrado (chi2, 90% acumulado):")
    st.write(selected_filter.tolist())
    
    st.write("\nVariables seleccionadas por incrustado (Random Forest, 90% acumulado):")
    st.write(selected_embedded.tolist())
    
    st.write("\nVariables seleccionadas por envoltura (RFECV coef, 90% acumulado):")
    st.write(selected_wrap_90.tolist())
    
    # --- Graficas comparativas ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5)) 
    
    axes[0].bar(range(len(sorted_scores_filter)), sorted_scores_filter, color='skyblue')
    axes[0].set_xticks(range(len(sorted_features_filter)), sorted_features_filter, rotation=90)
    axes[0].set_ylabel("Puntuación (chi2)")
    axes[0].set_title('Filtrado (SelectKBest chi2)')
    axes[0].axvline(cutoff_filter-1, color='red', linestyle='--', label='90% acumulado')
    axes[0].legend()
    
    axes[1].bar(range(len(sorted_importances_embedded)), sorted_importances_embedded, color='lightgreen')
    axes[1].set_xticks(range(len(sorted_features_embedded)), sorted_features_embedded, rotation=90)
    axes[1].set_ylabel("Importancia de Característica")
    axes[1].set_title('Incrustado (Random Forest)')
    axes[1].axvline(cutoff_embedded-1, color='red', linestyle='--', label='90% acumulado')
    axes[1].legend()
    
    axes[2].bar(range(len(abs_coefs_sorted)), abs_coefs_sorted, color='salmon')
    axes[2].set_xticks(range(len(selected_wrap_sorted_by_coefs)), selected_wrap_sorted_by_coefs, rotation=90)
    axes[2].set_ylabel("Valor absoluto del Coeficiente")
    axes[2].set_title('Envoltura (RFECV coef)')
    axes[2].axvline(cutoff_wrap-1, color='red', linestyle='--', label='90% acumulado')
    axes[2].legend()
    
    fig.tight_layout()
    st.pyplot(fig)

    st.markdown("### Conclusión")

    st.markdown("""
    Teniendo en cuenta los tres métodos de selección de características de la imagen anterior (Filtrado con Chi2, Random Forest e Envoltura con RFECV), podemos concluir que existe una fuerte coincidencia entre los tres métodos en que las variables "Discomfort Eye-strain", "Redness in eye" y "Itchiness/Irritation in eye" son las más relevantes. Estas podrían considerarse las variables clave a conservar o utilizar para la realización de los modelos de clasificación para la variable "Dry Eye Disease".

    Aunque el método de Random Forest sugiere otras variables con cierta importancia, los métodos de filtrado y envoltura indican que esas tres variables explican al menos el 90% de la varianza relevante, por lo tanto y siguiendo el principio de parsimonia, se opta por generar una base de datos con sólo esas tres variables ya que se prefiere un modelo más simple si su rendimiento es similar al de uno más complejo.
    """)

    # Base de datos seleccionando las variables que aparecen en la gráfica de chi2 y de regresión logistica
    Base_X_train_final = X_train_processed_filter[selected_filter.tolist()]
    
    Base_X_train_final

with tab4:
    st.subheader("Modelos de clasificación para predecir si un paciente tiene enfermedad de ojos secos")
    st.markdown("""
    Teniendo en cuenta el método de selector features, se desarrollarán diferentes modelos de clasificación para predecir la variable "Eye Dry Disease" con el objetivo de identificar por medio de las variables seleccionadas si una persona se encuentra enferma o no de ojos secos. Cabe mencionar que como la variable que se quiere predecir se encuentra desbalanceada, se realizará un oversampling para balancear los datos
    """)
    
    # Codificar la base y_train con 0 y 1 
    y_train = y_train.map({'N': 0, 'Y': 1})

    # Codificar la variable y_test y seleccionar sólo las variables independientes seleccionadas de x_test para el análisis
    y_test = y_test.map({'N': 0, 'Y': 1})
    X_test = X_test[selected_filter.tolist()]

    # Lista de modelos de ensamble a probar
    models = {
        "RandomForest": RandomForestClassifier(class_weight='balanced', random_state=42),
        "ExtraTrees": ExtraTreesClassifier(class_weight='balanced', random_state=42),
        "HistGradientBoosting": HistGradientBoostingClassifier(random_state=42),
        "LogisticRegression": LogisticRegression(class_weight='balanced', random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        }
    
    # Diccionario de espacios de búsqueda por modelo
    param_grids = {
        "RandomForest": {
            'classifier__n_estimators': randint(50, 200),
            'classifier__max_depth': randint(5, 30),
            'classifier__min_samples_split': randint(2, 20),
            'classifier__min_samples_leaf': randint(1, 20),
            'classifier__max_features': [None, 'sqrt', 'log2'],
            'classifier__bootstrap': [True, False]
        },
        "ExtraTrees": {
            'classifier__n_estimators': randint(50, 200),
            'classifier__max_depth': randint(5, 30),
            'classifier__min_samples_split': randint(2, 20),
            'classifier__min_samples_leaf': randint(1, 20),
            'classifier__max_features': ['sqrt', 'log2', None],
            'classifier__bootstrap': [True, False]
        },
        "HistGradientBoosting": {
            'classifier__max_iter': randint(50, 200),
            'classifier__learning_rate': [0.01, 0.05, 0.1, 0.2],
            'classifier__max_depth': randint(2, 10),
            'classifier__max_leaf_nodes': randint(10, 50)
        },
        "LogisticRegression": {
            'classifier__penalty': ['l2', 'none'],
            'classifier__C': uniform(0.01, 10),
            'classifier__solver': ['lbfgs'] 
        },
        "XGBoost": {
            'classifier__n_estimators': randint(50, 200),
            'classifier__max_depth': randint(3, 10),
            'classifier__gamma': uniform(0, 5),
            'classifier__learning_rate': [0.01, 0.05, 0.1, 0.2],
            'classifier__subsample': uniform(0.5, 0.5),
        }
    }

    def get_model_scores(estimator, X_test):
        """Devuelve la probabilidad o decision_function para el modelo, o None si no existe."""
        try:
            clf = estimator.named_steps['classifier']
            
            if hasattr(clf, "predict_proba"):
                return estimator.predict_proba(X_test)
            elif hasattr(clf, "decision_function"):
                return estimator.decision_function(X_test)
            else:
                print(f"[WARNING] El modelo {clf.__class__.__name__} no tiene ni predict_proba ni decision_function.")
                return None
        except Exception as e:
            print(f"[ERROR] No se pudo obtener y_score: {e}")
            return None
    
    # Para almacenar resultados
    results = {}
    
    for name, model in models.items():
        print(f"\n===== Entrenando {name} =====")
    
        # Definir pipeline
        pipeline = ImbPipeline(steps=[
            ('sample', BorderlineSMOTE()),  # Oversampling para balancear
            ('classifier', model)
        ])
    
        # RandomizedSearchCV
        random_search = RandomizedSearchCV(
            pipeline,
            param_distributions=param_grids[name],
            n_iter=15,
            cv=3,
            verbose=1,
            n_jobs=-1,
            random_state=42,
            scoring='f1_macro'
        )
    
        random_search.fit(Base_X_train_final, y_train)
    
        # Predicciones
        y_pred = random_search.predict(X_test)
    
        # Reporte
        report = classification_report(y_test, y_pred, output_dict=True)
    
        results[name] = {
            "best_params": random_search.best_params_,
            "classification_report": report,
            "cv_results": pd.DataFrame(random_search.cv_results_)[['params', 'mean_test_score', 'std_test_score']]
        }
    
        print("Mejores hiperparámetros:", random_search.best_params_)
        print(classification_report(y_test, y_pred))
    
        # Matriz de Confusión
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        fig, ax = plt.subplots()
        disp.plot(ax=ax, cmap="Blues")
        ax.set_title(f"Matriz de Confusión - {name}")
        st.pyplot(fig)
    
        # Curva ROC (binaria / multiclase)
        y_score = get_model_scores(random_search.best_estimator_, X_test)
    
        if y_score is not None:
            classes = np.unique(y_test)
    
            if len(classes) > 2:
                # Multiclase
                y_bin = label_binarize(y_test, classes=classes)
    
                fig, ax = plt.subplots(figsize=(8, 8))
                plt.style.use('seaborn-v0_8-paper')
    
                fpr, tpr, roc_auc = {}, {}, {}
                colors = matplotlib.colormaps['Set2'].resampled(len(classes))
    
                for i, color in zip(range(len(classes)), colors.colors):
                    fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_score[:, i])
                    roc_auc[i] = auc(fpr[i], tpr[i])
                    ax.plot(fpr[i], tpr[i], color=color, lw=1.5, alpha=0.8,
                             label=f"Clase {classes[i]} (AUC={roc_auc[i]:.2f})")
    
                # Micro-average
                fpr["micro"], tpr["micro"], _ = roc_curve(y_bin.ravel(), y_score.ravel())
                roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
                ax.plot(fpr["micro"], tpr["micro"], label=f"Micro-average (AUC={roc_auc['micro']:.2f})",
                         color="deeppink", linestyle=":", linewidth=2, alpha=0.9)
    
                # Macro-average
                all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(classes))]))
                mean_tpr = np.zeros_like(all_fpr)
                for i in range(len(classes)):
                    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
                mean_tpr /= len(classes)
    
                fpr["macro"] = all_fpr
                tpr["macro"] = mean_tpr
                roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
                ax.plot(fpr["macro"], tpr["macro"], label=f"Macro-average (AUC={roc_auc['macro']:.2f})",
                         color="navy", linestyle="--", linewidth=2, alpha=0.9)
    
                ax.plot([0, 1], [0, 1], "k--", lw=1)
                ax.set_title(f"Curva ROC Multiclase - {name}", fontsize=16, fontweight='bold')
                ax.set_xlabel("False Positive Rate", fontsize=14)
                ax.set_ylabel("True Positive Rate", fontsize=14)
                ax.legend(loc="lower right", fontsize=12)
                ax.set_xlim([0.0, 1.0])
                ax.set_ylim([0.0, 1.05])
                ax.grid(True, linestyle='--', alpha=0.5)
                st.pyplot(fig)
    
            else:
                # Binaria
                # En caso de decision_function que retorne 1D, adaptamos:
                if y_score.ndim == 1 or y_score.shape[1] == 1:
                    scores_for_roc = y_score.ravel()
                else:
                    scores_for_roc = y_score[:, 1]
    
                fpr, tpr, _ = roc_curve(y_test, scores_for_roc)
                roc_auc = auc(fpr, tpr)
                RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc).plot()
                ax.set_title(f"Curva ROC Binaria - {name}")
                st.pyplot(fig)
    
        else:
            st.write(f"No se pudo calcular la curva ROC para {name} porque el modelo no devuelve probabilidades ni decision_function.")
