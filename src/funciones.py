import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pickle

def eliminar_variables_alta_correlacion(df, target_col, umbral=0.9, umbral_target=0.9, cols_protegidas=[]):
    """
    Elimina variables del DataFrame que tienen una correlación mayor al umbral especificado, 
    excepto aquellas que están fuertemente correlacionadas con la variable objetivo o se indican como protegidas.
    """
    # Calcular la matriz de correlación en valor absoluto
    corr_matrix = df.corr().abs()
    
    # Seleccionar la parte superior de la matriz (sin duplicados)
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Identificar columnas para eliminar
    columnas_a_eliminar = []
    for col in upper.columns:
        # No evaluar la variable objetivo ni las columnas protegidas
        if col == target_col or col in cols_protegidas:
            continue
        # Si alguna correlación de esta columna con otra es mayor que el umbral
        if any(upper[col] > umbral):
            # Si la correlación con la variable target es alta, se protege la variable
            if corr_matrix.loc[target_col, col] > umbral_target:
                continue  # Se protege la variable porque tiene alta correlación con el target
            columnas_a_eliminar.append(col)
    
    print("Variables eliminadas:", columnas_a_eliminar)
    print("Número de variables eliminadas:", len(columnas_a_eliminar))
    
    # Eliminar las columnas identificadas y retornar el DataFrame resultante
    df_reducido = df.drop(columns=columnas_a_eliminar)
    return df_reducido



# Funcion para reemplazar los outliers de mis columnas
def reemplazar_outliers(column, df):
    stats = df[column].describe()
    iqr = stats["75%"] - stats["25%"]
    upper_limit = stats["75%"] + 1.5 * iqr
    lower_limit = stats["25%"] - 1.5 * iqr
    if lower_limit < 0:
        lower_limit = df[column].min()
    df[column] = df[column].apply(lambda x: x if x <= upper_limit else upper_limit)
    df[column] = df[column].apply(lambda x: x if x >= lower_limit else lower_limit)
    return df.copy(), [lower_limit, upper_limit]



# Funcion para normalizado de datos
def normalizar_datos(X_train, X_test, scaler_filename, numeric_columns= None ):
    # Si no se especifica, usamos todas las columnas
    if numeric_columns is None:
        numeric_columns = list(X_train.columns)

    scaler = StandardScaler()
    scaler.fit(X_train[numeric_columns])

    # Guardamos el scaler en un archivo para usarlo posteriormente
    with open(scaler_filename, "wb") as file:
        pickle.dump(scaler, file)

    X_train_num = pd.DataFrame(scaler.transform(X_train[numeric_columns]), 
                               index=X_train.index, columns=numeric_columns)
    X_test_num = pd.DataFrame(scaler.transform(X_test[numeric_columns]), 
                              index=X_test.index, columns=numeric_columns)
    X_train_norm = X_train.copy()
    X_test_norm = X_test.copy()
    X_train_norm[numeric_columns] = X_train_num
    X_test_norm[numeric_columns] = X_test_num
    return X_train_norm, X_test_norm




# Funcion para escalado Min/Max
def escalado_min_max(X_train, X_test, scaler_filename, numeric_columns=None):
    # Si no se especifica, usamos todas las columnas
    if numeric_columns is None:
        numeric_columns = list(X_train.columns)
    
    scaler = MinMaxScaler()
    scaler.fit(X_train[numeric_columns])
    
    # Guardamos el scaler en un archivo para usarlo posteriormente
    with open(scaler_filename, "wb") as file:
        pickle.dump(scaler, file)
    
    X_train_num = pd.DataFrame(scaler.transform(X_train[numeric_columns]), 
                               index=X_train.index, columns=numeric_columns)
    X_test_num = pd.DataFrame(scaler.transform(X_test[numeric_columns]), 
                              index=X_test.index, columns=numeric_columns)
    
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[numeric_columns] = X_train_num
    X_test_scaled[numeric_columns] = X_test_num
    
    return X_train_scaled, X_test_scaled

