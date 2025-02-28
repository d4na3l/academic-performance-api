# app/models/StudentPerformanceModel.py

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder, FunctionTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from app.core.config import settings


def get_feature_names(column_transformer: ColumnTransformer) -> List[str]:
    """
    Extrae los nombres de las columnas del ColumnTransformer.
    """
    output_features = []
    for name, transformer, columns in column_transformer.transformers_:
        if transformer == 'drop' or transformer is None:
            continue
        if transformer == 'passthrough':
            output_features.extend(columns)
        else:
            if isinstance(transformer, Pipeline):
                try:
                    names = transformer[-1].get_feature_names_out(columns)
                except AttributeError:
                    names = columns
            else:
                try:
                    names = transformer.get_feature_names_out(columns)
                except AttributeError:
                    names = columns
            output_features.extend(names)
    return output_features

# -----------------------
# Definición de preprocesamiento
# -----------------------


# [El código de preprocesamiento se mantiene igual]
# Variables numéricas
numeric_features = ["Hours_Studied", "Attendance", "Sleep_Hours", "Previous_Scores",
                    "Tutoring_Sessions", "Physical_Activity"]
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

# Variables ordinales (Low, Medium, High)
ordinal_features = ["Access_to_Resources", "Parental_Involvement",
                    "Motivation_Level", "Family_Income"]
ordinal_categories = [["Low", "Medium", "High"]] * len(ordinal_features)
ordinal_transformer = Pipeline(steps=[
    ('encoder', OrdinalEncoder(categories=ordinal_categories,
                               handle_unknown='use_encoded_value', unknown_value=-1)),
    ('scaler', StandardScaler())
])

# Variable especial: Peer_Influence (mapeo: Negative:-1, Neutral:0, Positive:1)
peer_influence_mapping = {"Negative": -1, "Neutral": 0, "Positive": 1}


def map_peer_influence(X):
    """
    Mapea los valores de Peer_Influence según el diccionario.
    """
    X_array = np.array(X)
    if X_array.ndim == 1:
        return np.array([peer_influence_mapping.get(val, 0) for val in X_array])
    else:
        return np.array([[peer_influence_mapping.get(val, 0) for val in row] for row in X_array])


peer_influence_feature = ["Peer_Influence"]
peer_influence_transformer = Pipeline(steps=[
    ('mapper', FunctionTransformer(map_peer_influence, validate=False)),
    ('scaler', StandardScaler())
])

# Variables binarias (Yes/No)
binary_features = ["Extracurricular_Activities",
                   "Internet_Access", "Learning_Disabilities"]
binary_categories = [["No", "Yes"]] * len(binary_features)
binary_transformer = Pipeline(steps=[
    ('encoder', OrdinalEncoder(categories=binary_categories))
])

# Variables nominales (School_Type y Gender)
nominal_features = ["School_Type", "Gender"]
nominal_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
])

# Construir el preprocesador completo
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('ord', ordinal_transformer, ordinal_features),
        ('peer', peer_influence_transformer, peer_influence_feature),
        ('bin', binary_transformer, binary_features),
        ('nom', nominal_transformer, nominal_features)
    ]
)

# -----------------------
# Clase para el modelo de rendimiento estudiantil
# -----------------------


class StudentPerformanceModel:
    """
    Modelo de Machine Learning para predecir el rendimiento estudiantil.
    """

    def __init__(self):
        """
        Inicializa la clase sin cargar ni entrenar el modelo.
        """
        self.pipeline = None
        self.is_trained = False

    @classmethod
    def load_from_file(cls, model_path: Optional[str] = None) -> 'StudentPerformanceModel':
        """
        Carga un modelo previamente entrenado desde un archivo.

        Args:
            model_path: Ruta al archivo del modelo. Si es None, usa settings.MODEL_PATH.

        Returns:
            Una instancia de StudentPerformanceModel con el modelo cargado.

        Raises:
            FileNotFoundError: Si el archivo no existe.
            Exception: Si hay un error al cargar el modelo.
        """
        model_path = model_path or settings.MODEL_PATH

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"No se encontró el modelo en {model_path}.")

        try:
            instance = cls()
            instance.pipeline = joblib.load(model_path)
            instance.is_trained = True
            return instance
        except Exception as e:
            raise Exception(
                f"Error al cargar el modelo desde {model_path}: {str(e)}")

    @classmethod
    def train_from_dataframe(cls, df: pd.DataFrame) -> 'StudentPerformanceModel':
        """
        Entrena un nuevo modelo usando un DataFrame.

        Args:
            df: DataFrame que debe incluir la columna 'Exam_Score' como objetivo.

        Returns:
            Una instancia de StudentPerformanceModel con el modelo entrenado.

        Raises:
            ValueError: Si el DataFrame no tiene la columna 'Exam_Score'.
            Exception: Si hay un error durante el entrenamiento.
        """
        if "Exam_Score" not in df.columns:
            raise ValueError(
                "El DataFrame debe contener la columna 'Exam_Score'")

        try:
            instance = cls()
            X = df.drop("Exam_Score", axis=1)
            y = df["Exam_Score"]

            # Dividir en entrenamiento y prueba
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, shuffle=True)

            # Crear y entrenar el pipeline
            instance.pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('regressor', LinearRegression())
            ])

            instance.pipeline.fit(X_train, y_train)
            instance.is_trained = True

            # Evaluar el modelo
            y_pred = instance.pipeline.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            evaluation = {
                "mse": mse,
                "r2": r2,
                "test_size": len(X_test),
                "train_size": len(X_train)
            }

            return instance, evaluation

        except Exception as e:
            raise Exception(
                f"Error durante el entrenamiento del modelo: {str(e)}")

    def save_model(self, model_path: Optional[str] = None) -> str:
        """
        Guarda el modelo entrenado en un archivo.

        Args:
            model_path: Ruta donde guardar el modelo. Si es None, usa settings.MODEL_PATH.

        Returns:
            La ruta donde se guardó el modelo.

        Raises:
            ValueError: Si el modelo no ha sido entrenado.
            Exception: Si hay un error al guardar el modelo.
        """
        if not self.is_trained or self.pipeline is None:
            raise ValueError(
                "No se puede guardar un modelo que no ha sido entrenado.")

        model_path = model_path or settings.MODEL_PATH

        # Asegurarse de que el directorio existe
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        try:
            joblib.dump(self.pipeline, model_path)
            return model_path
        except Exception as e:
            raise Exception(
                f"Error al guardar el modelo en {model_path}: {str(e)}")

    def predict(self, data):
        """
        Predice el puntaje de examen para uno o varios estudiantes.

        Args:
            data: Diccionario o DataFrame con las características del estudiante.

        Returns:
            Predicción del puntaje de examen o array de predicciones.

        Raises:
            ValueError: Si el modelo no ha sido entrenado.
        """
        if not self.is_trained or self.pipeline is None:
            raise ValueError(
                "El modelo debe ser entrenado o cargado antes de realizar predicciones.")

        # Si es un diccionario, conviértelo a DataFrame de una sola fila
        if isinstance(data, dict):
            df_input = pd.DataFrame([data])
            prediction = self.pipeline.predict(df_input)
            return prediction[0]

        # Si ya es un DataFrame, úsalo directamente
        elif isinstance(data, pd.DataFrame):
            predictions = self.pipeline.predict(data)
            return predictions

        else:
            raise ValueError(
                "Los datos deben ser un diccionario o un DataFrame.")

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Extrae la importancia de las características del modelo.

        Returns:
            DataFrame con las características y su importancia.

        Raises:
            ValueError: Si el modelo no ha sido entrenado.
        """
        if not self.is_trained or self.pipeline is None:
            raise ValueError(
                "El modelo debe ser entrenado o cargado para obtener la importancia de características.")

        preproc = self.pipeline.named_steps['preprocessor']
        feature_names = get_feature_names(preproc)
        coefs = self.pipeline.named_steps['regressor'].coef_

        importance_df = pd.DataFrame({
            'feature': feature_names,
            'coefficient': coefs,
            'importance': np.abs(coefs)
        }).sort_values('importance', ascending=False)

        return importance_df

    def preprocess_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocesa datos de entrada y los convierte al formato esperado por el modelo.
        """
        processed_data = {}

        # Mapas para variables categóricas
        level_map = {"bajo": "Low", "medio": "Medium", "alto": "High",
                     "low": "Low", "medium": "Medium", "high": "High"}
        yesno_map = {"no": "No", "sí": "Yes", "si": "Yes", "yes": "Yes"}

        # Variables numéricas
        processed_data["Hours_Studied"] = float(data.get("hours_studied", 0))
        processed_data["Attendance"] = float(data.get("attendance", 0))
        processed_data["Sleep_Hours"] = float(data.get("sleep_hours", 0))
        processed_data["Previous_Scores"] = float(
            data.get("previous_scores", 0))
        processed_data["Tutoring_Sessions"] = float(
            data.get("tutoring_sessions", 0))
        processed_data["Physical_Activity"] = float(
            data.get("physical_activity", 0))

        # Variables ordinales
        parental = data.get("parental_involvement", "medium").lower()
        processed_data["Parental_Involvement"] = level_map.get(
            parental, "Medium")

        resources = data.get("access_to_resources", "medium").lower()
        processed_data["Access_to_Resources"] = level_map.get(
            resources, "Medium")

        motivation = data.get("motivation_level", "medium").lower()
        processed_data["Motivation_Level"] = level_map.get(
            motivation, "Medium")

        income = data.get("family_income", "medium").lower()
        processed_data["Family_Income"] = level_map.get(income, "Medium")

        # Variables binarias
        extracurricular = data.get("extracurricular_activities", "no").lower()
        processed_data["Extracurricular_Activities"] = yesno_map.get(
            extracurricular, "No")

        internet = data.get("internet_access", "no").lower()
        processed_data["Internet_Access"] = yesno_map.get(internet, "No")

        disabilities = data.get("learning_disabilities", "no").lower()
        processed_data["Learning_Disabilities"] = yesno_map.get(
            disabilities, "No")

        # Variable especial: Peer_Influence
        peer_mapping = {
            "negativa": "Negative", "negative": "Negative",
            "neutral": "Neutral", "neutra": "Neutral",
            "positiva": "Positive", "positive": "Positive"
        }
        peer = data.get("peer_influence", "neutral").lower()
        processed_data["Peer_Influence"] = peer_mapping.get(peer, "Neutral")

        # Variables nominales
        school_type_map = {
            "publica": "Public", "pública": "Public", "public": "Public",
            "privada": "Private", "private": "Private"
        }
        school = data.get("school_type", "public").lower()
        processed_data["School_Type"] = school_type_map.get(school, "Public")

        gender_map = {
            "masculino": "Male", "male": "Male", "m": "Male",
            "femenino": "Female", "female": "Female", "f": "Female"
        }
        gender = data.get("gender", "male").lower()
        processed_data["Gender"] = gender_map.get(gender, "Male")

        return processed_data
