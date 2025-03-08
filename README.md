# IDENTIFICADOR DE PATRONES DE RENDIMIENTO Y BIENESTAR ESTUDIANTIL

## Tabla de contenidos

1. [Descripción](#descripción)
2. [Objetivos](#objetivos)
3. [Conjuntos de Datos](#conjuntos-de-datos)
4. [Arquitectura del proyecto](#arquitectura-del-proyecto)
5. [Proceso de Desarrollo](#proceso-de-desarrollo)
6. [Endpoints API](#endpoints-api)
7. [Casos de Uso](#casos-de-uso)
8. [Futura Evolución](#futura-evolución)
9. [Repositorios](#repositorios)
10. [Instalación y Uso](#instalación-y-uso)

## Descripción

El Identificador de Patrones de Rendimiento y Bienestar Estudiantil es una plataforma de análisis basada en Machine Learning diseñada para ayudar a instituciones educativas a identificar factores clave que influyen en el rendimiento académico y el bienestar psicológico de los estudiantes. 

El sistema analiza datos estudiantiles provenientes de diferentes fuentes para detectar patrones, predecir resultados y generar recomendaciones personalizadas que apoyen el desarrollo integral de los estudiantes.

Este proyecto está concebido como un módulo complementario para sistemas de gestión educativa existentes, enfocándose especialmente en departamentos de asesoramiento estudiantil, permitiendo implementar estrategias personalizadas de apoyo basadas en evidencia.

## Objetivos

- Identificar los parámetros más importantes que afectan el rendimiento académico y bienestar estudiantil
- Estudiar las relaciones e impactos entre diferentes factores para generar predicciones precisas
- Proporcionar un sistema escalable y personalizable para diferentes contextos educativos
- Ofrecer una API de fácil integración con sistemas educativos existentes
- Generar recomendaciones específicas para mejorar el rendimiento y bienestar del estudiante

## Conjuntos de Datos

El proyecto utiliza tres conjuntos de datos principales:

1. **Rendimiento Estudiantil**: Contiene factores relacionados con el desempeño académico general de los estudiantes (hábitos de estudio, asistencia, recursos, etc.)
2. **Score Académico**: Enfocado en calificaciones y métricas específicas de evaluación
3. **Indicadores de Depresión**: Métricas relacionadas con el bienestar psicológico y emocional de los estudiantes

Estos conjuntos de datos se encuentran en el directorio `/app/db/` de la aplicación:
- `StudentPerformanceFactors.csv`
- `ExamScore.csv`
- `StudentDepression.csv`

## Arquitectura del proyecto

```
└── academic-performance-api
    └── app
        └── api
            └── api_v1
                └── api_router.py
                └── endpoints
                    └── performance
                        └── predictions.py
                    └── score
                        └── predictions.py
        └── core
            └── config.py
            └── security.py
        └── db
            └── ExamScore.csv
            └── StudentDepression.csv
            └── StudentPerformanceFactors.csv
        └── models
            └── ExamScorePredictModel.py
            └── StudentPerformanceModel.py
            └── trained
        └── schemas
            └── exam_score_data.py
            └── student_performance_data.py
        └── services
    └── tests
        └── api
        └── models
    └── __init__.py
    └── main.py
    └── requirements.txt
```

El proyecto sigue una arquitectura modular y escalable:

- **api**: Contiene los endpoints y rutas de la API
- **core**: Configuraciones generales y seguridad
- **db**: Datos de entrenamiento y validación
- **models**: Modelos de Machine Learning y lógica de predicción
- **schemas**: Definición de estructuras de datos
- **services**: Lógica de negocio y servicios adicionales
- **tests**: Pruebas automatizadas

## Proceso de Desarrollo

### 1. Elección y Preprocesamiento de Datos

- Identificación de conjuntos de datos relevantes para el rendimiento académico y bienestar estudiantil
- Limpieza y normalización de datos (manejo de valores nulos, outliers, etc.)
- Ingeniería de características para mejorar la calidad predictiva
- Análisis exploratorio para identificar tendencias y correlaciones iniciales
- Preparación de conjuntos de entrenamiento y validación

### 2. Elección de modelos de Machine Learning

- Evaluación de diferentes algoritmos supervisados para predicción (regresión, clasificación)
- Selección basada en métricas de rendimiento (precisión, recall, F1-score)
- Implementación de técnicas de validación cruzada para evaluar robustez
- Optimización de hiperparámetros para cada modelo
- Comparación de rendimiento entre diferentes enfoques

### 3. Identificación de casos de uso

- Predicción de rendimiento académico basado en factores comportamentales
- Estimación de probabilidad de bajo rendimiento en exámenes
- Detección temprana de indicadores de depresión o problemas de bienestar
- Generación de recomendaciones personalizadas para mejora
- Análisis de impacto de distintas intervenciones

### 4. Desarrollo de API

- Implementación de endpoints RESTful para cada modelo
- Documentación de API con especificaciones OpenAPI/Swagger
- Validación de entradas y manejo de errores
- Implementación de seguridad y autenticación
- Optimización de rendimiento para solicitudes concurrentes

### 5. Implementación en módulo de prueba

- Desarrollo de interfaz de usuario en Laravel para demostración
- Integración con la API de predicción
- Implementación de flujos de asesoramiento estudiantil
- Pruebas de usabilidad y experiencia de usuario
- Documentación de proceso de integración

## Endpoints API

La API ofrece varios endpoints de predicción, cada uno especializado en un aspecto diferente:

1. **/prediction**: Genera predicciones individuales basadas en datos de estudiantes
2. **/prediction/batch**: Procesa múltiples predicciones en una sola solicitud
3. **/prediction/explain**: Proporciona predicciones con explicaciones sobre los factores más influyentes
4. **/prediction/recommendation**: Ofrece predicciones con recomendaciones personalizadas

Cada conjunto de datos (rendimiento, score, depresión) tiene su propio modelo y endpoints específicos.

## Casos de Uso

### Caso de uso principal: Asesoramiento Estudiantil

El sistema se integra en departamentos de asesoramiento estudiantil para:

1. Identificar tempranamente estudiantes en riesgo académico
2. Proporcionar recomendaciones personalizadas a consejeros y tutores
3. Monitorear la efectividad de intervenciones a lo largo del tiempo
4. Apoyar decisiones basadas en datos para mejorar políticas educativas

### Demostración: Módulo de Asesoramiento

Se ha desarrollado un prototipo funcional utilizando Laravel que demuestra cómo integrar la API en un sistema existente. Este módulo muestra las principales funcionalidades:

- Visualización de predicciones para asesores
- Interfaz para ingreso de datos estudiantiles
- Generación de recomendaciones personalizadas

## Futura Evolución

El proyecto está diseñado para crecer en las siguientes direcciones:

1. **Personalización avanzada**: Permitir que especialistas definan sus propios parámetros de evaluación
2. **Aprendizaje continuo**: Implementación de técnicas de aprendizaje incremental para mejorar modelos con nuevos datos
3. **Expansión de conjuntos de datos**: Incorporación de fuentes adicionales como datos socioeconómicos, actividades extracurriculares, etc.
4. **Análisis predictivo a largo plazo**: Seguimiento de trayectorias académicas completas
5. **Interfaces especializadas**: Desarrollo de dashboards específicos para diferentes roles (estudiantes, profesores, asesores)

## Repositorios

El proyecto está dividido en tres repositorios principales:

1. **Preprocesamiento de Datos y Entrenamiento de Modelos**
   - [https://github.com/d4na3l/sic-ia-project](https://github.com/d4na3l/sic-ia-project)
   - Contiene notebooks y scripts para el análisis exploratorio y entrenamiento inicial

2. **API de Rendimiento Académico**
   - [https://github.com/d4na3l/academic-performance-api](https://github.com/d4na3l/academic-performance-api)
   - Implementación completa de la API y modelos finales

3. **Interfaz de Asesoramiento (Caso de Uso)**
   - [https://github.com/d4na3l/student-performance-interface](https://github.com/d4na3l/student-performance-interface)
   - Prototipo funcional que demuestra la integración con sistemas educativos

## Instalación y Uso

### Requisitos

- Python 3.8+
- Dependencias listadas en `requirements.txt`

### Instalación

```bash
# Clonar el repositorio
git clone https://github.com/d4na3l/academic-performance-api.git
cd academic-performance-api

# Instalar dependencias
pip install -r requirements.txt

# Iniciar la aplicación
python main.py
```

### Uso básico

Para realizar una predicción simple:

```python
import requests
import json

# Datos de ejemplo
data = {
    "study_hours": 8,
    "attendance": 85,
    "previous_scores": 75,
    # Otros parámetros según el modelo
}

# Realizar petición
response = requests.post(
    "http://localhost:8000/api/v1/performance/prediction",
    headers={"Content-Type": "application/json"},
    data=json.dumps(data)
)

# Mostrar resultado
print(response.json())
```

Para más ejemplos, consulte la documentación de la API disponible en `/docs` después de iniciar el servidor.