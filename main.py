from app.api.api_v1.api_router import api_router
import os
import uvicorn
import logging
import pandas as pd
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException
from contextlib import asynccontextmanager
from app.core.config import settings
import sys

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Validar configuración necesaria


def validate_settings():
    required_settings = ["API_V1_STR", "PROJECT_NAME", "VERSION"]
    missing = [setting for setting in required_settings if not getattr(
        settings, setting, None)]
    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(
            f"Configuración incompleta. Faltan los siguientes valores: {missing_str}")

# Función para cargar y preparar recursos al iniciar la aplicación


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gestiona el ciclo de vida de la aplicación:
    1. Intenta cargar el modelo pre-entrenado
    2. Si no existe, intenta entrenar uno nuevo con el dataset
    3. Guarda el estado del modelo en app.state para uso en los endpoints
    """
    logger.info(f"Iniciando {settings.PROJECT_NAME} v{settings.VERSION}")

    try:
        # Validar configuración primero
        validate_settings()
        logger.info("Configuración validada correctamente")

        # Importar el modelo
        from app.models.StudentPerformanceModel import StudentPerformanceModel

        # Intentar cargar el modelo existente
        try:
            logger.info(
                f"Intentando cargar modelo desde: {settings.MODEL_PATH}")
            model = StudentPerformanceModel.load_from_file()
            logger.info(
                "Modelo cargado exitosamente desde archivo pre-entrenado")
            app.state.model = model

        except FileNotFoundError:
            # El modelo no existe, hay que entrenarlo con el dataset
            logger.warning(
                f"No se encontró un modelo en {settings.MODEL_PATH}, se entrenará uno nuevo")

            # Ruta completa al archivo CSV
            csv_path = os.path.join(
                settings.CSV_PATH, 'StudentPerformanceFactors.csv')
            logger.info(f"Cargando datos desde: {csv_path}")

            if not os.path.exists(csv_path):
                raise FileNotFoundError(
                    f"No se encontró el archivo de datos: {csv_path}")

            # Cargar dataset
            try:
                df = pd.read_csv(csv_path)
                logger.info(
                    f"Dataset cargado correctamente con {len(df)} registros")
            except Exception as e:
                raise Exception(f"Error al cargar el archivo CSV: {str(e)}")

            # Entrenar modelo
            logger.info("Entrenando nuevo modelo...")
            model, evaluation = StudentPerformanceModel.train_from_dataframe(
                df)

            # Guardar métricas de evaluación
            logger.info(
                f"Modelo entrenado - Métricas: MSE={evaluation['mse']:.4f}, R²={evaluation['r2']:.4f}")

            # Guardar el modelo entrenado
            saved_path = model.save_model()
            logger.info(f"Modelo guardado en: {saved_path}")

            # Guardar el modelo en el estado de la app
            app.state.model = model

    except Exception as e:
        logger.error(
            f"Error fatal durante la inicialización de la aplicación: {str(e)}", exc_info=True)
        # En producción, podrías querer permitir que la app inicie sin modelo
        # o usar un modelo de fallback básico
        raise

    # Todo correcto, ceder el control a la aplicación
    yield

    # Código de limpieza al cerrar la aplicación
    logger.info("Liberando recursos...")

# Crear la aplicación FastAPI
app = FastAPI(
    title=settings.PROJECT_NAME,
    description=settings.DESCRIPTION,
    version=settings.VERSION,
    lifespan=lifespan
)

# Configurar CORS
if settings.BACKEND_CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin)
                       for origin in settings.BACKEND_CORS_ORIGINS],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Manejadores de excepciones para respuestas consistentes


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Manejador personalizado para errores de validación.
    Formatea el error para que sea más legible y asegura que sea serializable a JSON.
    """
    logger.warning(f"Validación fallida: {exc.errors()}")

    # Crear una versión serializable de los errores
    formatted_errors = []
    for error in exc.errors():
        formatted_error = {
            "type": error.get("type", ""),
            "loc": " > ".join([str(loc) for loc in error.get("loc", [])]),
            "msg": error.get("msg", ""),
            "input": error.get("input", "")
        }
        # Evitar incluir el objeto ValueError directamente
        if "ctx" in error and isinstance(error["ctx"], dict) and "error" in error["ctx"]:
            # Si ctx.error es un objeto de excepción, solo incluir su mensaje como string
            if isinstance(error["ctx"]["error"], Exception):
                formatted_error["detail"] = str(error["ctx"]["error"])
            else:
                formatted_error["detail"] = error["ctx"]["error"]

        formatted_errors.append(formatted_error)

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": formatted_errors}
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    logger.warning(f"HTTP Exception: {exc.status_code} - {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Error no manejado: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Error interno del servidor"}
    )

# Importar routers
app.include_router(api_router, prefix=settings.API_V1_STR)


@app.get("/")
async def root():
    """Endpoint raíz que proporciona información básica sobre la API"""
    return {
        "message": "API de Predicción de Rendimiento Estudiantil",
        "version": settings.VERSION,
        "status": "online",
        "docs": "/docs",
        "redoc": "/redoc"
    }


@app.get("/health")
async def health_check():
    """Endpoint para verificar el estado de salud de la API"""
    return {
        "status": "healthy",
        "model_loaded": hasattr(app.state, "model") and app.state.model is not None
    }

if __name__ == "__main__":
    try:
        logger.info(f"Iniciando servidor en http://0.0.0.0:8000")
        uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
    except Exception as e:
        logger.critical(
            f"Error al iniciar el servidor: {str(e)}", exc_info=True)
        sys.exit(1)
