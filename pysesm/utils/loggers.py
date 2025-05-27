'''
Copyright (C) 2023-2025 Tecnológico de Costa Rica

Helpers for setting up loggers

Authors: The SESM Team 

License: 
'''

import logging

# Variable to ensure that the configuration is made just once
_logger_configured = False

def setup_logger(name: str = 'pysesm', level: int = logging.INFO) -> logging.Logger:
    """
    Configure and return a logger instance.

    Ensures that handlers are not duplicated if the same logger is called several times.
    You can directly provide the desired logging level.

    Args:
        name (str): El nombre del logger a configurar. Por defecto es 'pysesm'.
        level (int): El nivel mínimo de logging a procesar (ej., logging.INFO, logging.DEBUG, logging.WARNING).
                     Por defecto es logging.INFO.

    Returns:
        logging.Logger: La instancia de logger configurada.
    """
    global _logger_configured

    logger = logging.getLogger(name)
    logger.setLevel(level) # Establece el nivel para este logger específico

    # Si ya hemos configurado handlers para este logger o para el root, no añadir más
    if not logger.handlers and not _logger_configured: # Comprobar también la flag global
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Crear y configurar el handler de consola
        console_handler = logging.StreamHandler()
        # El nivel del handler debe ser al menos tan permisivo como el nivel general que deseas ver.
        # Aquí lo ponemos en DEBUG para que no filtre nada antes de que el logger principal lo haga.
        # O podrías ponerlo en level, pero si quieres que sea global lo mejor es DEBUG.
        console_handler.setLevel(logging.DEBUG) 
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # Crear y configurar el handler de archivo
        file_handler = logging.FileHandler("log.log")
        file_handler.setLevel(logging.DEBUG) # Igual que el de consola
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Establecer la flag a True después de la primera configuración
        _logger_configured = True
        
        # Opcional: Desactivar la propagación al logger root si no quieres que sus handlers (si los tiene) también manejen estos mensajes
        # logger.propagate = False 

    return logger

# Si quieres que el logger root no haga nada por defecto, puedes configurarlo explícitamente
# Esto evita que logging.basicConfig() (si se usara en otro lado) añada handlers al root
# logging.getLogger().setLevel(logging.WARNING)
# logging.getLogger().addHandler(logging.NullHandler()) # Añadir un handler nulo para que no haga nada
