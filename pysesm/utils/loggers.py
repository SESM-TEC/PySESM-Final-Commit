"""
Logging Configuration Utilities.

Provides helper functions to set up and configure Python loggers with standard
formatting and handlers (console, file) for the library.

Copyright (c) 2023-2025, Tecnológico de Costa Rica
All rights reserved.

This source code is licensed under the BSD 3-Clause License found in the
LICENSE file in the root directory of this source tree.

SPDX-License-Identifier: BSD-3-Clause
"""

import logging

# Variable to ensure that the configuration is made just once
_configured_loggers = {}

def setup_logger(name: str = 'pysesm', level: int = logging.INFO, 
                 log_file: str | None = None) -> logging.Logger:
    """
    Configure and return a logger instance.

    Ensures that handlers are not duplicated if the same logger is called several times.
    You can directly provide the desired logging level.

    Args:
        name (str): Logger name. Default: 'pysesm'.
        level (int): Minimum logging level (e.g. logging.INFO, logging.DEBUG, logging.WARNING).
                     Default:  logging.INFO.
        log_file: log file name

    Returns:
        logging.Logger: Configured logger instance.
    """
    # global _configured_loggers

    logger = logging.getLogger(name)
    logger.setLevel(level)  

    # If the logger is alredy configured, return
    if name in _configured_loggers:
        return logger
    
    # Avoid duplicated handlers
    if not logger.handlers:

        # Ensure the root logger has no handlers to avoid duplicated messages
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]: 
            root_logger.removeHandler(handler)
        
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Concole handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG) 
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # Configurar el handler de archivo SOLO SI se proporciona un nombre de archivo
        if log_file:
            try:
                file_handler = logging.FileHandler(log_file)
                file_handler.setLevel(logging.DEBUG) # Nivel bajo para que el logger principal filtre
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
            except Exception as e:
                # Loggear una advertencia si no se puede crear el archivo de log
                # Usamos el logger root temporalmente ya que nuestro logger aún no está completamente configurado
                root_logger.warning("Could not set up file handler for %s: %s",log_file, e)
        
        
        # Opcional: Desactivar la propagación al logger root
        logger.propagate = False 

    _configured_loggers[name] = logger
    return logger
