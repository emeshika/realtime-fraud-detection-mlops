import logging
import yaml

def load_yaml(file_path: str) -> dict:
    """Safely loads a YAML configuration file."""
    with open(file_path, "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            logging.error(f"Error loading YAML file: {exc}")
            raise exc

def get_logger(name: str) -> logging.Logger:
    """Returns a standardized logger instance."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger