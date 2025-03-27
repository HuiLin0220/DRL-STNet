import yaml

# Utility Functions
def load_config(config_path):
    """
    Loads the configuration file in YAML format.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        dict: Parsed configuration dictionary.
    """
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)