import yaml
import numpy as np

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
        
        
def adaptive_normalize(arr):
    max_p = 1 - 0.0001 * arr.shape[-1]
    arr = arr.astype(np.float32)
    PixelArr = arr[arr > 0]
    PixelArr.sort()
    max_v = PixelArr[int((len(PixelArr) - 1) * max_p + 0.5)]
    arr = np.clip(arr, 0, max_v) / max_v
    return arr