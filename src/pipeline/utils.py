"""
Pipeline Utility Functions

This module provides common utility functions for configuration loading,
file operations, image processing, device management, and other helpers
used throughout the food spoilage detection pipeline.
"""

import os
import yaml
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime
import logging


# ============================================================================
# Configuration Loading and Management
# ============================================================================

def load_config(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load YAML configuration file with error handling.
    
    Args:
        path (str or Path): Path to YAML configuration file
    
    Returns:
        dict: Parsed configuration dictionary
    
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid
    
    Example:
        >>> config = load_config('configs/pipeline_config.yaml')
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config if config is not None else {}
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML file {path}: {e}")


def save_config(config: Dict[str, Any], path: Union[str, Path]) -> None:
    """
    Save configuration dictionary to YAML file.
    
    Args:
        config (dict): Configuration dictionary
        path (str or Path): Output path for YAML file
    
    Example:
        >>> save_config(config, 'output/config.yaml')
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple configuration dictionaries.
    Later configs override earlier ones.
    
    Args:
        *configs: Variable number of config dictionaries
    
    Returns:
        dict: Merged configuration
    
    Example:
        >>> base_config = load_config('base.yaml')
        >>> user_config = load_config('user.yaml')
        >>> config = merge_configs(base_config, user_config)
    """
    merged = {}
    for config in configs:
        merged = _deep_update(merged, config)
    return merged


def _deep_update(base: Dict, update: Dict) -> Dict:
    """Recursively update nested dictionaries."""
    result = base.copy()
    for key, value in update.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = value
    return result


def validate_config(config: Dict[str, Any], required_keys: List[str]) -> bool:
    """
    Validate that configuration contains required keys.
    
    Args:
        config (dict): Configuration dictionary
        required_keys (list): List of required key paths (use '.' for nested keys)
    
    Returns:
        bool: True if valid
    
    Raises:
        ValueError: If required keys are missing
    
    Example:
        >>> validate_config(config, ['models.yolo.path', 'models.resnet.path'])
    """
    missing_keys = []
    
    for key_path in required_keys:
        keys = key_path.split('.')
        value = config
        
        try:
            for key in keys:
                value = value[key]
        except (KeyError, TypeError):
            missing_keys.append(key_path)
    
    if missing_keys:
        raise ValueError(f"Missing required config keys: {missing_keys}")
    
    return True


# ============================================================================
# Path and File Operations
# ============================================================================

def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Create directory if it doesn't exist.
    
    Args:
        path (str or Path): Directory path
    
    Returns:
        Path: Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_project_root() -> Path:
    """
    Get the project root directory.
    
    Returns:
        Path: Project root path
    """
    current = Path(__file__).resolve()
    # Go up from src/pipeline/utils.py to project root
    return current.parent.parent.parent


def resolve_path(path: Union[str, Path], relative_to: Optional[Path] = None) -> Path:
    """
    Resolve path relative to project root or specified directory.
    
    Args:
        path (str or Path): Path to resolve
        relative_to (Path, optional): Base directory (default: project root)
    
    Returns:
        Path: Resolved absolute path
    """
    path = Path(path)
    
    if path.is_absolute():
        return path
    
    base = relative_to if relative_to else get_project_root()
    return (base / path).resolve()


def get_file_list(directory: Union[str, Path], 
                  extensions: Optional[List[str]] = None,
                  recursive: bool = False) -> List[Path]:
    """
    Get list of files in directory with optional extension filtering.
    
    Args:
        directory (str or Path): Directory to search
        extensions (list, optional): File extensions to include (e.g., ['.jpg', '.png'])
        recursive (bool): Search recursively in subdirectories
    
    Returns:
        list: List of Path objects
    
    Example:
        >>> images = get_file_list('data/images', ['.jpg', '.png'])
    """
    directory = Path(directory)
    
    if not directory.exists():
        return []
    
    if recursive:
        pattern = '**/*'
    else:
        pattern = '*'
    
    files = [f for f in directory.glob(pattern) if f.is_file()]
    
    if extensions:
        extensions = [ext.lower() if ext.startswith('.') else f'.{ext.lower()}' 
                     for ext in extensions]
        files = [f for f in files if f.suffix.lower() in extensions]
    
    return sorted(files)


# ============================================================================
# Device Management
# ============================================================================

def get_device(device: Optional[str] = None, verbose: bool = True) -> torch.device:
    """
    Get PyTorch device (CPU/CUDA) with automatic detection.
    
    Args:
        device (str, optional): Device string ('cuda', 'cpu', 'cuda:0', etc.)
                               If None, auto-detect
        verbose (bool): Print device information
    
    Returns:
        torch.device: PyTorch device
    
    Example:
        >>> device = get_device()  # Auto-detect
        >>> device = get_device('cuda:0')  # Specific GPU
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    device_obj = torch.device(device)
    
    if verbose:
        if device_obj.type == 'cuda':
            gpu_name = torch.cuda.get_device_name(device_obj.index or 0)
            print(f"Using device: {device} ({gpu_name})")
        else:
            print(f"Using device: {device}")
    
    return device_obj


def get_device_memory_info(device: Union[str, torch.device] = 'cuda') -> Dict[str, float]:
    """
    Get GPU memory information.
    
    Args:
        device (str or torch.device): Device to query
    
    Returns:
        dict: Memory info with 'allocated', 'cached', 'free', 'total' in GB
    """
    if not torch.cuda.is_available():
        return {}
    
    device = torch.device(device) if isinstance(device, str) else device
    
    if device.type != 'cuda':
        return {}
    
    device_idx = device.index or 0
    
    return {
        'allocated_gb': torch.cuda.memory_allocated(device_idx) / 1e9,
        'cached_gb': torch.cuda.memory_reserved(device_idx) / 1e9,
        'total_gb': torch.cuda.get_device_properties(device_idx).total_memory / 1e9,
    }


# ============================================================================
# Logging Setup
# ============================================================================

def setup_logger(name: str = 'pipeline',
                log_file: Optional[Union[str, Path]] = None,
                level: str = 'INFO',
                console_output: bool = True) -> logging.Logger:
    """
    Setup logging with file and console handlers.
    
    Args:
        name (str): Logger name
        log_file (str or Path, optional): Path to log file
        level (str): Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        console_output (bool): Whether to output to console
    
    Returns:
        logging.Logger: Configured logger
    
    Example:
        >>> logger = setup_logger('pipeline', 'logs/pipeline.log')
        >>> logger.info('Pipeline started')
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


# ============================================================================
# Data Serialization
# ============================================================================

def save_json(data: Any, path: Union[str, Path], indent: int = 2) -> None:
    """
    Save data to JSON file.
    
    Args:
        data: Data to save (must be JSON serializable)
        path (str or Path): Output file path
        indent (int): JSON indentation level
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_json(path: Union[str, Path]) -> Any:
    """
    Load data from JSON file.
    
    Args:
        path (str or Path): JSON file path
    
    Returns:
        Loaded data
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


# ============================================================================
# Image Processing Helpers
# ============================================================================

def get_image_extensions() -> List[str]:
    """
    Get list of supported image file extensions.
    
    Returns:
        list: Image extensions
    """
    return ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp']


def is_image_file(path: Union[str, Path]) -> bool:
    """
    Check if file is a supported image.
    
    Args:
        path (str or Path): File path
    
    Returns:
        bool: True if file is an image
    """
    return Path(path).suffix.lower() in get_image_extensions()


def crop_bbox(image: np.ndarray, 
              bbox: Tuple[int, int, int, int],
              padding: int = 0) -> np.ndarray:
    """
    Crop image using bounding box with optional padding.
    
    Args:
        image (np.ndarray): Input image (H, W, C)
        bbox (tuple): Bounding box (x1, y1, x2, y2)
        padding (int): Padding pixels to add around bbox
    
    Returns:
        np.ndarray: Cropped image
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox
    
    # Add padding
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)
    
    return image[y1:y2, x1:x2]


def calculate_iou(box1: Tuple[float, ...], box2: Tuple[float, ...]) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        box1 (tuple): First box (x1, y1, x2, y2)
        box2 (tuple): Second box (x1, y1, x2, y2)
    
    Returns:
        float: IoU value between 0 and 1
    """
    x1_max = max(box1[0], box2[0])
    y1_max = max(box1[1], box2[1])
    x2_min = min(box1[2], box2[2])
    y2_min = min(box1[3], box2[3])
    
    # Calculate intersection area
    intersection = max(0, x2_min - x1_max) * max(0, y2_min - y1_max)
    
    # Calculate union area
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection
    
    return intersection / union if union > 0 else 0


# ============================================================================
# Timing and Performance
# ============================================================================

class Timer:
    """
    Simple timer context manager for measuring execution time.
    
    Example:
        >>> with Timer('Processing'):
        ...     process_images()
        Processing completed in 2.35s
    """
    
    def __init__(self, name: str = 'Operation', verbose: bool = True):
        self.name = name
        self.verbose = verbose
        self.start_time = None
        self.elapsed = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        return self
    
    def __exit__(self, *args):
        self.elapsed = (datetime.now() - self.start_time).total_seconds()
        if self.verbose:
            print(f"{self.name} completed in {self.elapsed:.2f}s")


# ============================================================================
# Miscellaneous Helpers
# ============================================================================

def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility.
    
    Args:
        seed (int): Random seed value
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def format_time(seconds: float) -> str:
    """
    Format seconds into human-readable string.
    
    Args:
        seconds (float): Time in seconds
    
    Returns:
        str: Formatted time string
    
    Example:
        >>> format_time(3665.5)
        '1h 1m 5.5s'
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.1f}s"


def get_timestamp(format: str = '%Y%m%d_%H%M%S') -> str:
    """
    Get current timestamp as formatted string.
    
    Args:
        format (str): strftime format string
    
    Returns:
        str: Formatted timestamp
    
    Example:
        >>> timestamp = get_timestamp()  # '20260203_143052'
    """
    return datetime.now().strftime(format)


def count_parameters(model: torch.nn.Module, trainable_only: bool = False) -> int:
    """
    Count parameters in PyTorch model.
    
    Args:
        model (torch.nn.Module): PyTorch model
        trainable_only (bool): Count only trainable parameters
    
    Returns:
        int: Number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())
