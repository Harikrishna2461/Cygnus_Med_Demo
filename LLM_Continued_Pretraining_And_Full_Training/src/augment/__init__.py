"""
Medical CPT Data Augmentation Pipeline

This module provides safe augmentation techniques for medical training data:
- back_translate: Back-translation via intermediate language
- extractive_restatement: Key point extraction from medical texts
- merge_dataset: Merging and deduplication
- validate_dataset: Comprehensive validation before training
"""

__version__ = "1.0.0"
__all__ = [
    "back_translate",
    "extractive_restatement",
    "merge_dataset",
    "validate_dataset"
]
