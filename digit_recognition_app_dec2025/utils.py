"""
Utility functions for digit recognition app
"""
import os
from constants import I18N_APP_NAME_DEFAULT


def get_app_name():
    """Get application name from environment variable or default"""
    return os.getenv("APP_NAME") or I18N_APP_NAME_DEFAULT

