import types
import os
import sys
from functools import wraps
from contextlib import contextmanager

def load_settings(settings_path="settings.json"):
    import json
    with open(settings_path, "r") as f:
        settings = json.load(f)
    return settings

@contextmanager
def silence_stdio():
    devnull = open(os.devnull, "w", encoding="utf-8")
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    try:
        sys.stdout = devnull
        sys.stderr = devnull
        yield
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        devnull.close()


def silence():
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)

            # If function returns generator → wrap iteration
            if hasattr(result, "__iter__") and not isinstance(result, (bytes, str)):
                def silent_generator():
                    with silence_stdio():
                        for item in result:
                            yield item
                return silent_generator()
            else:
                with silence_stdio():
                    return result

        return wrapper
    return decorator