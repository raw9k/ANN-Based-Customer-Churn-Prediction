import ssl
import certifi
import sys

_original_create_default_context = ssl.create_default_context

def _patched_create_default_context(*args, **kwargs):
    if 'cafile' not in kwargs:
        kwargs['cafile'] = certifi.where()
    return _original_create_default_context(*args, **kwargs)

ssl.create_default_context = _patched_create_default_context
ssl._create_default_https_context = _patched_create_default_context

from streamlit.web.cli import main

if __name__ == "__main__":
    sys.argv = ["streamlit", "run", "app.py"] # Replace app.py with your script's name
    sys.exit(main())