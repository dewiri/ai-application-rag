# tests/conftest.py

import sys
from pathlib import Path

# Projekt-Root (eine Ebene oberhalb von tests/) zum Import-Pfad hinzufügen
root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(root))