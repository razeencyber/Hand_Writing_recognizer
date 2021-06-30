from pathlib import Path
import os
from decouple import config

BASE_DIR = Path(__file__).resolve(strict=True).parent.parent
DIR = os.path.join(BASE_DIR,'source', 'templates')
print(BASE_DIR)
print(DIR)