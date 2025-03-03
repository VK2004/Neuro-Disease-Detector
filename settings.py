from pathlib import Path
import sys

# Get the absolute path of the current file
FILE = Path(__file__).resolve()
# Get the parent directory of the current file
ROOT = FILE.parent
# Add the root path to the sys.path list if it is not already there
if ROOT not in sys.path:
    sys.path.append(str(ROOT))
# Get the relative path of the root directory with respect to the current working directory
ROOT = ROOT.relative_to(Path.cwd())

# Sources
IMAGE = 'Image'


SOURCES_LIST = [IMAGE]

# Images config
IMAGES_DIR = ROOT / 'images'
DEFAULT_IMAGE = IMAGES_DIR / 'Original.jpg'
DEFAULT_DETECT_IMAGE = IMAGES_DIR / 'Detect.jpg'



# ML Model config
MODEL_DIR = ROOT / 'weights'
EX_MODEL = MODEL_DIR / 'neuro_disease_detector.pt'
#EX_MODEL = MODEL_DIR / 'ut.pt'
# In case of your custome model comment out the line above and
# Place your custom model pt file name at the line below 
# DETECTION_MODEL = MODEL_DIR / 'my_detection_model.pt'

PT_MODEL = MODEL_DIR / 'best.pt'

# Webcam
WEBCAM_PATH = 0
