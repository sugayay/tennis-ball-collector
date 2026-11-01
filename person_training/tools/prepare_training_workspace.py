"""Prepare a new training workspace without changing archived evidence or running training."""
from pathlib import Path
import argparse, hashlib, shutil
ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
parser = argparse.ArgumentParser()
parser.add_argument('--output', required=True, type=Path)
parser.add_argument('--render-new', action='store_true', help='Prepare fresh rendering instead of training from the recorded tiles')
args = parser.parse_args()
target = args.output.resolve()
if target.exists():
    raise FileExistsError('Choose a new workspace')
weights = REPO / 'models/yolo26n.pt'
if not weights.is_file() or hashlib.sha256(weights.read_bytes()).hexdigest() != '9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef':
    raise RuntimeError('Place and verify the official base model first, using the repository fetch_yolo_model.py')
data_rel = Path('outputs/tennis-ball-collector-suite/person-training-v1-20261024')
if not args.render_new:
    records = ROOT / data_rel / 'tiles/images/train'
    if not records.is_dir():
        raise RuntimeError('The supplied training tiles are missing')
target.mkdir(parents=True)
shutil.copytree(ROOT / 'scripts', target / 'scripts')
shutil.copytree(ROOT / 'work', target / 'work')
for name in ('tennis-drive-v49', 'tennis-drive-v50'):
    dest = target / 'work' / name / 'models'
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(weights, dest / 'yolo26n.pt')
suite = ROOT / 'outputs/tennis-ball-collector-suite'
for name in ('v42-random-dev731-20261024', 'person-tile-context-r2', 'v47-person-render-set-20261024'):
    shutil.copytree(suite / name, target / 'outputs/tennis-ball-collector-suite' / name)
if not args.render_new:
    dest = target / data_rel
    dest.mkdir(parents=True)
    for name in ('manifest.json', 'tile-manifest.json', 'predeclared.json'):
        shutil.copyfile(ROOT / data_rel / name, dest / name)
    shutil.copytree(ROOT / data_rel / 'tiles', dest / 'tiles')
    yaml_text = 'path: ' + (dest / 'tiles').as_posix() + '\ntrain: images/train\nval: images/tune\nnames:\n  0: person\n'
    (dest / 'dataset.yaml').write_text(yaml_text, encoding='utf-8')
print('Prepared a new workspace. No training or simulation has run.')
