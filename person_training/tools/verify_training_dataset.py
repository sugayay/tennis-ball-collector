"""Verify the exact supplied training tiles and labels without model loading."""
from pathlib import Path
import hashlib, json
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'outputs/tennis-ball-collector-suite/person-training-v1-20261024'
records = json.loads((DATA / 'tile-manifest.json').read_text(encoding='utf-8'))['records']
for row in records:
    path = DATA / 'tiles/images' / row['split'] / (row['name'] + '.png')
    if hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
        raise RuntimeError('Tile hash mismatch: ' + row['name'])
    label = DATA / 'tiles/labels' / row['split'] / (row['name'] + '.txt')
    if not label.is_file():
        raise RuntimeError('Missing label: ' + row['name'])
print(json.dumps({'verified_tiles': len(records), 'all_png_sha256_match_historical_manifest': True, 'model_loaded': False, 'training_run': False}))
