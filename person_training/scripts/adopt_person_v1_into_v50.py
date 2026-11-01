"""Gate candidate adoption on frozen offline checks, retaining original ball weights."""
from pathlib import Path
import hashlib,json,shutil
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/tennis-ball-collector-suite'
TRAIN=OUT/'person-training-v1-20261024';SOURCE=ROOT/'work/tennis-drive-v50'
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert not (OUT/'v50-acceptance-queue-20261024.json').exists(),'Never alter a running or evaluated source'
trained=read(TRAIN/'training-result.json')
for name in ('development','smoke','validation'):
    report=read(TRAIN/f'evaluation-{name}.json')
    assert report['all_supplied_images_pass'] and report['weights_sha256']==trained['best_sha256']
    assert report['threshold']==.75 and report['tile_size']==1024
path=trained['best_weights']
if path.startswith('/mnt/c/') and ':' in str(ROOT):path='C:/'+path[len('/mnt/c/'):]
weights=Path(path);assert sha(weights)==trained['best_sha256']
assert sha(SOURCE/'models/yolo26n.pt')==read(TRAIN/'training-plan.json')['input_weights_sha256']
target=SOURCE/'models/yolo26n-person-synthetic-v1.pt';assert not target.exists()
shutil.copy2(weights,target);assert sha(target)==trained['best_sha256']
cfg=SOURCE/'fixed_camera_config.json';fixed=read(cfg)
assert 'person_model' not in fixed
fixed['person_model']=target.name;cfg.write_text(json.dumps(fixed,indent=2)+'\n')
provenance=dict(person_model_sha256=sha(target),original_ball_model_sha256=sha(SOURCE/'models/yolo26n.pt'),
    evaluations={name:sha(TRAIN/f'evaluation-{name}.json') for name in ('development','smoke','validation')},
    scope='Candidate only. Frozen synthetic offline sets passed. Collection integration and real people not verified.')
(SOURCE/'models/person-model-provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
preparation=read(ROOT/'work/v50-preparation.json')
preparation['person_model']='Synthetic person-v1 accepted for integration after three offline gates; ball model unchanged'
preparation['source_sha256']={p.relative_to(SOURCE).as_posix():sha(p) for p in sorted(SOURCE.rglob('*')) if p.is_file() and '__pycache__' not in p.parts}
(ROOT/'work/v50-preparation.json').write_text(json.dumps(preparation,indent=2)+'\n')
print(json.dumps(provenance,indent=2))
