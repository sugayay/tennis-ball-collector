"""Local-only fine-tuning; existing ball model and external validation stay untouched."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,time
os.environ['WANDB_DISABLED']='true'
os.environ['YOLO_AUTOINSTALL']='false'
os.environ['YOLO_OFFLINE']='true'
import yaml
from ultralytics import YOLO,settings

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/tennis-ball-collector-suite/person-training-v1-20261024'
WEIGHTS=ROOT/'work/tennis-drive-v49/models/yolo26n.pt'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
original=sha(WEIGHTS)
for key in ('sync','wandb','mlflow','clearml','comet','neptune','raytune','tensorboard','dvc','hub'):
    if key in settings:settings.update({key:False})
tiles=json.loads((OUT/'tile-manifest.json').read_text())
assert tiles['scene_split_disjoint'] and not tiles['original_reserved_validation_used']
model=YOLO(str(WEIGHTS))
data=yaml.safe_load((OUT/'dataset.yaml').read_text())
# Preserve the pretrained class head. Only person labels are fitted and only the
# person outputs may be used at runtime; the separate original model finds balls.
assert model.names[0]=='person'
data['names']=model.names
(OUT/'training-dataset.yaml').write_text(yaml.safe_dump(data,sort_keys=False))
arguments=dict(data=str(OUT/'training-dataset.yaml'),epochs=40,patience=12,imgsz=1280,batch=4,
    device=0,workers=0,amp=False,deterministic=True,seed=20261024,optimizer='AdamW',lr0=.0005,lrf=.1,
    cos_lr=True,freeze=10,classes=[0],mosaic=.4,close_mosaic=8,degrees=0.,translate=.1,scale=.2,
    fliplr=.5,flipud=0.,hsv_h=.015,hsv_s=.4,hsv_v=.3,cache='ram',plots=False,
    project=str(OUT/'runs'),name='n-person-v1',exist_ok=False,save=True,save_period=-1)
plan=dict(scope='synthetic person-only candidate; no runtime adoption or real-world claim',
    input_weights_sha256=original,dataset_manifest_sha256=sha(OUT/'tile-manifest.json'),
    dataset_yaml_sha256=sha(OUT/'training-dataset.yaml'),arguments=arguments,
    script_sha256=sha(Path(__file__)),started_at_utc=datetime.now(timezone.utc).isoformat(),
    official_training_reference='https://docs.ultralytics.com/modes/train/',
    acceptance='zero false positives and zero missed persons on existing development and smoke; then reserved validation once at confidence 0.75 and 1024px full-image tiles')
(OUT/'training-plan.json').write_text(json.dumps(plan,indent=2)+'\n')
started=time.monotonic()
result=model.train(**arguments)
best=Path(model.trainer.best)
assert best.is_file() and sha(WEIGHTS)==original
proof=dict(status='training_completed_not_accepted',best_weights=str(best),best_sha256=sha(best),
    wall_seconds=time.monotonic()-started,original_ball_weights_unchanged=sha(WEIGHTS)==original,
    metrics={str(k):float(v) for k,v in result.results_dict.items()},
    completed_at_utc=datetime.now(timezone.utc).isoformat())
(OUT/'training-result.json').write_text(json.dumps(proof,indent=2)+'\n')
print(json.dumps(proof),flush=True)
