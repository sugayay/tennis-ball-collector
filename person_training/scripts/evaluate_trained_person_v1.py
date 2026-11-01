"""Evaluate the trained person model with the unchanged runtime camera pipeline."""
from pathlib import Path
import argparse,hashlib,json,sys,time
import numpy as np
from PIL import Image,ImageDraw
import torch

ROOT=Path(__file__).resolve().parents[1];SOURCE=ROOT/'work/tennis-drive-v49'
OUT=ROOT/'outputs/tennis-ball-collector-suite';TRAIN=OUT/'person-training-v1-20261024'
sys.path.insert(0,str(SOURCE))
from fixed_detection import FixedBallDetector
parser=argparse.ArgumentParser();parser.add_argument('--set',choices=('development','smoke','validation'),required=True)
args=parser.parse_args()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
trained=read(TRAIN/'training-result.json');weights=Path(trained['best_weights'])
assert weights.is_file() and sha(weights)==trained['best_sha256']
output=TRAIN/f'evaluation-{args.set}.json'
if output.exists():raise FileExistsError(output)
if args.set=='validation':
    for name in ('development','smoke'):
        gate=read(TRAIN/f'evaluation-{name}.json')
        assert gate['all_supplied_images_pass'] and gate['weights_sha256']==trained['best_sha256'], 'Keep validation sealed until development passes'
manifest=read(OUT/'v47-person-render-set-20261024/manifest.json')
if args.set=='smoke':
    calibration=read(OUT/'v42-random-dev731-20261024/calibration.json');items=[]
    for path in sorted((OUT/'v42-random-dev731-20261024').glob('person_*_rgb.png')):
        items.append(dict(image=path.relative_to(ROOT).as_posix(),camera_id='B',people=[],state_id=path.stem))
    for episode in (0,1,2):
        for camera in ('A','B'):
            path=OUT/f'person-tile-context-r2/{episode}_{camera}_rgb.png'
            items.append(dict(image=path.relative_to(ROOT).as_posix(),camera_id=camera,state_id=f'smoke_{episode}',
                people=[] if episode==0 else [[-10.4,-3.5,0.] if episode==1 else [10.4,3.5,0.]]))
else:
    calibration=manifest['calibration'];items=[x for x in manifest['items'] if x['split']==args.set]
by_camera={c['id']:c for c in calibration}
fixed=read(SOURCE/'fixed_camera_config.json');vision=read(SOURCE/'vision_config.json')
assert fixed['person_confidence']==.75 and fixed['person_tile_size_pixels']==1024
fixed['person_model']=str(weights)
detector=FixedBallDetector(SOURCE/'models/yolo26n.pt',vision,fixed)
predeclared=dict(weights_sha256=sha(weights),set=args.set,images=len(items),threshold=.75,tile_size=1024,
    criteria='zero false positives and zero missed persons; unchanged 1m localization match',
    runtime_source_sha256=sha(SOURCE/'fixed_detection.py'))
marker=TRAIN/f'evaluation-{args.set}-predeclared.json'
if marker.exists():assert read(marker)==predeclared
else:marker.write_text(json.dumps(predeclared,indent=2)+'\n')
first=items[0]
detector.detect(np.asarray(Image.open(ROOT/first['image']).convert('RGB')),by_camera[first['camera_id']],.0335,tiled=True,include_balls=False)
torch.cuda.synchronize();rows=[];positive_previews=0
for item in items:
    path=ROOT/item['image']
    if item.get('sha256'):assert sha(path)==item['sha256']
    rgb=np.asarray(Image.open(path).convert('RGB'))
    torch.cuda.synchronize();started=time.perf_counter()
    result=detector.detect(rgb,by_camera[item['camera_id']],.0335,tiled=True,include_balls=False)
    torch.cuda.synchronize();elapsed=time.perf_counter()-started
    expected=[np.asarray(p) for p in item['people']];matched=set();false=result['unlocalized_person_count']
    for observed in result['persons']:
        distances=[np.linalg.norm(np.asarray(observed['position_m'][:2])-p[:2]) if i not in matched else 1e9 for i,p in enumerate(expected)]
        if distances and min(distances)<=1.:matched.add(int(np.argmin(distances)))
        else:false+=1
    row=dict(image=item['image'],state_id=item['state_id'],camera_id=item['camera_id'],sha256=sha(path),
        expected=len(expected),matched=len(matched),false_positive=false,missed=len(expected)-len(matched),
        people=result['persons'],elapsed_seconds=elapsed,person_tile_count=result['person_tile_count'])
    rows.append(row)
    # Actual detector boxes only. No expected positions are drawn into RGB.
    if (expected and positive_previews<4) or row['missed'] or false or len(rows)==1:
        preview=Image.fromarray(rgb);draw=ImageDraw.Draw(preview)
        for observed in result['persons']:
            draw.rectangle(observed['bbox_xyxy'],outline='#ffe45e',width=7)
        preview.thumbnail((1280,720))
        previews=TRAIN/f'evaluation-{args.set}-previews';previews.mkdir(exist_ok=True)
        preview.save(previews/f'{len(rows):03d}_{path.stem}.png')
        if expected:positive_previews+=1
    print(json.dumps({k:row[k] for k in ('image','matched','false_positive','missed','elapsed_seconds')}),flush=True)
positive={r['state_id'] for r in rows if r['expected']}
seen={r['state_id'] for r in rows if r['matched']}
report={**predeclared, 'scope':'offline synthetic RGB; known negative regression images were included in training; external development and validation images were not',
    'true_positives':sum(r['matched'] for r in rows),'false_negatives':sum(r['missed'] for r in rows),'false_positives':sum(r['false_positive'] for r in rows),
    'positive_states':len(positive),'positive_states_seen_in_any_camera':len(positive&seen),
    'wall_median_seconds':float(np.median([r['elapsed_seconds'] for r in rows])),
    'wall_max_seconds':max(r['elapsed_seconds'] for r in rows),'rows':rows,
    'all_supplied_images_pass':all(not r['false_positive'] and not r['missed'] for r in rows)}
output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='rows'}),flush=True)
