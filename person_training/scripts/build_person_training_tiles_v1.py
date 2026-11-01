"""Keep complete-person tiles and robot/background hard negatives, split by scene."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1];SOURCE=ROOT/'work/tennis-drive-v49'
OUT=ROOT/'outputs/tennis-ball-collector-suite/person-training-v1-20261024'
sys.path.insert(0,str(SOURCE))
from fixed_detection import tile_starts,tile_owns_pixel
manifest=json.loads((OUT/'manifest.json').read_text())
DEST=OUT/'tiles'
if DEST.exists():raise FileExistsError(DEST)
for split in ('train','tune'):
    (DEST/'images'/split).mkdir(parents=True);(DEST/'labels'/split).mkdir(parents=True)
items=list(manifest['items'])
smoke=ROOT/'outputs/tennis-ball-collector-suite/v42-random-dev731-20261024'
negative_paths=list(sorted(smoke.glob('person_*_rgb.png')))
negative_paths += [ROOT/f'outputs/tennis-ball-collector-suite/person-tile-context-r2/0_{camera}_rgb.png' for camera in ('A','B')]
for index,path in enumerate(negative_paths):
    items.append(dict(image=path.relative_to(ROOT).as_posix(),state_id=f'known_negative_{index}',camera_id='X',split='train',boxes_xyxy=[],robot_pixel_centres=[],all_tiles=True,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
rng=np.random.default_rng(2026102402);records=[]
for item in items:
    path=ROOT/item['image'];assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256']
    image=Image.open(path).convert('RGB');w,h=image.size
    boxes=[np.asarray(b) for b in item['boxes_xyxy']]
    tiles=[(x,y,min(w,x+1024),min(h,y+1024)) for y in tile_starts(h,1024,.2) for x in tile_starts(w,1024,.2)]
    positives=[];negatives=[];robot_negatives=[]
    for tile in tiles:
        t=np.asarray(tile);intersects=[np.all(np.minimum(b[2:],t[2:])-np.maximum(b[:2],t[:2])>0) for b in boxes]
        contained=[np.all(b[:2]>=t[:2]+8) and np.all(b[2:]<=t[2:]-8) for b in boxes]
        if any(intersects):
            if all((not hit) or contained[i] for i,hit in enumerate(intersects)):
                positives.append((tile,[b.tolist() for b,hit in zip(boxes,intersects) if hit]))
            # Partial humans never become negative labels.
        else:
            negatives.append(tile)
            if any(tile_owns_pixel(tile,p,(w,h),103) for p in item['robot_pixel_centres']):robot_negatives.append(tile)
    if boxes:assert positives, f'No complete tile for {path}'
    chosen=list(positives)
    if item.get('all_tiles'):chosen += [(t,[]) for t in negatives]
    else:
        selected=list(dict.fromkeys(robot_negatives))[:2]
        pool=[t for t in negatives if t not in selected]
        if pool:selected.append(pool[int(rng.integers(len(pool)))])
        chosen += [(t,[]) for t in selected]
    for index,(tile,labels) in enumerate(chosen):
        name=f'{item["state_id"]}_{item["camera_id"]}_{index:02d}'
        crop=image.crop(tile);dest=DEST/'images'/item['split']/f'{name}.png';crop.save(dest)
        lines=[]
        for box in labels:
            b=np.asarray(box)-np.array([tile[0],tile[1],tile[0],tile[1]])
            centre=(b[:2]+b[2:])/2/np.array(crop.size);size=(b[2:]-b[:2])/np.array(crop.size)
            values=np.r_[centre,size];assert np.all(values>0) and np.all(values<1)
            lines.append('0 '+' '.join(f'{v:.8f}' for v in values))
        (DEST/'labels'/item['split']/f'{name}.txt').write_text('\n'.join(lines)+'\n' if lines else '')
        records.append(dict(name=name,state_id=item['state_id'],split=item['split'],source_image=item['image'],tile=tile,people=len(labels),sha256=hashlib.sha256(dest.read_bytes()).hexdigest()))
train_scenes={r['state_id'] for r in records if r['split']=='train'}
tune_scenes={r['state_id'] for r in records if r['split']=='tune'}
assert not train_scenes & tune_scenes
report=dict(records=records,scene_split_disjoint=True,counts={split:dict(tiles=sum(r['split']==split for r in records),positive_tiles=sum(r['split']==split and r['people']>0 for r in records)) for split in ('train','tune')},
    original_reserved_validation_used=False,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(OUT/'tile-manifest.json').write_text(json.dumps(report,indent=2)+'\n')
# Absolute Linux path for the existing WSL trainer; no downloadable dataset URL.
linux='/mnt/c/'+DEST.resolve().as_posix().split(':',1)[1].lstrip('/') if ':' in str(DEST) else str(DEST)
(OUT/'dataset.yaml').write_text(f'path: {linux}\ntrain: images/train\nval: images/tune\nnames:\n  0: person\n')
print(json.dumps(report['counts']))
