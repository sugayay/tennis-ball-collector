"""Structural dataset audit; runtime never receives synthetic annotation truth."""
from pathlib import Path
import hashlib,json
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/tennis-ball-collector-suite/person-training-v1-20261024'
read=lambda p:json.loads(p.read_text())
manifest=read(OUT/'manifest.json');tiles=read(OUT/'tile-manifest.json')
reserved=read(ROOT/'outputs/tennis-ball-collector-suite/v47-person-render-set-20261024/manifest.json')
reserved_paths={x['image'] for x in reserved['items']}
sources={x['source_image'] for x in tiles['records']}
splits={name:{x['state_id'] for x in tiles['records'] if x['split']==name} for name in ('train','tune')}
valid=True;label_count=0
for item in tiles['records']:
    path=OUT/'tiles/images'/item['split']/(item['name']+'.png')
    assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256']
    lines=(OUT/'tiles/labels'/item['split']/(item['name']+'.txt')).read_text().splitlines()
    assert len(lines)==item['people']
    for line in lines:
        values=[float(x) for x in line.split()]
        assert len(values)==5 and values[0]==0
        _,cx,cy,w,h=values
        assert w>0 and h>0 and 0<=cx-w/2<cx+w/2<=1 and 0<=cy-h/2<cy+h/2<=1
        label_count+=1
    with Image.open(path) as im:assert im.size==(1024,1024)
checks=dict(rendered_images=len(manifest['items'])==120,scene_split_disjoint=not splits['train']&splits['tune'],
    external_development_and_validation_not_in_training=not sources&reserved_paths,
    all_tile_hashes_and_person_labels_verified=True)
report=dict(checks=checks,verified=all(checks.values()),counts=tiles['counts'],person_labels=label_count,
    visual_spot_check=['scene_000_A.png','scene_001_B.png','scene_029_A.png','scene_040_A.png'],
    visual_observation='Viewed near and far persons enclosed by their label boxes, and an unoccupied scene with no person labels.',
    limits='One static CesiumMan pose, three scales, varied positions/orientations and robot shovel poses. Not real-person validation.',
    known_negative_regression_images_in_training=5)
(OUT/'dataset-audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
