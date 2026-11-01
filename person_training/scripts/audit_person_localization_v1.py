"""Derive localization error from saved inference, without rerunning sealed images."""
from pathlib import Path
import hashlib,json,math,statistics
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/tennis-ball-collector-suite'
TRAIN=OUT/'person-training-v1-20261024';read=lambda p:json.loads(p.read_text())
expected={x['image']:x['people'] for x in read(OUT/'v47-person-render-set-20261024/manifest.json')['items']}
for episode,position in ((1,[-10.4,-3.5,0.]),(2,[10.4,3.5,0.])):
    for camera in ('A','B'):expected[f'outputs/tennis-ball-collector-suite/person-tile-context-r2/{episode}_{camera}_rgb.png']=[position]
results={}
for name in ('development','smoke','validation'):
    report=read(TRAIN/f'evaluation-{name}.json');errors=[]
    for row in report['rows']:
        if not row['expected']:continue
        assert row['expected']==1 and len(row['people'])==1
        error=math.dist(row['people'][0]['position_m'][:2],expected[row['image']][0][:2])
        errors.append(dict(image=row['image'],error_m=error))
    values=sorted(r['error_m'] for r in errors)
    results[name]=dict(count=len(values),median_m=statistics.median(values),maximum_m=max(values),
        p95_m=values[min(len(values)-1,math.ceil(.95*len(values))-1)],
        over_025m=sum(v>.25 for v in values),rows=errors,
        input_sha256=hashlib.sha256((TRAIN/f'evaluation-{name}.json').read_bytes()).hexdigest())
proof=dict(scope='Post-evaluation diagnostics from saved detections; no inference, model selection or threshold changes',sets=results)
(TRAIN/'localization-audit.json').write_text(json.dumps(proof,indent=2)+'\n')
print(json.dumps({k:{x:v for x,v in r.items() if x not in ('rows','input_sha256')} for k,r in results.items()},indent=2))
