"""Replay raw120Hz measurements. No recorded containment bool is trusted."""
from pathlib import Path
from collections import deque
import argparse,json,hashlib,math
import numpy as np
R=Path(__file__).resolve().parents[1]
def contained_ids(row):
 p=np.asarray(row['basket_position_m'],float);rot=np.asarray(row['basket_rotation'],float)
 bp=np.asarray(row['base_position_m'],float);br=np.asarray(row['base_rotation'],float)
 q=np.asarray(row['q'],float);v=np.asarray(row['v'],float)
 assert p.shape==(3,) and rot.shape==br.shape==(3,3) and bp.shape==(3,) and q.shape==v.shape==(4,)
 assert np.isfinite(np.concatenate((p,rot.ravel(),bp,br.ravel(),q,v))).all()
 assert np.max(np.abs(rot.T@rot-np.eye(3)))<1e-5
 c,s=math.cos(q[2]),math.sin(q[2]);expected_p=bp+br@np.array([-.08-q[1],0.,.04+q[0]])
 expected_r=br@np.array([[c,0.,-s],[0.,1.,0.],[s,0.,c]])
 assert np.linalg.norm(p-expected_p)<=2e-5 and np.max(np.abs(rot-expected_r))<=2e-5
 ids=[]
 for b in row['balls']:
  world=np.asarray(b['world_position_m'],float);assert world.shape==(3,) and np.isfinite(world).all()
  x,y,z=np.array([-.08,0.,.04])+rot.T@(world-p)
  if -.1945<=x<=.0345 and -.1045<=y<=.1045 and .080<=z<=.147:ids.append(b['id'])
 assert len(ids)==len(set(ids));return set(ids)
def interval_hold(rows):
 tail=list(rows)[-361:]
 valid=len(tail)==361 and abs(tail[-1]['seconds']-tail[0]['seconds']-3.)<1e-6
 ids=set.intersection(*(r['ids'] for r in tail)) if valid else set()
 return dict(valid_three_second_interval=valid,samples=len(tail),interval=[tail[0]['seconds'],tail[-1]['seconds']] if tail else None,ids=sorted(ids))
def audit(folder):
 report=json.loads((folder/'mission_check.json').read_text(encoding='utf8'))
 state={rid:dict(previous=None,count=0,tail=deque(maxlen=361),hold=[],cycles={},last_cycle=0) for rid in ('A','B')}
 trace=folder/'actual-basket-retention-trace.jsonl';before=hashlib.sha256(trace.read_bytes()).hexdigest();errors=[]
 try:
  for line in trace.open(encoding='utf8'):
   raw=json.loads(line);rid=raw['robot'];st=state[rid];row=dict(seconds=raw['seconds'],ids=contained_ids(raw));prev=st['previous']
   assert abs(row['seconds']-st['count']/120)<1e-6
   st['count']+=1;st['tail'].append(row)
   if raw['previous_stage']=='retain':
    if not st['hold'] and prev is not None:st['hold'].append(prev)
    st['hold'].append(row)
   if raw['cycle']>st['last_cycle']:
    assert raw['cycle']==st['last_cycle']+1
    st['cycles'][raw['cycle']]=interval_hold(st['hold']);st['hold']=[];st['last_cycle']=raw['cycle']
   st['previous']=row
 except Exception as e:errors.append({'type':type(e).__name__,'message':str(e),'sample_counts':{k:v['count'] for k,v in state.items()}})
 decisions=[];final={};all_ids=[]
 for rid,st in state.items():
  final[rid]=interval_hold(st['tail']);all_ids+=final[rid]['ids'];prior=set()
  for d in report.get('basket_confirmation',{}).get('cycles',{}).get(rid,[]):
   h=st['cycles'].get(d['cycle'],{});ids=set(h.get('ids',[]));positive=d['status']=='confirmed_increase'
   added=ids-prior;lost=prior-ids
   decisions.append(dict(robot=rid,attempt=d['cycle'],image_status=d['status'],hold=h,
    no_false_success=not positive or h.get('valid_three_second_interval') is True and len(added)==1 and not lost))
   prior=ids
 checks={'raw_measurements_complete_and_valid':not errors,'all120Hz_samples_present':all(s['count']==round(report['simulated_seconds']*120)+1 for s in state.values()),
  'same_actual_basket_frame_recorded':report.get('physical_evaluation_frame')=='actual_authored_basket_link_full_sphere',
  'all_four_rgb_increases_have_independent_last3s_hold':len(decisions)==4 and all(d['no_false_success'] for d in decisions),
  'final3s_each_basket_two_full_spheres':all(v['valid_three_second_interval'] and len(v['ids'])==2 for v in final.values()),
  'four_unique_final_spheres':sorted(all_ids)==[0,1,2,3],
  'original_trial_reports_passed':report.get('verified') is True,
  'trace_unchanged':before==hashlib.sha256(trace.read_bytes()).hexdigest()}
 return dict(verified=all(checks.values()),checks=checks,decisions=decisions,final=final,errors=errors,
  sample_counts={rid:s['count'] for rid,s in state.items()},trace_sha256=before,
  scope='Independent replay only;full ball spheres in actual measured basket;no control outputs or world mutations.')
def main():
 p=argparse.ArgumentParser();p.add_argument('folder',type=Path);a=p.parse_args();out=audit(a.folder.resolve())
 (a.folder/'full-basket-independent-audit.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf8')
 print(json.dumps({'verified':out['verified'],'failed':[k for k,v in out['checks'].items() if not v],'errors':out['errors']}));return 0 if out['verified'] else 2
if __name__=='__main__':raise SystemExit(main())
