"""Generate a predeclared, scene-split person-training set, never controlling balls."""
from pathlib import Path
import hashlib, json, math, sys
import genesis as gs
import numpy as np
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'work/tennis-drive-v49'
OUT=ROOT/'outputs/tennis-ball-collector-suite/person-training-v1-20261024'
sys.path.insert(0,str(SOURCE))
from fleet_scene import build_fleet,hold_fleet
from camera_geometry import world_to_pixel
from mission_control import PICKUP,RAISED,CARRY,TIPPED

if OUT.exists():raise FileExistsError(OUT)
OUT.mkdir(parents=True)
rng=np.random.default_rng(2026102401)
poses=[(PICKUP,0.),(-.20,CARRY),(RAISED,CARRY),(RAISED,TIPPED)]
states=[]
for index in range(60):
    person=None
    if index<40:
        person=[float(rng.uniform(-14.,14.)),float(rng.uniform(-6.,6.)),float(rng.uniform(0,360))]
    states.append(dict(id=f'scene_{index:03d}',split='tune' if index%5==4 else 'train',
        robot=[float(rng.uniform(-14.1,-10.8)),float(rng.uniform(-7.2,-6.0)),float(rng.uniform(-45,45))],
        pose=poses[index%4],person=person,actor_index=index%3))
manifest=dict(seed=2026102401,states=states,items=[],
    scope='synthetic static-pose person training; three heights, random locations and yaw; not real people',
    reserved_test='v47-person-render-set-20261024 validation: never used for fitting or checkpoint selection',
    split_rule='scene index modulo five equals four: tune; all cameras and crops of a scene share its split',
    bbox_method='projection of actual visual mesh vertices; amodal full body; labels only for offline training',
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(OUT/'predeclared.json').write_text(json.dumps(manifest,indent=2)+'\n')
read=lambda n:json.loads((SOURCE/n).read_text())
config,fleet,vision,fixed=[read(n) for n in ('scene_config.json','fleet_config.json','vision_config.json','fixed_camera_config.json')]
config.update(ball_positions_m=[],robot_urdf='collector_beveled.urdf',physics_substeps=8)
gs.init(backend=gs.amdgpu,seed=13,logging_level='warning')
try:
    scene,robots,*_=build_fleet(config,fleet,vision)
    cameras=[scene.add_camera(res=tuple(s['resolution']),pos=tuple(s['position_m']),lookat=tuple(s['lookat_m']),up=(0,0,1),fov=s['vertical_fov_degrees'],near=.05,far=200,GUI=False) for s in fixed['cameras']]
    actors=[scene.add_entity(gs.morphs.Mesh(file=str(SOURCE/'assets/person-cesium/CesiumMan.glb'),pos=(0,0,-8),
        scale=h/1.5065499544143677,fixed=True,collision=False,decimate=False),name=f'training_person_{i}') for i,h in enumerate((1.55,1.75,1.95))]
    scene.build();hold_fleet(robots)
    for _ in range(120):scene.step()
    calibration=[{**s,'intrinsics':np.asarray(c.intrinsics).tolist(),'transform':np.asarray(c.transform).tolist()} for s,c in zip(fixed['cameras'],cameras)]
    manifest['calibration']=calibration
    (OUT/'images').mkdir();(OUT/'label-previews').mkdir()
    for state in states:
        x,y,heading=state['robot'];arm,bucket=state['pose']
        for side,robot in enumerate(robots):
            sign=1 if side==0 else -1;angle=math.radians(heading+180*side)/2
            robot.set_pos(np.array([sign*x,sign*y,.205]));robot.set_quat(np.array([math.cos(angle),0,0,math.sin(angle)]))
            joints=[robot.get_joint(n).dofs_idx_local[0] for n in ('shovel_lift_joint','shovel_tilt_joint')]
            robot.set_dofs_position(np.array([arm,bucket-arm]),dofs_idx_local=joints)
            robot.control_dofs_position(np.array([arm,bucket-arm]),dofs_idx_local=joints)
        for actor in actors:actor.set_pos(np.array([0.,0.,-8.]))
        actor=actors[state['actor_index']]
        if state['person']:
            px,py,yaw=state['person'];angle=math.radians(yaw)/2
            actor.set_pos(np.array([px,py,.01]));actor.set_quat(np.array([math.cos(angle),0,0,math.sin(angle)]))
        scene.step()
        vertices=actor.get_vverts().detach().cpu().numpy().reshape(-1,3) if state['person'] else None
        for camera,cal in zip(cameras,calibration):
            image=Image.fromarray(np.asarray(camera.render(antialiasing=True)[0],dtype=np.uint8))
            path=OUT/'images'/f'{state["id"]}_{cal["id"]}.png';image.save(path)
            boxes=[]
            if vertices is not None:
                local=np.linalg.solve(np.asarray(cal['transform']),np.column_stack((vertices,np.ones(len(vertices)))).T).T[:,:3]
                depth=-local[:,2];assert np.all(depth>0)
                k=np.asarray(cal['intrinsics']);uv=np.column_stack((k[0,0]*local[:,0]/depth+k[0,2],k[1,2]-k[1,1]*local[:,1]/depth))
                box=np.r_[uv.min(axis=0),uv.max(axis=0)]
                assert 1<=box[0]<box[2]<image.width-1 and 1<=box[1]<box[3]<image.height-1, 'Keep training people fully in frame'
                boxes=[box.tolist()]
            centres=[world_to_pixel([s*x,s*y,.4],np.asarray(cal['intrinsics']),np.asarray(cal['transform'])).tolist() for s in (1,-1)]
            manifest['items'].append(dict(image=path.relative_to(ROOT).as_posix(),camera_id=cal['id'],split=state['split'],state_id=state['id'],
                people=[] if vertices is None else [[px,py,0.]],boxes_xyxy=boxes,robot_pixel_centres=centres,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            if int(state['id'][-3:]) in (0,1,2,9,19,29,39,40,49):
                preview=image.copy();draw=ImageDraw.Draw(preview)
                for box in boxes:draw.rectangle(box,outline='#ff5500',width=5)
                preview.thumbnail((1280,720));preview.save(OUT/'label-previews'/path.name)
        print(json.dumps(dict(rendered=state['id'],images=len(manifest['items']))),flush=True)
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
finally:gs.destroy()
