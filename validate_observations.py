"""Numerical access audit, independent of benchmark scores."""
import dataclasses, importlib, json, sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from motion_challenge.resources import lafan_paths
from motion_challenge.data.dataset import load_csv_motion
from motion_challenge.simulation.physics import World,forward_kinematics
from motion_challenge.runtime.observations import PROPRIOCEPTIVE,controller_state

import argparse
parser=argparse.ArgumentParser(description='Compare the restricted path with its author-verified base adapter.')
parser.add_argument('--runtime',type=Path,required=True,help='Resolved runtime.json from the evaluation output')
parser.add_argument('--data',type=Path,default=Path('~/.cache/MotionChallenge/lafan').expanduser())
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
repo=Path(__file__).resolve().parent;config=args.runtime;out=args.output;sys.path.insert(0,str(repo))
runtime=json.loads(config.read_text());runtime['adapter_root']=str(repo)
native=importlib.import_module('adapter').create(repo,runtime)
restricted=importlib.import_module('restricted').create(repo,runtime)
logs=[[],[]]
class Capture:
 def __init__(self,session,log,label):self.session,self.log,self.label=session,log,label
 def __getattr__(self,name):return getattr(self.session,name)
 def run(self,names,feed,*args,**kwargs):
  snapshot={k:np.asarray(v).copy() for k,v in feed.items()}
  result=self.session.run(names,feed,*args,**kwargs)
  self.log.append((self.label,snapshot,[np.asarray(v).copy() for v in result]))
  return result
class TorchCapture:
 def __init__(self,model,log):self.model,self.log=model,log
 def __getattr__(self,name):return getattr(self.model,name)
 def __call__(self,*args,**kwargs):
  result=self.model(*args,**kwargs)
  values=list(result) if isinstance(result,(list,tuple)) else [result]
  self.log.append(('torch',{str(i):v.detach().cpu().numpy().copy() for i,v in enumerate(args)},[v.detach().cpu().numpy().copy() for v in values]))
  return result
for obj,log in [(native,logs[0]),(restricted.inner,logs[1])]:
 if hasattr(obj,'actor'):
  obj.actor=TorchCapture(obj.actor,log)
 for name,session in getattr(obj,'extra',{}).items():
  if hasattr(session,'run'):obj.extra[name]=Capture(session,log,'extra.'+name)
 for name in ('session','encoder','decoder'):
  value=getattr(obj,name,None)
  if value is not None and hasattr(value,'run') and hasattr(value,'get_inputs'):
   setattr(obj,name,Capture(value,log,name))
paths=lafan_paths();manifest=json.loads(paths['suite'].read_text())
data_root=args.data
rng=np.random.default_rng(731942)
maxima={'observation':0.,'network_output':0.,'motor_command':0.};calls=0;cases=0
report={'status':'running','cases':0,'resets':0,'tolerance_abs':2e-5,'checkpoint_sha256':runtime['weights_sha256']}
def compare(a,b,label):
 if np.asarray(a).shape!=np.asarray(b).shape:raise AssertionError(f'{label}: shape mismatch')
 if np.asarray(a).size:
  av=np.asarray(a);bv=np.asarray(b)
  delta=float(np.max(np.abs(av.astype(np.float64)-bv.astype(np.float64))))
  if not np.isfinite(delta) or delta>2e-5:raise AssertionError(f'{label}: error {delta}')
  maxima[label]=max(maxima[label],delta)
try:
 for clip in manifest['files'][:4]:
  world=World(paths['robot']);motion=load_csv_motion(data_root/clip['path'],manifest['joint_names'],manifest['source_fps']);world.reset(motion)
  def state_at(t):
   frame=world.reference(t);d=world.data;d.qpos[:]=world._qpos(frame)
   d.qpos[:3]+=rng.normal(0,3,3)
   d.qpos[world.qadr]+=rng.normal(0,.02,len(world.joints))
   rot=Rotation.from_quat(frame.root_orientation_xyzw)*Rotation.from_rotvec(rng.normal(0,.025,3));q=rot.as_quat();d.qpos[3:7]=q[[3,0,1,2]]
   d.qvel[:]=rng.normal(0,.3,world.model.nv);d.time=t;forward_kinematics(world.model,d)
   return world.state()
  def window(t):return tuple(world.reference(t+x) for x in runtime['reference_offsets_s'])
  state=state_at(0);refs=window(0)
  native.reset(state,refs,0);restricted.reset(controller_state(state,PROPRIOCEPTIVE),refs,0);report['resets']+=1
  for step in range(40):
   # Beginning and clamped endpoint exercise histories, repeat time and reset.
   t=step*runtime['control_period_s'] if step<30 else max(0,motion.duration_s-(39-step)*runtime['control_period_s'])
   state=state_at(t);refs=window(t)
   logs[0].clear();logs[1].clear()
   a=native.act(state,refs);b=restricted.act(controller_state(state,PROPRIOCEPTIVE),refs)
   for field in dataclasses.fields(a):compare(getattr(a,field.name),getattr(b,field.name),'motor_command')
   if len(logs[0])!=len(logs[1]):raise AssertionError('different network call counts')
   for first,second in zip(logs[0],logs[1]):
    if first[0]!=second[0] or first[1].keys()!=second[1].keys():raise AssertionError('different network input keys')
    for key in first[1]:compare(first[1][key],second[1][key],'observation')
    if len(first[2])!=len(second[2]):raise AssertionError('different network output count')
    for x,y in zip(first[2],second[2]):compare(x,y,'network_output')
    calls+=1
   # Core boundary check: poison all forbidden fields, keep allowed state fixed.
   poisoned=dataclasses.replace(state,root_position_world_m=(1e7,-1e7,1e7),bodies=())
   assert controller_state(poisoned,PROPRIOCEPTIVE)==controller_state(state,PROPRIOCEPTIVE)
   cases+=1
 if calls==0:raise AssertionError('no observed native network calls')
 report.update(status='passed',cases=cases,network_calls=calls,max_abs_error=maxima,
  checks=['160 matched state sequences / 4 resets','large actual translation perturbations','reference windows preserved including endpoints','all captured ONNX inputs and outputs','all PD command fields','forbidden-field counterfactual before transport'],
  limits='Finite numerical regression plus source audit; not a proof against malicious code or implicit neural odometry.')
except Exception as error:
 report.update(status='failed',cases=cases,max_abs_error=maxima,error=repr(error));raise
finally:
 out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2)+'\n');native.close();restricted.close()
