from adapters import *
import traceback
C=pathlib.Path(__file__).parent
try:
 eid='pick_up_the_bbq_sauce_and_place_it_in_the_basket__demo_0'
 work=C/('controller_audit_'+str(time.time_ns()));work.mkdir()
 render=LiveRender(eid,work,capacity=1)
 env=render.env
 for i in range(8):
  if hasattr(env,'robots'):break
  env=env.env
 robot=env.robots[0];ctrl=robot.controller
 result={'controller_type':type(ctrl).__name__,'controller_module':type(ctrl).__module__}
 for name in ['input_min','input_max','output_min','output_max','action_scale','action_input_transform','action_output_transform','control_freq','impedance_mode','control_delta','use_delta','position_limits','orientation_limits']:
  value=getattr(ctrl,name,None)
  if value is not None:result[name]=value.tolist() if hasattr(value,'tolist') else value
 result['gripper_type']=type(robot.gripper).__name__
 result['gripper_module']=type(robot.gripper).__module__
 (C/'controller_units.json').write_text(json.dumps(result,indent=2))
 render.close()
except Exception as ex:
 (C/'controller_units_failure.json').write_text(json.dumps({'type':type(ex).__name__,'message':str(ex)[:250],'frames':[{'file':x.filename,'line':x.lineno} for x in traceback.extract_tb(ex.__traceback__)]}))
