"""Robot-only visual-depth exclusion from measured q and camera calibration. No object or simulator contact state."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
from pathlib import Path
import xml.etree.ElementTree as ET
import importlib.util,numpy as np,mujoco
from scipy.spatial.transform import Rotation
class RobotDepth:
 def __init__(self,size):
  root=Path(importlib.util.find_spec('robosuite').origin).parent/'models/assets/grippers'
  xml=ET.parse(root/'panda_gripper.xml').getroot()
  for a in xml.findall('./asset/mesh'):a.set('file',str(root/a.get('file')))
  ET.SubElement(xml,'visual')
  ET.SubElement(xml.find('visual'),'global',offwidth=str(size),offheight=str(size))
  ET.SubElement(xml.find('worldbody'),'camera',name='probe',pos='0 0 1',fovy='75')
  self.model=mujoco.MjModel.from_xml_string(ET.tostring(xml,encoding='unicode'))
  self.data=mujoco.MjData(self.model)
  self.renderer=mujoco.Renderer(self.model,height=size,width=size)
  self.option=mujoco.MjvOption();self.option.geomgroup[:]=0;self.option.geomgroup[1]=1
  self.renderer.enable_depth_rendering()
  self.camera=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_CAMERA,'probe')
 def render(self,q,cam_in_hand,K):
  self.data.qpos[:]=q
  self.model.cam_pos[self.camera]=cam_in_hand[:3,3]
  rot=cam_in_hand[:3,:3]@np.diag([1,-1,-1])
  quat=Rotation.from_matrix(rot).as_quat()
  self.model.cam_quat[self.camera]=quat[[3,0,1,2]]
  self.model.cam_fovy[self.camera]=np.degrees(2*np.arctan(self.renderer.height/(2*K[1,1])))
  mujoco.mj_forward(self.model,self.data)
  self.renderer.update_scene(self.data,camera=self.camera,scene_option=self.option)
  return self.renderer.render()
 def close(self):
  renderer=self.renderer;self.renderer=None
  del renderer
