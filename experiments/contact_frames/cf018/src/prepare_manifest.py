from pathlib import Path
import json,hashlib,h5py,numpy as np,time
R=Path('/mnt/nvme1/drh/D_VLA')
E=R/'experiments/contact_frames/cf018'
CASES=[
('alphabet_soup_demo_0','data/v2/stage1_rgbd/alphabet_soup/demo_0.h5','data/raw/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5',0),
('alphabet_soup_demo_1','data/v2/stage1_rgbd/alphabet_soup/demo_1.h5','data/raw/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5',1),
('porcelain_mug_demo_9','data/v3/trainval_rgbd_448/porcelain_mug/demo_9.h5','data/raw/libero_90/STUDY_SCENE3_pick_up_the_white_mug_and_place_it_to_the_right_of_the_caddy_demo.hdf5',9),
('white_yellow_mug_demo_5','data/v3/trainval_rgbd_448/white_yellow_mug/demo_5.h5','data/raw/libero_90/LIVING_ROOM_SCENE5_put_the_yellow_and_white_mug_on_the_right_plate_demo.hdf5',5),
('tomato_sauce_demo_3','data/v3/trainval_rgbd_448/tomato_sauce/demo_3.h5','data/raw/libero_object/pick_up_the_tomato_sauce_and_place_it_in_the_basket_demo.hdf5',3),
('two_objects_demo_0','experiments/contact_frames/cf017/rgbd/two_objects_demo_0.h5','experiments/contact_frames/cf017/raw/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket_demo.hdf5',0),
('drawer_demo_0','experiments/contact_frames/cf017/rgbd/drawer_demo_0.h5','experiments/contact_frames/cf017/raw/KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet_demo.hdf5',0),
('stove_demo_0','experiments/contact_frames/cf017/rgbd/stove_demo_0.h5','experiments/contact_frames/cf017/raw/KITCHEN_SCENE9_turn_on_the_stove_demo.hdf5',0)]
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for x in iter(lambda:f.read(8*1024*1024),b''):h.update(x)
 return h.hexdigest()
rows=[]
for name,rgbpath,rawpath,demo in CASES:
 rgb,raw=R/rgbpath,R/rawpath
 assert rgb.exists() and raw.exists()
 with h5py.File(rgb) as h,h5py.File(raw) as d:
  n=len(h['frame_indices'])
  ids=h['frame_indices'][:]
  assert np.array_equal(ids,np.arange(n))
  assert h['agentview/rgb'].shape==(n,448,448,3)
  assert h['robot0_eye_in_hand/rgb'].shape==(n,448,448,3)
  obs=d[f'data/demo_{demo}/obs']
  acts=d[f'data/demo_{demo}/actions']
  assert len(acts)==n and len(obs['ee_pos'])==n
  lang=json.loads(d['data'].attrs['problem_info'])['language_instruction']
  assert lang and isinstance(lang,str)
  env=json.loads(d['data'].attrs['env_args'])
  ctrl=env.get('env_kwargs',{}).get('controller_configs',{})
  rows.append({'case':name,'rgbd_path':str(rgb),'raw_path':str(raw),'demo_index':demo,'frames':n,'rgbd_sha256':sha(rgb),'raw_sha256':sha(raw),'instruction':lang,'time_mapping':'render t->obs t-1 for t>=1; t0 state unknown','control_type':ctrl.get('type'),'camera_views':['agentview','robot0_eye_in_hand']})
(E/'episode_manifest.json').write_text(json.dumps({'created_at':time.strftime('%Y-%m-%d %H:%M:%S%z'),'cases':rows,'manual_ranges_used':False,'reserved_test_access':False},ensure_ascii=False,indent=2))
print(json.dumps({'stage':'manifest_ready','cases':len(rows),'frames':sum(x['frames'] for x in rows),'case_counts':{x['case']:x['frames'] for x in rows}},ensure_ascii=False))
