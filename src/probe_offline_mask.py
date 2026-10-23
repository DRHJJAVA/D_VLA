#!/usr/bin/env python3
"""Audit offline element segmentation against RGB-D render and target geom IDs."""
import json
from pathlib import Path
import h5py
import numpy as np
from restore_audit import OffScreenRenderEnv,libero_utils
from asset_paths import localize_assets
import robosuite.macros as macros

root=Path('/mnt/nvme1/drh/D_VLA')
raw=root/'data/raw/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5'
video=root/'data/rendered/alphabet_soup_demo_0_full_rgbd.h5'
libero=Path('/home/drh/project/ECV-VLA/vendor/LIBERO')
with h5py.File(raw) as f,h5py.File(video) as v:
 d=f['data'];ep=d['demo_0'];xml=libero_utils.postprocess_model_xml(ep.attrs['model_file'],{})
 xml,_=localize_assets(xml,libero)
 bddl=libero/'libero/libero/bddl_files/libero_object'/Path(d.attrs['bddl_file_name']).name
 env=OffScreenRenderEnv(bddl_file_name=str(bddl),camera_names=['agentview','robot0_eye_in_hand'],
  camera_heights=448,camera_widths=448,camera_depths=True,camera_segmentations='element')
 try:
  env.reset();env.reset_from_xml_string(xml);env.sim.reset()
  target={i:env.sim.model.geom_id2name(i) for i in range(env.sim.model.ngeom)
          if (env.sim.model.geom_id2name(i) or '').startswith('alphabet_soup_1')}
  print('target_geoms',target,flush=True)
  frame=17;env.sim.set_state_from_flattened(np.asarray(ep['states'][frame],dtype=np.float64));env.sim.forward();env._update_observables(force=True)
  obs=env.env._get_observations();report={}
  for camera in ('agentview','robot0_eye_in_hand'):
   rgb=np.asarray(obs[camera+'_image']);seg=np.asarray(obs[camera+'_segmentation_element'])[...,0]
   if macros.IMAGE_CONVENTION=='opengl':rgb=np.flipud(rgb);seg=np.flipud(seg)
   match=np.mean(rgb==v[camera]['rgb'][frame])
   values,counts=np.unique(seg,return_counts=True)
   histogram={int(k):int(n) for k,n in zip(values,counts) if k in target or k-1 in target}
   report[camera]={'rgb_match_fraction':float(match),'target_pixel_counts':histogram,
       'unique_seg_ids':len(values),'first_ids':values[:15].astype(int).tolist()}
  print(json.dumps(report),flush=True)
 finally:env.close()
