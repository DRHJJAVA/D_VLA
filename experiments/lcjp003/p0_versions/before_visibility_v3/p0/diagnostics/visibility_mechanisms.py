import pathlib,json,numpy as np,h5py,hashlib,datetime,traceback
from scipy.ndimage import maximum_filter,minimum_filter
ROOT=pathlib.Path('/mnt/nvme1/drh/D_VLA')
P0=ROOT/'experiments/lcjp003/p0'
OFFSET=np.array([.375,.875])

def quant(v):
    v=np.asarray(v);v=v[np.isfinite(v)]
    return {'n':int(len(v)),'min':float(v.min()),'median':float(np.median(v)),'p90':float(np.quantile(v,.9)),'p95':float(np.quantile(v,.95)),'max':float(v.max())} if len(v) else {'n':0}

def project_world(world,K,C):
    q=(world-C[:3,3])@C[:3,:3]
    return q[:,:2]/q[:,2,None]*[K[0,0],K[1,1]]+[K[0,2],K[1,2]]-.5

def run():
    audit=json.loads((P0/'data_audit.json').read_text())
    codepath=ROOT/'experiments/lcjp003/src/data.py'
    source_sha=hashlib.sha256(codepath.read_bytes()).hexdigest()
    if source_sha!=audit['adapter_sha256']:raise RuntimeError('Frozen adapter hash mismatch')
    arrays=[];sourcefiles=[]
    for sample in audit['samples']:
        if sample['case'] not in ('start_complete','middle','late_complete'):continue
        path=ROOT/sample['path'];sha=hashlib.sha256(path.read_bytes()).hexdigest()
        if sha!=sample['sha256']:raise RuntimeError('Frozen input hash mismatch')
        eid=sample['episode_id'];t=sample['current_frame'];sourcefiles.append({'episode':eid,'frame':t,'sample':sample['path'],'sha256':sha})
        with np.load(path) as win:
            r={k:win[k][-1] for k in ('xyz_base','point_valid','camera_id','depth_uv','uv','projected_uv','projected_depth_uv','view_visible','object_id','K','camera_to_world')}
            original_ids=win['candidate_original_ids']
        K=r['K'];C=r['camera_to_world']
        with h5py.File(ROOT/'experiments/contact_joint_object/cache/rgbd'/f'{eid}.h5','r') as h:
            names=json.loads(h.attrs['robot_metadata'])['body_names'];bi=names.index('robot0_base');br=h['robot/body_rotation'][t,bi];bp=h['robot/body_position'][t,bi]
            world=r['xyz_base']@br.T+bp
            depth=np.stack([h[c+'/depth_m'][t] for c in ('agentview','robot0_eye_in_hand')])
            localrange=maximum_filter(depth,size=(1,3,3))-minimum_filter(depth,size=(1,3,3))
            for src,dst in ((0,1),(1,0)):
                ids=np.flatnonzero(r['point_valid']&(r['camera_id']==src)&r['view_visible'][:,dst]);uv=r['projected_depth_uv'][ids,dst].astype(np.float64);p=np.rint(uv).astype(int)
                srcuv=np.rint(r['depth_uv'][ids]).astype(int);sm=(localrange[src,srcuv[:,1],srcuv[:,0]]<.01)&(localrange[dst,p[:,1],p[:,0]]<.01)
                predicted=((world[ids]-C[dst,:3,3])@C[dst,:3,:3])[:,2];srcz=((world[ids]-C[src,:3,3])@C[src,:3,:3])[:,2]
                nearest=depth[dst,p[:,1],p[:,0]]
                cuv=np.clip(uv,0,255);lo=np.floor(cuv).astype(int);hi=np.minimum(lo+1,255);frac=cuv-lo
                d4=np.column_stack([depth[dst,lo[:,1],lo[:,0]],depth[dst,lo[:,1],hi[:,0]],depth[dst,hi[:,1],lo[:,0]],depth[dst,hi[:,1],hi[:,0]]])
                weights=np.column_stack([(1-frac[:,0])*(1-frac[:,1]),frac[:,0]*(1-frac[:,1]),(1-frac[:,0])*frac[:,1],frac[:,0]*frac[:,1]])
                interp=(d4*weights).sum(1)
                physical=uv+OFFSET
                ray=np.column_stack([(physical[:,0]-K[dst,0,2])/K[dst,0,0],(physical[:,1]-K[dst,1,2])/K[dst,1,1],np.ones(len(ids))])
                def source_from_depth(z):
                    w=(ray*z[:,None])@C[dst,:3,:3].T+C[dst,:3,3]
                    return project_world(w,K[src],C[src])
                err_interp=np.linalg.norm(source_from_depth(interp)-r['uv'][ids],axis=1)
                err_nearest_depth=np.linalg.norm(source_from_depth(nearest)-r['uv'][ids],axis=1)
                # This second nearest comparison includes actual different pixel footprint instead of only its depth.
                pr=p+OFFSET
                qn=np.column_stack([(pr[:,0]-K[dst,0,2])*nearest/K[dst,0,0],(pr[:,1]-K[dst,1,2])*nearest/K[dst,1,1],nearest])
                wn=qn@C[dst,:3,:3].T+C[dst,:3,3]
                err_nearest_pixel=np.linalg.norm(project_world(wn,K[src],C[src])-r['uv'][ids],axis=1)
                eps=1e-5
                derivative=np.linalg.norm(source_from_depth(predicted+eps)-source_from_depth(predicted-eps),axis=1)/(2*eps)*.001
                offsets=np.array([(x,y) for y in (-1,0,1) for x in (-1,0,1)])
                neighbor=np.clip(p[:,None,:]+offsets[None],0,255)
                nz=depth[dst,neighbor[:,:,1],neighbor[:,:,0]]
                npix=neighbor+OFFSET
                points=np.stack([(npix[:,:,0]-K[dst,0,2])*nz/K[dst,0,0],(npix[:,:,1]-K[dst,1,2])*nz/K[dst,1,1],nz],axis=-1)
                center=points.mean(1);centered=points-center[:,None]
                _,_,vh=np.linalg.svd(centered,full_matrices=False);normal=vh[:,-1,:]
                fit_rmse=np.sqrt(np.mean(np.einsum('nki,ni->nk',centered,normal)**2,axis=1))
                den=np.einsum('ni,ni->n',ray,normal)
                plane_depth=np.einsum('ni,ni->n',center,normal)/den
                err_plane=np.linalg.norm(source_from_depth(plane_depth)-r['uv'][ids],axis=1)
                for j,idx in enumerate(ids):
                    arrays.append({'episode':eid,'frame':int(t),'source_camera':src,'destination_camera':dst,'point_index':int(idx),'candidate_id':int(original_ids[r['object_id'][idx]]),'smooth_10mm':bool(sm[j]),'source_uv':r['uv'][idx].tolist(),'destination_depth_uv':uv[j].tolist(),'source_depth_m':float(srcz[j]),'predicted_destination_depth_m':float(predicted[j]),'nearest_destination_depth_m':float(nearest[j]),'interpolated_destination_depth_m':float(interp[j]),'neighbor4_depths_m':d4[j].tolist(),'neighbor4_weights':weights[j].tolist(),'neighbor4_range_mm':float(np.ptp(d4[j])*1000),'source_range3_mm':float(localrange[src,srcuv[j,1],srcuv[j,0]]*1000),'destination_range3_mm':float(localrange[dst,p[j,1],p[j,0]]*1000),'cycle_interp_px':float(err_interp[j]),'cycle_nearest_depth_px':float(err_nearest_depth[j]),'cycle_nearest_pixel_px':float(err_nearest_pixel[j]),'cycle_local_plane_px':float(err_plane[j]),'plane_fit_rmse_mm':float(fit_rmse[j]*1000),'plane_depth_m':float(plane_depth[j]),'depth_to_source_pixel_sensitivity_px_per_mm':float(derivative[j]),'visibility_tolerance_mm':float(max(.005,.02*nearest[j])*1000)})
    groups={
      'all_actual_fused':arrays,
      'all_gt2px':[r for r in arrays if r['cycle_interp_px']>2],
      'smooth_actual_fused':[r for r in arrays if r['smooth_10mm']],
      'smooth_gt2px':[r for r in arrays if r['smooth_10mm'] and r['cycle_interp_px']>2],
      'edge_gt2px':[r for r in arrays if not r['smooth_10mm'] and r['cycle_interp_px']>2]}
    summaries={}
    for name,rows in groups.items():
        if not rows:continue
        def count(fn):return int(sum(fn(r) for r in rows))
        summaries[name]={
          'n':len(rows),
          'cycle_error_px':quant([r['cycle_interp_px'] for r in rows]),
          'depth_abs_residual_mm':quant([abs(r['interpolated_destination_depth_m']-r['predicted_destination_depth_m'])*1000 for r in rows]),
          'pixel_sensitivity_px_per_mm':quant([r['depth_to_source_pixel_sensitivity_px_per_mm'] for r in rows]),
          'source_depth_m':quant([r['source_depth_m'] for r in rows]),
          'neighbor4_range_mm':quant([r['neighbor4_range_mm'] for r in rows]),
          'plane_fit_rmse_mm':quant([r['plane_fit_rmse_mm'] for r in rows]),
          'bilinear_gt2_but_same_ray_nearest_le2':count(lambda r:r['cycle_interp_px']>2 and r['cycle_nearest_depth_px']<=2),
          'both_bilinear_and_same_ray_nearest_gt2':count(lambda r:r['cycle_interp_px']>2 and r['cycle_nearest_depth_px']>2),
          'bilinear_gt2_but_local_plane_le2':count(lambda r:r['cycle_interp_px']>2 and r['cycle_local_plane_px']<=2),
          'all_four_neighbors_nearer_by_gt_0p1mm':count(lambda r:r['predicted_destination_depth_m']-max(r['neighbor4_depths_m'])>.0001),
          'all_four_neighbors_farther_by_gt_0p1mm':count(lambda r:min(r['neighbor4_depths_m'])-r['predicted_destination_depth_m']>.0001),
          'predicted_depth_between_four_neighbor_extrema':count(lambda r:min(r['neighbor4_depths_m'])<=r['predicted_destination_depth_m']<=max(r['neighbor4_depths_m'])),
          'abs_depth_residual_lt_1mm':count(lambda r:abs(r['interpolated_destination_depth_m']-r['predicted_destination_depth_m'])<.001),
          'abs_depth_residual_lt_0p5mm':count(lambda r:abs(r['interpolated_destination_depth_m']-r['predicted_destination_depth_m'])<.0005),
          'near_planar_rmse_le_0p1mm':count(lambda r:r['plane_fit_rmse_mm']<=.1),
          'near_planar_and_source_behind_fit_gt_0p1mm':count(lambda r:r['plane_fit_rmse_mm']<=.1 and r['predicted_destination_depth_m']-r['plane_depth_m']>.0001),
          'wrist_as_source':count(lambda r:r['source_camera']==1),
          'fallback_candidate':count(lambda r:r['candidate_id']>=1000000),
        }
    smoothbad=groups['smooth_gt2px']
    report={'created_beijing':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(timespec='seconds'),'source_sha256':source_sha,'frozen_samples':sourcefiles,'readonly':True,'original_sources_modified':False,'model_loss_training_modified':False,
      'scope':'Only six existing P0 windows and their current frames, same 4517 actual view_visible correspondences; no added scenes and no point removal.',
      'interpretation_rules':['Diagnostic 10mm smooth split is not a segmentation label or physical same-surface guarantee.','Bilinear-only means the audit interpolation changes the outcome compared with the exact same ray using nearest depth; it is not proof nearest depth is correct.','Four neighbors all nearer is observed occlusion evidence conditional on local surface continuity; no true object/contact labels are read.','Local plane is a diagnostic alternative fitted to 3x3 observed depths only; not used to change inputs or force pass.','Sensitivity expresses camera geometry and source resolution amplification; it does not prove the fused descriptor belongs to the correct surface.'],
      'summaries':summaries,'smooth_outlier_representatives':sorted(smoothbad,key=lambda r:r['cycle_interp_px'],reverse=True)[:12]}
    dst=P0/'diagnostics';(dst/'visibility_mechanisms.json').write_text(json.dumps(report,indent=2));(dst/'visibility_mechanisms_points.json').write_text(json.dumps(arrays,separators=(',',':')))
    print(json.dumps({'counts':{k:len(v) for k,v in groups.items()},'smooth_outliers':summaries['smooth_gt2px'],'all_outliers':summaries['all_gt2px']}))

if __name__=='__main__':
    try:run()
    except Exception as e:print(json.dumps({'error_type':type(e).__name__,'locations':[{'file':x.filename,'line':x.lineno} for x in traceback.extract_tb(e.__traceback__)]}))
