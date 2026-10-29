from pathlib import Path
import h5py,numpy as np,json,hashlib,datetime,re
from PIL import Image
root=Path('/mnt/nvme1/drh/D_VLA')
out=root/'experiments/contact_preference/fixed_camera_pilot_v1'
docs=root/'docs'
cfg=json.loads((out/'config.json').read_text())
audit=json.loads((out/'audit.json').read_text())
manifest=json.loads((out/'gallery_manifest.json').read_text())
oldlock=json.loads((root/'experiments/contact_preference/stage012_v1/completed_locked.json').read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
old_bad=[]
for rel,digest in oldlock.items():
    if rel=='docs/夹爪物体接触偏好学习_完整方案.md':continue
    q=root/rel
    if not q.exists() or sha(q)!=digest:old_bad.append(rel)
assert not old_bad,old_bad
source={}
max_reprojection_px=0.0
fixed_ok=True
for name,c in cfg['cases'].items():
    for path in [root/c['raw'],root/'experiments/contact_frames/cf017/rgbd'/(name+'.h5'),root/'experiments/contact_frames/cf017/cases'/name/'artifacts/reference.npz']:
        source[str(path.relative_to(root))]=sha(path)
    with h5py.File(out/(name+'.h5'),'r') as h:
        frames=h['frame_indices'][:].tolist()
        assert frames==c['frames'] and len(frames)==len(set(frames))
        for cam in cfg['new_fixed_cameras']:
            g=h[cam]
            assert len(g['rgb'])==len(frames)
            assert g['rgb'].shape[1:]==(448,448,3)
            assert g['depth_m'].shape[1:]==(448,448)
            assert np.allclose(g['camera_to_world'][:],g['camera_to_world'][0],atol=1e-7)
            assert np.allclose(g['K'][:],g['K'][0],atol=1e-7)
            for idx in range(len(frames)):
                z=np.asarray(g['depth_m'][idx],float)
                valid=np.isfinite(z)&(z>0)
                assert valid.mean()>0.95
                K=np.asarray(g['K'][idx],float)
                C=np.asarray(g['camera_to_world'][idx],float)
                assert abs(np.linalg.det(C[:3,:3])-1)<1e-5
                vv,uu=np.mgrid[5:448:32,5:448:32]
                u=uu.ravel().astype(float);v=vv.ravel().astype(float);d=z[v.astype(int),u.astype(int)]
                Xc=np.stack([(u-K[0,2])*d/K[0,0],(v-K[1,2])*d/K[1,1],d],axis=1)
                Xw=Xc@C[:3,:3].T+C[:3,3]
                back=(Xw-C[:3,3])@np.linalg.inv(C[:3,:3]).T
                u2=K[0,0]*back[:,0]/back[:,2]+K[0,2]
                v2=K[1,1]*back[:,1]/back[:,2]+K[1,2]
                max_reprojection_px=max(max_reprojection_px,float(np.max(np.abs(np.stack([u-u2,v-v2])))))
        assert audit['cases'][name]['frames']==frames
for name,arr in manifest['images'].items():
    for mode,digest in arr.items():
        p=docs/'图示/接触偏好_固定相机试验'/(name+'_'+mode+'.jpg')
        assert p.exists() and sha(p)==digest
        with Image.open(p) as im:assert im.size==(1340,560)
page=root/manifest['page']
assert sha(page)==manifest['page_sha256']
html=page.read_text()
assert html.count('<canvas id=')==3
assert html.count('新增正面')>=1 and html.count('新增侧面')>=1 and html.count('新增俯视')>=1
sourcefile=out/'source_locked.json';sourcefile.write_text(json.dumps(source,ensure_ascii=False,indent=2))
now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(timespec='seconds')
integrity={'created_beijing':now,'old_stage012_locked_files_checked_excluding_evolving_scheme_md':len(oldlock)-1,'old_stage012_mismatches':old_bad,'source_files_sha256':source,'new_fixed_cameras':cfg['new_fixed_cameras'],'all_fixed_camera_extrinsics_constant':fixed_ok,'max_depth_world_pixel_roundtrip_error_px':max_reprojection_px,'gallery_images':6,'gallery_cloud_modes':6,'reserved_test_read':False,'new_model_training':False}
(out/'integrity.json').write_text(json.dumps(integrity,ensure_ascii=False,indent=2))
doc=docs/'夹爪物体接触偏好学习_完整方案.md'
with doc.open('a') as w:
    w.write('\n结果与验收（'+now+'）：三路固定视角在炉灶开关、抽屉和双物体搬运各一段已开发示范的预选帧上完成重放，均保留真实 RGB、米制深度、K、camera_to_world 与原始帧号。内外参跨所选帧固定，深度→世界→像素最大往返误差 '+format(max_reprojection_px,'.2e')+' px；阶段012旧锁除持续迭代的本文档外 '+str(len(oldlock)-1)+' 文件未变。来源帧0的既有手部邻域候选，新增正面/侧面/俯视深度一致可见数依次为：炉灶103/6/83（共116），抽屉302/271/226（共319），双物体568/288/0（共678）。操作帧夹爪中心8厘米内，每3像素抽样的3毫米格，与原主/腕相机并集不重合的新格为炉灶292、抽屉84、双物体138；格点可能属于夹爪、背景或其他物体，且受抽样影响，不等于新增真实接触面积。\n\n查看页：docs/接触偏好_固定RGBD多视角与点云核查.html；包含五视角同帧RGB/深度、夹前候选可见性和可旋转的来源点云/操作帧附近点云。绿色仅为旧阶段2来源候选，操作帧尚无点级接触判定。本轮通过“固定多视角派生数据与几何对齐、可视化”的工程验收；未通过遮挡接触面定位或阶段2总体验收，不能进入接触偏好监督。\n')
paths=[out/'config.json',out/'src/render_fixed_sparse.py',out/'src/run_render.py',out/'src/audit.py',out/'src/build_gallery.py',out/'stove_demo_0.h5',out/'drawer_demo_0.h5',out/'two_objects_demo_0.h5',out/'audit.json',out/'gallery_manifest.json',out/'source_locked.json',out/'integrity.json',page]
paths+=list((docs/'图示/接触偏好_固定相机试验').glob('*.jpg'))
lock={str(p.relative_to(root)):sha(p) for p in paths}
(out/'completed_locked.json').write_text(json.dumps(lock,ensure_ascii=False,indent=2))
(out/'stage_state.json').write_text(json.dumps({'stage':'complete','created_beijing':now,'page':manifest['page'],'locked_files':len(lock),'scheme_md_sha256_at_completion':sha(doc),'stage2_contact_surface_accepted':False,'reserved_test_read':False},ensure_ascii=False,indent=2))
print(json.dumps({'stage':'complete','time':now,'locked_files':len(lock),'max_reproj_px':max_reprojection_px,'old_mismatch':len(old_bad),'page':manifest['page']},ensure_ascii=False))
