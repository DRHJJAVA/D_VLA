
from prepare import *
def main():
 work=max(OUT.glob('preflight_*'),key=lambda p:p.stat().st_mtime)
 rows=json.loads((ROOT/'experiments/contact_joint_object/manifests/episodes.json').read_text())['episodes']
 e=next(x for x in rows if x['id']=='pick_up_the_alphabet_soup_and_place_it_in_the_basket__demo_0')
 live=live_geometry(work)
 d,_,obj,meta=live.policy_input(e,0)
 ref,_,robj,rmeta=geometry.policy_input(e,0)
 equal={k:bool(torch.equal(d[k],ref[k])) for k in d}
 save('input_comparison.json',{'fields_equal':equal,'live_objects':len(obj),'reference_objects':len(robj),'work':str(work)})
 torch.set_num_threads(4)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 model=JointNetwork(common.build_expert('cpu').lm_expert)
 ck=torch.load(RUN/'checkpoints/step3000.pt',map_location='cpu',weights_only=False)
 model.load_state_dict(ck['model'],strict=True);model=model.cuda().eval()
 d={k:v.cuda() for k,v in d.items()};ref={k:v.cuda() for k,v in ref.items()}
 norm=json.loads((RUN/'normalization.json').read_text());sd=torch.tensor(norm['std'],device='cuda');mu=torch.tensor(norm['mean'],device='cuda')
 gen=torch.Generator(device='cpu').manual_seed(20261028);results=[]
 with torch.inference_mode():
  e1=model.encode(d);e2=model.encode(ref)
  for i in range(20):
   z=torch.randn(1,16,6,generator=gen).cuda()
   a,g=model.generate(e1,z);b,h=model.generate(e2,z)
   ap=(a[0]*sd+mu).clamp(-1,1);bp=(b[0]*sd+mu).clamp(-1,1)
   diff=ap[0]-bp[0]
   results.append({'normalized_sequence_mae':float((a-b).abs().mean()),'first_translation_target_mm':float(diff[:3].norm()*50),'first_rotation_vector_deg':float(diff[3:].norm()*.5*180/np.pi),'gripper_flip':bool(g[0,0].argmax()!=h[0,0].argmax())})
  a2,g2=model.generate(e1,z)
  repeat=float((a-a2).abs().max())
 passed=all(equal.values()) or (np.mean([x['normalized_sequence_mae'] for x in results])<=.01 and max(x['first_translation_target_mm'] for x in results)<=1 and max(x['first_rotation_vector_deg'] for x in results)<=.25 and not any(x['gripper_flip'] for x in results) and repeat<=1e-6)
 save('action_input_gate.json',{'passed':bool(passed),'all_inputs_exact':all(equal.values()),'repeat_max':repeat,'pairs':results,'criterion':'exact inputs or paired mean sequence MAE <=.01, every first translation target <=1mm, rotation vector <=.25deg, no gripper flip, repeat<=1e-6','note':'controller target difference, not measured trajectory error'})
 save('state.json',{'status':'input_gate_passed' if passed else 'input_gate_failed','formal_trials_started':False})
if __name__=='__main__':
 try:main()
 except Exception as e:save('failure.json',{'type':type(e).__name__,'summary':str(e)[:500],'frames':[{'file':x.filename,'line':x.lineno} for x in traceback.extract_tb(e.__traceback__)]})
