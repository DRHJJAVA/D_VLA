from adapter import *
def main():
 torch.set_num_threads(4);torch.use_deterministic_algorithms(False);torch.backends.cudnn.deterministic=False;front=Front()
 e=json.loads((RUN/'episodes.json').read_text())[0]
 ref=torch.load(RUN/'cache'/e['id']/'0.pt',map_location='cpu',weights_only=False)['inputs']
 im=(ref['rgb'][0].numpy()*255).round().astype(np.uint8) if ref['rgb'].max()<=1 else ref['rgb'][0].numpy().astype(np.uint8)
 results=[]
 from torch.nn.attention import sdpa_kernel,SDPBackend
 for tf,cudnn in [(False,False),(False,True),(True,False),(True,True)]:
  for mode in ['default','math']:
   torch.backends.cuda.matmul.allow_tf32=tf
   import contextlib
   with sdpa_kernel(SDPBackend.MATH) if mode=='math' else contextlib.nullcontext():
    torch.backends.cudnn.allow_tf32=cudnn
    ns={'self':types.SimpleNamespace(dino=front.dino),'preproc':front.preproc,'im':im,'torch':torch,'np':np}
    with torch.inference_mode():exec(front.feature_code,ns)
    ft=ns['ft']
   v=torch.from_numpy(ft).half();rr=ref['features'][0]
   results.append({'matmul_tf32':tf,'cudnn_tf32':cudnn,'sdpa':mode,'max_abs':float((v.float()-rr.float()).abs().max()),'mean_abs':float((v.float()-rr.float()).abs().mean())})
 save('feature_observation_backend_probe.json',results)
if __name__=='__main__':
 try:main()
 except Exception as e:fail('feature_probe',e)
