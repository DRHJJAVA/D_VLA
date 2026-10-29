"""Small full-sequence training check on available frozen features; never used for Stage-2 acceptance."""
from pathlib import Path
import json,time,torch,numpy as np
from stage2_crossview_teacher import R,Teacher,load_case
torch.set_num_threads(4);torch.manual_seed(2026)
rows={x['id']:x for x in [json.loads(x) for x in (R/'data/v3/manifests/original_sequences.jsonl').read_text().splitlines() if x]}
train=[rows[x] for x in ('alphabet_soup_demo_0','alphabet_soup_demo_3','alphabet_soup_demo_5')]
val=rows['alphabet_soup_demo_1']
net=Teacher('A').cuda();opt=torch.optim.AdamW(net.parameters(),lr=2e-4)
history=[];start=time.time()
for epoch in range(2):
 net.train();losses=[]
 for row in train:
  data=load_case(row,'cuda',False);opt.zero_grad(set_to_none=True);loss,_=net(*data);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1);opt.step()
  losses.append(float(loss.detach()));del data,loss
 net.eval()
 with torch.no_grad():v=float(net(*load_case(val,'cuda',False))[0])
 history.append(dict(epoch=epoch+1,train_loss=float(np.mean(losses)),val_loss=v))
 print(json.dumps(history[-1]),flush=True)
report=dict(status='smoke_training_completed',episodes_used=4,all_frames_per_episode=True,not_for_acceptance=True,method='A cross-view contrast',history=history,elapsed_s=round(time.time()-start,1))
out=R/'audits/v3/stage2_smoke_train.json';out.write_text(json.dumps(report,indent=2))
print(json.dumps(report),flush=True)
