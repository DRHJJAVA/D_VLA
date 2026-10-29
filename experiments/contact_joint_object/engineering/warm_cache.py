
import pathlib,sys,ast,json,time,traceback,concurrent.futures,os
B=pathlib.Path('/mnt/nvme1/drh/D_VLA/experiments/contact_joint_object')
sys.path[:0]=[str(B/'src'),str(B/'engineering')]
def one(job):
    split,row=job
    import input_cache,data,online_targets,torch
    torch.set_num_threads(1)
    from threadpoolctl import threadpool_limits
    limits=threadpool_limits(limits=1,user_api='blas')
    ds=data.Dataset(split);input_cache.install();started=time.time()
    if split=='dev':
        for t in range(row['frames']):
            for oid in ds.frame_objects(row,t):
                ds.get(row,t,oid,ci=t%2,point_limit=128)
    else:
        tree=ast.parse((B/'src/online_targets.py').read_text())
        original=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='refresh_episode')
        keep=[n for n in original.body if n.lineno in (16,17,20,21)]
        withnode=keep[-1];loop=next(n for n in withnode.body if n.lineno==28)
        withnode.body=[loop]
        class Prefix(ast.NodeTransformer):
            def visit_If(self,node):
                if node.lineno==36:
                    node.body=[ast.copy_location(ast.Continue(),node)]
                    node.orelse=[n for n in node.orelse if n.lineno<=42]
                    return node
                return self.generic_visit(node)
        loop=Prefix().visit(loop)
        # Line 42 is the original deterministic input constructor; retain no teacher operation.
        inner=next(n for n in ast.walk(loop) if isinstance(n,ast.For) and n.lineno==30)
        inner.body=[n for n in inner.body if n.lineno<=42]
        original.body=keep
        original.name='warm_selected'
        original.decorator_list=[]
        original.args=ast.arguments(posonlyargs=[],args=[ast.arg(arg=x) for x in ['ds','row','point_limit','prune']],kwonlyargs=[],kw_defaults=[],defaults=[ast.Constant(128),ast.Constant(True)])
        module=ast.fix_missing_locations(ast.Module(body=[original],type_ignores=[]))
        scope=dict(vars(online_targets));exec(compile(module,'<immutable_online_inputs>','exec'),scope)
        scope['warm_selected'](ds,row)
    result={'split':split,'id':row['id'],'seconds':time.time()-started,'stats':dict(input_cache.STATS)}
    (B/'engineering/warm_progress'/ (row['id']+'.json')).write_text(json.dumps(result))
    input_cache.close()
    return {'id':row['id'],'seconds':result['seconds']}
def main():
    import data
    (B/'engineering/warm_progress').mkdir(exist_ok=True)
    jobs=[(s,r) for s in ['dev','train'] for r in data.Dataset(s).rows]
    jobs=[j for j in jobs if not (B/'engineering/warm_progress'/(j[1]['id']+'.json')).exists()]
    started=time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=6) as pool:
        for result in pool.map(one,jobs):pass
    (B/'engineering/warm_complete.json').write_text(json.dumps({'time':time.time(),'seconds':time.time()-started,'episodes':450}))
if __name__=='__main__':
    try:main()
    except Exception as e:
        (B/'engineering/warm_failure.json').write_text(json.dumps({'error_type':type(e).__name__,'locations':[{'file':pathlib.Path(x.filename).name,'line':x.lineno} for x in traceback.extract_tb(e.__traceback__)]}))
