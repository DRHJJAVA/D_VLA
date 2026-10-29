import ast, pathlib, types, torch
def install(train, batch_size=4):
    import r5_model
    original_history=r5_model.R5Network.history_summary
    original_previous=train.previous_encoding
    original_objective=train.objective
    original_load=torch.load
    tree=ast.parse(pathlib.Path(r5_model.__file__).read_text())
    fn=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='history_summary')
    fn.name='_batched_history'
    class ChangeBatch(ast.NodeTransformer):
        def visit_Constant(self,n):
            return ast.copy_location(ast.Constant(batch_size),n) if type(n.value) is int and n.value==4 else n
    fn=ChangeBatch().visit(fn);ast.fix_missing_locations(fn)
    namespace=dict(vars(r5_model))
    exec(compile(ast.Module(body=[fn],type_ignores=[]),'<engineering_history_batch>','exec'),namespace)
    batched=namespace['_batched_history']
    def tagged_load(path,*args,**kwargs):
        obj=original_load(path,*args,**kwargs)
        if isinstance(path,(str,pathlib.Path)) and isinstance(obj,dict):
            d=obj.get('inputs',obj.get('d',obj))
            if isinstance(d,dict) and all(k in d for k in ('features','depth','words','ee')):
                d['_engineering_input_path']=str(pathlib.Path(path).resolve())
        return obj
    torch.load=tagged_load
    def history(self,frames):
        cache=getattr(self,'_engineering_history_cache',None)
        if cache is None:return original_history(self,frames)
        keys=[None if d is None else d.get('_engineering_input_path') for d in frames]
        if any(d is not None and k is None for d,k in zip(frames,keys)):
            pathlib.Path(__file__).with_name('missing_identity.json').write_text(__import__('json').dumps([{'keys':list(d.keys()),'shapes':{k:list(v.shape) for k,v in d.items() if torch.is_tensor(v)}} for d,k in zip(frames,keys) if d is not None and k is None])); raise RuntimeError('missing_engineering_input_identity')
        pending={}
        for d,k in zip(frames,keys):
            if k is not None and k not in cache:pending.setdefault(k,d)
        if pending:
            vals=batched(self,list(pending.values()))
            for k,v in zip(pending,vals):cache[k]=v.detach()
        self._engineering_reused=getattr(self,'_engineering_reused',0)+sum(k is not None for k in keys)-len(pending)
        self._engineering_computed=getattr(self,'_engineering_computed',0)+len(pending)
        out=next(self.parameters()).new_zeros((len(frames),15))
        for i,k in enumerate(keys):
            if k is not None:out[i]=cache[k]
        return out
    r5_model.R5Network.history_summary=history
    def previous(model,rec):
        model._engineering_history_cache={}
        return original_previous(model,rec)
    def objective(model,*args,**kwargs):
        try:return original_objective(model,*args,**kwargs)
        finally:model._engineering_history_cache=None
    train.previous_encoding=previous;train.objective=objective
    def uninstall():
        r5_model.R5Network.history_summary=original_history
        train.previous_encoding=original_previous;train.objective=original_objective;torch.load=original_load
    return uninstall
