import os, sys, pathlib, json, ast, copy, types, collections, time, hashlib
os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
ROOT = pathlib.Path('/mnt/nvme1/drh/D_VLA')
R = ROOT / 'experiments/lcjp003'
P = R / 'p1/v001'
OLD = ROOT / 'experiments/contact_joint_object'
HERE = P / 'closed_loop'
sys.path[:0] = [str(P), str(P / 'source'), str(ROOT / 'src'), str(ROOT / 'src/v3'), str(OLD / 'src'), str(OLD / 'dependencies/sam2'), str(OLD / 'dependencies/co-tracker')]
import numpy as np, h5py, torch
torch.set_num_threads(2)
torch.set_num_interop_threads(1)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
from common import Dataset, KEYS
from features import FrozenSmolFeatures
from model import LCJP003, ModelConfig
CAMS = ['agentview', 'robot0_eye_in_hand']

def sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()

def compile_nodes(nodes, filename):
    return compile(ast.fix_missing_locations(ast.Module(body=copy.deepcopy(nodes), type_ignores=[])), str(filename), 'exec')

def module_from(path, name, transform=None):
    tree = ast.parse(path.read_text())
    if transform:
        tree = transform.visit(tree)
    mod = types.ModuleType(name)
    mod.__file__ = str(path)
    sys.modules[name] = mod
    exec(compile(ast.fix_missing_locations(tree), str(path), 'exec'), mod.__dict__)
    return mod

class CPU(ast.NodeTransformer):

    def visit_Call(self, n):
        self.generic_visit(n)
        if isinstance(n.func, ast.Attribute) and n.func.attr == 'cuda':
            return ast.copy_location(ast.Call(func=ast.Attribute(value=n.func.value, attr='to', ctx=ast.Load()), args=[ast.Constant('cpu')], keywords=[]), n)
        return n

    def visit_Constant(self, n):
        if n.value == 'cuda':
            return ast.copy_location(ast.Constant('cpu'), n)
        return n

class LiveRender:

    def __init__(self, eid, work, capacity=600):
        self.eid = eid
        self.work = pathlib.Path(work)
        self.capacity = capacity
        self.work.mkdir(parents=True, exist_ok=True)
        self.ns = {'__name__': 'closed_render', '__file__': str(OLD / 'src/render_episode.py')}
        tree = ast.parse((OLD / 'src/render_episode.py').read_text())
        exec(compile_nodes([n for n in tree.body if not isinstance(n, (ast.FunctionDef, ast.If))], OLD / 'src/render_episode.py'), self.ns)
        fn = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main'))
        source_with = next((n for n in fn.body if isinstance(n, ast.With)))
        setup = []
        for n in source_with.body:
            if n.lineno == 42:
                break
            if n.lineno == 34:
                n = ast.parse("states = episode['states'][0:1]").body[0]
            if n.lineno == 35:
                continue
            setup.append(n)
        rows = json.loads((OLD / 'manifests/episodes.json').read_text())['episodes']
        row = next((x for x in rows if x['id'] == eid))
        assert row['episode'] in ['demo_' + str(i) for i in range(8)]
        self.source = h5py.File(ROOT / row['raw'], 'r')
        self.ns.update(source=self.source, args=types.SimpleNamespace(input=ROOT / row['raw'], episode=row['episode'], size=256, libero_root=pathlib.Path('/home/drh/project/ECV-VLA/vendor/LIBERO'), bddl_subdir='libero_object'), nframes=capacity)
        exec(compile_nodes(setup, OLD / 'src/render_episode.py'), self.ns)
        tr = next((n for n in source_with.body if isinstance(n, ast.Try)))
        exec(compile_nodes([n for n in tr.body if n.lineno in [43, 44, 45]], OLD / 'src/render_episode.py'), self.ns)
        self.env = self.ns['env']
        output_with = next((n for n in tr.body if isinstance(n, ast.With)))
        rgbdir = self.work / 'cache/rgbd'
        rgbdir.mkdir(parents=True, exist_ok=True)
        self.out = h5py.File(rgbdir / (eid + '.h5'), 'w')
        self.ns.update(output=self.out, nframes=capacity)
        init = [n for n in output_with.body if n.lineno < 94]
        exec(compile_nodes(init, OLD / 'src/render_episode.py'), self.ns)
        loop = next((n for n in output_with.body if isinstance(n, ast.For) and n.lineno == 94))
        self.capture_code = compile_nodes([n for n in loop.body if 99 <= n.lineno <= 125], OLD / 'src/render_episode.py')
        self.env.sim.set_state_from_flattened(self.ns['states'][0])
        self.env.sim.forward()
        self.env._update_observables(force=True)
        self.capture(0)
        self.source.close()

    def capture(self, t):
        assert t < self.capacity
        self.ns['index'] = t
        exec(self.capture_code, self.ns)
        self.out.flush()

    def step(self, action, t):
        action = np.asarray(action, dtype=np.float32)
        assert action.shape == (7,) and np.isfinite(action).all()
        (obs, reward, done, info) = self.env.step(action)
        self.capture(t)
        return (bool(self.env.check_success()), bool(done))

    def close(self):
        self.out.close()
        self.env.close()

class LiveProposals:

    def __init__(self, device='cpu'):
        path = OLD / 'src/observations.py'
        tree = ast.parse(path.read_text())
        tree.body = [n for n in tree.body if not (isinstance(n, ast.ImportFrom) and n.module == 'cotracker.predictor')]
        cls = next((n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Processor'))
        init = next((n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__init__'))
        init.body = [n for n in init.body if n.lineno in [34, 35, 36]]
        run = next((n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'run'))
        withnode = next((n for n in run.body if isinstance(n, ast.With)))
        loop = next((n for n in withnode.body if isinstance(n, ast.For) and n.lineno == 56))
        if device == 'cpu':
            module_from(ROOT / 'src/v3/stage2_extract_dino.py', 'stage2_extract_dino', CPU())
            tree = CPU().visit(tree)
        ns = {'__name__': 'live_proposals', '__file__': str(path)}
        exec(compile(ast.fix_missing_locations(tree), str(path), 'exec'), ns)
        old_tf32 = torch.backends.cudnn.allow_tf32
        torch.backends.cudnn.allow_tf32 = True
        self.proc = ns['Processor']()
        torch.backends.cudnn.allow_tf32 = old_tf32
        self.ns = ns
        self.ns.update(self=self.proc, previous={}, nextid=0, queries=[[], []], track_object=[[], []], counts=[])
        self.code = compile_nodes(loop.body, path)
        from torch.nn.attention import sdpa_kernel, SDPBackend
        original_generate = self.proc.sam.generate

        def deterministic_generate(*args, **kwargs):
            with sdpa_kernel(SDPBackend.MATH):
                return original_generate(*args, **kwargs)
        self.proc.sam.generate = deterministic_generate

    def frame(self, render, t):
        path = render.work / 'cache/observations' / (render.eid + '.h5')
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, 'a') as out:
            fr = out.require_group('frames')
            self.ns.update(h=render.out, out=out, frame_root=fr, t=t)
            old_tf32 = torch.backends.cudnn.allow_tf32
            try:
                torch.backends.cudnn.allow_tf32 = True
                with torch.inference_mode():
                    exec(self.code, self.ns)
            finally:
                torch.backends.cudnn.allow_tf32 = old_tf32
        return len(self.ns['current'])

class LiveInput:

    def __init__(self, eid, work, capacity=600, device='cpu', encoder=None):
        from data import INPUT_PROTOCOL_VERSION
        self.ds = Dataset.__new__(Dataset)
        self.ds.cached = False
        self.ds.old = pathlib.Path(work)
        self.ds._cache = collections.OrderedDict()
        self.ds.read_log = []
        self.ds.adapter_sha256 = sha(P / 'source/data.py')
        self.ds.input_protocol_version = INPUT_PROTOCOL_VERSION
        norm = json.loads((P / 'inputs.json').read_text())['normalization']
        self.ds.action_mean = np.asarray(norm['mean'], np.float32)
        self.ds.action_std = np.asarray(norm['std'], np.float32)
        self.ds.actions = {eid: np.zeros((capacity, 7), np.float32)}
        rows = json.loads((OLD / 'manifests/episodes.json').read_text())['episodes']
        self.ds.rows = {eid: dict(next((x for x in rows if x['id'] == eid)))}
        self.ds.rows[eid]['frames'] = capacity
        self.ds.instructions = {eid: 'pick up the bbq sauce and place it in the basket'}
        self.eid = eid
        self.frames = {}
        self.device = device
        self._geometry_frame = self.ds.frame
        self.ds.frame = lambda eid, t, n_points=2048: self.frames[t]
        self.enc = encoder if encoder is not None else FrozenSmolFeatures(R / 'weights/smolvlm2', device=device)
        text = np.load(P / 'text/train_task.npz')
        self.text = {k: text[k] for k in text.files}

    def frame(self, t):
        fr = self._geometry_frame(self.eid, t)
        sem = self.enc.point_features(fr['rgb'][None], fr['projected_uv'][None], fr['view_visible'][None])[0]
        fr['semantic'] = sem
        self.frames[t] = fr
        for old_t in list(self.frames):
            if old_t < t - 15:
                del self.frames[old_t]
        return fr

    def batch(self, t):
        w = self.ds.window(self.eid, t)
        return {k: torch.from_numpy(np.asarray(w[k])[None]).to(self.device) for k in KEYS + ['text_features', 'text_valid']}