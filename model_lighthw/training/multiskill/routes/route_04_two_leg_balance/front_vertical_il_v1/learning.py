"""Train only on independently verified, actually executed front teacher rollouts."""
from pathlib import Path
import numpy as np
import torch
from .common import read, write, sha, require, VERSION
from .model import Actor, basis
from .settings import OBSERVATION


def load_actor(path):
    value = torch.load(path, map_location='cpu', weights_only=False)
    if value['version'] != VERSION or value['observation_contract'] != OBSERVATION:
        raise ValueError('Only R4b vertical IL checkpoints are accepted')
    actor = Actor(); actor.load_state_dict(value['actor'], strict=True)
    if not all(torch.isfinite(v).all() for v in actor.state_dict().values()):
        raise ValueError('Nonfinite Actor')
    return actor.eval(), value


def fit(run, steps=3000):
    run = Path(run); require(run)
    evidence = read(run/'collect/verification.json')
    if evidence['passes'] != 2 or evidence['hold_seconds'] != 60:
        raise ValueError('Two physically verified 60s front teacher cases required before IL')
    trace = run/'collect/trace.npz'
    if sha(trace) != evidence['trace_sha256']: raise ValueError('Verified trace changed')
    with np.load(trace) as z:
        active = z['policy_active']; x = torch.from_numpy(z['policy_observation'][active]).float()
        y = torch.from_numpy(z['policy_action'][active]).float()
    torch.manual_seed(42); torch.set_num_threads(4)
    actor = Actor()
    pre = x[:, 305] < .5
    # Both entry and the executed rise are represented; long steady labels cannot swamp them.
    entry = torch.nonzero(pre & (x[:, 299] >= .5)).flatten()
    rise = torch.nonzero(pre & (x[:, 299] < .5)).flatten()
    if min(len(entry), len(rise)) < 10: raise ValueError('Insufficient executed entry/rise labels')
    opt = torch.optim.Adam(actor.prefix.parameters(), lr=3e-4)
    for step in range(steps):
        ids = torch.cat([group[torch.randint(len(group), (128,))] for group in (entry, rise)])
        loss = (actor.prefix(x[ids, :307])-y[ids]).square().mean()
        if not torch.isfinite(loss): raise ValueError('Nonfinite BC loss')
        opt.zero_grad(); loss.backward(); opt.step()
        if (step+1) % 500 == 0: print(f'[R4b IL] {step+1}/{steps} prefix_mse={loss.item():.8g}', flush=True)
    steady = x[:, 306] >= .999
    if int(steady.sum()) < 200: raise ValueError('Insufficient executed capture feedback labels')
    b = basis(x[steady]).double(); target = y[steady].double()
    with torch.no_grad():
        for action in range(16):
            cols = actor.mask[action].bool()
            weights = torch.linalg.lstsq(b[:, cols], target[:, action], driver='gelsd').solution
            actor.coefficients[action, cols] = weights.float()
        mse = float((actor(x)-y).square().mean())
    actor.prefix.requires_grad_(False)
    checkpoint = run/'student.pt'
    if checkpoint.exists(): raise FileExistsError(checkpoint)
    torch.save(dict(version=VERSION, observation_contract=OBSERVATION,
        actor=actor.state_dict(), teacher=read(run/'selected_teacher.json'),
        dataset_trace=str(trace.resolve()), dataset_sha256=sha(trace),
        bc_optimizer_steps=steps, feedback_closed_form_fits=1), checkpoint)
    write(run/'il_result.json', dict(checkpoint=str(checkpoint.resolve()), sha256=sha(checkpoint),
        frames=len(x), prefix_frames=int(pre.sum()), feedback_frames=int(steady.sum()),
        bc_optimizer_steps=steps, feedback_closed_form_fits=1, mse=mse,
        autonomous_verified=False, ppo_admission=False))
    print(f'[R4b IL] saved {checkpoint}; MSE is not autonomous standing evidence', flush=True)


def require_admission(run):
    run = Path(run); require(run)
    actor, saved = load_actor(run/'student.pt')
    report = read(run/'admission.json')
    if not report['autonomous_admission'] or report['actor_sha256'] != sha(run/'student.pt'):
        raise ValueError('Missing matching pure-student autonomous admission')
    for item in report['evidence']:
        path = Path(item['path'])
        if sha(path) != item['sha256']: raise ValueError('Admission evidence changed')
        verified = read(path)
        if verified['passes'] != 2 or not verified['teacher_off']:
            raise ValueError('Teacher evidence cannot admit PPO')
        if sha(path.parent/'trace.npz') != verified['trace_sha256']:
            raise ValueError('Admission trace changed')
    return actor, saved
