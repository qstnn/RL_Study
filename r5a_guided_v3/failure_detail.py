"""Explain the first native termination from recorded data without changing it."""
import json
from pathlib import Path
import numpy as np

def explain(folder, label='teacher'):
    folder = Path(folder)
    meta = json.loads((folder / (label + '_trace.json')).read_text())
    with np.load(folder / (label + '_trace.npz'), allow_pickle=False) as a:
        bad = np.flatnonzero(a['online_failed'])
        if not len(bad):
            return dict(failed=False, first_failure_s=None, contacts=[])
        i = int(bad[0]); t = a['t']
        origin = a['com'][np.flatnonzero(t <= .4)[-1], :2]
        contacts = []
        for j, name in enumerate(meta['contact_names']):
            f = a['contact_force'][i, j]
            if np.linalg.norm(f) > 5:
                contacts.append(dict(body=name, norm_N=float(np.linalg.norm(f)), vector_N=f.tolist()))
        nonwheel = float(a['body_force'][i]) > 5 and t[i] > .1
        out_of_bounds = np.linalg.norm(a['com'][i, :2] - origin) > 1.5
        reason = ('nonwheel_contact' if nonwheel else 'horizontal_bounds' if out_of_bounds
                  else 'launch_deadline' if t[i] >= 3 and not a['online_launched'][i] else 'grounded_fall')
        return dict(failed=True, first_failure_s=float(t[i]), reason=reason, contacts=contacts,
                    nonwheel_force_N=float(a['body_force'][i]), base_height_m=float(a['base_height'][i]))
