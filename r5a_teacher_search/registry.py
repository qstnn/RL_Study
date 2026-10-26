import importlib
MODULES={'latent':'control','actions':'control_actions','pose':'control_pose','sequence':'control_sequence','pulse':'control_pulse','dense':'control_dense'}
def controller(family):
 if family not in MODULES:raise ValueError('Unknown controller family: '+str(family))
 return importlib.import_module('r5a_teacher_search.'+MODULES[family])
