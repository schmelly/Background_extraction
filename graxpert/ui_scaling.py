import logging

from screeninfo import get_monitors
from platform import system
import os
from appdirs import user_config_dir
from graxpert.preferences import load_preferences

scaling_factor = None

prefs_filename = os.path.join(user_config_dir(appname="GraXpert"), "preferences.json")
prefs = load_preferences(prefs_filename)
factor = 1.0

if "scaling" in prefs:
    factor = prefs["scaling"]

def get_scaling_factor():
    global scaling_factor

    if scaling_factor is not None:
        return scaling_factor

    try:
        monitors = get_monitors()

        monitor = None
        if len(monitors) == 1:
            # use the only available monitor
            monitor = monitors[0]
        else:
            try:
                # try to query the primary monitor...
                monitor = next(mon for mon in monitors if mon.is_primary)
            except:
                # ... if that fails try the first one in the list
                monitor = monitors[0]
        
        dpi = monitor.width / (monitor.width_mm / 25.4)
        scaling_factor = dpi / 96.0
        
    except BaseException as e:
        logging.warning("WARNING: could not calculate monitor dpi, {}".format(e))
        scaling_factor = 1.0
    
    if isinstance(scaling_factor, float):
        return scaling_factor * factor
    else:
        return 1.0 * factor
