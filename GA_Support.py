import json
import os

def make_results_json(start_time, script_path):
    filename = 'evolution_results-%s.json' % (start_time)
    destdir = os.path.join(script_path, 'ga_runs')
    filepath = os.path.join(destdir, filename)
    if not os.path.exists(destdir):
        os.makedirs(destdir)
    json.dump(results, open(filepath, 'w'), indent=4)
    
    return filename

class UnsupportedBackendDecorator(Exception):
    """Exception raised for errors in the backend 'type' decoration.

    Attributes:
        bknd_type -- input backend decoration that caused the error
        message -- explanation of the error
    """

    def __init__(self, backend_type, message="Backend type not supported. Options are: 'high', 'mid', and 'low'."):
        self.bk_type = backend_type
        self.message = message
        super().__init__(self.message)