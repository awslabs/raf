import subprocess
import re

def cleanup(f_name, ind_dir):
    raw = subprocess.check_output(["gcc", "-E", f_name, ind_dir]).decode('utf-8')

    raw = re.sub('__attribute__\\(\\(.*\\)\\)', '', raw)
    raw = raw.replace('__host__', '')
    raw = raw.replace('__inline__', '')
    return raw
