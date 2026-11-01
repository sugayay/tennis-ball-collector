"""Check only the supplied person checkpoint's bytes and static serialization metadata."""
from pathlib import Path
import hashlib, json, pickletools, re, zipfile
ROOT = Path(__file__).resolve().parents[1]
path = ROOT / 'models/yolo26n-person-synthetic-v1.pt'
expected = '2591906d7ca45eeb9cd199380cf42a73e42b82af0bce30f138e5ce8aa3abc1d5'
actual = hashlib.sha256(path.read_bytes()).hexdigest()
if actual != expected:
    raise RuntimeError('Person model SHA-256 mismatch')
with zipfile.ZipFile(path) as archive:
    pickle_name = next(n for n in archive.namelist() if n.endswith('/data.pkl'))
    strings = [arg for op, arg, pos in pickletools.genops(archive.read(pickle_name)) if isinstance(arg, str)]
    if any(re.search(r'(?:(?<![A-Za-z])[A-Za-z]:[\\/]|/mnt/[a-z]/|/home/|/root/|/Users/)', value) for value in strings):
        raise RuntimeError('Absolute path in static checkpoint strings')
print(json.dumps({'person_model_sha256': actual, 'verified': True, 'pickle_executed': False}))
