"""Download the official pretrained weight and verify the recorded SHA-256."""
import hashlib
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parent
MODEL = ROOT/'models'/'yolo26n.pt'
URL = 'https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt'
SHA256 = '9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef'


def main():
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    if MODEL.exists() and hashlib.sha256(MODEL.read_bytes()).hexdigest() == SHA256:
        print('Verified official YOLO26n weight already present.')
        return
    temporary = MODEL.with_suffix('.download')
    urllib.request.urlretrieve(URL, temporary)
    if hashlib.sha256(temporary.read_bytes()).hexdigest() != SHA256:
        temporary.unlink()
        raise RuntimeError('Downloaded model hash does not match; no model was installed.')
    temporary.replace(MODEL)
    print('Downloaded and verified official YOLO26n weight.')


if __name__ == '__main__':
    main()
