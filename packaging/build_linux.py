"""Build an onedir Linux x86_64 bundle from an isolated build environment."""
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
from datetime import datetime,timezone


def main():
    if sys.platform!='linux' or platform.machine()!='x86_64':
        raise SystemExit('This builder targets Linux x86_64')
    root=Path(__file__).resolve().parents[1];build=root/'build';build.mkdir(exist_ok=True)
    sys.path.insert(0,str(root/'src'))
    from trichodesmium import __version__
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    dirty=bool(subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip())
    data=dict(program_version=__version__,git_commit=revision,git_dirty=dirty,
              build_utc=datetime.now(timezone.utc).isoformat(),platform=platform.platform(),
              python=sys.version,glibc=platform.libc_ver(),pyinstaller=importlib.metadata.version('pyinstaller'),
              installed_packages=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True).splitlines())
    info=build/'linux-build-info.json';info.write_text(json.dumps(data,indent=2))
    args=[sys.executable,'-m','PyInstaller','--noconfirm','--clean','--onedir','--name','trichodesmium',
          '--distpath',str(build/'linux-dist'),'--workpath',str(build/'pyinstaller-work'),
          '--specpath',str(build),'--paths',str(root/'src'),
          '--collect-all','trichodesmium','--collect-all','pillow_heif','--collect-all','skimage',
          '--add-data',f'{info}:trichodesmium/assets', '--exclude-module','pytest',
          '--exclude-module','matplotlib','--exclude-module','tkinter']
    for name in ['numpy','scipy','opencv-python-headless','scikit-image','Pillow','pillow-heif','openpyxl','trichodesmium-morphometry']:
        args += ['--copy-metadata',name]
    args.append(str(root/'packaging/entry.py'))
    subprocess.run(args,cwd=root,check=True)
    print(build/'linux-dist/trichodesmium')


if __name__=='__main__':main()
