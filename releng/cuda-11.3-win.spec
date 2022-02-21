# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_data_files

block_cipher = None

datas = []
datas += collect_data_files("gpytorch", include_py_files=True)

a = Analysis(['..\\src\\gui.py'],
             pathex=[],
             binaries=[],
             datas=datas,
             hiddenimports=[],
             hookspath=[],
             hooksconfig={},
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)

exe = EXE(pyz,
          a.scripts,  
          a.binaries,
          a.zipfiles,
          a.datas,  
          [],
          # exclude_binaries=True,
          name='background-extraction-win-c11.3',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=False,
          console=True,
          disable_windowed_traceback=False,
          target_arch=None,
          codesign_identity=None,
          entitlements_file=None )
#coll = COLLECT(exe,
#               a.binaries,
#               a.zipfiles,
#               a.datas, 
#               strip=False,
#               upx=True,
#               upx_exclude=[],
#               name='background-extraction-win-c11.3')
