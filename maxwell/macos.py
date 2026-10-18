"""Keep Cocoa preferences volatile; never change the user's defaults on disk."""
import sys


def restrict_file_writes():
    """Run the macOS GUI under a project-only Seatbelt write policy.

    Cocoa may attempt asynchronous saved-state or cache writes despite its
    restoration settings. Enforce the boundary at the OS, not just in Python.
    This applies only to this process and its children, never system settings.
    """
    if sys.platform != 'darwin':
        return
    import json
    import os
    from pathlib import Path
    from . import ROOT
    marker = 'MAXWELL_PROJECT_WRITE_SANDBOX'
    if os.environ.get(marker) == str(ROOT):
        # Validate the actual kernel policy, not merely an inherited marker.
        # This is a permission query; it does not create or open an outside file.
        import ctypes
        check = ctypes.CDLL(None).sandbox_check
        # sandbox_check is variadic. On ARM64 its path argument must use the
        # variadic calling convention, so only the three fixed args go here.
        check.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
        check.restype = ctypes.c_int
        outside = os.fsencode(ROOT.parent/'maxwell-write-policy-check')
        if check(os.getpid(), b'file-write-data', 1, ctypes.c_char_p(outside)) <= 0:
            raise RuntimeError('Project-only write restriction could not be verified; refusing GUI launch.')
        return
    launcher = Path('/usr/bin/sandbox-exec')
    if not launcher.is_file():
        raise RuntimeError('macOS sandbox-exec is required to enforce project-only GUI writes.')
    policy = ('(version 1)\n(allow default)\n(deny file-write*)\n'
              f'(allow file-write* (subpath {json.dumps(str(ROOT))}))\n')
    os.environ[marker] = str(ROOT)
    os.execv(str(launcher), [str(launcher), '-p', policy, sys.executable,
                            str(ROOT/'main.py'), *sys.argv[1:]])


def prepare_cocoa():
    if sys.platform != 'darwin':
        return
    # Must run BEFORE pyglet imports app/window/gl (which initialize Cocoa).
    # Stock pyglet writes two preferences if they do not already exist.
    from pyglet.libs.darwin import cocoapy
    values = cocoapy.ObjCClass('NSMutableDictionary').dictionary()
    boolean = cocoapy.ObjCClass('NSNumber')
    for name, value in (('ApplePersistenceIgnoreState', True),
                        ('ApplePressAndHoldEnabled', False),
                        ('NSQuitAlwaysKeepsWindows', False)):
        values.setObject_forKey_(boolean.numberWithBool_(value), cocoapy.CFSTR(name))
    defaults = cocoapy.ObjCClass('NSUserDefaults').standardUserDefaults()
    defaults.setVolatileDomain_forName_(values, cocoapy.CFSTR('NSArgumentDomain'))


def disable_restoration():
    if sys.platform != 'darwin':
        return
    from pyglet.app.cocoa import _AppDelegate_Implementation
    delegate = _AppDelegate_Implementation._AppDelegate

    @delegate.method('B@')
    def applicationShouldSaveApplicationState_(self, app):
        return False

    @delegate.method('B@')
    def applicationShouldSaveSecureApplicationState_(self, app):
        return False

    @delegate.method('B@')
    def applicationShouldRestoreApplicationState_(self, app):
        return False

    @delegate.method('B@')
    def applicationShouldRestoreSecureApplicationState_(self, app):
        return False
