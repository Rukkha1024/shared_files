"""
Compatibility shim for legacy imports.

The shared library package was renamed from `scr` to `src`.
New code should import from `src.*`, but existing scripts that import `scr.*`
continue to work via these re-export modules.
"""

