"""Run the unchanged evaluator and record OS process peak resident memory.

Pass the same arguments as evaluate.py, including an explicit --output path.
Resource results are written beside it as *.resources.json.
"""
import ctypes
import json
import os
from pathlib import Path
import sys
import evaluate


def peak_rss_bytes():
    if os.name == 'nt':
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in (
                    'PeakWorkingSetSize', 'WorkingSetSize', 'QuotaPeakPagedPoolUsage',
                    'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage',
                    'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage')]
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        psapi = ctypes.WinDLL('psapi', use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return counters.PeakWorkingSetSize
    import resource
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == 'darwin' else peak * 1024


if __name__ == '__main__':
    if '--output' not in sys.argv:
        raise SystemExit('Supply --output to identify the resource report.')
    output = Path(sys.argv[sys.argv.index('--output') + 1])
    evaluate.main()
    peak = peak_rss_bytes()
    result = {'peak_process_rss_bytes': peak, 'peak_process_rss_gib': peak / 2**30,
              'scope': 'whole process including imports, checkpoint and data loading, and scoring'}
    output.with_suffix('.resources.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))
