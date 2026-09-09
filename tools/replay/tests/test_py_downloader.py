"""Compile the real C++ bridge with only its logging dependency stubbed.

The child is a local fake downloader: no network, credentials, CAN, or live Params.
"""
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


def test_cpp_downloader_progress_results_and_cancellation(tmp_path):
  compiler = shutil.which('c++')
  if compiler is None:
    pytest.skip('requires a C++ compiler')
  root = Path(__file__).resolve().parents[3]
  include = tmp_path / 'tools/replay'
  include.mkdir(parents=True)
  (include / 'util.h').write_text('''#pragma once
#include <cstdarg>
#include <cstdio>
inline void rWarning(const char *fmt, ...) {
  va_list args; va_start(args, fmt); vfprintf(stderr, fmt, args); va_end(args);
}
''')
  child = tmp_path / 'python3'
  child.write_text(f'#!{sys.executable}\n' + '''import sys, time
mode = sys.argv[4]
print('ordinary diagnostic', file=sys.stderr, flush=True)
print('PROGRESS:bad', file=sys.stderr, flush=True)
print('PROGRESS:1:0', file=sys.stderr, flush=True)
print('PROGRESS:-1:10', file=sys.stderr, flush=True)
print('PROGRESS:11:10', file=sys.stderr, flush=True)
print('PROGRESS:4:10', file=sys.stderr, flush=True)
if mode == 'cancel':
  time.sleep(30)
if mode == 'fail':
  sys.exit(1)
for _ in range(3000):
  print('PROGRESS:10:10', file=sys.stderr)
print('/local/result')
''')
  child.chmod(0o755)
  harness = tmp_path / 'check.cc'
  harness.write_text('''#include <cassert>
#include <chrono>
#include <thread>
#include "tools/replay/py_downloader.h"
int main() {
  int updates = 0, failures = 0;
  uint64_t last = 0;
  std::atomic<bool> abort{false};
  bool cancel_on_progress = false;
  installDownloadProgressHandler([&](uint64_t cur, uint64_t total, bool ok) {
    if (!ok) { ++failures; return; }
    assert(total == 10 && cur <= total);
    ++updates; last = cur;
    if (cancel_on_progress) abort = true;
  });
  assert(PyDownloader::download("ok") == "/local/result");
  assert(updates == 3001 && last == 10 && failures == 0);
  assert(PyDownloader::download("fail").empty());
  assert(failures == 1);
  cancel_on_progress = true;
  assert(PyDownloader::download("cancel", true, &abort).empty());
  assert(failures == 2);
  installDownloadProgressHandler(nullptr);
  assert(PyDownloader::download("ok") == "/local/result");
}
''')
  binary = tmp_path / 'check'
  subprocess.run([compiler, '-std=c++17', '-pthread', '-Wall', '-Wextra', '-Werror',
                  '-I', str(tmp_path), '-I', str(root), str(harness),
                  str(root / 'tools/replay/py_downloader.cc'), '-o', str(binary)], check=True, timeout=60)
  result = subprocess.run([str(binary)], env={**os.environ, 'PATH': f'{tmp_path}{os.pathsep}{os.environ["PATH"]}'},
                          capture_output=True, text=True, timeout=15)
  assert result.returncode == 0, result.stderr
  assert result.stdout == ''
  assert 'ordinary diagnostic' in result.stderr
  assert 'PROGRESS:' not in result.stderr
