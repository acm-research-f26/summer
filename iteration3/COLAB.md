# Iteration 3, run 2 — v4 calibration run (Colab run sheet)

This run re-scores VISION for all 546 screens with the new **v4 calibrated
prompt** (the fix for the VLM's 0.5 hedge), then rebuilds every iteration-3
output on top of it and writes a v3-vs-v4 calibration report.

**This run needs a GPU runtime** (Runtime → Change runtime type → T4 GPU is
fine): unlike run 1, the VLM actually executes — 546 fresh screens, expect
roughly 2–4 h on a T4. It is fully resumable: every image is cached to Drive
the moment it finishes, so a disconnect costs only the image in flight —
reconnect, rerun Cells 1–4, and rerun the long cell; it skips everything
cached.

What is reused vs fresh:

- **Reused (never re-run)**: OCR/text scores, the frozen v3 vision results
  (they become the calibration baseline), ground-truth labels.
- **Fresh**: v4 vision predictions (new cache keys `*_v4_*` — the v3 cache is
  untouched), corpus scores, all reports, stimulus set, synthetic run (now
  n=800).

Assumes the same two Drive prerequisites as run 1: an `acmwork*.zip` in
MyDrive root (zip the project AFTER these code changes), and the iteration-2
cache at `MyDrive/acmwork_outputs/cache/`.

---

## Cell 1 — mount Drive, locate + unpack the project

```python
from google.colab import drive
drive.mount('/content/drive')

import os, shutil, sys, zipfile
from pathlib import Path

DRIVE = Path('/content/drive/MyDrive')
DEST = Path('/content/acmwork')

if (DEST / 'iteration3/config.py').exists():
    print('project already unpacked at', DEST)
else:
    src = None
    zips = sorted(DRIVE.glob('acmwork*.zip'), key=lambda p: p.stat().st_mtime,
                  reverse=True)
    if zips:
        print('unzipping', zips[0].name, '…')
        shutil.rmtree('/content/_unzip', ignore_errors=True)
        with zipfile.ZipFile(zips[0]) as z:
            z.extractall('/content/_unzip')
        hit = next(Path('/content/_unzip').rglob('iteration2/config.py'), None)
        if hit:
            src = hit.parent.parent
    if src is None:
        hit = next(DRIVE.rglob('iteration2/config.py'), None)
        if hit:
            print('copying project folder from Drive:', hit.parent.parent)
            src = hit.parent.parent
    if src is None:
        raise FileNotFoundError(
            'could not find the project: put acmwork*.zip in MyDrive root, '
            'or an unzipped folder containing iteration2/config.py'
        )
    shutil.rmtree(DEST, ignore_errors=True)
    if src.is_relative_to('/content/_unzip'):
        shutil.move(str(src), DEST)
    else:
        shutil.copytree(src, DEST)
    print('project ready at', DEST)

# this run sheet only works with the v4 code — catch a stale zip immediately
cfg = (DEST / 'iteration2/config.py').read_text()
assert 'PROMPT_VERSION = "v4"' in cfg, (
    'iteration2/config.py still says v3 — you zipped the project before the '
    'calibration changes; re-zip and re-upload'
)
print('zip carries the v4 calibration code: OK')

for rel in ('iteration2/src/vlm_model.py', 'iteration2/src/dataset_contextdp.py',
            'iteration2/data/own_screenshots/labels.csv',
            'iteration3/src/calibration_report.py'):
    status = 'ok' if (DEST / rel).exists() else 'MISSING'
    print(f'  {status:8s} {rel}')
    assert status == 'ok', f'{rel} missing — re-zip the project with it included'
```

## Cell 2 — get the CONTEXTDP dataset (clone if the zip doesn't carry it)

```python
import shutil, subprocess
from pathlib import Path

DEST = Path('/content/acmwork')
DRIVE = Path('/content/drive/MyDrive')
CTX = DEST / 'iteration2/AidUI/evaluation/evaluation_dataset'

def ctx_ok(root: Path) -> bool:
    return all((root / p / 'result.json').exists() for p in ('web', 'mobile'))

if not ctx_ok(CTX):
    drive_candidates = [
        DRIVE / 'AidUI/evaluation/evaluation_dataset',
        DRIVE / 'acmwork_outputs/AidUI/evaluation/evaluation_dataset',
    ]
    src = next((c for c in drive_candidates if ctx_ok(c)), None)
    if src is not None:
        print('copying CONTEXTDP from Drive:', src)
        CTX.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, CTX, dirs_exist_ok=True)
    else:
        print('cloning SageSELab/AidUI (shallow) …')
        shutil.rmtree('/content/_aidui', ignore_errors=True)
        subprocess.run(['git', 'clone', '--depth', '1',
                        'https://github.com/SageSELab/AidUI', '/content/_aidui'],
                       check=True)
        src = Path('/content/_aidui/evaluation/evaluation_dataset')
        assert ctx_ok(src), ('clone succeeded but evaluation_dataset layout is '
                             f'unexpected under {src} — inspect /content/_aidui')
        CTX.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), CTX)
        shutil.rmtree('/content/_aidui', ignore_errors=True)

assert ctx_ok(CTX), f'CONTEXTDP still incomplete at {CTX}'
for plat in ('web', 'mobile'):
    n = len(list((CTX / plat / 'images').glob('*')))
    print(f'CONTEXTDP {plat}: {n} images')
n_own = len(list((DEST / 'iteration2/data/own_screenshots').glob('*.png')))
print(f'own set: {n_own} screenshots (want 45)')
```

## Cell 3 — link the iteration-2 cache from Drive and inspect it

```python
import json, os, shutil, sys
from pathlib import Path

DEST = Path('/content/acmwork')
DRIVE = Path('/content/drive/MyDrive')

# symlink iteration2/data/cache -> Drive. OCR and the v3 vision results are
# REUSED from here; the v4 run WRITES new per-image files here as it goes,
# which is exactly what makes the long cell resumable.
cache_src = DRIVE / 'acmwork_outputs/cache'
assert (cache_src / 'ocr').exists() and (cache_src / 'vlm').exists(), (
    f'iteration-2 cache not found at {cache_src}'
)
cache_dst = DEST / 'iteration2/data/cache'
if cache_dst.is_symlink():
    cache_dst.unlink()
elif cache_dst.exists():  # a real dir from the zip — set aside, don't delete
    shutil.move(str(cache_dst), str(cache_dst) + '_local_backup')
cache_dst.parent.mkdir(parents=True, exist_ok=True)
cache_dst.symlink_to(cache_src)
print('cache linked:', cache_dst, '->', cache_src)

# baseline (v3) inputs that must already exist, and v4 progress so far
must_exist = {
    'ocr/system_a_contextdp.json  (want 501)': cache_dst / 'ocr/system_a_contextdp.json',
    'ocr/system_a_own.json        (want 45)':  cache_dst / 'ocr/system_a_own.json',
    'vlm/system_b_contextdp_v3.json (want ~501)': cache_dst / 'vlm/system_b_contextdp_v3.json',
    'vlm/system_b_own_v3.json     (want 45)':  cache_dst / 'vlm/system_b_own_v3.json',
}
for label, p in must_exist.items():
    n = len(json.loads(p.read_text())) if p.exists() else 'MISSING'
    print(f'  {label:45s} -> {n}')
    assert n != 'MISSING', f'{p} is required (v3 baseline / text scores)'

n_v3 = len(list((cache_dst / 'vlm').glob('*_v3_*.json')))
n_v4 = len(list((cache_dst / 'vlm').glob('*_v4_*.json')))
print(f'per-image vlm cache: v3 {n_v3} (frozen) | v4 {n_v4} '
      f'(0 on the first attempt; >0 means an earlier v4 run partially '
      f'completed and will be resumed, not redone)')

# stale run-1 artifacts inside the zip would carry local paths — always rebuild
for rel in ('iteration3/data/truth_contextdp.csv', 'iteration3/data/truth_own.csv',
            'iteration3/data/corpus_scores.csv', 'iteration3/data/stimuli.csv',
            'iteration3/data/responses/responses.csv'):
    p = DEST / rel
    if p.exists():
        p.unlink()
        print('removed stale', rel)
```

## Cell 4 — dependencies + GPU check

```python
%pip -q install "transformers>=4.45" accelerate bitsandbytes qwen-vl-utils \
    pandas numpy scipy matplotlib flask pytest
import torch
assert torch.cuda.is_available(), (
    'no GPU — Runtime → Change runtime type → T4 GPU, then rerun from Cell 1'
)
print('GPU:', torch.cuda.get_device_name(0))
```

## Cell 5 — preliminary checks (must be green BEFORE any GPU time)

```python
import os, sys
os.chdir('/content/acmwork/iteration3')
!python -m pytest tests -q
```

```python
# version wiring: both configs on v4, baseline still v3
import os, sys
os.chdir('/content/acmwork/iteration3')
if '/content/acmwork/iteration3' not in sys.path:
    sys.path.insert(0, '/content/acmwork/iteration3')
import config as c3
assert c3.IT2_PROMPT_VERSION == 'v4' and c3.IT2_PROMPT_VERSION_BASELINE == 'v3'
print('iteration3: primary', c3.IT2_PROMPT_VERSION,
      '| baseline', c3.IT2_PROMPT_VERSION_BASELINE)
for k, p in c3.BASELINE_VISION_AGGREGATES.items():
    assert p.exists(), f'v3 baseline aggregate missing: {p}'
print('v3 baseline aggregates present')
```

## Cell 6 — PART 1: dev-subset gate (25 screens, ~10 GPU-min)

Run v4 on the documented 25-screen dev subset first. **Do not start the full
run unless this prints GO.**

```python
import os
os.chdir('/content/acmwork/iteration2')
!python -m src.run_comparison --dataset contextdp --limit 25 --system b
```

```python
# hedge check on those 25: v4 vs the frozen v3 scores for the same screens
import json
from pathlib import Path

cache = Path('/content/acmwork/iteration2/data/cache/vlm')
v4 = json.loads((cache / 'system_b_contextdp_dev25_v4.json').read_text())
v3_full = json.loads((cache / 'system_b_contextdp_v3.json').read_text())
cats = ['urgency', 'scarcity', 'social_proof', 'guilt_wording', 'other']

def band_share(preds, names):
    scores = [p['categories'][c] for f in names if (p := preds.get(f)) for c in cats]
    return sum(0.45 <= s <= 0.55 for s in scores) / len(scores), scores

names = list(v4)
v3_share, _ = band_share(v3_full, names)
v4_share, v4_scores = band_share(v4, names)
n_fail = sum(bool(p.get('parse_failed')) for p in v4.values())
print(f'score mass in [0.45, 0.55]:  v3 {v3_share:.1%}  ->  v4 {v4_share:.1%}')
print(f'v4 parse failures: {n_fail}/25')
print(f'v4 distinct scores used: {sorted(set(round(s, 2) for s in v4_scores))}')
if v4_share <= v3_share / 2 and n_fail <= 2:
    print('\nGO — hedging clearly reduced; run Cell 7.')
else:
    print('\nNO-GO — v4 did not clearly reduce hedging (or parsing broke).')
    print('Stop here: back up with Cell 10 and revisit the prompt before')
    print('spending GPU time on the full run.')
```

## Cell 7 — PART 2: full v4 vision run (546 screens — the long cell)

Safe to interrupt/resume: rerunning skips every cached image. Uses
`--system b` only, which never touches the run-1 text-vs-vision reports
(the `write_headline_report` clobber needs both systems).

```python
import os
os.chdir('/content/acmwork/iteration2')
!python -m src.run_comparison --dataset contextdp --system b
!python -m src.run_comparison --dataset own --system b
```

Expected finish line: `system_b_contextdp_v4.json` with 501 screens and
`system_b_own_v4.json` with 45 (both land in the Drive cache automatically).

## Cell 8 — PART 3: rebuild iteration-3 on v4 + calibration report

```python
import os
os.chdir('/content/acmwork/iteration3')
!python -m src.score_corpus
!python -m src.calibration_report
```

Check the printouts: 546 rows; `vision scores from prompt v4`; on the
`[calibration]` line, hedge band share should drop sharply, severity
std/range should widen, and macro-F1 should hold within 0.05.

## Cell 9 — PART 4: map, stimuli, synthetic participants, analysis

```python
import os
os.chdir('/content/acmwork/iteration3')
!python -m src.prevalence_potency
!python -m src.stimulus_select
!python -m src.synthetic_participants
!python -m src.analysis
```

`synthetic_participants` now defaults to n=800, so no flag is needed. The
stimulus set will differ from run 1: it is re-selected on the new, wider v4
severities.

```python
from IPython.display import Image, display
for f in ('reports/calibration_v3_v4.png', 'reports/prevalence_potency.png',
          'reports/severity_vs_behavior.png'):
    display(Image(f))
```

## Cell 10 — back up everything to Drive as it3outputsv2

```python
import shutil
from pathlib import Path

out = Path('/content/drive/MyDrive/it3outputsv2')
out.mkdir(parents=True, exist_ok=True)

it3 = Path('/content/acmwork/iteration3')
it2 = Path('/content/acmwork/iteration2')

if (it3 / 'reports').exists():
    shutil.copytree(it3 / 'reports', out / 'reports', dirs_exist_ok=True)
for f in (it3 / 'data').glob('*.csv'):
    shutil.copy(f, out / f.name)
if (it3 / 'data/responses').exists():
    shutil.copytree(it3 / 'data/responses', out / 'responses', dirs_exist_ok=True)

# v4 run artifacts from iteration2, for offline analysis of the raw scores
agg = out / 'v4_aggregates'
agg.mkdir(exist_ok=True)
for pat in ('system_b_*_v4.json',):
    for f in (it2 / 'data/cache/vlm').glob(pat):
        shutil.copy(f, agg / f.name)
for name in ('vlm_parse_failures.md', 'dev_subset.txt'):
    p = it2 / 'reports' / name
    if p.exists():
        shutil.copy(p, out / name)

print('backed up to', out)
!ls -laR /content/drive/MyDrive/it3outputsv2
```
