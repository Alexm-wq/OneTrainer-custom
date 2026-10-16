# Multi-resolution image caching

The shared text-to-image dataloader, including Krea 2 / Kroma, now uses the
multi-resolution cache from `preview` commit
`7e1899241dcf5f6cf068f5f5c614b02dbbdbf89b`, also used by the
`mage-flow-full` integration.

Enable **Image Caching** and enter the desired comma-separated resolutions, for
example `512,768,1024`, in the training resolution field. Concept resolution
overrides are respected. Each image variation is encoded at every configured
resolution. During training one resolution is selected per image per epoch;
the selection visits all resolutions before starting a new cycle. Two
resolutions alternate. Image Variations controls augmentation variants, not
the number of resolutions.

The number of training samples per epoch is unchanged. DPO chosen/rejected
latents and masks share the selected resolution. Text caching retains the
existing separate cache.

Use a new cache directory or enable **Clear Cache Before Training** for the
first run after updating, and whenever source data, resolution settings, or
augmentation settings change. Afterwards disable that option to reuse the
completed cache between runs. Disk use and initial VAE encoding work grow
with the number of distinct resolutions.

## Implementation provenance

The cache, resolution selector, aspect calculation, and cache I/O are copied
from `preview/src/mgds` into `modules/dataLoader/cache`. They use the existing
pinned MGDS pipeline, so no untracked local MGDS changes or import-path
overrides are required. Imported class paths are adjusted for this location.
Resolution lists are deduplicated before selecting variants; non-bucketed
calculation preserves leading frame dimensions used by Krea 2's image VAE.
The original cache I/O helper uses the cryptography dependency already present
in the supported locked OneTrainer environments. Encryption and source-free
cache-only UI settings are not enabled by this port.

Aspect calculation derives from Nerogar/MGDS, licensed under Apache-2.0; see
`modules/dataLoader/cache/LICENSE-MGDS.txt`.

## Tests

```sh
python -m unittest discover -s tests -p 'test_multi_resolution_cache.py' -v
python -m unittest discover -s tests -p 'test_krea2_checkpoint.py' -v
```
