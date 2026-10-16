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

On each trainer start, the cache rebuilds its row mapping from the currently
enumerated dataset. An old `cache_layout.pt` can no longer fix the epoch size
at a previous image count. The log reports source rows and balanced epoch samples.

Image and text caches use ordered, per-concept source fingerprints. Adding,
removing or renaming an image rebuilds both caches for the affected concept.
Changing an image, rejected DPO image, mask or conditioning image rebuilds its
image cache; changing a caption rebuilds the text cache. Other concepts keep
their valid cache. Global resolution/frame settings also identify the image
cache. Balancing is recalculated without reencoding unchanged sources.

Fingerprints use file paths, sizes and nanosecond modification/change timestamps.
**Clear Cache Before Training** remains useful when switching the VAE/text encoder
or deliberately restoring files while preserving their metadata. Existing old
cache groups are not deleted automatically; the first run after this update
builds groups with the new source identities.

Files added during an active run are picked up after restarting the trainer.
The saved resume position is retained, so the first resumed epoch can have fewer
remaining batches than its full epoch size. With batch size 1, repeats 1 and no
filtering, 923 discovered normal images give 923 samples per full epoch.
Larger batches, fixed SAMPLES balancing, holdout and aspect buckets can change
the displayed batch count. Disk use and initial VAE encoding work grow with the
number of distinct resolutions.

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
python -m unittest discover -s tests -p 'test_krea2_multires_integration.py' -v
```
