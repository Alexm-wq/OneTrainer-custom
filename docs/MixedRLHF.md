# Mixed preference training

The RLHF tab supports offline preference pairs alongside ordinary training images.
Krea 2/Kroma, Flux and Flux 2 use their native flow-matching error as a preference
score proxy. These scores are not exact image log probabilities.

## Three independent weights

- **Normal Training Weight**: supervised loss on unpaired normal images.
- **DPO Weight**: the selected preference objective, including its optional
  reward anchors. This does not multiply the Chosen Training Weight.
- **Chosen Training Weight**: supervised loss on the preferred half of each pair.

For a batch with N normal rows and P paired rows, the objective is:

`N/(N+P) * normal_weight * normal_loss + P/(N+P) * (dpo_weight * preference_loss + chosen_weight * chosen_loss)`

Each component is averaged over its own subbatch. A pair counts as one dataset
row. These weights do not change how frequently concepts are sampled; use concept
balancing for that. An all-normal batch still respects Normal Training Weight.
Zero disables that component's contribution. With RLHF disabled, normal training
retains its existing behavior.

Set the three new weights explicitly or use the new preset. The removed
Supervised Mix field is not imported. Internal backups can be resumed even
when optional RLHF state files are absent; see Backups and resume below.

## Starting settings

Select **Krea 2 > kroma Mixed DPO LoRA** for a new run. It uses the native Kroma
v0.3 Sensei Teacher Velocity transformer override
(`kroma-sensei-booru-e6-teacher-velocity-v0.3.safetensors`) and Krea 2's
supporting components. The [Kroma model card](https://huggingface.co/lodestones/Kroma#lora-training)
documents both teacher and base as suitable LoRA training checkpoints, with the
resulting LoRAs also usable on v0.3.1 Turbo OPD. This preset defaults to the teacher.
Loading the preset itself does not download weights; starting the trainer can download the
configured models on the machine running OneTrainer.

| Setting | Starting value |
| --- | --- |
| Objective | DPO / Sigmoid |
| Normal / DPO / Chosen weights | 1 / 1 / 0.1 |
| Beta | 300 |
| Beta Gradient Decouple / Reference | On / 1 |
| Label smoothing | 0.05 |
| Learning rate | 1e-5 |
| Reference | New Adapter / Base Reference |
| Batch / gradient accumulation | 1 / 4 |
| Resolution | 1024 |
| Adaptive beta / extra anchors / curriculum | Off |
| Validation | 10% holdout, seed 42, 4 fixed draws per pair |
| Validate every | 100 training microsteps, at the next completed optimizer step |
| Save Best / early stopping | On / Off |

These are conservative starting values, not benchmark-proven optimal settings.
Memory settings derive from the existing 24 GB Krea LoRA preset, but preference
training processes two images per pair and can require more VRAM. Adapt memory
and resolution settings to the training machine. Select your own concepts and
output paths. For continuing an existing LoRA, choose Existing Adapter Snapshot;
it captures that adapter before the first training step and persists in backups.

With gradient decoupling enabled and reference 1, beta controls the sigmoid
comparison while its explicit multiplicative gradient factor is canceled.
DPO Weight then controls the outer preference strength. Beta still affects
saturation and hence which pairs receive gradient.

## Streamed DPO (Low VRAM)

In the paired concept's General tab, enable **Streamed DPO (Low VRAM)**.
The setting is per concept, defaults to off, and is saved in concept files and
training backups. For example, use Path `/workspace/train/1440`, Include
Subdirectories on, Chosen Pattern `winner/{}`, Rejected Pattern `loser/{}`.
Patterns are relative to the concept root and must not start with a slash.

This ports preview's sequential reference forwards and policy replay/backward.
Each expensive model call processes B images rather than a combined 2B pair
batch. Policy value forwards retain no transformer activation graph; backward
recomputes and differentiates one branch at a time. It trades extra forwards
for lower activation memory. Native flow losses, the independent Normal/DPO/
Chosen weights, beta decoupling, anchors, validation and gradient accumulation
keep their existing semantics. Random state is restored during replay. Switching
the execution mode may change the random draws compared with a 2B forward, but
chosen/rejected and reference/policy within a pair share their draws.

Mixed batches can contain ordinary, batched-DPO and streamed-DPO concepts.
Each group's backward completes before the next reference swap/offload forward.
The optimizer still steps once per accumulation window. For MOSS, ordinary
captions and layout JSON use the same final cached conditioning on both sides;
the flag is read from the current concept and does not change cache identities.

**Fused Back Pass must be off.** On multiple GPUs, Fused Gradient Reduce must
also be off: reducing/updating inside individual branch backward calls would
act on incomplete gradients. Regular accumulation and end-of-step reduction
remain supported. Preview's restriction on combining multi-GPU streamed DPO
with layer/activation offloading is retained. Single-GPU reentrant checkpointing
uses direct gradient accumulation.

Focused CPU tests compare loss values and LoRA gradients across all existing
objectives/references, mixed group weights, accumulation, checkpoint replay,
validation, concept persistence and path diagnostics. A mocked multi-GPU route
checks its vector-Jacobian computation; this is not a real distributed/GPU
memory or performance benchmark.

## Correctness and validation

Concept weights and batch/accumulation loss scaling are applied outside the
preference objective. Changing them no longer changes beta's effective temperature,
the reward margin or anchor units. Native masks and timestep loss weights remain
part of the error proxy.

Reference swapping precedes every policy graph in a mixed batch. Adapter dropout
is disabled throughout RLHF forward and backward, including normal rows within
that step, so checkpoint recomputation uses the same dropout behavior. Ordinary
training with RLHF off keeps its configured dropout.

Pair IDs include their concept root. Curriculum weights only reduce rejection
pressure; chosen protection and pair-order corrections retain full weight.
Curriculum state uses version 3 with the new pair IDs and score units.

The holdout is selected by a stable hash before augmentation, caching, balancing
or resolution expansion. All variants of a pair stay in one split. The percentage
is approximate; small datasets may need a larger percentage or explicit validation
concepts. Explicit validation concepts are included in addition to automatic
holdout. Avoid duplicating the same images under different dataset roots.

Validation reuses its dataset and fixed noise/timestep draws, disables random
augmentations, uses fixed base beta, and never updates training curriculum,
adaptive beta or pair CSV histories. TensorBoard reports pair objective, relative
preference accuracy, margin, chosen/rejected rewards and native errors. Accuracy
here measures improvement relative to the reference; inspect generated samples
as well before judging image quality.

Save Best writes `save/<prefix>best-dpo<extension>` and a metrics JSON with the
weights actually evaluated, without substituting EMA. Patience counts validation
evaluations without a lower unweighted pair objective. It stops only after a
completed optimizer step, synchronously across GPUs. Adaptive beta updates once
per optimizer step from the global pair-weighted mean margin; state and validation
progress are included in internal backups.

The cache namespace changes to avoid reusing legacy pair IDs or stale split
membership. Training and DPO validation use separate cache directories. Existing
cache files are not deleted automatically.

## Backups and resume

Internal backups save model/adapter weights, optimizer and training progress
together with these RLHF sidecars:

- `onetrainer_rlhf_state.json`: adaptive-beta controller, runtime beta,
  best validation score and patience counter. The JSON is written atomically.
- `onetrainer_dpo_reference.pt`: the original frozen reference for Existing
  Adapter mode, rather than the current trained policy.
- `onetrainer_dpo_hard_pair_curriculum.json`: per-pair statistics when
  Hard-Pair Curriculum is active.

A missing sidecar does not abort resume. This also allows enabling RLHF when
continuing an older or supervised backup. The trainer logs the missing path
and initializes only that part of RLHF: beta/validation tracking starts fresh,
missing curriculum statistics start empty, and a missing Existing Adapter
reference is captured from the resumed adapter before the first training step.
That last fallback changes the preference anchor and is explicitly reported;
the unavailable original reference cannot be reconstructed. Beta, validation and
curriculum statistics also restart when re-anchoring because they depend on the
old reference. New Adapter mode
continues to use the base model and does not need a reference snapshot.

Existing sidecars are restored, so missing trainer JSON never replaces an
available fixed reference. Subsequent backups include the newly initialized
state. Model weights, optimizer state and training progress are not reset by
these fallbacks. Corrupt/incompatible files that are present still report
their errors rather than being silently discarded.

## Regression coverage

GitHub Actions runs CPU autograd tests with real hooked LoRA layers, including
mixed batches, existing-adapter references, checkpointed backward, dropout,
independent weights, all preference objectives, native Krea/Flux scores,
deterministic validation, holdout selection, adaptive beta, best-checkpoint
selection, patience, config serialization and UI bindings. It also runs the existing
Krea checkpoint and multi-resolution cache tests. This does not replace a
full Kroma training run on a GPU.
