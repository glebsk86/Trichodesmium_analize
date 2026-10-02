# Working on this project

Read SPEC.md and README.md before changing the pipeline. Preserve traceability
from output rows to original photographs and object masks.

- Keep this runnable on Python 3.10+, macOS and Linux, without mandatory GPU.
- Do not silently assign a physical scale, reconstruct hidden filament parts,
  classify species from width alone, or promote texture peaks to verified cells.
- Separate segmentation, calibration, geometry and reporting modules.
- Preserve missing values and per-image errors. Never overwrite a run directory.
- When a bug or algorithm change affects measurements, add a test with known
  geometry or a reviewed real example, run the relevant tests, and update docs.
- Synthetic success does not establish performance on biological photographs.
- Keep source images and generated measurements out of Git unless deliberately
  included as documented reference fixtures. Do not commit large private datasets.
- Use a feature branch for substantial changes and record the final change in Git.
- The report must continue recording code revision, dirty state, parameters,
  dependency versions and input hashes.
