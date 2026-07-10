"""vision_model: aerial object detection + geo-tagging pipeline.

See `vision_model_architecture.md` (repo root) for the full design and
phased roadmap. Current implementation status:

    Phase 0 (scaffolding)      - done
    Phase 1 (MVP detection)    - done (datasets, models, inference, training,
                                  evaluation, visualization)
    Phase 2 (MVP geo-tagging)  - done (telemetry, camera model, projection,
                                  frame transforms, exporters)
    Phase 3 (hardening)         - done (config validation, checkpoint resume,
                                  CLI error handling, CI)
    Phase 4 (video/batch)        - done (streaming video inference, annotated
                                  video export)
    Phase 5+                     - not started (see the roadmap for scope)
"""

__version__ = "0.4.0"
