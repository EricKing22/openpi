"""Caching pi0 / pi0.5 hidden states during inference.

Submodules are imported explicitly rather than re-exported here, because
`openpi.probe.capture` is imported by the PyTorch model code on every run and should stay
cheap.

    capture    model-side hooks, CaptureConfig, session     (needs torch)
    recorder   ActivationRecorder writes shards, load_site reads them back
    codec      how bfloat16 is stored on disk               (numpy only)
"""
