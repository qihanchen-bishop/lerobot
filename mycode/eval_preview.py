"""Bounded background segmentation; the worker never accesses Tk or robot hardware."""
from concurrent.futures import ThreadPoolExecutor
import time


class AsyncSegmentationPreview:
    def __init__(self, model_factory, interval_s=.2):
        self.model_factory = model_factory
        self.interval_s = interval_s
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="eval-segmentation")
        self.future = None
        self.context = None
        self.generation = 0
        self.last_submitted = -float("inf")
        self.models = {}
        self.signature = None
        self.closed = False

    def reset(self):
        self.generation += 1
        self.last_submitted = -float("inf")

    def submit(self, images, paths, device, eval_token, *, recorder=None, source=None):
        now = time.monotonic()
        if self.closed or self.future is not None or now - self.last_submitted < self.interval_s:
            return False
        self.context = (self.generation, eval_token, now)
        snapshot = {key: image for key, image in images.items() if key in ("front", "side")}
        self.future = self.executor.submit(self._compute, snapshot, dict(paths), device, recorder, source)
        self.last_submitted = now
        return True

    def _compute(self, images, paths, device, recorder=None, source=None):
        signature = (tuple(sorted(paths.items())), device)
        if signature != self.signature:
            self.models = {}
            self.signature = signature
        overlays, summaries, measurements, loaded = {}, {}, {}, {}
        for view, image in images.items():
            model = self.models.get(view)
            if model is None:
                model = self.model_factory(paths[view], device)
                self.models[view] = model
            colors, descriptions, metrics = model.predict({view: image})
            if recorder is not None and source is not None:
                # Reuse IDs before visualization's boxes/text; no extra model call.
                palette = getattr(model.model, "bundle_config", {}).get("palette_rgb")
                if palette is not None and view in model.latest_class_ids:
                    recorder.update(view, model.latest_class_ids[view], palette, source, model)
            overlays.update(colors)
            summaries.update(descriptions)
            measurements.update(metrics)
            loaded[view] = (paths[view], model.loaded_device)
        return overlays, summaries, measurements, loaded

    def poll(self):
        if self.future is None or not self.future.done():
            return None
        future, context = self.future, self.context
        self.future = None
        self.context = None
        if context[0] != self.generation:
            # Still observe worker errors, but never apply old models to a new run.
            future.result()
            return None
        return context, future.result()

    def pending_for(self, eval_token):
        return (self.future is not None and self.context[0] == self.generation
                and self.context[1] == eval_token)

    def close(self):
        self.closed = True
        self.executor.shutdown(wait=False, cancel_futures=True)
