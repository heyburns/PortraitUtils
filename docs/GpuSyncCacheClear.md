# GpuSyncCacheClear

`GpuSyncCacheClear` forces a quick CUDA sync and clears selected GPU caches. It’s a maintenance node that helps recover VRAM between heavy steps so long pipelines stay stable.

---

## Inputs
- `sync_cuda` – When enabled, calls `torch.cuda.synchronize()` to finish outstanding GPU work before clearing memory.
- `empty_cache` – Clears PyTorch’s allocator cache so unused memory is released back to the driver.
- `collect_garbage` – Runs Python’s garbage collector to free objects that might still hold GPU references.
- `delay_ms` – Optional pause (in milliseconds) after the cleanup. Useful when the next node needs a short breather.

---

## Outputs
- `log` – Text summary of actions taken and how long each step required.

---

## Where It Fits

Insert this node between VRAM-heavy stages—e.g., after a large upscaler, before a memory-intensive diffusion pass, or while alternating between different models on a shared GPU.

---

## Tuning Tips

- Leave `sync_cuda` on when running asynchronous operations; it ensures everything finishes before cleaning.
- Use the delay only when you know a downstream node immediately restarts another heavy task.
- Pair with status logging (e.g., ComfyUI console or a text overlay) so you can confirm the cleanup is happening during long renders.

---

## Troubleshooting

- **No VRAM freed** – Some drivers hold onto memory until the next allocation; keep `empty_cache` and `collect_garbage` enabled for the best chance of recovery.
- **Pipeline slows down** – Reduce how often you call the node or disable the delay; each cleanup adds a small pause.
- **Unexpected errors** – Make sure the GPU is still initialised when the node runs; this helper assumes CUDA is available.

---

Screenshot: `docs/screenshots/gpu_sync_cache_clear.png`
