# SeedVR2Prep

`SeedVR2Prep` prepares heavy SeedVR2 workflows by warming up caches, priming key models, and cleaning up afterward. Think of it as a one-button “get ready to render” helper for VRAM-intensive runs.

---

## Inputs
- `run_warmup` – Execute the warmup pass, loading and touching key models so the first real render doesn’t spike.
- `warmup_steps` – Number of dummy steps to run during warmup. Higher values build a larger cache but take longer.
- `clear_after` – Free caches once the main job finishes. Leave on if you’re about to switch projects.
- `log_only` – When `True`, print what would happen without actually running warmup or cleanup. Useful for testing.
- `notes` – Free-form text stored alongside the log output.

---

## Outputs
- `log` – Summary of warmup and cleanup actions.
- `status` – Simple text flag (`ready`, `skipped`, `cleared`) for downstream branching.

---

## Where It Fits

Drop SeedVR2Prep at the start of VRAM-heavy SeedVR2 graphs when you need stable runtimes or when the first render tends to hitch due to lazy model loading. Run it again at the end to clean up before switching to a different project or model set.

---

## Tuning Tips

- Start with a modest `warmup_steps` count (e.g., 5) and increase only if the first render still stutters.
- Leave `clear_after` enabled on shared machines so other users get a clean slate.
- Toggle `log_only` during setup to verify the node sees the right models before committing to a full warmup.

---

## Troubleshooting

- **Warmup takes too long** – Lower `warmup_steps` or limit which models participate in warmup within your workflow.
- **VRAM still spikes on first render** – Ensure `run_warmup` is enabled and the node executes before the main generation block.
- **Cleanup skipped** – Check that `clear_after` is toggled on and the node executes after your main render branch.

---

Screenshot: `docs/screenshots/seedvr2_prep.png`
