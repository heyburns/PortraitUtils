# ComparisonGate

`ComparisonGate` makes sure your comparison viewer only updates when two real images are ready. It prevents half-populated A/B viewers while batches spin up or when one branch lags behind the others.

---

## Inputs
- `source_image` – Optional image you want to keep on the “reference” side. Leave it unplugged to let the node pick one of the comparisons instead.
- `image_a`, `image_b`, `image_c` – Up to three comparison streams. They can arrive in any order and at any time; empty values are ignored.

All sockets accept `None`, empty lists, or valid image tensors. The gate filters out anything that isn’t a real image.

---

## Outputs
- `final_image` – The first populated comparison stream (`image_a` first, then `b`, then `c`).
- `source_image` – The dedicated source input if it exists, otherwise the second non-empty comparison stream.

Both outputs return `None` until at least two images are available.

---

## Where It Fits

Drop ComparisonGate right before a two-input viewer or saver whenever you rely on asynchronous branches—classic example: A/B testing two upscalers or feeding matched before/after renders to a gallery.

---

## Tuning Tips

- Use it together with `PairedImageLoader` when you want before/after stills to advance in lockstep.
- If you only need a single “winner” image, you can connect just `image_a`; the gate will pass it through once it sees valid data.
- Combine the debug output of upstream nodes with this gate to log why a branch is delayed.

---

## Troubleshooting

- **Viewer still updates with one image** – Check that the viewer itself isn’t caching inputs; the gate always emits `(None, None)` until it sees two valid tensors.
- **Wrong image shows up on the reference side** – Supply the dedicated `source_image`; otherwise the gate promotes the second populated comparison by design.
- **Node never fires** – Make sure at least one branch produces an actual image tensor rather than an empty list or placeholder.

---

Screenshot: `docs/screenshots/comparison_gate.png`
