# MultiPromptNode

`MultiPromptNode` cycles through a list of prompt variants, blending each option with a weight. It’s a fast way to explore phrasing tweaks or maintain a rotating pool of detail prompts without editing text nodes manually.

---

## Inputs
- `seed` – Optional number to lock the random order. Leave at `0` to use the global seed.
- `prompts` – List of prompt strings. Each entry corresponds to a slot in the UI.
- `weights` – Matching list of weights; higher numbers make that prompt more likely to appear.
- `mode` – Choose how the node selects variants:
  - `Cycle` walks through the list in order.
  - `Random` picks based on weight distribution.
  - `Shuffle` seeds a random order once, then cycles through it.
- `fallback_prompt` – Used when all prompts are blank or disabled.

Slots can be enabled or disabled individually from the widget panel.

---

## Outputs
- `prompt` – The selected prompt string ready for downstream text-to-image nodes.
- `index` – Slot number that triggered this run (useful for logging).

---

## Where It Fits

Use MultiPromptNode when iterating around a core idea—e.g., alternating between lighting descriptions, swapping wardrobe notes, or rotating through portrait adjectives while keeping the base prompt steady.

---

## Tuning Tips

- Set weights to `0` to temporarily mute a slot without removing it.
- Combine `Cycle` mode with `ComparisonGate` to run predictable A/B/C tests.
- Use `Random` with a fixed seed when you want reproducible randomness across multiple runs.

---

## Troubleshooting

- **Same prompt repeats** – Check that `mode` is set to the behaviour you expect and that weights aren’t skewed heavily toward one slot.
- **Output is empty** – Make sure at least one prompt slot contains text or set a `fallback_prompt`.
- **Order resets unexpectedly** – When using `Shuffle`, keep the seed fixed; changing it generates a new permutation.

---

Screenshot: `docs/screenshots/multi_prompt_node.png`
