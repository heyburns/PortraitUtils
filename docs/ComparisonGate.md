# ComparisonGate

## Overview
`ComparisonGate` keeps comparison viewers tidy by only emitting images when a complete pair is available. It listens to a designated `source_image` input together with up to three comparison sockets and mirrors ComfyUI's unplugged behaviour (`None`, `None`) until two populated tensors are ready to forward.

## Inputs
- `source_image` (`IMAGE`, optional): Preferred reference image for the secondary output. Leave disconnected when you want the first populated comparison to feed both sockets.
- `image_a` (`IMAGE`, optional): Primary comparison feed. Becomes the `final_image` output whenever populated.
- `image_b` (`IMAGE`, optional): Secondary comparison feed used either as a fallback `final_image` or to backfill the `source_image` output when the dedicated source socket is empty.
- `image_c` (`IMAGE`, optional): Tertiary comparison feed that participates in the fallback cascade.

All inputs accept `None`; the node filters out empty lists/tuples as well as bare `None` values so placeholder widgets will not trigger accidental forwards.

## Outputs
- `final_image` (`IMAGE`): The first populated comparison input (`image_a` › `image_b` › `image_c`). Returns `None` until at least one comparison produces data.
- `source_image` (`IMAGE`): Mirrors the dedicated `source_image` socket, or falls back to the next populated comparison input so the viewer always receives a pair. Returns `None` alongside `final_image` when fewer than two valid images are present.

## Forwarding Rules
- The node inspects every input on each execution and counts how many look like real image tensors. If fewer than two are found it emits `(None, None)` to keep downstream nodes blank.
- When only comparison sockets are populated, the first comparison feeds `final_image` and the second feeds `source_image`, ensuring A/B layouts remain synchronised.
- When `source_image` is connected, it always feeds the `source_image` output unless it is empty—in that case the node automatically falls back to the second non-empty comparison input.
- Containers such as one-item tuples or lists are unwrapped only when they contain real data; sequences of `None` values are ignored.

## Tips
- Place the gate directly before two-input comparison viewers to prevent half-populated renders during workflow warm-up.
- Combine with `PairedImageLoader` for workflow QA: let the loader drive comparisons and feed its outputs through the gate so the viewer only refreshes when both sockets advance.
- Because the node simply relays its inputs, it adds no meaningful processing cost; you can safely sprinkle it anywhere you need to guarantee paired images.*** End Patch
