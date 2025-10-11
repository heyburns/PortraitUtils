# MultiPromptNode
<div align="center"><img src="screenshots/multi_prompt_node.png" alt="Screenshot" width="300" /></div>


## Overview
`MultiPromptNode` exposes four independent multiline string sockets that can hold prompt snippets, masks, or any free-form text. It does not alter or combine the strings; the node simply passes them through so they can be reused elsewhere in a workflow.

## Inputs
- `subject_mask` (`STRING`, default empty): Free-form field typically used to describe subject-masking instructions.
- `inpaint_prompt` (`STRING`, default empty): Slot for inpainting-specific guidance.
- `outpaint_prompt` (`STRING`, default empty): Slot for outpainting instructions.
- `stitch_mask` (`STRING`, default empty): Slot for notes or textual encodings related to stitching.

## Outputs
- `subject_mask` (`STRING`)
- `inpaint_prompt` (`STRING`)
- `outpaint_prompt` (`STRING`)
- `stitch_mask` (`STRING`)

## Usage Notes
- All four sockets are optional; the node returns whatever was entered, including empty strings.
- Because outputs are plain strings, they can feed directly into conditioning nodes, prompt editors, or scripting helpers.
- Rename the node instance in your workflow to reflect the actual meaning of each socket if you repurpose them for other tasks.*** End Patch
