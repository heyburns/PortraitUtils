# The Universal Config Suite

Have you ever looked at a complex ComfyUI workflow and felt like you were looking at a giant bowl of spaghetti? Wires crossing everywhere, getting tangled, making it impossible to see what is actually happening? 

That's exactly what the **Universal Project Config** and **Extract Config Value** nodes are designed to fix!

## What do they do?

Instead of using individual nodes to set your seeds, sizes, or custom text, you put all your master settings into the **Universal Project Config** node. Think of it as the main control panel for your entire workflow. 

It takes all those settings and bundles them into a single, neat package (a tiny wire called `config_bundle`). 

Instead of dragging 20 different wires across your screen, you drag just that *one* `config_bundle` wire. When it reaches a distant part of your workflow that needs a setting, you drop down an **Extract Config Value** node, connect the bundle, and simply pick what you need from the dropdown list.

## The Nodes

### Universal Project Config
This is your master control board. It reads your project's settings file and automatically creates all the sliders and text boxes you need. 

<div align="center"><img src="screenshots/PLACEHOLDER_UNIVERSAL_CONFIG.png" alt="Universal Project Config" width="500" /></div>

### Extract Config Value
This is your wireless receiver. Drop it down anywhere in your workflow, connect the master bundle, and pick the setting you want to extract. It acts as a clean, invisible bridge to your master control panel.

<div align="center"><img src="screenshots/PLACEHOLDER_EXTRACT_CONFIG.png" alt="Extract Config Value" width="500" /></div>
