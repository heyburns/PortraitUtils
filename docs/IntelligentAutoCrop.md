# Intelligent AutoCrop (GPU)

Sometimes you have an image with thick, ugly letterboxing (black bars on the top and bottom like a movie) or strange solid color borders left over from an old save format. Getting rid of those manually for hundreds of images is incredibly tedious.

The **Intelligent AutoCrop** node acts like a smart pair of scissors. It looks at your image, figures out where the real content starts and where the empty borders end, and automatically snips the borders away for you!

<div align="center"><img src="screenshots/PLACEHOLDER_INTELLIGENT_AUTO_CROP.png" alt="Intelligent AutoCrop" width="500" /></div>

## Why is it "Intelligent"?

Unlike simple croppers that just guess a size, this node actually analyzes the pixels of the image. 

- **It ignores logos**: If a movie has black bars but there is a tiny watermark or logo sitting in the black bar, the node is smart enough to ignore the logo and still crop the bar away!
- **It's lightning fast**: It uses your computer's graphics card (GPU) to do the math, meaning it can process huge batches of images in the blink of an eye.

## What do the settings do?

- **border_color**: You can tell the node exactly what color the border is (like black or white), or set it to `auto` to let the node guess.
- **tolerance**: If the border isn't perfectly solid black (maybe it's a bit noisy or dark gray), turning this up helps the node grab the messy bits too.
- **min_crop_size**: Prevents the node from accidentally cropping away the entire image if it gets confused.

Use this node anytime you want to clean up messy framing without touching an image editor!
