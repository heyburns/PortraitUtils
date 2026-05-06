# Process Scanned Photo

When you scan an old Polaroid or a physical photo, it's almost never perfectly straight. Maybe it's tilted, maybe there are ugly black scanner borders around the edges, or maybe it's just off-center.

The **Process Scanned Photo** node acts as an automated digital photo assistant. It looks at your messy scanned images, figures out where the actual photo is, straightens it out perfectly, and trims off the ugly borders—all in one go!

<div align="center"><img src="screenshots/PLACEHOLDER_PROCESS_SCANNED_PHOTO.png" alt="Process Scanned Photo" width="500" /></div>

## What do the settings do?

- **straighten**: Turn this on to have the node automatically rotate the image to fix tilt.
- **crop_mode**: 
  - `tight`: Crops right to the edge of the photo, sometimes trimming a tiny bit of the picture to ensure no scanner border is left.
  - `safe`: Keeps more of the photo but might leave a tiny sliver of scanner border.
  - `none`: Just straightens the photo without cutting anything off.
- **padding**: Adds a little extra breathing room (in pixels) around the edges after cropping.
- **threshold**: This helps the node see the edges of the photo. If it's cutting off too much of a dark photo, try adjusting this number.

Just plug your scanned image in, and let the node do the heavy lifting!
