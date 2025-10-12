import os
import re


def _strip_extensions(path: str, strip_all: bool = True) -> tuple[str, str]:
    """
    Return (directory, base_without_extension).
    If strip_all is True, remove every trailing extension segment (e.g. .tar.gz).
    """
    path = path.strip()
    directory = os.path.dirname(path)
    basename = os.path.basename(path)
    if strip_all:
        basename = re.sub(r"(?:\.[A-Za-z0-9]{1,5})+$", "", basename)
    else:
        basename = os.path.splitext(basename)[0]
    return directory, basename


class FilenameAppendSuffix:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "filename": ("STRING", {"multiline": False, "default": ""}),
                "suffix": ("STRING", {"multiline": False, "default": ""}),
                "separator": ("STRING", {"multiline": False, "default": "-"}),
                "strip_all_extensions": ("BOOLEAN", {"default": True}),
                "preserve_directory": ("BOOLEAN", {"default": True}),
                "trim_whitespace": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filename",)
    FUNCTION = "build"
    CATEGORY = "Utils/IO"

    def build(
        self,
        filename: str,
        suffix: str = "supir",
        separator: str = "-",
        strip_all_extensions: bool = True,
        preserve_directory: bool = True,
        trim_whitespace: bool = True,
    ):
        fn = filename if isinstance(filename, str) else str(filename)
        sx = suffix if isinstance(suffix, str) else str(suffix)
        sep = separator if isinstance(separator, str) else "-"

        if trim_whitespace:
            fn = fn.strip()
            sx = sx.strip()
            sep = sep.strip() or "-"

        directory, base = _strip_extensions(fn, strip_all=strip_all_extensions)

        if not sx:
            out = base
        else:
            out = f"{base}{sx}" if base.endswith(sep) else f"{base}{sep}{sx}"

        if preserve_directory and directory:
            out = os.path.join(directory, out)

        return (out,)


NODE_CLASS_MAPPINGS = {
    "FilenameAppendSuffix": FilenameAppendSuffix,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FilenameAppendSuffix": "Filename: Append Suffix",
}
