"""Map recording-machine XML asset paths to this server's pinned LIBERO checkout."""
import os
from pathlib import Path
import xml.etree.ElementTree as ET


LIBERO_MARKER = "/chiliocosm/assets/"


def localize_assets(xml, libero_checkout):
    tree = ET.fromstring(xml)
    assets = (Path(libero_checkout) / "libero/libero/assets").resolve()
    rewritten = 0
    missing = []
    for element in tree.iter():
        old = element.get("file")
        if not old:
            continue
        if LIBERO_MARKER in old:
            suffix = old.split(LIBERO_MARKER, 1)[1]
            target = (assets / suffix).resolve()
            if not target.is_relative_to(assets):
                raise ValueError(f"Asset escapes checkout: {old}")
            element.set("file", str(target))
            rewritten += 1
        resolved = element.get("file")
        if os.path.isabs(resolved) and not Path(resolved).is_file():
            missing.append(resolved)
    if missing:
        raise FileNotFoundError(f"{len(missing)} XML assets missing; first: {missing[:3]}")
    return ET.tostring(tree, encoding="unicode"), rewritten
