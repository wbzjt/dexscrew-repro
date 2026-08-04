#!/usr/bin/env python3
"""Build a DexH13 MuJoCo include with URDF-like fixed tactile/tip bodies."""

from __future__ import annotations

import argparse
import copy
import xml.etree.ElementTree as ET
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = REPO_ROOT / "assets/dexh13_right_description2/urdf/dexh13_right_fixed_fingertips.xml"
DEFAULT_OUTPUT = REPO_ROOT / "assets/dexh13_right_description2/urdf/dexh13_right_isaacgym_parity_fingertips.xml"

TIP_PARENT_MESH = {
    "right_index_tip": "right_index_tactile_link_2",
    "right_middle_tip": "right_middle_tactile_link_2",
    "right_ring_tip": "right_ring_tactile_link_2",
    "right_thumb_tip": "right_thumb_tactile_link_1",
}
TIP_OFFSET = "0 0.01 0.005"


def is_tactile_mesh(mesh_name: str) -> bool:
    return "_tactile_link_" in mesh_name


def body_children(body: ET.Element) -> list[ET.Element]:
    return [child for child in list(body) if child.tag == "body"]


def transform_body(body: ET.Element) -> None:
    body_name = body.get("name", "")
    if "_tactile_link_" in body_name or body_name.endswith("_tip"):
        return

    children = list(body)
    pending_tips: dict[str, ET.Element] = {}
    tactile_bodies: dict[str, ET.Element] = {}

    for child in children:
        if child.tag == "geom" and child.get("name") in TIP_PARENT_MESH:
            body.remove(child)
            pending_tips[child.get("name", "")] = child

    children = list(body)
    for child in children:
        if child.tag != "geom":
            continue
        mesh_name = child.get("mesh", "")
        if not is_tactile_mesh(mesh_name):
            continue
        body.remove(child)
        tactile_body = ET.Element("body", {"name": mesh_name})
        if child.get("pos") is not None:
            tactile_body.set("pos", child.get("pos", ""))
        if child.get("quat") is not None:
            tactile_body.set("quat", child.get("quat", ""))
        tactile_geom = copy.deepcopy(child)
        tactile_geom.set("name", mesh_name)
        tactile_geom.attrib.pop("pos", None)
        tactile_geom.attrib.pop("quat", None)
        tactile_body.append(tactile_geom)
        body.append(tactile_body)
        tactile_bodies[mesh_name] = tactile_body

    for tip_name, parent_mesh in TIP_PARENT_MESH.items():
        tip_geom = pending_tips.pop(tip_name, None)
        tactile_body = tactile_bodies.get(parent_mesh)
        if tip_geom is None or tactile_body is None:
            continue
        tip_body = ET.Element("body", {"name": tip_name, "pos": TIP_OFFSET})
        tip_geom = copy.deepcopy(tip_geom)
        tip_geom.attrib.pop("pos", None)
        tip_geom.attrib.pop("quat", None)
        tip_body.append(tip_geom)
        tactile_body.append(tip_body)

    for child in body_children(body):
        transform_body(child)


def build(source: Path, output: Path) -> None:
    tree = ET.parse(source)
    root = tree.getroot()
    root.set("model", "dexh13_right_isaacgym_parity")
    for worldbody in root.findall("worldbody"):
        for body in body_children(worldbody):
            transform_body(body)
    ET.indent(tree, space="  ")
    output.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output, encoding="utf-8", xml_declaration=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    build(args.source.expanduser().resolve(), args.output.expanduser().resolve())
    print(args.output.expanduser().resolve())


if __name__ == "__main__":
    main()
