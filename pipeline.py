"""VisionForge headless Blender pipeline. Runs INSIDE Blender (3.6 LTS and 4.x).
Usage: blender -b --python pipeline.py -- --job job.json
"""
import bpy
import bmesh
import sys
import os
import json
import math
import time
import subprocess
from mathutils import Vector

VERSION = bpy.app.version
IS_4X = VERSION >= (4, 0, 0)
EEVEE_ID = "BLENDER_EEVEE_NEXT" if VERSION >= (4, 2, 0) else "BLENDER_EEVEE"

SOCKET_ALIASES = {
    "Specular": ["Specular IOR Level", "Specular"],
    "Subsurface": ["Subsurface Weight", "Subsurface"],
    "Emission": ["Emission Color", "Emission"],
    "Emission Strength": ["Emission Strength"],
    "Transmission": ["Transmission Weight", "Transmission"],
    "Clearcoat": ["Coat Weight", "Clearcoat"],
    "Sheen": ["Sheen Weight", "Sheen"],
    "Base Color": ["Base Color"],
    "Metallic": ["Metallic"],
    "Roughness": ["Roughness"],
    "Normal": ["Normal"],
    "Alpha": ["Alpha"],
}


def log(msg):
    print(f"[VF {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def socket(node, name):
    for alias in SOCKET_ALIASES.get(name, [name]):
        if alias in node.inputs:
            return node.inputs[alias]
    raise KeyError(f"socket {name} not found on {node.bl_idname} (Blender {VERSION})")


def parse_job():
    argv = sys.argv
    args = argv[argv.index("--") + 1:] if "--" in argv else []
    job_path = None
    for i, a in enumerate(args):
        if a == "--job" and i + 1 < len(args):
            job_path = args[i + 1]
    if not job_path or not os.path.exists(job_path):
        raise SystemExit("usage: blender -b --python pipeline.py -- --job job.json")
    with open(job_path, "r", encoding="utf-8") as f:
        job = json.load(f)
    job.setdefault("object_class", "object")
    job.setdefault("target_faces", 12000)
    job.setdefault("bake_resolution", 4096)
    job.setdefault("bake_samples", 16)
    job.setdefault("cage_extrusion", 0.02)
    job.setdefault("max_ray_distance", 0.2)
    job.setdefault("target_height_m", 1.0)
    job.setdefault("rotate_z_deg", 0.0)
    job.setdefault("use_mesh_symmetry", False)
    job.setdefault("animation_candidate", False)
    job.setdefault("keep_highpoly", False)
    job.setdefault("maps", ["albedo", "normal", "roughness", "metallic", "curvature"])
    job.setdefault("export_previews", True)
    job["output_blend"] = os.path.abspath(job["output_blend"])
    job["input_mesh"] = os.path.abspath(job["input_mesh"])
    return job


# --------------------------------------------------------------------------- #
# Scene / import                                                              #
# --------------------------------------------------------------------------- #
def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 1.0
    scene.unit_settings.length_unit = "METERS"
    return scene


def override(obj, extra=None):
    ctx = {"object": obj, "active_object": obj, "selected_objects": [obj], "selected_editable_objects": [obj]}
    if extra:
        ctx.update(extra)
    return bpy.context.temp_override(**ctx)


def select_only(objs, active):
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = active


def import_mesh(path):
    ext = os.path.splitext(path)[1].lower()
    before = set(bpy.data.objects)
    if ext in (".glb", ".gltf"):
        bpy.ops.import_scene.gltf(filepath=path)
    elif ext == ".obj":
        if hasattr(bpy.ops.wm, "obj_import"):
            bpy.ops.wm.obj_import(filepath=path)
        else:
            bpy.ops.import_scene.obj(filepath=path)
    elif ext == ".ply":
        if hasattr(bpy.ops.wm, "ply_import"):
            bpy.ops.wm.ply_import(filepath=path)
        else:
            bpy.ops.import_mesh.ply(filepath=path)
    elif ext == ".stl":
        if hasattr(bpy.ops.wm, "stl_import"):
            bpy.ops.wm.stl_import(filepath=path)
        else:
            bpy.ops.import_mesh.stl(filepath=path)
    elif ext == ".fbx":
        bpy.ops.import_scene.fbx(filepath=path)
    else:
        raise RuntimeError(f"unsupported mesh format: {ext}")
    new = [o for o in bpy.data.objects if o not in before]
    meshes = [o for o in new if o.type == "MESH"]
    if not meshes:
        raise RuntimeError("no mesh objects imported")
    for o in new:
        if o.type != "MESH":
            bpy.data.objects.remove(o, do_unlink=True)
    for o in meshes:
        o.hide_set(False)
        o.hide_viewport = False
        o.hide_render = False
    select_only(meshes, meshes[0])
    with bpy.context.temp_override(active_object=meshes[0], selected_objects=meshes,
                                   selected_editable_objects=meshes, object=meshes[0]):
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        if len(meshes) > 1:
            bpy.ops.object.join()
    obj = bpy.context.view_layer.objects.active
    obj.name = "VF_Source"
    obj.data.name = "VF_Source"
    log(f"imported {path} -> {len(obj.data.polygons)} faces")
    return obj


# --------------------------------------------------------------------------- #
# World-space alignment                                                       #
# --------------------------------------------------------------------------- #
def world_bbox(obj):
    pts = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    mn = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
    mx = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
    return mn, mx


def normalize_and_ground(obj, target_height_m, rotate_z_deg):
    if rotate_z_deg:
        obj.rotation_euler[2] += math.radians(rotate_z_deg)
    mn, mx = world_bbox(obj)
    height = max(mx.z - mn.z, 1e-6)
    s = target_height_m / height
    obj.scale = obj.scale * s
    bpy.context.view_layer.update()
    mn, mx = world_bbox(obj)
    center = (mn + mx) * 0.5
    obj.location -= Vector((center.x, center.y, mn.z))
    bpy.context.view_layer.update()
    select_only([obj], obj)
    with override(obj):
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    bpy.context.scene.cursor.location = (0.0, 0.0, 0.0)
    with override(obj):
        bpy.ops.object.origin_set(type="ORIGIN_CURSOR", center="MEDIAN")
    mn, mx = world_bbox(obj)
    dims = mx - mn
    log(f"grounded: dims={tuple(round(v, 4) for v in dims)} origin={tuple(obj.location)} forward=-Y")
    return {"dimensions_m": [round(dims.x, 5), round(dims.y, 5), round(dims.z, 5)], "origin": [0.0, 0.0, 0.0],
            "forward_axis": "-Y", "up_axis": "+Z", "units": "meters"}


# --------------------------------------------------------------------------- #
# Remesh + topology checks                                                    #
# --------------------------------------------------------------------------- #
def make_clean_copy(obj, name):
    dup = obj.copy()
    dup.data = obj.data.copy()
    dup.name = name
    dup.data.name = name
    bpy.context.scene.collection.objects.link(dup)
    return dup


def quadriflow(obj, target_faces, symmetry):
    select_only([obj], obj)
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-5)
    bmesh.ops.triangulate(bm, faces=bm.faces)
    bm.to_mesh(obj.data)
    bm.free()
    try:
        with override(obj):
            bpy.ops.object.quadriflow_remesh(use_mesh_symmetry=symmetry, use_preserve_sharp=False,
                                             use_preserve_boundary=False, smooth_normals=True, mode="FACES",
                                             target_faces=int(target_faces), seed=0)
        log(f"quadriflow ok: {len(obj.data.polygons)} faces")
        return "quadriflow"
    except RuntimeError as exc:
        log(f"quadriflow failed ({exc}); voxel pre-pass then retry")
        mn, mx = world_bbox(obj)
        obj.data.remesh_voxel_size = max((mx - mn).length / 250.0, 0.002)
        obj.data.remesh_voxel_adaptivity = 0.0
        with override(obj):
            bpy.ops.object.voxel_remesh()
            bpy.ops.object.quadriflow_remesh(use_mesh_symmetry=symmetry, use_preserve_sharp=False,
                                             use_preserve_boundary=False, smooth_normals=True, mode="FACES",
                                             target_faces=int(target_faces), seed=0)
        log(f"quadriflow after voxel ok: {len(obj.data.polygons)} faces")
        return "voxel+quadriflow"


def topology_report(obj):
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    faces = len(bm.faces)
    quads = sum(1 for f in bm.faces if len(f.verts) == 4)
    tris = sum(1 for f in bm.faces if len(f.verts) == 3)
    ngons = faces - quads - tris
    non_manifold = sum(1 for e in bm.edges if not e.is_manifold)
    boundary = sum(1 for e in bm.edges if e.is_boundary)
    poles = sum(1 for v in bm.verts if len(v.link_edges) > 5 or (len(v.link_edges) == 3 and not v.is_boundary))
    verts = len(bm.verts)
    # Edge-loop continuity: walk quad loops from random seeds and measure closed/terminated ratio
    closed_loops = 0
    sampled = 0
    for e in list(bm.edges)[:: max(1, len(bm.edges) // 200)]:
        if len(e.link_faces) != 2:
            continue
        sampled += 1
        start = e
        cur = e
        prev_face = None
        steps = 0
        closed = False
        while steps < 4000:
            faces_ = [f for f in cur.link_faces if f is not prev_face and len(f.verts) == 4]
            if not faces_:
                break
            f = faces_[0]
            opposite = [x for x in f.edges if not set(x.verts) & set(cur.verts)]
            if not opposite:
                break
            prev_face = f
            cur = opposite[0]
            steps += 1
            if cur is start:
                closed = True
                break
        if closed:
            closed_loops += 1
    bm.free()
    quad_ratio = quads / faces if faces else 0.0
    pole_ratio = poles / verts if verts else 1.0
    loop_ratio = closed_loops / sampled if sampled else 0.0
    ok = quad_ratio >= 0.95 and non_manifold == 0 and pole_ratio < 0.03
    return {"faces": faces, "quads": quads, "tris": tris, "ngons": ngons, "verts": verts,
            "quad_ratio": round(quad_ratio, 4), "non_manifold_edges": non_manifold, "boundary_edges": boundary,
            "pole_ratio": round(pole_ratio, 4), "closed_loop_ratio": round(loop_ratio, 4),
            "animation_topology_ok": bool(ok), "edge_loop_check_passed": bool(ok and loop_ratio >= 0.5)}


# --------------------------------------------------------------------------- #
# UVs                                                                          #
# --------------------------------------------------------------------------- #
def unwrap(obj):
    select_only([obj], obj)
    if not obj.data.uv_layers:
        obj.data.uv_layers.new(name="UVMap")
    with override(obj):
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.uv.smart_project(angle_limit=math.radians(66.0), island_margin=0.003, area_weight=0.0,
                                 correct_aspect=True, scale_to_bounds=False)
        try:
            bpy.ops.uv.pack_islands(margin=0.003, rotate=True)
        except TypeError:
            bpy.ops.uv.pack_islands(margin=0.003)
        bpy.ops.object.mode_set(mode="OBJECT")
    log("uv unwrap done")


# --------------------------------------------------------------------------- #
# Cycles device                                                               #
# --------------------------------------------------------------------------- #
def setup_cycles(scene, samples):
    scene.render.engine = "CYCLES"
    scene.cycles.samples = int(samples)
    scene.cycles.use_denoising = False
    scene.render.bake.margin = 16
    scene.render.bake.use_clear = True
    prefs = bpy.context.preferences.addons.get("cycles")
    device_used = "CPU"
    if prefs:
        cp = prefs.preferences
        for dtype in ("OPTIX", "CUDA", "HIP", "METAL", "ONEAPI"):
            try:
                cp.compute_device_type = dtype
                cp.get_devices()
            except (TypeError, AttributeError):
                continue
            gpus = [d for d in cp.devices if d.type != "CPU"]
            if gpus:
                for d in cp.devices:
                    d.use = True
                scene.cycles.device = "GPU"
                device_used = dtype
                break
    if device_used == "CPU":
        scene.cycles.device = "CPU"
    log(f"cycles device: {device_used}")
    return device_used


# --------------------------------------------------------------------------- #
# Materials                                                                   #
# --------------------------------------------------------------------------- #
def ensure_source_materials(obj):
    if not obj.data.materials or all(m is None for m in obj.data.materials):
        mat = bpy.data.materials.new("VF_SourceMat")
        mat.use_nodes = True
        obj.data.materials.clear()
        obj.data.materials.append(mat)
    for mat in obj.data.materials:
        if mat and not mat.use_nodes:
            mat.use_nodes = True
    # glTF imports with vertex colors but no texture: wire Color Attribute into Base Color
    for mat in obj.data.materials:
        if mat is None:
            continue
        nt = mat.node_tree
        bsdf = next((n for n in nt.nodes if n.bl_idname == "ShaderNodeBsdfPrincipled"), None)
        if bsdf is None:
            continue
        bc = socket(bsdf, "Base Color")
        if not bc.is_linked and obj.data.color_attributes:
            attr = nt.nodes.new("ShaderNodeVertexColor")
            attr.layer_name = obj.data.color_attributes[0].name
            nt.links.new(attr.outputs["Color"], bc)


def principled_of(mat):
    return next((n for n in mat.node_tree.nodes if n.bl_idname == "ShaderNodeBsdfPrincipled"), None)


def new_bake_image(name, res, non_color):
    img = bpy.data.images.new(name, width=res, height=res, alpha=False, float_buffer=False)
    img.generated_color = (0.0, 0.0, 0.0, 1.0)
    if non_color:
        img.colorspace_settings.name = "Non-Color"
    return img


def set_active_bake_target(low, img):
    mat = low.data.materials[0]
    nt = mat.node_tree
    for n in nt.nodes:
        n.select = False
    node = nt.nodes.new("ShaderNodeTexImage")
    node.image = img
    node.name = f"BAKE_{img.name}"
    node.label = node.name
    node.select = True
    nt.nodes.active = node
    return node


def run_bake(high, low, bake_type, pass_filter=None, normal_space="TANGENT", cage=0.02, max_ray=0.2):
    select_only([high, low], low)
    kwargs = {"type": bake_type, "margin": 16, "use_clear": True, "use_selected_to_active": True,
              "cage_extrusion": float(cage), "max_ray_distance": float(max_ray)}
    if pass_filter:
        kwargs["pass_filter"] = set(pass_filter)
    if bake_type == "NORMAL":
        kwargs["normal_space"] = normal_space
    with bpy.context.temp_override(object=low, active_object=low, selected_objects=[high, low],
                                   selected_editable_objects=[high, low]):
        bpy.ops.object.bake(**kwargs)


class EmissionSwap:
    """Temporarily route a scalar Principled input (or a custom node output) to Emission for EMIT bakes."""

    def __init__(self, obj):
        self.obj = obj
        self.saved = []
        self.temp_nodes = []

    def route_input(self, input_name):
        for mat in self.obj.data.materials:
            if mat is None:
                continue
            bsdf = principled_of(mat)
            if bsdf is None:
                continue
            nt = mat.node_tree
            src = socket(bsdf, input_name)
            emis = socket(bsdf, "Emission")
            strength = socket(bsdf, "Emission Strength")
            self.saved.append((mat, [(l.from_socket, l.to_socket) for l in nt.links if l.to_socket in (emis, strength)],
                               tuple(emis.default_value), float(strength.default_value)))
            for l in [l for l in nt.links if l.to_socket in (emis, strength)]:
                nt.links.remove(l)
            if src.is_linked:
                nt.links.new(src.links[0].from_socket, emis)
            else:
                v = float(src.default_value)
                emis.default_value = (v, v, v, 1.0)
            strength.default_value = 1.0

    def route_curvature(self):
        for mat in self.obj.data.materials:
            if mat is None:
                continue
            bsdf = principled_of(mat)
            if bsdf is None:
                continue
            nt = mat.node_tree
            emis = socket(bsdf, "Emission")
            strength = socket(bsdf, "Emission Strength")
            self.saved.append((mat, [(l.from_socket, l.to_socket) for l in nt.links if l.to_socket in (emis, strength)],
                               tuple(emis.default_value), float(strength.default_value)))
            for l in [l for l in nt.links if l.to_socket in (emis, strength)]:
                nt.links.remove(l)
            geo = nt.nodes.new("ShaderNodeNewGeometry")
            ramp = nt.nodes.new("ShaderNodeValToRGB")
            ramp.color_ramp.elements[0].position = 0.42
            ramp.color_ramp.elements[1].position = 0.58
            nt.links.new(geo.outputs["Pointiness"], ramp.inputs["Fac"])
            nt.links.new(ramp.outputs["Color"], emis)
            strength.default_value = 1.0
            self.temp_nodes.extend([(nt, geo), (nt, ramp)])

    def restore(self):
        for nt, node in self.temp_nodes:
            nt.nodes.remove(node)
        self.temp_nodes = []
        for mat, links, emis_val, strength_val in self.saved:
            bsdf = principled_of(mat)
            nt = mat.node_tree
            emis = socket(bsdf, "Emission")
            strength = socket(bsdf, "Emission Strength")
            for l in [l for l in nt.links if l.to_socket in (emis, strength)]:
                nt.links.remove(l)
            for from_s, to_s in links:
                nt.links.new(from_s, to_s)
            emis.default_value = emis_val
            strength.default_value = strength_val
        self.saved = []


def bake_all(high, low, job, textures_dir):
    res = int(job["bake_resolution"])
    cage = float(job["cage_extrusion"])
    max_ray = float(job["max_ray_distance"])
    scene = bpy.context.scene
    os.makedirs(textures_dir, exist_ok=True)
    mat = bpy.data.materials.new("VF_Baked")
    mat.use_nodes = True
    low.data.materials.clear()
    low.data.materials.append(mat)
    nt = mat.node_tree
    bsdf = principled_of(mat)
    base = f"{job['object_class']}_{res}"
    images = {}
    nodes = {}

    def do(map_name, image_non_color, bake_fn, samples):
        scene.cycles.samples = samples
        img = new_bake_image(f"{base}_{map_name}", res, image_non_color)
        node = set_active_bake_target(low, img)
        t0 = time.time()
        bake_fn()
        img.filepath_raw = os.path.join(textures_dir, f"{img.name}.png")
        img.file_format = "PNG"
        img.save()
        img.pack()
        images[map_name] = img
        nodes[map_name] = node
        log(f"baked {map_name} {res}x{res} in {time.time() - t0:.1f}s")

    maps = job["maps"]
    if "albedo" in maps:
        do("albedo", False, lambda: run_bake(high, low, "DIFFUSE", {"COLOR"}, cage=cage, max_ray=max_ray), job["bake_samples"])
    if "normal" in maps:
        do("normal", True, lambda: run_bake(high, low, "NORMAL", normal_space="TANGENT", cage=cage, max_ray=max_ray), 1)
    if "roughness" in maps:
        do("roughness", True, lambda: run_bake(high, low, "ROUGHNESS", cage=cage, max_ray=max_ray), 1)
    swap = EmissionSwap(high)
    if "metallic" in maps:
        swap.route_input("Metallic")
        try:
            do("metallic", True, l
