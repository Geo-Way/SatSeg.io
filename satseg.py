#!/usr/bin/env python3
"""
GeoWay SatSeg — Building Footprint Extractor
=============================================
Extrae footprints de edificios desde imágenes satelitales/aéreas
y los exporta como GeoJSON con coordenadas reales.

INSTALACIÓN (una sola vez):
  pip install torch torchvision transformers pillow rasterio shapely opencv-python numpy

USO:
  python satseg.py imagen.tif              # GeoTIFF con georeferencia
  python satseg.py imagen.jpg --lon -0.09 --lat 51.505 --zoom 18   # imagen normal
  python satseg.py imagen.jpg --bbox -0.095,51.500,-0.085,51.510   # bbox exacto

SALIDA:
  buildings_<timestamp>.geojson   ← polígonos vectorizados
  mask_<timestamp>.png            ← máscara visual
"""

import sys, os, json, time, argparse
import numpy as np
from pathlib import Path
from datetime import datetime

# ── Imports opcionales (se importan cuando se necesitan) ──────
def require(pkg, install_hint=""):
    import importlib
    try:
        return importlib.import_module(pkg)
    except ImportError:
        hint = install_hint or f"pip install {pkg}"
        print(f"\n❌ Falta: {pkg}\n   Instala con: {hint}\n")
        sys.exit(1)


# ══════════════════════════════════════════════════════════════
# 1. CARGA DE IMAGEN
# ══════════════════════════════════════════════════════════════

def load_geotiff(path):
    """Carga GeoTIFF retornando (array RGB uint8, transform, crs)"""
    rasterio = require("rasterio")
    import rasterio as rio
    from rasterio.enums import Resampling

    with rio.open(path) as src:
        transform = src.transform
        crs = src.crs

        # Leer hasta 3 bandas
        count = min(src.count, 3)
        data  = src.read(list(range(1, count + 1)), resampling=Resampling.bilinear)

        # Normalizar a uint8
        if data.dtype != np.uint8:
            p2, p98 = np.percentile(data, 2), np.percentile(data, 98)
            data = np.clip((data - p2) / (p98 - p2 + 1e-6) * 255, 0, 255).astype(np.uint8)

        if count == 1:
            data = np.stack([data[0], data[0], data[0]])  # gris → RGB

        # HWC
        rgb = np.transpose(data[:3], (1, 2, 0))
        print(f"  GeoTIFF: {src.width}×{src.height} · CRS: {crs} · {src.count} bandas")
        return rgb, transform, crs


def load_regular_image(path):
    """Carga JPG/PNG como array RGB"""
    from PIL import Image as PILImage
    img = PILImage.open(path).convert("RGB")
    return np.array(img), None, None


def bbox_to_transform(west, south, east, north, width, height):
    """Crea un Affine transform desde un bounding box"""
    from rasterio.transform import from_bounds
    return from_bounds(west, south, east, north, width, height)


def zoom_to_bbox(lon, lat, zoom, w, h):
    """Aproxima el bbox en grados para una imagen de w×h pixels a zoom dado"""
    import math
    # Metros por pixel a ese zoom (aprox)
    mpp = 156543.03392 * math.cos(math.radians(lat)) / (2 ** zoom)
    # Grados por pixel
    deg_per_px = mpp / 111320
    half_w = deg_per_px * w / 2
    half_h = deg_per_px * h / 2
    return lon - half_w, lat - half_h, lon + half_w, lat + half_h


# ══════════════════════════════════════════════════════════════
# 2. SEGMENTACIÓN — MODELO ESPECIALIZADO EN EDIFICIOS
# ══════════════════════════════════════════════════════════════

MODELS = {
    "segformer-building": {
        "id":   "borhansam/segformer_b3_finetuned_segments_building",
        "desc": "SegFormer-B3 finetuneado en segmentación de edificios (dataset Segments.ai)",
        "type": "hf-segmentation",
    },
    "segformer-ade":  {
        "id":   "nvidia/segformer-b5-finetuned-ade-640-640",
        "desc": "SegFormer-B5 ADE20k (150 clases, incluye building/wall)",
        "type": "hf-segmentation",
        "building_labels": ["building", "house", "wall", "skyscraper", "tower"],
    },
    "mask2former-ade": {
        "id":   "facebook/mask2former-swin-base-ade-semantic",
        "desc": "Mask2Former-Base ADE semantic (incluye building)",
        "type": "hf-segmentation",
        "building_labels": ["building", "house", "wall", "skyscraper", "tower"],
    },
}

BUILDING_LABELS = {"building", "house", "wall", "skyscraper", "tower", "roof",
                   "edificio", "construcci", "build"}


def run_segmentation(rgb_array, model_key="segformer-ade", device=None):
    """Ejecuta segmentación y devuelve máscara binaria de edificios (H×W bool)"""
    torch = require("torch")
    transformers = require("transformers")
    from transformers import pipeline, AutoImageProcessor, AutoModelForSemanticSegmentation
    from PIL import Image as PILImage
    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = MODELS.get(model_key, MODELS["segformer-ade"])
    model_id = cfg["id"]
    print(f"\n  Modelo : {model_id}")
    print(f"  Device : {device}")
    print(f"  Desc   : {cfg['desc']}")

    pil_img = PILImage.fromarray(rgb_array)

    # Redimensionar a máx 1024px para no exceder memoria
    max_side = 1024
    w, h = pil_img.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        pil_img = pil_img.resize((int(w * scale), int(h * scale)), PILImage.LANCZOS)
        print(f"  Resize : {w}×{h} → {pil_img.width}×{pil_img.height}")

    seg_pipeline = pipeline(
        "image-segmentation",
        model=model_id,
        device=0 if device == "cuda" else -1,
    )

    print("  Inferencia...", flush=True)
    t0 = time.time()
    results = seg_pipeline(pil_img)
    elapsed = time.time() - t0
    print(f"  Tiempo : {elapsed:.1f}s · {len(results)} segmentos detectados")

    # Mostrar todas las clases detectadas
    print("\n  Clases detectadas:")
    for r in results:
        print(f"    [{r['score']:.2f}] {r['label']}")

    # Construir máscara binaria uniendo segmentos de edificios
    building_labels = set(cfg.get("building_labels", []))
    # Si el modelo está especializado en edificios, usar TODOS los segmentos
    specialized = ("building" in model_id.lower() or
                   len(building_labels) == 0 or
                   all("build" in r['label'].lower() or "background" in r['label'].lower()
                       for r in results))

    W, H = pil_img.size
    mask = np.zeros((H, W), dtype=bool)

    for r in results:
        label_lower = r["label"].lower()
        is_building = (
            specialized and "background" not in label_lower
        ) or any(b in label_lower for b in BUILDING_LABELS) or \
           any(b in label_lower for b in building_labels)

        if is_building and r.get("mask"):
            seg_mask = np.array(r["mask"])  # PIL Image → numpy
            if seg_mask.dtype == np.uint8:
                mask |= (seg_mask > 128)
            else:
                mask |= seg_mask.astype(bool)

    coverage = mask.sum() / mask.size * 100
    print(f"\n  Cobertura edificios: {coverage:.1f}%")

    # Escalar máscara de vuelta a tamaño original si se redimensionó
    if mask.shape != (rgb_array.shape[0], rgb_array.shape[1]):
        import cv2
        mask = cv2.resize(
            mask.astype(np.uint8),
            (rgb_array.shape[1], rgb_array.shape[0]),
            interpolation=cv2.INTER_NEAREST
        ).astype(bool)

    return mask


# ══════════════════════════════════════════════════════════════
# 3. VECTORIZACIÓN (máscara → polígonos)
# ══════════════════════════════════════════════════════════════

def mask_to_polygons(mask, transform, simplify_tolerance=1.5, min_area_px=30):
    """
    Convierte máscara binaria a lista de polígonos Shapely en coordenadas pixel.
    Aplica simplificación Douglas-Peucker para footprints limpios.
    """
    cv2    = require("cv2", "pip install opencv-python")
    shapely = require("shapely")
    from shapely.geometry import Polygon, MultiPolygon
    from shapely.ops import unary_union
    import cv2

    # Morfología: cerrar pequeños huecos, eliminar ruido
    kernel  = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE,  kernel, iterations=2)
    cleaned = cv2.morphologyEx(cleaned,               cv2.MORPH_OPEN,   kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned,               cv2.MORPH_DILATE, kernel, iterations=1)

    # Contornos
    contours, hierarchy = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    print(f"\n  Contornos encontrados: {len(contours)}")

    polys = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area_px:
            continue

        pts = cnt.squeeze()
        if pts.ndim != 2 or len(pts) < 3:
            continue

        coords = [(float(p[0]), float(p[1])) for p in pts]
        if len(coords) < 3:
            continue

        try:
            poly = Polygon(coords)
            if not poly.is_valid:
                poly = poly.buffer(0)
            if poly.is_empty or poly.area < min_area_px:
                continue

            # Simplificar contorno (Douglas-Peucker)
            poly = poly.simplify(simplify_tolerance, preserve_topology=True)
            if poly.is_empty:
                continue

            polys.append(poly)
        except Exception:
            continue

    print(f"  Polígonos válidos (>{min_area_px}px²): {len(polys)}")
    return polys


def pixels_to_geo(polys, transform, img_height):
    """
    Convierte polígonos en coordenadas pixel a lon/lat usando el Affine transform.
    El transform de rasterio mapea: (col, row) → (x, y) donde y crece hacia arriba.
    """
    from shapely.geometry import mapping, Polygon
    import rasterio.transform as rtransform

    geo_polys = []
    for poly in polys:
        if transform is None:
            # Sin georeferencia: devolver coordenadas pixel como están
            geo_polys.append(poly)
            continue

        def px_to_lonlat(x, y):
            # rasterio: (col, row) → (lon, lat)
            lon, lat = transform * (x, y)
            return lon, lat

        if hasattr(poly, 'geoms'):
            # MultiPolygon
            new_geoms = []
            for g in poly.geoms:
                new_coords = [px_to_lonlat(x, y) for x, y in g.exterior.coords]
                new_geoms.append(Polygon(new_coords))
            geo_polys.extend(new_geoms)
        else:
            new_coords = [px_to_lonlat(x, y) for x, y in poly.exterior.coords]
            geo_polys.append(Polygon(new_coords))

    return geo_polys


def polys_to_geojson(polys, transform, img_height, source_file, model_id):
    """Genera FeatureCollection GeoJSON"""
    from shapely.geometry import mapping

    has_geo = transform is not None
    geo_polys = pixels_to_geo(polys, transform, img_height) if has_geo else polys

    features = []
    for i, poly in enumerate(geo_polys):
        if poly.is_empty:
            continue
        features.append({
            "type": "Feature",
            "id":   i + 1,
            "properties": {
                "id":     i + 1,
                "area":   round(poly.area, 4),
                "source": str(source_file),
                "model":  model_id,
                "tool":   "GeoWay SatSeg v3.1",
                "date":   datetime.utcnow().isoformat() + "Z",
                "georef": has_geo,
            },
            "geometry": mapping(poly),
        })

    crs_block = {
        "type": "name",
        "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}
    } if has_geo else None

    fc = {
        "type": "FeatureCollection",
        "name": "building_footprints",
        "features": features,
    }
    if crs_block:
        fc["crs"] = crs_block

    return fc


# ══════════════════════════════════════════════════════════════
# 4. GUARDAR OUTPUTS
# ══════════════════════════════════════════════════════════════

def save_mask_overlay(rgb, mask, output_path):
    """Guarda imagen RGB con overlay de la máscara en cian"""
    from PIL import Image as PILImage, ImageDraw

    overlay = rgb.copy()
    # Tinte cian sobre edificios
    overlay[mask, 0] = (overlay[mask, 0] * 0.2 + 0 * 0.8).astype(np.uint8)
    overlay[mask, 1] = (overlay[mask, 1] * 0.2 + 229 * 0.8).astype(np.uint8)
    overlay[mask, 2] = (overlay[mask, 2] * 0.2 + 255 * 0.8).astype(np.uint8)

    PILImage.fromarray(overlay).save(output_path)
    print(f"  Overlay guardado: {output_path}")


def save_geojson(fc, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(fc, f, indent=2, ensure_ascii=False)
    print(f"  GeoJSON guardado: {output_path}  ({len(fc['features'])} features)")


# ══════════════════════════════════════════════════════════════
# 5. MAIN
# ══════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="GeoWay SatSeg — Building Footprint Extractor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python satseg.py foto.tif
  python satseg.py imagen.jpg --lon -0.09 --lat 51.505 --zoom 18
  python satseg.py imagen.jpg --bbox -0.095,51.500,-0.085,51.510
  python satseg.py foto.tif --model segformer-ade --simplify 2.0
        """
    )
    p.add_argument("image",   help="Imagen de entrada (.tif, .tiff, .jpg, .png)")
    p.add_argument("--model", default="segformer-ade",
                   choices=list(MODELS.keys()),
                   help="Modelo a usar (default: segformer-ade)")
    p.add_argument("--lon",   type=float, help="Longitud central (para JPG/PNG)")
    p.add_argument("--lat",   type=float, help="Latitud central (para JPG/PNG)")
    p.add_argument("--zoom",  type=int,   default=18, help="Nivel de zoom (default: 18)")
    p.add_argument("--bbox",  type=str,   help="Bounding box: west,south,east,north")
    p.add_argument("--simplify", type=float, default=1.5,
                   help="Tolerancia simplificación Douglas-Peucker en píxeles (default: 1.5)")
    p.add_argument("--min-area", type=int, default=30,
                   help="Área mínima en píxeles² para incluir un polígono (default: 30)")
    p.add_argument("--output", type=str, default=None,
                   help="Prefijo para archivos de salida")
    p.add_argument("--no-mask", action="store_true",
                   help="No guardar imagen de máscara")
    return p.parse_args()


def main():
    args = parse_args()
    img_path = Path(args.image)

    if not img_path.exists():
        print(f"\n❌ Archivo no encontrado: {img_path}")
        sys.exit(1)

    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix  = args.output or f"geoway_{img_path.stem}_{ts}"
    out_json = Path(prefix + ".geojson")
    out_mask = Path(prefix + "_mask.png")

    print(f"\n{'='*55}")
    print(f"  GeoWay SatSeg — Building Footprint Extractor")
    print(f"{'='*55}")
    print(f"  Imagen : {img_path}")
    print(f"  Modelo : {args.model}")

    # ── Cargar imagen ──────────────────────────────────────────
    transform, crs = None, None
    ext = img_path.suffix.lower()

    if ext in (".tif", ".tiff"):
        rgb, transform, crs = load_geotiff(str(img_path))
    else:
        rgb, _, _ = load_regular_image(str(img_path))
        h, w = rgb.shape[:2]
        # Establecer georeferencia si se proporcionan coordenadas
        if args.bbox:
            try:
                import rasterio
                w_b, s, e, n = [float(v) for v in args.bbox.split(",")]
                transform = bbox_to_transform(w_b, s, e, n, w, h)
                print(f"  BBox   : {w_b:.4f},{s:.4f},{e:.4f},{n:.4f}")
            except Exception as ex:
                print(f"  ⚠ No se pudo establecer transform desde bbox: {ex}")
        elif args.lon and args.lat:
            try:
                import rasterio
                west, south, east, north = zoom_to_bbox(args.lon, args.lat, args.zoom, w, h)
                transform = bbox_to_transform(west, south, east, north, w, h)
                print(f"  Centro : {args.lon}, {args.lat}  zoom={args.zoom}")
                print(f"  BBox   : {west:.5f},{south:.5f},{east:.5f},{north:.5f}")
            except Exception as ex:
                print(f"  ⚠ No se pudo establecer transform: {ex}")

    if transform is None:
        print("  ⚠ Sin georeferencia — el GeoJSON tendrá coordenadas en píxeles")

    # ── Segmentar ──────────────────────────────────────────────
    print(f"\n[1/4] Segmentando edificios...")
    mask = run_segmentation(rgb, model_key=args.model)

    # ── Vectorizar ─────────────────────────────────────────────
    print(f"\n[2/4] Vectorizando máscara...")
    polys = mask_to_polygons(mask, transform,
                              simplify_tolerance=args.simplify,
                              min_area_px=args.min_area)

    if not polys:
        print("\n  ⚠ No se detectaron edificios. Prueba otro modelo o ajusta --min-area")
        sys.exit(0)

    # ── Georeferencia ──────────────────────────────────────────
    print(f"\n[3/4] Generando GeoJSON...")
    model_id = MODELS[args.model]["id"]
    fc = polys_to_geojson(polys, transform, rgb.shape[0], img_path.name, model_id)
    save_geojson(fc, out_json)

    # ── Guardar máscara visual ─────────────────────────────────
    if not args.no_mask:
        print(f"\n[4/4] Guardando overlay...")
        save_mask_overlay(rgb, mask, str(out_mask))

    print(f"\n{'='*55}")
    print(f"  ✅ Completado")
    print(f"  📄 GeoJSON  : {out_json}")
    if not args.no_mask:
        print(f"  🖼  Máscara  : {out_mask}")
    print(f"  🏢 Edificios: {len(fc['features'])}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
