# GeoWay SatSeg — Building Footprint Extractor

Herramienta de extracción de footprints de edificios desde imágenes satelitales
y aéreas. Genera polígonos vectoriales georeferenciados en formato GeoJSON.

---

## ¿Por qué Python y no la web?

Los modelos de segmentación satelital de edificios **no están disponibles en la
Inference API gratuita de Hugging Face** — requieren GPU dedicada o se ejecutan
localmente. Este script corre el modelo directamente en tu máquina.

---

## Instalación

```bash
# 1. Python 3.9+ requerido
python --version

# 2. Instalar dependencias
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install transformers pillow rasterio shapely opencv-python numpy

# 3. (Opcional) Si tienes GPU NVIDIA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## Uso

### GeoTIFF con georeferencia (recomendado)
```bash
python satseg.py mi_imagen.tif
```

### JPG/PNG con coordenadas de centro
```bash
python satseg.py captura.jpg --lon -0.09 --lat 51.505 --zoom 18
```

### JPG/PNG con bounding box exacto
```bash
python satseg.py captura.jpg --bbox -0.095,51.500,-0.085,51.510
```

### Opciones avanzadas
```bash
python satseg.py imagen.tif \
  --model segformer-ade \     # modelo a usar
  --simplify 1.0 \            # tolerancia Douglas-Peucker (px)
  --min-area 50 \             # área mínima de edificio (px²)
  --output mis_edificios      # prefijo de archivos de salida
```

---

## Modelos disponibles

| `--model`           | Descripción                                | Mejor para          |
|---------------------|--------------------------------------------|---------------------|
| `segformer-ade`     | SegFormer-B5 ADE20k (150 clases) ← **default** | Imágenes variadas |
| `segformer-building`| SegFormer-B3 finetuneado en edificios      | Satelital urbano    |
| `mask2former-ade`   | Mask2Former-Base ADE semantic              | Alta precisión      |

**Recomendación:** empieza con `segformer-building` para imágenes satelitales urbanas.

---

## Outputs

| Archivo                          | Contenido                                      |
|----------------------------------|------------------------------------------------|
| `geoway_<imagen>_<ts>.geojson`   | Polígonos de edificios en WGS84 (si hay georef)|
| `geoway_<imagen>_<ts>_mask.png`  | Imagen con overlay cian de edificios detectados|

### Estructura del GeoJSON
```json
{
  "type": "FeatureCollection",
  "crs": { "type": "name", "properties": { "name": "urn:ogc:def:crs:OGC:1.3:CRS84" }},
  "features": [
    {
      "type": "Feature",
      "id": 1,
      "properties": {
        "id": 1,
        "area": 245.3,
        "model": "nvidia/segformer-b5-finetuned-ade-640-640",
        "tool": "GeoWay SatSeg v3.1",
        "georef": true
      },
      "geometry": {
        "type": "Polygon",
        "coordinates": [[ [-0.091, 51.506], ... ]]
      }
    }
  ]
}
```

---

## Flujo del proceso

```
Imagen de entrada
      │
      ▼
  Carga + normalización
  (GeoTIFF preserva transform)
      │
      ▼
  Modelo de segmentación
  (SegFormer / Mask2Former)
      │
      ▼
  Máscara binaria de edificios
      │
      ▼
  Morfología (cierre, apertura)
  → elimina ruido, rellena huecos
      │
      ▼
  findContours (OpenCV)
      │
      ▼
  Simplificación Douglas-Peucker
  (Shapely)
      │
      ▼
  Conversión px → lon/lat
  (Affine transform rasterio)
      │
      ▼
  GeoJSON export
```

---

## Abrir el GeoJSON en QGIS

1. Arrastra el `.geojson` a QGIS
2. Clic derecho → Propiedades → Simbología → Relleno simple, sin relleno, borde cian
3. Superpón con capa satelital (XYZ Tiles → Google/Mapbox)

---

## Limitaciones del prototipo

- La precisión depende del modelo y de la resolución de la imagen
- Funciona mejor con imágenes aéreas/satelitales de alta resolución (< 0.5 m/px)
- Imágenes de baja resolución (Sentinel-2, 10 m/px) producen resultados pobres
- No distingue construcciones pequeñas de sombras o pavimento oscuro

---

© 2025 GeoWay Initiative · MIT License
