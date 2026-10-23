#!/usr/bin/env python3
"""
MONITOR DE GRANIZO - MALARGÜE
==============================
Cruza dos fuentes abiertas y te avisa al celular (app ntfy):

1. RAYOS DEL SATÉLITE GOES-19 (sensor GLM de la NOAA)
   Archivos nuevos cada 20 s, cubre toda Argentina. Detecta:
   - tormenta eléctrica a menos de 100 km
   - tormenta que SE ACERCA (con dirección y tiempo estimado de llegada)
   - "salto de rayos": aumento brusco de actividad -> precursor de granizo
   - tormenta encima (< 10 km)
   - tormenta que se fue (45 min sin rayos)

2. PRONÓSTICO POR HORA para tus coordenadas (Open-Meteo)
   Inestabilidad atmosférica (CAPE, índice de levantamiento, tormentas)
   -> aviso de "día con riesgo" una vez por día.

Uso:
  python granizo_monitor.py           # una revisión (para GitHub Actions / cron)
  python granizo_monitor.py --loop    # revisa cada 5 min sin parar (PC encendida)
  python granizo_monitor.py --test    # manda una notificación de prueba

Variables de entorno:
  NTFY_TOPIC  nombre del tema en la app ntfy (obligatorio para recibir avisos)
  LAT, LON    coordenadas de tu casa (por defecto: centro de Malargüe)
"""

import datetime as dt
import json
import math
import os
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests
from netCDF4 import Dataset

# ---------------------------------------------------------------- CONFIG
LAT = float(os.getenv("LAT") or "-35.475")
LON = float(os.getenv("LON") or "-69.585")
NTFY_TOPIC = os.getenv("NTFY_TOPIC", "").strip()
STATE_FILE = os.getenv("STATE_FILE") or "state.json"

RADIO_KM = 60           # radio de vigilancia de rayos
RADIO_SALTO_KM = 60     # radio para detectar "salto de rayos"
DIST_CERCA_KM = 40      # tormenta "cerca"
DIST_ENCIMA_KM = 10     # tormenta "encima"
VENTANA_MIN = 10        # se comparan dos ventanas de 10 min (reciente vs previa)
SALTO_MIN_RAYOS = 25    # mínimo de rayos en la ventana reciente para hablar de salto
SALTO_FACTOR = 2.0      # la ventana reciente tiene que duplicar a la previa
COOLDOWN_MIN = 20       # no repetir el mismo nivel de aviso antes de esto
CALMA_MIN = 45          # sin rayos este tiempo = "se fue"
DIST_PROXIMA_KM = 20    # tormenta "cerca" aunque no se mueva (celda que nace cerca)
VEL_MAX_KMH = 80        # más rápido que esto no es movimiento real: es una celda nueva
ENCIMA_MIN_RAYOS = 5    # rayos a menos de DIST_ENCIMA_KM para el aviso urgente

BUCKET = "https://noaa-goes19.s3.amazonaws.com"
PRODUCT = "GLM-L2-LCFA"
S3_NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
TZ_LOCAL = dt.timezone(dt.timedelta(hours=-3))

NIVELES = {0: "calma", 1: "detectada", 2: "cerca", 3: "salto", 4: "encima"}

# ---------------------------------------------------------------- UTILIDADES


def dist_km(lat1, lon1, lat2, lon2):
    """Distancia haversine en km (acepta arrays numpy)."""
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dlmb = np.radians(np.asarray(lon2) - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def rumbo(lat1, lon1, lat2, lon2):
    """Rumbo en grados desde (lat1, lon1) hacia (lat2, lon2)."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    x = math.sin(dl) * math.cos(p2)
    y = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def cardinal(grados):
    return ["norte", "noreste", "este", "sureste", "sur", "suroeste", "oeste", "noroeste"][
        int(round(grados / 45.0)) % 8
    ]


def cargar_estado():
    try:
        with open(STATE_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def guardar_estado(estado):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(estado, f, indent=2)


def avisar(titulo, mensaje, prioridad=3, tags=("zap",)):
    """Notificación por ntfy. Prioridad: 1 mín, 2 baja, 3 normal, 4 alta, 5 urgente."""
    print(f"[AVISO p{prioridad}] {titulo}: {mensaje}")
    if not NTFY_TOPIC:
        print("  (NTFY_TOPIC vacío: no se envía al celular)")
        return
    try:
        r = requests.post(
            "https://ntfy.sh/",
            json={
                "topic": NTFY_TOPIC,
                "title": titulo,
                "message": mensaje,
                "priority": prioridad,
                "tags": list(tags),
                "click": "https://map.blitzortung.org/",
            },
            timeout=20,
        )
        r.raise_for_status()
    except requests.RequestException as e:
        print(f"  ERROR enviando aviso: {e}")


# ---------------------------------------------------------------- RAYOS GOES-19


def inicio_archivo(key):
    """Hora UTC de inicio del archivo, sacada del nombre (_sYYYYDDDHHMMSSt_)."""
    try:
        s = key.rsplit("/", 1)[-1].split("_s", 1)[1][:13]
        return dt.datetime.strptime(s, "%Y%j%H%M%S").replace(tzinfo=dt.timezone.utc)
    except (IndexError, ValueError):
        return None


def listar_archivos(desde, hasta):
    horas = []
    h = desde.replace(minute=0, second=0, microsecond=0)
    while h <= hasta:
        horas.append(h)
        h += dt.timedelta(hours=1)

    archivos = []
    for h in horas:
        prefix = f"{PRODUCT}/{h:%Y}/{h.timetuple().tm_yday:03d}/{h:%H}/"
        token = None
        while True:
            params = {"list-type": "2", "prefix": prefix}
            if token:
                params["continuation-token"] = token
            r = requests.get(BUCKET + "/", params=params, timeout=30)
            r.raise_for_status()
            root = ET.fromstring(r.content)
            for c in root.findall("s3:Contents", S3_NS):
                key = c.findtext("s3:Key", namespaces=S3_NS)
                t0 = inicio_archivo(key)
                if t0 and desde <= t0 < hasta:
                    archivos.append((t0, key))
            if root.findtext("s3:IsTruncated", namespaces=S3_NS) == "true":
                token = root.findtext("s3:NextContinuationToken", namespaces=S3_NS)
            else:
                break
    return sorted(archivos)


def descargar(item):
    """Descarga un archivo GLM (esto sí se puede hacer en paralelo)."""
    t0, key = item
    r = requests.get(f"{BUCKET}/{key}", timeout=60)
    r.raise_for_status()
    return t0, r.content


def leer_rayos(t0, contenido):
    """Devuelve (hora, lats, lons) de los rayos de buena calidad cerca de la casa.
    OJO: netCDF4/HDF5 no es thread-safe -> llamar de a uno, nunca en hilos."""
    fd, path = tempfile.mkstemp(suffix=".nc")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(contenido)
        with Dataset(path) as ds:
            if "flash_lat" not in ds.variables or ds.variables["flash_lat"].shape[0] == 0:
                return t0, np.array([]), np.array([])
            lat = np.ma.filled(ds.variables["flash_lat"][:].astype(float), np.nan)
            lon = np.ma.filled(ds.variables["flash_lon"][:].astype(float), np.nan)
            if "flash_quality_flag" in ds.variables:
                q = np.ma.filled(ds.variables["flash_quality_flag"][:], 1)
                ok = q == 0
                lat, lon = lat[ok], lon[ok]
    finally:
        os.remove(path)

    # Recorte rápido por caja (~1.5° alrededor) antes de calcular distancias
    caja = (np.abs(lat - LAT) < 1.5) & (np.abs(lon - LON) < 1.8)
    return t0, lat[caja], lon[caja]


def resumen_ventana(lats, lons):
    """Estadísticas de una ventana de rayos dentro del radio de vigilancia."""
    if len(lats) == 0:
        return {"n": 0, "n_salto": 0}
    d = dist_km(LAT, LON, lats, lons)
    dentro = d <= RADIO_KM
    d, la, lo = d[dentro], lats[dentro], lons[dentro]
    if len(d) == 0:
        return {"n": 0, "n_salto": 0}
    # "Frente" de la tormenta: percentil 10 de distancias (robusto a rayos sueltos)
    frente = float(np.percentile(d, 10))
    idx = np.argsort(d)[: max(5, len(d) // 5)]
    lat_c, lon_c = float(np.median(la[idx])), float(np.median(lo[idx]))
    return {
        "n": int(len(d)),
        "n_salto": int(np.sum(d <= RADIO_SALTO_KM)),
        "min": float(d.min()),
        "n_encima": int(np.sum(d <= DIST_ENCIMA_KM)),
        "frente": frente,
        "rumbo": rumbo(LAT, LON, lat_c, lon_c),
    }


def analizar_rayos(estado):
    ahora = dt.datetime.now(dt.timezone.utc)
    corte = ahora - dt.timedelta(minutes=VENTANA_MIN)
    desde = ahora - dt.timedelta(minutes=2 * VENTANA_MIN)

    archivos = listar_archivos(desde, ahora)
    if not archivos:
        print("Sin archivos GLM recientes (¿demora de la NOAA?).")
        return

    with ThreadPoolExecutor(max_workers=8) as ex:
        descargados = list(ex.map(descargar, archivos))
    resultados = [leer_rayos(t0, contenido) for t0, contenido in descargados]

    rec_la, rec_lo, prev_la, prev_lo = [], [], [], []
    for t0, la, lo in resultados:
        if t0 >= corte:
            rec_la.append(la); rec_lo.append(lo)
        else:
            prev_la.append(la); prev_lo.append(lo)

    cat = lambda xs: np.concatenate(xs) if xs else np.array([])
    rec = resumen_ventana(cat(rec_la), cat(rec_lo))
    prev = resumen_ventana(cat(prev_la), cat(prev_lo))
    print(f"Archivos: {len(archivos)} | rayos <{RADIO_KM} km: previa={prev['n']} reciente={rec['n']}")

    ahora_s = ahora.timestamp()
    nivel_ant = estado.get("nivel", 0)
    ultimo_aviso = estado.get("ultimo_aviso", 0)

    # ---------- Sin rayos recientes ----------
    if rec["n"] == 0:
        if prev["n"] > 0:
            estado["ultimo_rayo"] = corte.timestamp()
        ultimo_rayo = estado.get("ultimo_rayo", 0)
        if nivel_ant > 0 and ahora_s - ultimo_rayo > CALMA_MIN * 60:
            avisar("Tormenta se fue",
                   f"Sin rayos a menos de {RADIO_KM} km en los últimos {CALMA_MIN} min.",
                   2, ("white_check_mark",))
            estado["nivel"] = 0
            estado["ultimo_aviso"] = ahora_s
        estado["acerc_prev"] = False
        return

    estado["ultimo_rayo"] = ahora_s

    # ---------- Hay rayos: evaluar nivel ----------
    frente, dirc = rec["frente"], cardinal(rec["rumbo"])

    # Acercamiento: el frente baja >3 km entre ventanas, a una velocidad creíble,
    # y eso se repite en DOS revisiones seguidas (filtra celdas que "burbujean").
    vel_kmh = (prev["frente"] - frente) * (60 / VENTANA_MIN) if prev["n"] > 0 else 0
    acerc_ahora = prev["n"] > 0 and frente < prev["frente"] - 3 and vel_kmh <= VEL_MAX_KMH
    acercandose = acerc_ahora and estado.get("acerc_prev", False)
    estado["acerc_prev"] = acerc_ahora
    eta = frente / vel_kmh * 60 if acercandose and vel_kmh > 0 else None  # minutos

    salto = (rec["n_salto"] >= SALTO_MIN_RAYOS and
             rec["n_salto"] >= SALTO_FACTOR * max(prev["n_salto"], 1))

    if rec["n_encima"] >= ENCIMA_MIN_RAYOS:
        nivel = 4
    elif salto:
        nivel = 3
    elif (frente <= DIST_CERCA_KM and acercandose) or frente <= DIST_PROXIMA_KM:
        nivel = 2
    else:
        nivel = 1

    # ¿Avisar? Si sube el nivel, o si se mantiene alto y pasó el cooldown.
    paso_cooldown = ahora_s - ultimo_aviso > COOLDOWN_MIN * 60
    # Repetir el mismo nivel solo si es grave o si la tormenta efectivamente viene.
    repetible = nivel >= 3 or (nivel == 2 and acercandose)
    avisar_ahora = (nivel > nivel_ant) or (repetible and nivel == nivel_ant and paso_cooldown)

    if not avisar_ahora:
        # Si bajó de nivel y pasó el cooldown, lo registramos (así un nuevo
        # empeoramiento vuelve a avisar, sin spam por oscilaciones cortas).
        if nivel < nivel_ant and paso_cooldown:
            estado["nivel"] = nivel
        print(f"Nivel {NIVELES[nivel]} (sin aviso nuevo)")
        return

    estado["nivel"] = nivel
    estado["ultimo_aviso"] = ahora_s
    llegada = f" Llegaría en ~{int(round(eta))} min." if eta and eta < 120 else ""
    tendencia = " y se ACERCA" if acercandose else ""

    if nivel == 4:
        avisar("TORMENTA ENCIMA",
               f"{rec['n_encima']} rayos a menos de {DIST_ENCIMA_KM} km en 10 min "
               f"({rec['n']} en total). Riesgo de granizo: guardá el auto YA.",
               5, ("rotating_light", "zap"))
    elif nivel == 3:
        avisar("Salto de rayos - posible granizo",
               f"La actividad eléctrica se disparó: {rec['n_salto']} rayos en 10 min "
               f"(antes {prev['n_salto']}). Frente a ~{frente:.0f} km al {dirc}{tendencia}.{llegada}",
               4, ("warning", "zap"))
    elif nivel == 2 and acercandose:
        avisar("Tormenta se acerca",
               f"Frente a ~{frente:.0f} km al {dirc}, acercándose ~{vel_kmh:.0f} km/h.{llegada}",
               4, ("warning", "zap"))
    elif nivel == 2:
        avisar("Tormenta cerca",
               f"Rayos a ~{frente:.0f} km al {dirc} ({rec['n']} en 10 min). "
               "Por ahora no se ve que se esté moviendo hacia acá.",
               4, ("warning", "zap"))
    else:
        avisar("Tormenta detectada",
               f"Rayos a ~{frente:.0f} km al {dirc} ({rec['n']} en 10 min){tendencia}.{llegada}",
               3, ("zap",))


# ---------------------------------------------------------------- PRONÓSTICO


def analizar_pronostico(estado):
    r = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": LAT,
            "longitude": LON,
            "hourly": "weather_code,cape,lifted_index,precipitation_probability",
            "forecast_hours": 12,
            "timezone": "America/Argentina/Mendoza",
        },
        timeout=30,
    )
    r.raise_for_status()
    h = r.json()["hourly"]

    horas_riesgo, peor = [], 0
    for i, hora in enumerate(h["time"]):
        cape = h["cape"][i] or 0
        li = h["lifted_index"][i]
        code = h["weather_code"][i] or 0
        pp = h["precipitation_probability"][i] or 0
        p = 0
        p += 2 if cape >= 1500 else (1 if cape >= 800 else 0)
        if li is not None:
            p += 2 if li <= -4 else (1 if li <= -2 else 0)
        p += 2 if code in (95, 96, 99) else 0
        p += 1 if pp >= 50 else 0
        if p >= 3:
            horas_riesgo.append((hora[11:16], p, cape))
            peor = max(peor, p)

    if not horas_riesgo:
        print("Pronóstico: sin señales de tormenta fuerte en 12 h.")
        return

    hoy = dt.datetime.now(TZ_LOCAL).date().isoformat()
    nivel = "ALTO" if peor >= 5 else "moderado"
    ya = estado.get("pronostico", {})
    if ya.get("fecha") == hoy and (ya.get("nivel") == nivel or ya.get("nivel") == "ALTO"):
        print(f"Pronóstico: riesgo {nivel} (ya avisado hoy).")
        return

    desde, hasta = horas_riesgo[0][0], horas_riesgo[-1][0]
    cape_max = max(c for _, _, c in horas_riesgo)
    avisar(f"Riesgo {nivel} de tormentas hoy",
           f"Atmósfera inestable entre las {desde} y las {hasta} (CAPE máx {cape_max:.0f}). "
           "Posibles tormentas con granizo: te aviso si aparecen rayos cerca.",
           4 if nivel == "ALTO" else 3, ("cloud_with_lightning",))
    estado["pronostico"] = {"fecha": hoy, "nivel": nivel}


# ---------------------------------------------------------------- MAIN


def una_revision():
    estado = cargar_estado()
    try:
        for tarea in (analizar_rayos, analizar_pronostico):
            try:
                tarea(estado)
            except Exception as e:  # una fuente caída no frena a la otra
                print(f"ERROR en {tarea.__name__}: {e}")
    finally:
        guardar_estado(estado)


def main():
    if "--test" in sys.argv:
        avisar("Monitor de granizo", "Notificación de prueba: el monitor está funcionando.",
               3, ("satellite",))
        return
    if "--loop" in sys.argv:
        while True:
            print(f"\n== {dt.datetime.now(TZ_LOCAL):%Y-%m-%d %H:%M} ==")
            una_revision()
            time.sleep(300)
    else:
        una_revision()


if __name__ == "__main__":
    main()
