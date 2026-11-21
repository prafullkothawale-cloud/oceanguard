import requests
import json
import random
import sqlite3
import tempfile
import threading
from datetime import datetime

import av
import cv2
import folium
import numpy as np
import pandas as pd
import streamlit as st
from folium.plugins import HeatMap
from PIL import Image
from streamlit_folium import st_folium
from streamlit_geolocation import streamlit_geolocation
from streamlit_webrtc import RTCConfiguration, webrtc_streamer
from ultralytics import YOLO

MODEL_PATH = "yolo11n_ocean_v3_best.pt"
DB_PATH = "detections.db"
POLLUTION = {"plastic", "plastic_bottle", "plastic_bag", "net"}

st.set_page_config(page_title="OceanGuard", page_icon="🌊", layout="wide")

# ---------- look and feel ----------
st.markdown(
    """
<style>
.hero {background: linear-gradient(100deg,#0A3550 0%,#0E6E8C 55%,#19C3C3 100%);
       padding: 22px 28px; border-radius: 16px; margin-bottom: 14px;}
.hero h1 {margin: 0; font-size: 2.1rem; color: #fff;}
.hero p {margin: 4px 0 0 0; color: #D8F3F7; font-size: 1.02rem;}
.card {background:#0A3550; border:1px solid #17607F; border-radius:14px; padding:14px 18px; margin-bottom:10px;}
.card b {color:#19C3C3;}
.stTabs [data-baseweb="tab"] {font-size: 1.02rem; padding: 10px 18px;}
.small {color:#9FC4D4; font-size:0.85rem;}
div[data-testid="stMetric"] {background:#0A3550; border:1px solid #17607F; border-radius:12px; padding:10px 14px;}
</style>
<div class="hero">
  <h1>🌊 OceanGuard</h1>
  <p>AI-powered marine plastic detection and pollution hotspot mapping</p>
</div>
""",
    unsafe_allow_html=True,
)


# ---------- model ----------
@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)


try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load model '{MODEL_PATH}'. Put the .pt file next to app.py. Details: {e}")
    st.stop()

POLL_IDS = [i for i, n in model.names.items() if n in POLLUTION]


def detect(bgr_image, conf, only_plastic):
    """bgr_image: BGR numpy array (what Ultralytics expects). Returns annotated RGB image and counts."""
    res = model.predict(
        bgr_image, conf=conf, agnostic_nms=True,
        classes=POLL_IDS if only_plastic else None, verbose=False,
    )[0]
    annotated = res.plot()[:, :, ::-1]
    counts = {}
    for c in res.boxes.cls.tolist():
        name = res.names[int(c)]
        counts[name] = counts.get(name, 0) + 1
    return annotated, counts


def split_counts(counts):
    plastic = sum(v for k, v in counts.items() if k in POLLUTION)
    animals = sum(v for k, v in counts.items() if k not in POLLUTION)
    return plastic, animals


def exif_gps(img):
    try:
        gps = img.getexif().get_ifd(0x8825)
        if not gps:
            return None

        def conv(v):
            d, m, s = [float(x) for x in v]
            return d + m / 60 + s / 3600

        lat, lon = conv(gps[2]), conv(gps[4])
        if gps.get(1) == "S":
            lat = -lat
        if gps.get(3) == "W":
            lon = -lon
        return lat, lon
    except Exception:
        return None

# ---------- database (Supabase online storage, SQLite fallback) ----------
COLS = ["id", "ts", "lat", "lon", "source", "plastic", "animals", "details"]


def cloud_on():
    try:
        return "SUPABASE_URL" in st.secrets and "SUPABASE_KEY" in st.secrets
    except Exception:
        return False


def _hdr():
    k = st.secrets["SUPABASE_KEY"]
    return {"apikey": k, "Authorization": f"Bearer {k}", "Content-Type": "application/json"}


def _url():
    return st.secrets["SUPABASE_URL"].rstrip("/") + "/rest/v1/detections"


def db():
    con = sqlite3.connect(DB_PATH)
    con.execute(
        """CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, lat REAL, lon REAL, source TEXT,
            plastic INTEGER, animals INTEGER, details TEXT)"""
    )
    return con


def insert_rows(rows):
    """rows: list of dicts with ts, lat, lon, source, plastic, animals, details."""
    if cloud_on():
        r = requests.post(_url(), headers={**_hdr(), "Prefer": "return=minimal"}, json=rows, timeout=20)
        r.raise_for_status()
        return
    con = db()
    con.executemany(
        "INSERT INTO detections (ts, lat, lon, source, plastic, animals, details) VALUES (?,?,?,?,?,?,?)",
        [(x["ts"], x["lat"], x["lon"], x["source"], x["plastic"], x["animals"], x["details"]) for x in rows],
    )
    con.commit()
    con.close()


def save_row(lat, lon, source, counts):
    plastic, animals = split_counts(counts)
    insert_rows([{
        "ts": datetime.now().isoformat(timespec="seconds"), "lat": float(lat), "lon": float(lon),
        "source": source, "plastic": int(plastic), "animals": int(animals), "details": json.dumps(counts),
    }])


def load_rows():
    if cloud_on():
        try:
            r = requests.get(_url(), headers=_hdr(),
                             params={"select": "*", "order": "id.asc", "limit": 1000}, timeout=20)
            r.raise_for_status()
            df = pd.DataFrame(r.json())
            return df[COLS] if not df.empty else pd.DataFrame(columns=COLS)
        except Exception as e:
            st.error(f"Could not read online storage: {e}")
            return pd.DataFrame(columns=COLS)
    con = db()
    df = pd.read_sql_query("SELECT * FROM detections", con)
    con.close()
    return df


def add_demo_data():
    random.seed(7)
    centers = [(18.95, 72.80, 12), (19.10, 72.75, 6), (18.80, 72.90, 4)]
    rows = []
    for clat, clon, n in centers:
        for _ in range(n):
            rows.append({
                "ts": datetime.now().isoformat(timespec="seconds"),
                "lat": clat + random.uniform(-0.06, 0.06), "lon": clon + random.uniform(-0.06, 0.06),
                "source": "demo", "plastic": max(0, int(random.gauss(8 if n > 8 else 4, 3))),
                "animals": random.choice([0, 0, 1, 2, 3]), "details": "{}",
            })
    insert_rows(rows)


def _delete(sql_where, params, rest_query):
    if cloud_on():
        r = requests.delete(_url() + rest_query, headers=_hdr(), timeout=20)
        r.raise_for_status()
        return
    con = db()
    con.execute("DELETE FROM detections" + sql_where, params)
    con.commit()
    con.close()


def clear_demo_data():
    _delete(" WHERE source='demo'", (), "?source=eq.demo")


def clear_all_data():
    _delete("", (), "?id=gt.0")


def delete_record(rec_id):
    _delete(" WHERE id = ?", (int(rec_id),), f"?id=eq.{int(rec_id)}")


ESRI_IMG = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
ESRI_LABELS = "https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}"


def base_map(lat, lon, zoom=9):
    m = folium.Map(location=[lat, lon], zoom_start=zoom, tiles=None)
    folium.TileLayer("OpenStreetMap", name="Street map", show=False).add_to(m)
    folium.TileLayer(tiles=ESRI_IMG, attr="Esri", name="Satellite", show=True).add_to(m)
    folium.TileLayer(tiles=ESRI_LABELS, attr="Esri", name="Place names", overlay=True, show=True).add_to(m)
    return m


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_currents(lats, lons, hours):
    """Hourly ocean current (speed km/h, direction the current is heading towards) from Open-Meteo."""
    try:
        r = requests.get(
            "https://marine-api.open-meteo.com/v1/marine",
            params={
                "latitude": ",".join(f"{x:.4f}" for x in lats),
                "longitude": ",".join(f"{x:.4f}" for x in lons),
                "hourly": "ocean_current_velocity,ocean_current_direction",
                "forecast_hours": hours,
                "cell_selection": "sea",
            },
            timeout=20,
        )
        r.raise_for_status()
        data = r.json()
        if isinstance(data, dict):
            data = [data]
        out = []
        for d in data:
            h = d.get("hourly", {})
            out.append((h.get("ocean_current_velocity", []), h.get("ocean_current_direction", [])))
        return out
    except Exception:
        return None


def drift_path(lat, lon, speeds, dirs):
    """Move a point hour by hour. speeds in km/h, dirs in degrees (direction of travel, 0 = north)."""
    path = [(lat, lon)]
    for s, d in zip(speeds, dirs):
        if s is None or d is None:
            continue
        lat += s * np.cos(np.radians(d)) / 111.0
        lon += s * np.sin(np.radians(d)) / (111.0 * max(0.2, np.cos(np.radians(lat))))
        path.append((lat, lon))
    return path


# ---------- sidebar ----------
st.sidebar.markdown("## 🌊 OceanGuard")
st.sidebar.caption("Detect plastic. Protect marine life.")
st.sidebar.caption("☁️ Online storage connected" if cloud_on() else "💾 Local storage (not permanent online)")
conf = st.sidebar.slider("Confidence threshold", 0.10, 0.90, 0.40, 0.05)
only_plastic = st.sidebar.checkbox("Show only plastic / debris", value=False)
st.sidebar.markdown("---")
st.sidebar.subheader("Demo data")
st.sidebar.caption("Simulated points for presenting the map. Not real measurements.")
if st.sidebar.button("Load simulated demo data"):
    add_demo_data()
    st.sidebar.success("Demo data added")
if st.sidebar.button("Remove demo data"):
    clear_demo_data()
    st.sidebar.success("Demo data removed")
confirm = st.sidebar.checkbox("Confirm delete")
if st.sidebar.button("Delete ALL saved records"):
    if confirm:
        clear_all_data()
        st.sidebar.success("All records deleted")
    else:
        st.sidebar.warning("Tick 'Confirm delete' first")    

tab1, tab2, tab3, tab4 = st.tabs(["📷 Photo / video", "📹 Live camera", "🗺️ Pollution map", "🔥 Hotspots"])


def result_panel(col, counts, lat, lon, key, source):
    plastic, animals = split_counts(counts)
    col.metric("Plastic / debris items", plastic)
    col.metric("Marine life detected", animals)
    if counts:
        col.dataframe(pd.DataFrame(counts.items(), columns=["class", "count"]), hide_index=True)
    if plastic and animals:
        col.warning("Plastic and marine life in the same scene: animals at risk.")
    col.caption(f"Location: {lat:.5f}, {lon:.5f}")
    if col.button("Save to pollution map", key=key):
        save_row(lat, lon, source, counts)
        col.success("Saved")


# ---------- tab 1: photo / video ----------
with tab1:
    st.subheader("Detect from a photo or video")
    files = st.file_uploader("Upload underwater images or a short video",
                             type=["jpg", "jpeg", "png", "mp4", "mov", "avi"], accept_multiple_files=True)
    c1, c2 = st.columns(2)
    manual_lat = c1.number_input("Latitude (used if the photo has no GPS)", value=18.95, format="%.5f", key="lat1")
    manual_lon = c2.number_input("Longitude (used if the photo has no GPS)", value=72.80, format="%.5f", key="lon1")

    for i, f in enumerate(files or []):
        st.markdown(f"##### {f.name}")
        ext = f.name.lower().rsplit(".", 1)[-1]
        left, right = st.columns([3, 2])
        if ext in ("jpg", "jpeg", "png"):
            img = Image.open(f).convert("RGB")
            gps = exif_gps(Image.open(f))
            lat, lon = gps if gps else (manual_lat, manual_lon)
            bgr = np.ascontiguousarray(np.array(img)[:, :, ::-1])
            annotated, counts = detect(bgr, conf, only_plastic)
            left.image(annotated, width="stretch")
        else:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix="." + ext)
            tmp.write(f.read())
            tmp.close()
            cap = cv2.VideoCapture(tmp.name)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
            step = max(1, total // 24)
            counts, shown, idx = {}, [], 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if idx % step == 0:
                    ann, c = detect(frame, conf, only_plastic)
                    for k, v in c.items():
                        counts[k] = max(counts.get(k, 0), v)
                    if c and len(shown) < 6:
                        shown.append(ann)
                idx += 1
            cap.release()
            lat, lon = manual_lat, manual_lon
            if shown:
                left.image(shown, width=220)
            else:
                left.info("No detections in the sampled frames.")
            right.caption("Video counts = highest count seen in any one sampled frame.")
        result_panel(right, counts, lat, lon, f"save_{i}_{f.name}", f.name)


# ---------- tab 2: live camera ----------
class LiveProcessor:
    def __init__(self):
        self.conf = 0.4
        self.classes = None
        self.lock = threading.Lock()
        self.counts = {}

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        res = model.predict(img, conf=self.conf, imgsz=416, agnostic_nms=True,
                            classes=self.classes, verbose=False)[0]
        c = {}
        for k in res.boxes.cls.tolist():
            n = res.names[int(k)]
            c[n] = c.get(n, 0) + 1
        with self.lock:
            self.counts = c
        return av.VideoFrame.from_ndarray(res.plot(), format="bgr24")

    def latest(self):
        with self.lock:
            return dict(self.counts)


@st.fragment(run_every=1)
def live_counts(ctx):
    p = ctx.video_processor
    if not p:
        st.caption("Press START and allow camera access.")
        return
    counts = p.latest()
    plastic, animals = split_counts(counts)
    a, b = st.columns(2)
    a.metric("Plastic / debris in view", plastic)
    b.metric("Marine life in view", animals)


with tab2:
    st.subheader("Live camera detection")
    st.markdown(
        '<div class="card"><b>On your phone:</b> open this page in Chrome, press START, allow the camera '
        "and location, and point the rear camera at the scene. Press <b>Mark hotspot point</b> to log what "
        "the camera sees at your current GPS location.</div>",
        unsafe_allow_html=True,
    )
    ctx = webrtc_streamer(
        key="live",
        video_processor_factory=LiveProcessor,
        rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
        media_stream_constraints={"video": {"width": {"ideal": 640}, "height": {"ideal": 480}}, "audio": False},
        async_processing=True,
    )
    if ctx.video_processor:
        ctx.video_processor.conf = conf
        ctx.video_processor.classes = POLL_IDS if only_plastic else None
    live_counts(ctx)

    st.markdown("##### Mark a hotspot point")
    st.caption("1. Tap the location button below and allow location.  2. Press Mark hotspot point.")
    loc = streamlit_geolocation()
    glat, glon = (loc or {}).get("latitude"), (loc or {}).get("longitude")
    if glat is None:
        c1, c2 = st.columns(2)
        glat = c1.number_input("Latitude (if GPS is not available)", value=18.95, format="%.5f", key="lat2")
        glon = c2.number_input("Longitude (if GPS is not available)", value=72.80, format="%.5f", key="lon2")
    else:
        st.success(f"GPS: {glat:.5f}, {glon:.5f}")
    if st.button("📍 Mark hotspot point"):
        if ctx.video_processor:
            counts = ctx.video_processor.latest()
            save_row(glat, glon, "live-camera", counts)
            p, a = split_counts(counts)
            st.success(f"Saved: {p} plastic / debris, {a} marine life at {glat:.5f}, {glon:.5f}")
        else:
            st.warning("Start the camera first.")

    with st.expander("Camera not connecting? Use snapshot mode"):
        snap = st.camera_input("Take a photo")
        if snap:
            img = Image.open(snap).convert("RGB")
            bgr = np.ascontiguousarray(np.array(img)[:, :, ::-1])
            annotated, counts = detect(bgr, conf, only_plastic)
            l, r = st.columns([3, 2])
            l.image(annotated, width="stretch")
            result_panel(r, counts, glat, glon, "save_snap", "snapshot")


# ---------- tab 3: map ----------
with tab3:
    st.subheader("Pollution map")
    df = load_rows()
    if df.empty:
        st.info("No data yet. Save a detection, mark a hotspot point, or load demo data from the sidebar.")
    else:
        m = base_map(df.lat.mean(), df.lon.mean())
        HeatMap(df[["lat", "lon", "plastic"]].values.tolist(), radius=25, blur=18).add_to(m)
        for _, r in df.iterrows():
            folium.CircleMarker([r.lat, r.lon], radius=4, color="#0b5394", fill=True,
                                tooltip=f"{r.source}: {r.plastic} plastic, {r.animals} animals").add_to(m)
        folium.LayerControl().add_to(m)
        st_folium(m, height=520, use_container_width=True)
        a, b, c = st.columns(3)
        a.metric("Records", len(df))
        b.metric("Total plastic items", int(df.plastic.sum()))
        c.metric("Total marine life", int(df.animals.sum()))
        with st.expander("Manage records (delete one)"):
            st.dataframe(df, hide_index=True)
            options = {f"#{r.id} | {r.ts} | {r.source} | plastic {r.plastic}": int(r.id) for _, r in df.iterrows()}
            pick = st.selectbox("Select a record to delete", list(options.keys()))
            if st.button("Delete selected record"):
                delete_record(options[pick])
                st.rerun()

# ---------- tab 4: hotspots ----------
with tab4:
    st.subheader("Hotspots and cleanup priority")
    st.caption("Estimated from the density of plastic detections in each map cell, boosted where marine life is also present.")
    df = load_rows()
    if df.empty:
        st.info("No data yet.")
    else:
        cell = st.select_slider("Grid cell size (degrees)", options=[0.02, 0.05, 0.1, 0.2], value=0.1)
        d = df.copy()
        d["glat"] = (d.lat // cell) * cell + cell / 2
        d["glon"] = (d.lon // cell) * cell + cell / 2
        g = d.groupby(["glat", "glon"]).agg(records=("id", "count"), plastic=("plastic", "sum"),
                                            animals=("animals", "sum")).reset_index()
        g["score"] = (g.plastic * (1 + 0.5 * (g.animals > 0))).round(1)
        g = g.sort_values("score", ascending=False).reset_index(drop=True)
        top = g.score.max() or 1
        g["level"] = pd.cut(g.score / top, [-0.01, 0.33, 0.66, 1.0], labels=["Low", "Medium", "High"])

        m = base_map(g.glat.mean(), g.glon.mean())
        colors = {"Low": "#2e8b57", "Medium": "#f0a500", "High": "#c0392b"}
        for _, r in g.iterrows():
            folium.Circle([r.glat, r.glon], radius=cell * 111000 / 2, color=colors[r.level], fill=True,
                          fill_opacity=0.45, tooltip=f"{r.level}: score {r.score}, {int(r.plastic)} plastic").add_to(m)
        folium.LayerControl().add_to(m)
        st.markdown("##### Drift forecast")
        mode = st.radio("Current data", ["Real forecast (Open-Meteo)", "Manual"], horizontal=True)
        hours = st.slider("Hours ahead", 6, 72, 24)
        cand = g[g.level != "Low"].head(8).reset_index(drop=True)
        drift_rows = []
        if len(cand) == 0:
            st.info("No Medium or High hotspots to forecast yet.")
        elif mode.startswith("Real"):
            series = fetch_currents(tuple(float(x) for x in cand.glat), tuple(float(x) for x in cand.glon), hours)
            if not series or not any(len(sp) for sp, _ in series):
                st.warning("Could not get current data right now. Try again later or switch to Manual.")
            else:
                for (_, r), (sp, di) in zip(cand.iterrows(), series):
                    vals = [s for s in sp if s is not None]
                    drift_rows.append((r, drift_path(r.glat, r.glon, sp, di), sum(vals) / len(vals) if vals else 0))
        else:
            c1, c2 = st.columns(2)
            bearing = c1.slider("Current direction (degrees, 0 = north, 90 = east)", 0, 359, 90)
            speed = c2.slider("Current speed (km/h)", 0.0, 5.0, 1.0, 0.1)
            for _, r in cand.iterrows():
                drift_rows.append((r, drift_path(r.glat, r.glon, [speed] * hours, [bearing] * hours), speed))

        table = []
        for r, path, avg in drift_rows:
            if len(path) < 2:
                continue
            folium.PolyLine(path, color="#8e44ad", weight=3, dash_array="6", tooltip=f"Drift path ({hours} h)").add_to(m)
            end = path[-1]
            folium.Circle(end, radius=cell * 111000 / 2, color="#8e44ad", fill=True, fill_opacity=0.25,
                          tooltip=f"Expected position in {hours} h (score {r.score})").add_to(m)
            km = float(np.hypot((end[0] - path[0][0]) * 111.0,
                                (end[1] - path[0][1]) * 111.0 * np.cos(np.radians(path[0][0]))))
            table.append({"from lat": round(r.glat, 3), "from lon": round(r.glon, 3),
                          "to lat": round(end[0], 3), "to lon": round(end[1], 3),
                          "drift (km)": round(km, 1), "avg current (km/h)": round(avg, 2)})
        if drift_rows and not table:
            st.info("No current data for these cells (they may be too close to land). Try Manual.")
        st_folium(m, height=480, use_container_width=True)
        if table:
            st.dataframe(pd.DataFrame(table), hide_index=True, width="stretch")
        st.caption("Current data: Open-Meteo Marine API (Copernicus Marine, about 8 km resolution), CC BY 4.0. "
                   "Floating plastic is also pushed by wind and waves, which this estimate ignores, and "
                   "accuracy near the coast is limited.")

        st.markdown("##### Cleanup priority list")
        out = g.rename(columns={"glat": "lat", "glon": "lon"})
        out.insert(0, "rank", range(1, len(out) + 1))
        st.dataframe(out, hide_index=True, width="stretch")
        st.caption("Priority score = plastic count x 1.5 if marine life is also present.")

st.markdown('<p class="small">OceanGuard | YOLO11n model, 9 classes | SY AI&DS PBL project</p>', unsafe_allow_html=True)

