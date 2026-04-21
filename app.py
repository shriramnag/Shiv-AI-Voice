# Shiv AI v4.1 — Professional Voice Cloning
# PAISAWALA | Shri Ram Nag

import os, re, gc, json, glob
import numpy as np
import torch
import gradio as gr
import requests
from TTS.api import TTS
from huggingface_hub import hf_hub_download
from pydub import AudioSegment, effects
from scipy.signal import butter, filtfilt, find_peaks
from scipy.io import wavfile

os.environ["COQUI_TOS_AGREED"] = "1"

# ── Device ──────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32       = True
    except: pass

REPO = "Shriramnag/My-Shriram-Voice"
GRAW = "https://raw.githubusercontent.com/shriramnag/Shiv-AI-Voice/main/voices/"
DFILE = "custom_dict.json"

print("Shiv AI v4.1 starting...")
try:
    hf_hub_download(repo_id=REPO, filename="Ramai.pth")
    hf_hub_download(repo_id=REPO, filename="config.json")
except: pass

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
print(f"Ready on {device.upper()}")

try:
    import librosa, librosa.effects
    HAS_LIBROSA = True
    print("librosa OK")
except:
    HAS_LIBROSA = False

# ── Default voices download ─────────────────────────────────────────
os.makedirs("voices", exist_ok=True)
for vf in ["aideva.wav", "Joanne.wav", "Reginald voice.wav", "cloning .wav"]:
    lp = os.path.join("voices", vf)
    if os.path.exists(lp) and os.path.getsize(lp) > 1000:
        continue
    for url in [GRAW + vf.replace(" ", "%20"),
                GRAW + requests.utils.quote(vf, safe="")]:
        try:
            r = requests.get(url, timeout=15)
            if r.status_code == 200 and len(r.content) > 1000:
                open(lp, "wb").write(r.content)
                print(f"Got: {vf}")
                break
        except: pass

# ── Dictionary ──────────────────────────────────────────────────────
def load_dict():
    try:
        if os.path.exists(DFILE):
            return json.load(open(DFILE, encoding="utf-8"))
    except: pass
    return {}

def save_dict(d):
    json.dump(d, open(DFILE, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

# ── Number → Hindi words ─────────────────────────────────────────────
_H = ["","एक","दो","तीन","चार","पाँच","छह","सात","आठ","नौ","दस","ग्यारह","बारह",
      "तेरह","चौदह","पंद्रह","सोलह","सत्रह","अठारह","उन्नीस","बीस","इक्कीस","बाईस",
      "तेईस","चौबीस","पच्चीस","छब्बीस","सत्ताईस","अट्ठाईस","उनतीस","तीस","इकतीस",
      "बत्तीस","तैंतीस","चौंतीस","पैंतीस","छत्तीस","सैंतीस","अड़तीस","उनतालीस","चालीस",
      "इकतालीस","बयालीस","तैंतालीस","चौंतालीस","पैंतालीस","छियालीस","सैंतालीस",
      "अड़तालीस","उनचास","पचास","इक्यावन","बावन","तिरपन","चौवन","पचपन","छप्पन",
      "सत्तावन","अट्ठावन","उनसठ","साठ","इकसठ","बासठ","तिरसठ","चौंसठ","पैंसठ",
      "छियासठ","सड़सठ","अड़सठ","उनहत्तर","सत्तर","इकहत्तर","बहत्तर","तिहत्तर",
      "चौहत्तर","पचहत्तर","छिहत्तर","सतहत्तर","अठहत्तर","उनासी","अस्सी","इक्यासी",
      "बयासी","तिरासी","चौरासी","पचासी","छियासी","सत्तासी","अट्ठासी","नवासी","नब्बे",
      "इक्यानवे","बानवे","तिरानवे","चौरानवे","पचानवे","छियानवे","सत्तानवे","अट्ठानवे","निन्यानवे"]

def n2h(n):
    if n == 0: return "शून्य"
    if n < 0: return "ऋण " + n2h(-n)
    if n <= 99: return _H[n]
    if n < 1000:
        h, r = divmod(n, 100)
        return _H[h] + " सौ" + (" " + _H[r] if r else "")
    if n < 100000:
        h, r = divmod(n, 1000)
        return _H[h] + " हज़ार" + (" " + n2h(r) if r else "")
    if n < 10000000:
        h, r = divmod(n, 100000)
        return _H[h] + " लाख" + (" " + n2h(r) if r else "")
    h, r = divmod(n, 10000000)
    return _H[h] + " करोड़" + (" " + n2h(r) if r else "")

# ── Text Processing ──────────────────────────────────────────────────
_SK = {"dharma":"धर्म","karma":"कर्म","yoga":"योग","shakti":"शक्ति","om":"ॐ",
       "namaste":"नमस्ते","guru":"गुरु","mantra":"मंत्र","atma":"आत्मा","maya":"माया",
       "moksha":"मोक्ष","ahimsa":"अहिंसा","satya":"सत्य","seva":"सेवा","bhakti":"भक्ति",
       "veda":"वेद","puja":"पूजा","pooja":"पूजा"}

_EN = {"AI":"ए आई","ML":"एम एल","API":"ए पी आई","GPU":"जी पी यू",
       "YouTube":"यूट्यूब","Instagram":"इंस्टाग्राम","Facebook":"फेसबुक",
       "WhatsApp":"व्हाट्सऐप","Google":"गूगल","Internet":"इंटरनेट",
       "Online":"ऑनलाइन","Software":"सॉफ्टवेयर","Computer":"कंप्यूटर",
       "Mobile":"मोबाइल","App":"ऐप","Website":"वेबसाइट","Download":"डाउनलोड",
       "Upload":"अपलोड","Channel":"चैनल","Video":"वीडियो","Content":"कंटेंट",
       "Subscribe":"सब्सक्राइब","Life":"लाइफ","Dream":"ड्रीम","Mindset":"माइंडसेट",
       "Success":"सक्सेस","Fail":"फेल","Goal":"गोल","Focus":"फोकस","Power":"पावर",
       "Money":"मनी","Business":"बिज़नेस","Market":"मार्केट","Smart":"स्मार्ट",
       "Team":"टीम","Plan":"प्लान","Skill":"स्किल","OK":"ओके","okay":"ओके",
       "hello":"हेलो","bye":"बाय","thanks":"थैंक्स"}

_ORD = {"1ला":"पहला","1ली":"पहली","2रा":"दूसरा","3रा":"तीसरा","4था":"चौथा",
        "5वीं":"पाँचवीं","5वें":"पाँचवें","6वें":"छठवें","7वें":"सातवें",
        "8वीं":"आठवीं","8वें":"आठवें","9वें":"नौवें","10वें":"दसवें"}

def process_text(text, custom):
    if not text: return ""
    # Custom dict
    for k, v in custom.items():
        text = re.sub(rf'(?<![a-zA-Z\u0900-\u097F]){re.escape(k)}(?![a-zA-Z\u0900-\u097F])',
                      v, text, flags=re.IGNORECASE)
    # Sanskrit
    for k, v in _SK.items():
        text = re.sub(rf'(?<![a-zA-Z]){re.escape(k)}(?![a-zA-Z])', v, text, flags=re.IGNORECASE)
    # English
    for k, v in _EN.items():
        text = re.sub(rf'(?<![a-zA-Z]){re.escape(k)}(?![a-zA-Z])', v, text, flags=re.IGNORECASE)
    # Ordinals
    for k, v in _ORD.items():
        text = text.replace(k, v)
    # Numbers
    text = re.sub(r'(\d+(?:\.\d+)?)%', lambda m: n2h(int(float(m.group(1)))) + " प्रतिशत", text)
    text = re.sub(r'(\d+)\.(\d+)', lambda m: n2h(int(m.group(1))) + " दशमलव " + n2h(int(m.group(2))), text)
    text = re.sub(r'\b(\d+)\b', lambda m: n2h(int(m.group(1))), text)
    # Punctuation → pauses
    text = re.sub(r'\n+',      '। ',  text)
    text = re.sub(r'\.\.\.+',  '। ',  text)
    text = re.sub(r'\?+',      '। ',  text)
    text = re.sub(r'!+',       '! ',  text)
    text = re.sub(r'[।\.]+',   '। ',  text)
    text = re.sub(r'[,،]+',    ', ',  text)
    text = re.sub(r'[-–—]+',   ', ',  text)
    text = re.sub(r'[;:]',     ', ',  text)
    text = re.sub(r'["""\'\'()\[\]{}*#@&^~`|<>]', '', text)
    text = re.sub(r'\s+',      ' ',   text)
    text = re.sub(r'(। ){2,}', '। ',  text)
    return text.strip()

# ── Chunker ──────────────────────────────────────────────────────────
def get_lang(words):
    if not words: return "hi"
    deva = sum(1 for w in words for c in w if '\u0900' <= c <= '\u097F')
    if deva == 0 and sum(1 for w in words if w.isalpha() and w.isascii()) > len(words) * 0.7:
        return "en"
    return "hi"

def make_chunks(text, max_w=30):
    sents = [s.strip() for s in re.split(r'(?<=।)\s+|\n+', text) if s.strip()]
    result, buf = [], []

    def flush():
        if buf:
            result.append((" ".join(buf), get_lang(buf)))
            buf.clear()

    for sent in sents:
        words = sent.split()
        if not words: continue
        if len(words) > max_w:
            flush()
            tmp = []
            for part in re.split(r',\s*', sent):
                pw = part.split()
                if len(tmp) + len(pw) > max_w and tmp:
                    result.append((" ".join(tmp), get_lang(tmp)))
                    tmp = pw
                else:
                    tmp.extend(pw)
            if tmp: result.append((" ".join(tmp), get_lang(tmp)))
        elif len(buf) + len(words) > max_w:
            flush()
            buf.extend(words)
        else:
            buf.extend(words)
    flush()

    # Merge tiny chunks
    out = []
    for chunk, lang in result:
        if len(chunk.split()) < 3 and out:
            out[-1] = (out[-1][0] + " " + chunk, out[-1][1])
        else:
            out.append((chunk, lang))
    return out

# ── Reference Audio ──────────────────────────────────────────────────
_REF_F0 = None

def measure_f0(path):
    try:
        sr, d = wavfile.read(path)
        if d.ndim == 2: d = d.mean(axis=1)
        d = d.astype(np.float32)
        seg = d[len(d)//4: len(d)//4 + sr//2]
        seg /= (np.max(np.abs(seg)) + 1e-9)
        corr = np.correlate(seg, seg, 'full')[len(seg)-1:]
        lo, hi = int(sr/500), int(sr/60)
        peaks, _ = find_peaks(corr[lo:hi], height=0.2)
        if len(peaks): return sr / (peaks[0] + lo)
    except: pass
    return None

def prep_reference(path, out="ref.wav"):
    global _REF_F0
    try:
        a = AudioSegment.from_file(path)
        a = a.set_channels(1).set_frame_rate(22050)
        try: a = effects.strip_silence(a, silence_thresh=-40, padding=300)
        except: pass
        # Normalize to -18 dBFS (XTTS optimal)
        a = a.apply_gain(-18.0 - a.dBFS)
        while len(a) < 8000: a = a + a
        if len(a) > 30000: a = a[:30000]
        a.export(out, format="wav")
        _REF_F0 = measure_f0(out)
        f0s = f" F0={_REF_F0:.0f}Hz" if _REF_F0 else ""
        print(f"Ref ready: {len(a)/1000:.1f}s{f0s}")
        return out
    except Exception as e:
        print(f"prep_reference: {e}")
        return path

def audio_quality(path):
    if not path or not os.path.exists(path):
        return "Awaaz upload karein (6-30 sec)"
    try:
        a = AudioSegment.from_file(path)
        dur = len(a) / 1000
        rms = np.sqrt(np.mean(np.array(a.get_array_of_samples(), dtype=np.float32)**2))
        ds = "OK" if 6 <= dur <= 30 else ("Chhota — 6s+ chahiye" if dur < 6 else "Lamba — 30s tak best")
        vs = "OK" if rms > 800 else "Bahut soft — louder record karein"
        return f"Duration: {dur:.1f}s  {ds}\nVolume: {rms:.0f}  {vs}\nSample rate: {a.frame_rate}Hz"
    except Exception as e:
        return f"Error: {e}"

def find_reference(up, git_ref):
    # Upload priority
    if up and os.path.exists(up):
        return prep_reference(up)
    # Local voices folder
    lp = os.path.join("voices", git_ref)
    if os.path.exists(lp) and os.path.getsize(lp) > 1000:
        return prep_reference(lp)
    # Download from GitHub / HuggingFace
    for url in [GRAW + git_ref.replace(" ", "%20"),
                GRAW + requests.utils.quote(git_ref, safe=""),
                f"https://huggingface.co/Shriramnag/My-Shriram-Voice/resolve/main/{requests.utils.quote(git_ref)}"]:
        try:
            r = requests.get(url, timeout=20)
            if r.status_code == 200 and len(r.content) > 1000:
                open("ref_dl.wav", "wb").write(r.content)
                open(lp, "wb").write(r.content)
                return prep_reference("ref_dl.wav")
        except: pass
    # Any local fallback
    for fb in ["ref.wav", "Ramai.wav"] + glob.glob("voices/*.wav") + glob.glob("/content/**/*.wav", recursive=True):
        if isinstance(fb, str) and os.path.exists(fb) and os.path.getsize(fb) > 5000:
            return prep_reference(fb)
    return None

# ── Audio Post-Processing ─────────────────────────────────────────────
def apply_eq(seg, bass=0.0, mid=1.5, treble=-1.5, sr=22050):
    """
    4-band EQ tuned from audio analysis:
    bass   = 80-250Hz   (already matched — default 0)
    lmid   = 250-800Hz  (body/warmth — +1.5 makes voice deeper/fuller)
    mid    = 800-2500Hz (presence/emotion — linked to mid param)
    treble = 5kHz+      (harshness/robotics — cut slightly)
    """
    try:
        seg = seg.set_frame_rate(sr).set_channels(1)
        s = np.array(seg.get_array_of_samples(), dtype=np.float64)
        nyq = sr / 2.0

        def bp(lo, hi):
            b1,a1 = butter(2, lo/nyq, btype='high')
            b2,a2 = butter(2, hi/nyq, btype='low')
            return filtfilt(b2, a2, filtfilt(b1, a1, s))

        def hp(lo):
            b, a = butter(2, lo/nyq, btype='high')
            return filtfilt(b, a, s)

        # Bass 80-250Hz
        if abs(bass) > 0.1:
            s += bp(80, 250) * (10**(bass/20) - 1)

        # Low-mid 250-800Hz — body/warmth of voice (most important for "mota" voice)
        lmid_gain = float(mid)  # mid param = low-mid boost
        if abs(lmid_gain) > 0.1:
            s += bp(250, 800) * (10**(lmid_gain/20) - 1)

        # Presence 1kHz-3kHz — emotion/clarity (+0.5 fixed)
        s += bp(1000, 3000) * (10**(0.5/20) - 1)

        # Treble 5kHz+ — cut for natural sound
        if abs(treble) > 0.1:
            s += hp(5000) * (10**(treble/20) - 1)

        s = np.clip(s, -32768, 32767).astype(np.int16)
        return AudioSegment(s.tobytes(), frame_rate=sr, sample_width=2, channels=1)
    except Exception as e:
        print(f"EQ skip: {e}")
        return seg

def apply_deess(seg, sr=22050):
    try:
        seg = seg.set_frame_rate(sr).set_channels(1)
        s = np.array(seg.get_array_of_samples(), dtype=np.float64)
        nyq = sr / 2.0
        b, a = butter(2, 6000/nyq, btype='high')
        hi = filtfilt(b, a, s)
        thr = 10**(-22/20) * 32768
        gain = np.where(np.abs(hi) > thr, thr/(np.abs(hi)+1e-9), 1.0)
        from scipy.ndimage import uniform_filter1d
        gain = uniform_filter1d(np.clip(gain, 0.2, 1.0), size=int(sr*0.005))
        s2 = np.clip(s - hi + hi * gain, -32768, 32767).astype(np.int16)
        return AudioSegment(s2.tobytes(), frame_rate=sr, sample_width=2, channels=1)
    except Exception as e:
        print(f"DeEss skip: {e}")
        return seg

def apply_compress(seg, sr=22050):
    """
    Gentle compressor — ratio 1.5:1, threshold -12dB
    Preserves natural dynamics (original DR=29.7dB)
    Only controls very loud peaks, not normal speech
    """
    try:
        seg = seg.set_frame_rate(sr).set_channels(1)
        s = np.array(seg.get_array_of_samples(), dtype=np.float64)
        # Gentler: threshold -12dB (was -18), ratio 1.5 (was 3.0)
        thr = 10**(-12/20) * 32768
        ratio = 1.5
        gain = np.where(
            np.abs(s) > thr,
            thr/np.abs(s) * (np.abs(s)/thr)**(1/ratio),
            1.0
        )
        from scipy.ndimage import uniform_filter1d
        gain = uniform_filter1d(np.clip(gain, 0.3, 1.0), size=int(sr*0.02))
        s2 = np.clip(s * gain, -32768, 32767).astype(np.int16)
        # Don't normalize after compress — preserve natural level
        return AudioSegment(s2.tobytes(), frame_rate=sr, sample_width=2, channels=1)
    except Exception as e:
        print(f"Compress skip: {e}")
        return seg

def apply_pitch(seg, semitones, sr=22050):
    if not HAS_LIBROSA or abs(semitones) < 0.05: return seg
    try:
        s = np.array(seg.get_array_of_samples(), dtype=np.float32) / 32768.0
        if seg.channels == 2: s = s.reshape(-1, 2).mean(axis=1)
        sh = librosa.effects.pitch_shift(s, sr=sr, n_steps=float(semitones), bins_per_octave=24)
        sh = np.clip(sh * 32768, -32768, 32767).astype(np.int16)
        return AudioSegment(sh.tobytes(), frame_rate=sr, sample_width=2, channels=1)
    except Exception as e:
        print(f"Pitch skip: {e}")
        return seg

def smart_join(segs, cf=60):
    """Volume-level each segment, then crossfade-join."""
    if not segs: return AudioSegment.silent(100)
    rms_vals = [np.sqrt(np.mean(np.array(s.get_array_of_samples(), dtype=np.float32)**2))
                for s in segs]
    rms_vals = [r for r in rms_vals if r > 100]
    target = float(np.median(rms_vals)) if rms_vals else 3000
    leveled = []
    for s in segs:
        rms = np.sqrt(np.mean(np.array(s.get_array_of_samples(), dtype=np.float32)**2))
        if rms > 100:
            g = np.clip(target / (rms + 1e-9), 0.3, 3.0)
            s = s.apply_gain(20 * np.log10(g))
        leveled.append(s)
    out = leveled[0]
    for s in leveled[1:]:
        c = min(cf, len(out)//2, len(s)//2)
        out = out.append(s, crossfade=max(c, 10))
    return out

# ── XTTS Parameter Setter ─────────────────────────────────────────────
STYLES = {
    # temp  = expressiveness (low=stable, high=emotional)
    # rep   = repetition penalty (high=no stutter)
    # speed = talking pace
    "Calm":     {"temp": 0.15, "rep": 8.0, "speed": 0.88},
    "Normal":   {"temp": 0.22, "rep": 7.0, "speed": 0.95},
    "Pro":      {"temp": 0.27, "rep": 6.0, "speed": 1.00},
    "Dramatic": {"temp": 0.33, "rep": 5.5, "speed": 1.05},
}

def set_xtts_params(temp, rep, gpt_len):
    t = min(float(temp), 0.28)
    try:
        cfg = tts.synthesizer.tts_config.model_args
        cfg.temperature        = t
        cfg.repetition_penalty = float(rep)
        cfg.gpt_cond_len       = max(int(gpt_len), 14)  # min 14 for voice match
        cfg.gpt_cond_chunk_len = 4
        cfg.top_p              = 0.85
        cfg.top_k              = 50
        return
    except: pass
    try: tts.tts_config.temperature = t
    except: pass

# ── Single Chunk Generator ────────────────────────────────────────────
def gen_one(text, lang, ref, out, speed, temp, rep, gpt_len, fallback_speed):
    spd = float(speed) if float(speed) >= 0.8 else fallback_speed

    def try_it(t, l, s):
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        gc.collect()
        set_xtts_params(temp, rep, gpt_len)
        tts.tts_to_file(text=t, speaker_wav=ref, language=l, file_path=out, speed=s)
        return os.path.exists(out) and os.path.getsize(out) > 500

    # Attempt 1
    try:
        if try_it(text, lang, spd): return True, None
    except Exception as e:
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        gc.collect()

    # Attempt 2 — shorter text, force Hindi
    try:
        w = text.split()
        t2 = " ".join(w[:18]) if len(w) > 18 else text
        if try_it(t2, "hi", 0.95): return True, "short"
    except Exception as e:
        return False, str(e)[:80]

    return False, "both attempts failed"

# ── Main Generate Function ────────────────────────────────────────────
_chunk_files = []

def generate(
    text, upload, git_ref,
    style_name, speed_override,
    pitch_on, pitch_manual, gpt_len,
    bass, mid, treble,
    do_norm, do_eq, do_deess, do_comp,
    out_format, custom_raw,
    progress=gr.Progress()
):
    global _chunk_files
    _chunk_files = []

    if not text or not text.strip():
        return None, "Text khaali hai.", "", gr.update(choices=[])

    style = STYLES.get(style_name, STYLES["Normal"])
    spd   = float(speed_override) if float(speed_override) >= 0.8 else style["speed"]

    # Custom words
    custom = load_dict()
    if custom_raw:
        for line in custom_raw.strip().splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                if k.strip() and v.strip():
                    custom[k.strip()] = v.strip()
        save_dict(custom)

    # Process text
    progress(0.02, desc="Text processing...")
    cleaned = process_text(text, custom)

    # Reference
    progress(0.05, desc="Reference audio preparing...")
    ref = find_reference(upload, git_ref)
    if not ref:
        return None, (
            "Reference audio nahi mila.\n"
            "Fix: Apni awaaz upload karein (6-30 sec, WAV ya MP3).\n"
            "Ya niche se default voice chunein."
        ), "", gr.update(choices=[])
    ref_info = audio_quality(ref)

    # Chunks
    progress(0.08, desc="Chunking...")
    chunks = make_chunks(cleaned, max_w=30)
    total  = len(chunks)
    if total == 0:
        return None, "Text process ke baad khaali ho gaya.", ref_info, gr.update(choices=[])

    est = total * 0.35
    progress(0.09, desc=f"{total} parts — ~{est:.0f} min GPU time")
    segs, errors = [], []

    for i, (chunk, lang) in enumerate(chunks):
        progress(0.10 + (i / total) * 0.74, desc=f"Part {i+1}/{total}")
        tmp = f"_c{i}.wav"

        if i > 0 and i % 5 == 0:
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            gc.collect()

        ok, err = gen_one(chunk, lang, ref, tmp, spd,
                          style["temp"], style["rep"], gpt_len, style["speed"])

        if ok and os.path.exists(tmp):
            try:
                seg = AudioSegment.from_wav(tmp)
                try: seg = effects.strip_silence(seg, silence_thresh=-45, padding=250)
                except: pass
                if len(seg) > 150:
                    segs.append(seg)
                    pf = f"_p{i+1}.wav"
                    seg.export(pf, format="wav")
                    _chunk_files.append(pf)
                    print(f"  Part {i+1}: {len(seg)}ms")
                else:
                    errors.append(f"Part {i+1}: too short {len(seg)}ms")
            except Exception as e:
                errors.append(f"Part {i+1}: {str(e)[:50]}")
            if os.path.exists(tmp): os.remove(tmp)
        else:
            errors.append(f"Part {i+1}: {err}")
            if os.path.exists(tmp): os.remove(tmp)

        if torch.cuda.is_available(): torch.cuda.empty_cache()
        gc.collect()

    if not segs:
        return None, (
            f"Koi part generate nahi hua ({total} try).\n"
            + "\n".join(errors[:6])
            + "\n\nFix: Apni awaaz upload karein (6-30 sec clear recording)."
        ), ref_info, gr.update(choices=[])

    # Join
    progress(0.85, desc=f"Joining {len(segs)} parts...")
    out = smart_join(segs)
    print(f"Joined: {len(out)/1000:.1f}s")

    # Volume match to -21 dBFS (matches original voice level)
    if do_norm:
        progress(0.87, desc="Volume matching...")
        out = out.set_frame_rate(22050).set_channels(1)
        # Target -22.7dBFS to match original (-20.9 - 1.8dB correction)
        out = out.apply_gain(max(-6.0, -22.7 - out.dBFS))

    # EQ
    if do_eq:
        progress(0.89, desc="EQ...")
        out = apply_eq(out, float(bass), float(mid), float(treble))

    # DeEsser
    if do_deess:
        progress(0.91, desc="DeEsser...")
        out = apply_deess(out)

    # Compressor
    if do_comp:
        progress(0.92, desc="Compressor...")
        out = apply_compress(out)

    # Pitch correction
    if pitch_on and HAS_LIBROSA:
        manual = float(pitch_manual)
        if abs(manual) > 0.1:
            progress(0.94, desc=f"Pitch {manual:+.1f}st...")
            out = apply_pitch(out, manual)
        elif _REF_F0:
            try:
                out.export("_ptmp.wav", format="wav")
                gf0 = measure_f0("_ptmp.wav")
                if gf0 and _REF_F0 and gf0 > 0:
                    auto = float(np.clip(12 * np.log2(_REF_F0 / gf0), -5, 5))
                    if abs(auto) > 0.4:
                        progress(0.94, desc=f"Auto pitch {auto:+.1f}st...")
                        out = apply_pitch(out, auto)
                        print(f"Auto pitch: ref={_REF_F0:.0f}Hz gen={gf0:.0f}Hz → {auto:+.1f}st")
                if os.path.exists("_ptmp.wav"): os.remove("_ptmp.wav")
            except: pass

    # Export
    fmt  = out_format.lower()
    name = f"ShivAI_v41.{fmt}"
    progress(0.97, desc=f"Saving {fmt.upper()}...")
    if fmt == "mp3":
        out.export(name, format="mp3", bitrate="192k")
    elif fmt == "ogg":
        out.export(name, format="ogg")
    else:
        out.export(name, format="wav", parameters=["-ar", "22050"])

    dur = len(out) / 1000
    ok_n = len(segs)
    status = (
        f"{'Done' if ok_n == total else 'Partial'}: {ok_n}/{total} parts\n"
        f"Duration: {dur:.1f}s ({dur/60:.1f} min)\n"
        f"Style: {style_name} | Speed: {spd:.2f}x"
    )
    if errors:
        status += f"\n\nFailed ({len(errors)}):\n" + "\n".join(errors[:4])

    choices = [f"Part {i+1}" for i in range(len(_chunk_files))]
    return name, status, ref_info, gr.update(choices=choices, value=None)


def play_chunk(label):
    if not label or not _chunk_files: return None
    try:
        i = int(label.split()[1]) - 1
        if 0 <= i < len(_chunk_files) and os.path.exists(_chunk_files[i]):
            return _chunk_files[i]
    except: pass
    return None

def dict_add(w, p):
    if not w or not p: return "Dono fields bharo.", _dict_md()
    d = load_dict(); d[w.strip()] = p.strip(); save_dict(d)
    return f"Saved: {w} → {p}", _dict_md()

def dict_del(w):
    d = load_dict()
    if w.strip() in d:
        del d[w.strip()]; save_dict(d)
        return f"Removed: {w}", _dict_md()
    return f"Not found: {w}", _dict_md()

def _dict_md():
    d = load_dict()
    return "\n".join(f"**{k}** → {v}" for k, v in d.items()) if d else "Dictionary khaali hai."

# ── UI ───────────────────────────────────────────────────────────────
CSS = """
* { box-sizing: border-box; }
.gradio-container {
    font-family: 'Segoe UI', Inter, Arial, sans-serif !important;
    background: #0d1117 !important;
    color: #e6edf3 !important;
    max-width: 1100px !important;
    margin: 0 auto !important;
}
.gr-panel, .gr-form, .gr-box, .gr-group, .main {
    background: #161b22 !important;
    border: 1px solid #21262d !important;
    border-radius: 10px !important;
}
label span { color: #8b949e !important; font-size: 0.85em !important; font-weight: 500 !important; }
textarea, input[type=text] {
    background: #0d1117 !important;
    border: 1px solid #30363d !important;
    color: #e6edf3 !important;
    border-radius: 8px !important;
    font-size: 0.95em !important;
}
textarea:focus, input:focus {
    border-color: #f7931a !important;
    outline: none !important;
    box-shadow: 0 0 0 2px rgba(247,147,26,0.15) !important;
}
input[type=range] { accent-color: #f7931a !important; }
.gr-button-primary {
    background: linear-gradient(135deg, #f7931a, #e06d00) !important;
    border: none !important;
    color: #fff !important;
    font-weight: 700 !important;
    font-size: 1em !important;
    border-radius: 10px !important;
    padding: 12px 32px !important;
    box-shadow: 0 4px 20px rgba(247,147,26,0.35) !important;
    transition: all 0.15s !important;
}
.gr-button-primary:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 28px rgba(247,147,26,0.5) !important;
}
.gr-button-secondary {
    background: #21262d !important;
    border: 1px solid #30363d !important;
    color: #c9d1d9 !important;
    border-radius: 8px !important;
}
select { background: #0d1117 !important; border: 1px solid #30363d !important; color: #e6edf3 !important; border-radius: 8px !important; }
.mono textarea { background: #0d1117 !important; color: #7ee787 !important; font-family: Consolas, monospace !important; font-size: 0.82em !important; }
input[type=checkbox] { accent-color: #f7931a !important; }
.gr-tab-item { background: #161b22 !important; color: #8b949e !important; font-size: 0.9em !important; }
.gr-tab-item.selected { color: #f7931a !important; border-bottom: 2px solid #f7931a !important; background: #1c2128 !important; }
.section-title { color: #8b949e; font-size: 0.78em; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; margin: 12px 0 6px; }
"""

HEADER = """
<div style="text-align:center;padding:18px 0 14px;border-bottom:1px solid #21262d;margin-bottom:18px">
  <div style="font-size:1.75em;font-weight:700;letter-spacing:0.5px;
    background:linear-gradient(90deg,#f7931a,#ffb347);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text">
    Shiv AI — v4.1
  </div>
  <div style="color:#8b949e;font-size:0.82em;margin-top:4px">
    Shri Ram Nag &nbsp;·&nbsp; PAISAWALA &nbsp;·&nbsp; Hindi Voice Cloning
  </div>
  <div style="display:flex;justify-content:center;gap:6px;margin-top:10px;flex-wrap:wrap">
    <span style="background:#1c2128;border:1px solid #2d5a2d;color:#7ee787;padding:2px 10px;border-radius:12px;font-size:0.72em">Voice Match</span>
    <span style="background:#1c2128;border:1px solid #2d5a2d;color:#7ee787;padding:2px 10px;border-radius:12px;font-size:0.72em">Auto Pitch</span>
    <span style="background:#1c2128;border:1px solid #2d5a2d;color:#7ee787;padding:2px 10px;border-radius:12px;font-size:0.72em">Hindi + English + Sanskrit</span>
    <span style="background:#1c2128;border:1px solid #2d5a2d;color:#7ee787;padding:2px 10px;border-radius:12px;font-size:0.72em">Long Audio</span>
    <span style="background:#1c2128;border:1px solid #2d5a2d;color:#7ee787;padding:2px 10px;border-radius:12px;font-size:0.72em">EQ · DeEss · Compress</span>
  </div>
</div>
"""

with gr.Blocks(css=CSS, title="Shiv AI v4.1") as demo:
    gr.HTML(HEADER)

    with gr.Tabs():

        # ══════════════════════════════════════════
        # TAB 1 — GENERATE
        # ══════════════════════════════════════════
        with gr.Tab("Generate"):
            with gr.Row(equal_height=False):

                # ── LEFT: Script ──────────────────────────
                with gr.Column(scale=5):
                    txt = gr.Textbox(
                        label="Script",
                        placeholder="Yahan Hindi / English / Sanskrit script paste karein...",
                        lines=18
                    )
                    with gr.Row():
                        wc = gr.Markdown("Words: **0**")
                        cc = gr.Markdown("Chars: **0**")
                    txt.change(
                        lambda x: (f"Words: **{len(x.split())}**", f"Chars: **{len(x)}**"),
                        [txt], [wc, cc]
                    )
                    with gr.Row():
                        btn_preview = gr.Button("Text Preview", size="sm", variant="secondary")
                        btn_gen     = gr.Button("Generate Karo", variant="primary", size="lg")

                    preview_out = gr.Textbox(
                        label="Cleaned Text (yahi bolega XTTS ko)",
                        lines=5, interactive=False, visible=False,
                        elem_classes=["mono"]
                    )
                    btn_preview.click(
                        lambda t: (process_text(t, load_dict()), gr.update(visible=True)),
                        [txt], [preview_out, preview_out]
                    )

                # ── RIGHT: Controls ───────────────────────
                with gr.Column(scale=3):

                    # Voice Upload
                    gr.HTML('<div class="section-title">Voice Upload</div>')
                    upload = gr.Audio(
                        label="Apni awaaz (6-30 sec WAV / MP3)",
                        type="filepath"
                    )
                    quality_box = gr.Textbox(
                        label="Quality check",
                        interactive=False, lines=3,
                        elem_classes=["mono"]
                    )
                    upload.change(
                        lambda f: audio_quality(f) if f else "Awaaz upload karein",
                        [upload], [quality_box]
                    )
                    git_v = gr.Dropdown(
                        choices=["aideva.wav", "Joanne.wav", "Reginald voice.wav", "cloning .wav"],
                        label="Ya default voice",
                        value="aideva.wav"
                    )

                    # Style
                    gr.HTML('<div class="section-title">Style</div>')
                    style = gr.Radio(
                        choices=list(STYLES.keys()),
                        value="Normal", label=""
                    )
                    spd = gr.Slider(0.0, 1.4, 0.0, step=0.05,
                                    label="Speed  (0 = auto)")
                    style.change(
                        lambda s: STYLES.get(s, STYLES["Normal"])["speed"],
                        [style], [spd]
                    )

                    # Voice Match
                    gr.HTML('<div class="section-title">Voice Match</div>')
                    gpt = gr.Slider(3, 30, 12, step=1,
                                    label="Match quality  (12 = fast · 24 = best)")
                    pitch_on = gr.Checkbox(label="Pitch correction (librosa)", value=True)
                    pitch_sl = gr.Slider(-6.0, 6.0, 0.0, step=0.5,
                                         label="Manual pitch  (0 = auto-detect)")

                    # EQ
                    gr.HTML('<div class="section-title">EQ</div>')
                    bass_sl   = gr.Slider(-6.0, 12.0,  0.0, step=0.5, label="Bass dB  (0 = matched to original)")
                    mid_sl    = gr.Slider(-6.0,  6.0,  1.5, step=0.5, label="Mid dB  (+1.5 = deeper/fuller voice)")
                    treble_sl = gr.Slider(-9.0,  3.0, -1.5, step=0.5, label="Treble dB")

                    # Options
                    gr.HTML('<div class="section-title">Options</div>')
                    with gr.Row():
                        do_norm  = gr.Checkbox(label="Normalize", value=True)
                        do_eq    = gr.Checkbox(label="EQ",        value=True)
                        do_deess = gr.Checkbox(label="DeEss",     value=True)
                        do_comp  = gr.Checkbox(label="Compress",  value=True)

                    out_fmt = gr.Radio(["wav", "mp3", "ogg"], value="wav", label="Format")
                    cwords  = gr.Textbox(
                        label="Custom words  (WORD = उच्चारण, ek per line)",
                        placeholder="PAISAWALA = पेसावाला",
                        lines=2
                    )

            # Output row
            with gr.Row():
                with gr.Column(scale=3):
                    out_audio = gr.Audio(label="Output", type="filepath", autoplay=True)
                with gr.Column(scale=2):
                    out_status = gr.Textbox(
                        label="Status", interactive=False,
                        lines=8, elem_classes=["mono"]
                    )

            with gr.Accordion("Chunk preview", open=False):
                with gr.Row():
                    ch_dd  = gr.Dropdown(label="Part", choices=[], interactive=True)
                    ch_btn = gr.Button("Play", size="sm")
                ch_out = gr.Audio(label="", type="filepath", autoplay=True)
                ch_btn.click(play_chunk, [ch_dd], [ch_out])

            btn_gen.click(
                generate,
                inputs=[txt, upload, git_v, style, spd,
                        pitch_on, pitch_sl, gpt,
                        bass_sl, mid_sl, treble_sl,
                        do_norm, do_eq, do_deess, do_comp,
                        out_fmt, cwords],
                outputs=[out_audio, out_status, quality_box, ch_dd]
            )

        # ══════════════════════════════════════════
        # TAB 2 — DICTIONARY
        # ══════════════════════════════════════════
        with gr.Tab("Dictionary"):
            gr.Markdown("Custom words — permanently save karo")
            with gr.Row():
                with gr.Column():
                    dw = gr.Textbox(label="Word",          placeholder="PAISAWALA")
                    dp = gr.Textbox(label="Pronunciation", placeholder="पेसावाला")
                    with gr.Row():
                        da_btn = gr.Button("Add",    variant="primary")
                        dd_btn = gr.Button("Remove", variant="secondary")
                    ds = gr.Textbox(label="Status", interactive=False, lines=2)
                with gr.Column():
                    dm = gr.Markdown(_dict_md())
            da_btn.click(dict_add, [dw, dp], [ds, dm])
            dd_btn.click(dict_del, [dw],     [ds, dm])

        # ══════════════════════════════════════════
        # TAB 3 — GUIDE
        # ══════════════════════════════════════════
        with gr.Tab("Guide"):
            gr.Markdown("""
### Best Voice Match ke liye

1. **6–30 second** ki saaf awaaz upload karein
2. Quiet room mein record karein — background noise se quality kharab hoti hai
3. **Pitch Correction ON** rakhein — auto F0 match karega
4. **Match Quality 12–24** — zyada = better match, slower generation

---

### Long Audio (30–40 min)

- Poori script ek baar mein paste karein
- Colab mein **GPU T4 runtime** zaroori hai
- ~1000 words = 8–10 min audio

---

### EQ Guide

| Setting | Effect |
|---------|--------|
| Bass +1 | Natural warmth |
| Treble -1.5 | Less robotic, smoother |
| DeEss ON | Harsh "s" sounds fix |
| Compress ON | Consistent loudness |

---

### Style Guide

| Style | Best for |
|-------|----------|
| Calm | Meditation, slow narration |
| Normal | YouTube, general content |
| Pro | News, professional |
| Dramatic | Stories, energetic |

---

### Formats

- **WAV** — best quality, editing ke liye
- **MP3** — YouTube upload, smaller file size
            """)

demo.launch(share=True, show_error=True)
