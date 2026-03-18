"""
╔══════════════════════════════════════════════════════════════════╗
║         शिव AI (Shiv AI) v3.5 — श्री राम नाग                   ║
║         PAISAWALA YouTube Channel                                ║
║                                                                  ║
║  v3.5 NEW:                                                       ║
║  ✅ Pitch Correction (librosa PSOLA)                             ║
║  ✅ Voice Match: gpt_cond_len tuning                             ║
║  ✅ Multi-Segment Reference (3 clips se better embedding)        ║
║  ✅ DeEsser (sibilance/harshness hatao)                          ║
║  ✅ Compressor (dynamic range fix)                               ║
║  ✅ Output Format: MP3 / WAV / OGG choice                        ║
║  ✅ Batch Script Mode (multiple files)                           ║
║  ✅ Preview: Text cleanup ka result dikhao                       ║
║  ✅ Speed per-chunk control                                      ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os, torch, gradio as gr, requests, re, gc, json, wave, struct
import numpy as np
from TTS.api import TTS
from huggingface_hub import hf_hub_download
from pydub import AudioSegment, effects
from scipy.signal import butter, filtfilt, welch
from scipy.io import wavfile
import soundfile as sf

# ══════════════════════════════════════════════════════════════════
# १. Setup
# ══════════════════════════════════════════════════════════════════
os.environ["COQUI_TOS_AGREED"] = "1"
torch.backends.cudnn.benchmark = True
device = "cuda" if torch.cuda.is_available() else "cpu"

REPO_ID   = "Shriramnag/My-Shriram-Voice"
G_RAW     = "https://raw.githubusercontent.com/shriramnag/Aivoicebox/main/%F0%9F%93%81%20voices/"
DICT_FILE = "custom_dict.json"

print("🚩 शिव AI v3.5 — Advanced Voice Match Engine...")
try:
    hf_hub_download(repo_id=REPO_ID, filename="Ramai.pth")
    hf_hub_download(repo_id=REPO_ID, filename="config.json")
except:
    pass

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
print(f"✅ Ready on {device.upper()}")

# librosa optional (pitch shift ke liye)
try:
    import librosa
    import librosa.effects
    LIBROSA_OK = True
    print("✅ librosa available — Pitch correction active")
except ImportError:
    LIBROSA_OK = False
    print("⚠️  librosa not found — install karo: pip install librosa")

# ══════════════════════════════════════════════════════════════════
# २. Custom Dictionary
# ══════════════════════════════════════════════════════════════════
def load_custom_dict():
    if os.path.exists(DICT_FILE):
        try:
            with open(DICT_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except: pass
    return {}

def save_custom_dict(d):
    with open(DICT_FILE, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)

CUSTOM_DICT = load_custom_dict()

# ══════════════════════════════════════════════════════════════════
# ३. Trilingual Dictionaries
# ══════════════════════════════════════════════════════════════════
SANSKRIT_DICT = {
    "dharma":"धर्म","karma":"कर्म","yoga":"योग","shakti":"शक्ति",
    "om":"ॐ","aum":"ॐ","namaste":"नमस्ते","namaskaar":"नमस्कार",
    "guru":"गुरु","mantra":"मंत्र","tantra":"तंत्र","yantra":"यंत्र",
    "atma":"आत्मा","brahma":"ब्रह्म","vishnu":"विष्णु","shiva":"शिव",
    "maya":"माया","moksha":"मोक्ष","nirvana":"निर्वाण","prana":"प्राण",
    "ahimsa":"अहिंसा","satya":"सत्य","seva":"सेवा","bhakti":"भक्ति",
    "jnana":"ज्ञान","gyan":"ज्ञान","veda":"वेद","sutra":"सूत्र",
    "shloka":"श्लोक","stotra":"स्तोत्र","pooja":"पूजा","puja":"पूजा",
    "aarti":"आरती","prasad":"प्रसाद","tilak":"तिलक","rudra":"रुद्र",
    "sadhana":"साधना","tapas":"तपस","satsang":"सत्संग",
}

ENGLISH_TO_HINDI = {
    "AI":"ए आई","ML":"एम एल","API":"ए पी आई","GPU":"जी पी यू",
    "CPU":"सी पी यू","URL":"यू आर एल",
    "YouTube":"यूट्यूब","Instagram":"इंस्टाग्राम","Facebook":"फेसबुक",
    "WhatsApp":"व्हाट्सऐप","Google":"गूगल","Twitter":"ट्विटर",
    "Internet":"इंटरनेट","Online":"ऑनलाइन","Offline":"ऑफलाइन",
    "Software":"सॉफ्टवेयर","Hardware":"हार्डवेयर","Computer":"कंप्यूटर",
    "Mobile":"मोबाइल","App":"ऐप","Website":"वेबसाइट",
    "Download":"डाउनलोड","Upload":"अपलोड","Password":"पासवर्ड",
    "Server":"सर्वर","Cloud":"क्लाउड","Data":"डेटा",
    "Channel":"चैनल","Video":"वीडियो","Content":"कंटेंट",
    "Subscribe":"सब्सक्राइब","Like":"लाइक","Share":"शेयर",
    "Comment":"कमेंट","Notification":"नोटिफिकेशन",
    "Life":"लाइफ","Dream":"ड्रीम","Mindset":"माइंडसेट",
    "Believe":"बिलीव","Success":"सक्सेस","Fail":"फेल",
    "Failure":"फेल्योर","Goal":"गोल","Focus":"फोकस",
    "Step":"स्टेप","Fear":"फियर","Simple":"सिंपल",
    "Practical":"प्रैक्टिकल","Strong":"स्ट्रॉन्ग","Turbo":"टर्बो",
    "Power":"पावर","Energy":"एनर्जी","Positive":"पॉजिटिव",
    "Negative":"नेगेटिव","Challenge":"चैलेंज","Time":"टाइम",
    "Work":"वर्क","Hard":"हार्ड","Smart":"स्मार्ट",
    "Money":"मनी","Business":"बिज़नेस","Market":"मार्केट",
    "Brand":"ब्रांड","Profit":"प्रॉफिट","Loss":"लॉस",
    "Investment":"इन्वेस्टमेंट","Invest":"इन्वेस्ट",
    "Strategy":"स्ट्रेटेजी","Plan":"प्लान","Team":"टीम",
    "Leader":"लीडर","Leadership":"लीडरशिप","Skill":"स्किल",
    "Training":"ट्रेनिंग","Course":"कोर्स",
    "OK":"ओके","okay":"ओके","hey":"हे","hi":"हाय",
    "hello":"हेलो","bye":"बाय","thanks":"थैंक्स",
    "please":"प्लीज़","sorry":"सॉरी","welcome":"वेलकम",
}

NUMBER_MAP = {
    "1000":"एक हज़ार","500":"पाँच सौ","200":"दो सौ",
    "100":"सौ","90":"नब्बे","80":"अस्सी","70":"सत्तर",
    "60":"साठ","50":"पचास","40":"चालीस","30":"तीस",
    "25":"पच्चीस","20":"बीस","19":"उन्नीस","18":"अठारह",
    "17":"सत्रह","16":"सोलह","15":"पंद्रह","14":"चौदह",
    "13":"तेरह","12":"बारह","11":"ग्यारह","10":"दस",
    "9":"नौ","8":"आठ","7":"सात","6":"छह",
    "5":"पाँच","4":"चार","3":"तीन","2":"दो","1":"एक","0":"शून्य",
}

# ══════════════════════════════════════════════════════════════════
# ४. Text Processor
# ══════════════════════════════════════════════════════════════════
def apply_all_dicts(text, custom_dict):
    for src, tgt in custom_dict.items():
        text = re.sub(rf'(?<![a-zA-Z\u0900-\u097F]){re.escape(src)}(?![a-zA-Z\u0900-\u097F])',
                      tgt, text, flags=re.IGNORECASE)
    for src, tgt in SANSKRIT_DICT.items():
        text = re.sub(rf'(?<![a-zA-Z]){re.escape(src)}(?![a-zA-Z])',
                      tgt, text, flags=re.IGNORECASE)
    for src, tgt in ENGLISH_TO_HINDI.items():
        text = re.sub(rf'(?<![a-zA-Z]){re.escape(src)}(?![a-zA-Z])',
                      tgt, text, flags=re.IGNORECASE)
    for num in sorted(NUMBER_MAP.keys(), key=lambda x: -len(x)):
        text = re.sub(rf'\b{num}\b', NUMBER_MAP[num], text)
    return text

def clean_punctuation(text):
    text = re.sub(r'[।\.](\s|$)', '। ', text)
    text = re.sub(r'[,،]\s*', ', ', text)
    text = re.sub(r'[!]\s*', '! ', text)
    text = re.sub(r'[?]\s*', '? ', text)
    text = re.sub(r'[-–—]+', ', ', text)
    text = re.sub(r'["""\'\'()\[\]{}]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def full_text_processor(text, custom_dict):
    if not text: return ""
    text = apply_all_dicts(text, custom_dict)
    text = clean_punctuation(text)
    return text

def preview_cleaned_text(text, custom_words_raw):
    """Text cleanup preview — user ko dikhao kya process hoga"""
    custom_dict = load_custom_dict()
    if custom_words_raw:
        for line in custom_words_raw.strip().splitlines():
            if "=" in line:
                s,t = line.split("=",1)
                if s.strip() and t.strip():
                    custom_dict[s.strip()] = t.strip()
    cleaned = full_text_processor(text, custom_dict)
    diff_count = sum(1 for a,b in zip(text.split(),cleaned.split()) if a!=b)
    return f"**Cleaned Text Preview:**\n\n{cleaned}\n\n---\n*{diff_count} words converted*"

# ══════════════════════════════════════════════════════════════════
# ५. Language-Aware Chunker
# ══════════════════════════════════════════════════════════════════
def language_aware_chunker(text, max_words=45):
    sentences = re.split(r'(?<=[।\.\!\?])\s+', text.strip())
    chunks_with_lang = []
    current_words, current_count = [], 0

    def commit(words):
        if not words: return
        chunk = ' '.join(words)
        latin = sum(1 for w in words
                    if sum(1 for c in w if c.isascii() and c.isalpha()) > len(w)//2)
        lang = "en" if latin > len(words)*0.5 else "hi"
        chunks_with_lang.append((chunk, lang))

    for sentence in sentences:
        words = sentence.split()
        wc = len(words)
        if wc > max_words:
            if current_words: commit(current_words); current_words, current_count = [], 0
            parts = re.split(r',\s*', sentence)
            tmp, tc = [], 0
            for part in parts:
                pw = part.split()
                if tc+len(pw) > max_words and tmp:
                    commit(tmp); tmp, tc = pw, len(pw)
                else:
                    tmp.extend(pw); tc += len(pw)
            if tmp: commit(tmp)
        elif current_count+wc > max_words:
            commit(current_words); current_words, current_count = words, wc
        else:
            current_words.extend(words); current_count += wc

    commit(current_words)
    return [(c.strip(), l) for c, l in chunks_with_lang if c.strip()]

# ══════════════════════════════════════════════════════════════════
# ६. Reference Audio — Prepare + Quality Check
# ══════════════════════════════════════════════════════════════════
def check_ref_quality(filepath):
    if not filepath or not os.path.exists(filepath):
        return "⚠️ Koi reference audio nahi."
    try:
        audio = AudioSegment.from_file(filepath)
        dur = len(audio)/1000
        sr  = audio.frame_rate
        samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
        rms = np.sqrt(np.mean(samples**2))
        msgs = []
        msgs.append(f"⏱ Duration: {dur:.1f}s {'✅' if 6<=dur<=25 else ('⚠️ Kam hai (6s+ chahiye)' if dur<6 else 'ℹ️ Thoda lamba')}")
        msgs.append(f"🎵 Sample Rate: {sr}Hz {'✅' if sr>=22050 else '⚠️ Kam hai'}")
        msgs.append(f"🔊 Volume RMS: {rms:.0f} {'✅' if rms>800 else '⚠️ Bahut soft — louder record karo'}")
        msgs.append(f"📢 Channels: {audio.channels} {'(mono ✅)' if audio.channels==1 else '(stereo — auto-convert hoga)'}")
        return "\n".join(msgs)
    except Exception as e:
        return f"⚠️ Check failed: {e}"

def prepare_reference(filepath, out="ref_ready.wav"):
    """Reference audio ko XTTS ke liye optimal banao"""
    try:
        a = AudioSegment.from_file(filepath)
        a = a.set_channels(1).set_frame_rate(22050)
        # Silence trim (gentle)
        a = effects.strip_silence(a, silence_thresh=-42, padding=200)
        a = effects.normalize(a)
        # 6-25 sec window
        if len(a) < 6000:
            # Repeat if too short
            while len(a) < 6000: a = a + a
        if len(a) > 25000:
            a = a[:25000]
        a.export(out, format="wav")
        return out
    except Exception as e:
        print(f"Ref prep error: {e}")
        return filepath

def merge_multi_refs(ref_files, out="ref_merged.wav"):
    """Multiple reference files ko merge karo — better voice embedding"""
    segments = []
    for f in ref_files:
        if f and os.path.exists(f):
            try:
                a = AudioSegment.from_file(f)
                a = a.set_channels(1).set_frame_rate(22050)
                a = effects.strip_silence(a, silence_thresh=-42, padding=100)
                a = effects.normalize(a)
                segments.append(a)
            except: pass
    if not segments:
        return None
    # 500ms silence between segments
    silence = AudioSegment.silent(duration=500, frame_rate=22050)
    merged = segments[0]
    for s in segments[1:]:
        merged = merged + silence + s
    # Max 25 sec
    if len(merged) > 25000:
        merged = merged[:25000]
    merged.export(out, format="wav")
    return out

# ══════════════════════════════════════════════════════════════════
# ७. Pitch Correction (NEW — voice match ka sabse bada upgrade)
# ══════════════════════════════════════════════════════════════════
def pitch_shift_audio(audio_seg, semitones, sr=22050):
    """
    Semitones mein pitch shift karo bina speed badhe.
    -2 to +2 semitones recommended for voice match.
    """
    if abs(semitones) < 0.1:
        return audio_seg
    if not LIBROSA_OK:
        print("librosa nahi hai — pitch shift skip")
        return audio_seg
    try:
        samples = np.array(audio_seg.get_array_of_samples(), dtype=np.float32) / 32768.0
        if audio_seg.channels == 2:
            samples = samples.reshape(-1,2).mean(axis=1)
        shifted = librosa.effects.pitch_shift(
            samples, sr=sr, n_steps=float(semitones),
            bins_per_octave=24  # finer control
        )
        shifted = np.clip(shifted * 32768, -32768, 32767).astype(np.int16)
        return AudioSegment(
            shifted.tobytes(), frame_rate=sr,
            sample_width=2, channels=1
        )
    except Exception as e:
        print(f"Pitch shift error: {e}")
        return audio_seg

# ══════════════════════════════════════════════════════════════════
# ८. DeEsser (sibilance/harsh 's' sound fix)
# ══════════════════════════════════════════════════════════════════
def deess(audio_seg, threshold_db=-20, freq=6000, sr=22050):
    """
    6kHz+ pe harsh sibilance reduce karo.
    Voice match better hogi — TTS ka artificial sharpness hatega.
    """
    try:
        samples = np.array(audio_seg.get_array_of_samples(), dtype=np.float64)
        if audio_seg.channels == 2:
            samples = samples.reshape(-1,2).mean(axis=1)
        nyq = sr / 2.0
        b, a = butter(2, freq/nyq, btype='high')
        high_band = filtfilt(b, a, samples)
        # Compress high band when loud
        threshold = 10**(threshold_db/20.0) * 32768
        gain = np.where(np.abs(high_band) > threshold,
                        threshold / (np.abs(high_band) + 1e-9), 1.0)
        gain = np.clip(gain, 0.2, 1.0)
        # Smooth gain
        from scipy.ndimage import uniform_filter1d
        gain = uniform_filter1d(gain, size=int(sr*0.005))
        processed = samples - high_band + high_band * gain
        processed = np.clip(processed, -32768, 32767).astype(np.int16)
        return AudioSegment(
            processed.tobytes(), frame_rate=sr,
            sample_width=2, channels=1
        )
    except Exception as e:
        print(f"DeEss error: {e}")
        return audio_seg

# ══════════════════════════════════════════════════════════════════
# ९. Compressor (dynamic range control)
# ══════════════════════════════════════════════════════════════════
def compress_audio(audio_seg, threshold_db=-18, ratio=3.0, sr=22050):
    """
    Loud parts thode kam karo, soft parts thode badhao.
    Result: consistent volume, natural feel.
    """
    try:
        samples = np.array(audio_seg.get_array_of_samples(), dtype=np.float64)
        if audio_seg.channels == 2:
            samples = samples.reshape(-1,2).mean(axis=1)
        threshold = 10**(threshold_db/20.0) * 32768
        abs_samples = np.abs(samples)
        gain = np.where(
            abs_samples > threshold,
            threshold / abs_samples * (abs_samples/threshold)**(1.0/ratio),
            1.0
        )
        gain = np.clip(gain, 0.1, 1.0)
        # Smooth (attack/release ~10ms)
        from scipy.ndimage import uniform_filter1d
        gain = uniform_filter1d(gain, size=int(sr*0.01))
        compressed = samples * gain
        compressed = np.clip(compressed, -32768, 32767).astype(np.int16)
        result = AudioSegment(
            compressed.tobytes(), frame_rate=sr,
            sample_width=2, channels=1
        )
        return effects.normalize(result)
    except Exception as e:
        print(f"Compress error: {e}")
        return audio_seg

# ══════════════════════════════════════════════════════════════════
# १०. EQ (Bass/Mid/Treble)
# ══════════════════════════════════════════════════════════════════
def apply_eq(audio_seg, bass_db=5.0, mid_db=0.0, treble_db=-2.0, sr=22050):
    try:
        audio_seg = audio_seg.set_frame_rate(sr).set_channels(1)
        samples = np.array(audio_seg.get_array_of_samples(), dtype=np.float64)
        nyq = sr / 2.0

        if abs(bass_db) > 0.1:
            b1,a1 = butter(2, 80/nyq,  btype='high')
            b2,a2 = butter(2, 300/nyq, btype='low')
            band = filtfilt(b2,a2, filtfilt(b1,a1, samples))
            samples += band * (10**(bass_db/20.0) - 1.0)

        if abs(mid_db) > 0.1:
            b1,a1 = butter(2, 500/nyq,  btype='high')
            b2,a2 = butter(2, 2000/nyq, btype='low')
            band = filtfilt(b2,a2, filtfilt(b1,a1, samples))
            samples += band * (10**(mid_db/20.0) - 1.0)

        if abs(treble_db) > 0.1:
            b,a = butter(2, 5000/nyq, btype='high')
            band = filtfilt(b, a, samples)
            samples += band * (10**(treble_db/20.0) - 1.0)

        samples = np.clip(samples, -32768, 32767).astype(np.int16)
        return effects.normalize(AudioSegment(
            samples.tobytes(), frame_rate=sr, sample_width=2, channels=1
        ))
    except Exception as e:
        print(f"EQ error: {e}")
        return effects.normalize(audio_seg)

# ══════════════════════════════════════════════════════════════════
# ११. Crossfade Join
# ══════════════════════════════════════════════════════════════════
def crossfade_join(segs, cf_ms=60):
    if not segs: return AudioSegment.silent(100)
    result = segs[0]
    for s in segs[1:]:
        cf = min(cf_ms, len(result)//2, len(s)//2)
        result = result.append(s, crossfade=max(cf,10))
    return result

# ══════════════════════════════════════════════════════════════════
# १२. Emotion Presets
# ══════════════════════════════════════════════════════════════════
EMOTION_PRESETS = {
    "🧘 शांत (Calm)":       {"temperature":0.20,"rep_pen":7.0,"speed":0.92},
    "😊 सामान्य (Normal)":  {"temperature":0.35,"rep_pen":6.0,"speed":1.00},
    "🎙️ प्रो (Pro)":        {"temperature":0.50,"rep_pen":5.0,"speed":1.05},
    "🔥 नाटकीय (Dramatic)": {"temperature":0.68,"rep_pen":4.0,"speed":1.10},
}

# ══════════════════════════════════════════════════════════════════
# १३. Main Generation Engine v3.5
# ══════════════════════════════════════════════════════════════════
_chunk_audios = []

def generate_v35(
    text,
    up_ref1, up_ref2, up_ref3,   # Multi-ref
    git_ref,
    emotion_mode, speed_override,
    pitch_semitones,              # NEW: pitch correction
    gpt_cond_len,                 # NEW: voice match quality
    bass_db, mid_db, treble_db,
    use_silence, use_normalize,
    use_eq, use_deess, use_compress,  # NEW: deess + compress
    pitch_shift_enable,           # NEW toggle
    output_format,                # NEW: wav/mp3/ogg
    custom_words_raw,
    progress=gr.Progress()
):
    global _chunk_audios
    _chunk_audios = []

    if not text or not text.strip():
        return None, "❌ Text khaali hai.", "", gr.update(choices=[])

    preset  = EMOTION_PRESETS.get(emotion_mode, EMOTION_PRESETS["😊 सामान्य (Normal)"])
    temperature = preset["temperature"]
    rep_pen     = preset["rep_pen"]
    speed_s     = float(speed_override) if float(speed_override) != 0 else preset["speed"]

    # Custom dict
    custom_dict = load_custom_dict()
    if custom_words_raw:
        for line in custom_words_raw.strip().splitlines():
            if "=" in line:
                s,t = line.split("=",1)
                if s.strip() and t.strip():
                    custom_dict[s.strip()] = t.strip()
        save_custom_dict(custom_dict)

    progress(0.02, desc="📝 Text processing...")
    p_text = full_text_processor(text, custom_dict)

    # Reference audio — multi-ref support
    progress(0.05, desc="🎤 Reference prepare...")
    refs_uploaded = [r for r in [up_ref1, up_ref2, up_ref3] if r and os.path.exists(r)]

    if refs_uploaded:
        if len(refs_uploaded) > 1:
            ref = merge_multi_refs(refs_uploaded)
            if not ref: ref = prepare_reference(refs_uploaded[0])
        else:
            ref = prepare_reference(refs_uploaded[0])
    else:
        raw = "ref_dl.wav"
        url = G_RAW + requests.utils.quote(git_ref)
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            with open(raw,"wb") as f: f.write(resp.content)
            ref = prepare_reference(raw)
        except Exception as e:
            return None, f"❌ Download failed: {e}", "", gr.update(choices=[])

    ref_quality = check_ref_quality(ref)

    # Chunking
    progress(0.08, desc="✂️ Smart chunking...")
    chunks = language_aware_chunker(p_text, max_words=45)
    total  = len(chunks)
    if total == 0:
        return None, "❌ Text empty after processing.", ref_quality, gr.update(choices=[])

    segments, errors = [], []

    for i, (chunk, lang) in enumerate(chunks):
        progress((i+1)/total*0.80, desc=f"🎙️ Part {i+1}/{total} [{lang.upper()}]")
        name = f"chunk_{i}.wav"
        try:
            # gpt_cond_len: voice match quality
            # Higher = better speaker similarity, slower
            try:
                tts.synthesizer.tts_config.model_args.temperature        = float(temperature)
                tts.synthesizer.tts_config.model_args.repetition_penalty = float(rep_pen)
                tts.synthesizer.tts_config.model_args.gpt_cond_len       = int(gpt_cond_len)
            except: pass

            tts.tts_to_file(
                text=chunk,
                speaker_wav=ref,
                language=lang,
                file_path=name,
                speed=float(speed_s),
            )

            seg = AudioSegment.from_wav(name)

            if use_silence:
                try:
                    seg = effects.strip_silence(seg, silence_thresh=-50, padding=180)
                except: pass

            if len(seg) > 80:
                segments.append(seg)
                cout = f"prev_{i+1}.wav"
                seg.export(cout, format="wav")
                _chunk_audios.append(cout)

            os.remove(name)

        except Exception as e:
            errors.append(f"Part {i+1}[{lang}]: {str(e)[:90]}")
            if os.path.exists(name): os.remove(name)

        torch.cuda.empty_cache(); gc.collect()

    if not segments:
        return None, f"❌ Generate nahi hua.\n{chr(10).join(errors[:5])}", ref_quality, gr.update(choices=[])

    # Join
    progress(0.83, desc="🔗 Crossfade join...")
    combined = crossfade_join(segments, cf_ms=60)

    # Normalize
    if use_normalize:
        progress(0.86, desc="🧹 Normalize...")
        combined = combined.set_frame_rate(22050).set_channels(1)
        combined = effects.normalize(combined)

    # EQ
    if use_eq:
        progress(0.89, desc="🎛️ EQ...")
        combined = apply_eq(combined, float(bass_db), float(mid_db), float(treble_db))

    # DeEsser (NEW)
    if use_deess:
        progress(0.91, desc="🔇 De-essing (harshness hatao)...")
        combined = deess(combined, threshold_db=-22, freq=6000)

    # Compressor (NEW)
    if use_compress:
        progress(0.93, desc="🗜️ Compressing...")
        combined = compress_audio(combined, threshold_db=-18, ratio=3.0)

    # Pitch shift (NEW)
    if pitch_shift_enable and abs(float(pitch_semitones)) > 0.05:
        progress(0.95, desc=f"🎵 Pitch shift {pitch_semitones:+.1f} semitones...")
        combined = pitch_shift_audio(combined, float(pitch_semitones))

    # Export
    progress(0.97, desc=f"💾 Export as {output_format.upper()}...")
    fmt = output_format.lower()
    final = f"ShivAI_v35_Output.{fmt}"

    if fmt == "wav":
        combined.export(final, format="wav", parameters=["-ar","22050"])
    elif fmt == "mp3":
        combined.export(final, format="mp3", bitrate="192k")
    elif fmt == "ogg":
        combined.export(final, format="ogg")
    else:
        combined.export(final, format="wav")
        final = final.replace(fmt,"wav")

    progress(1.0, desc="✅ Done!")

    status  = f"✅ {len(segments)}/{total} parts generated\n"
    status += f"⏱ Duration: {len(combined)/1000:.1f}s\n"
    status += f"🎵 Format: {fmt.upper()} | Speed: {speed_s:.2f}x\n"
    status += f"🎭 Emotion: {emotion_mode}\n"
    if pitch_shift_enable:
        status += f"🎵 Pitch: {float(pitch_semitones):+.1f} semitones\n"
    if errors:
        status += f"\n⚠️ {len(errors)} errors:\n" + "\n".join(errors[:3])

    chunk_choices = [f"Part {i+1}" for i in range(len(_chunk_audios))]
    return final, status, ref_quality, gr.update(choices=chunk_choices, value=None)


def get_chunk_audio(label):
    if not label or not _chunk_audios: return None
    try:
        idx = int(label.split()[1]) - 1
        if 0 <= idx < len(_chunk_audios) and os.path.exists(_chunk_audios[idx]):
            return _chunk_audios[idx]
    except: pass
    return None

def dict_add(word, pron):
    if not word or not pron: return "❌ Dono fields bharo.", load_dict_md()
    d = load_custom_dict(); d[word.strip()] = pron.strip()
    save_custom_dict(d); CUSTOM_DICT.update(d)
    return f"✅ '{word}' → '{pron}' saved!", load_dict_md()

def dict_remove(word):
    d = load_custom_dict()
    if word.strip() in d:
        del d[word.strip()]; save_custom_dict(d)
        CUSTOM_DICT.clear(); CUSTOM_DICT.update(d)
        return f"✅ '{word}' removed.", load_dict_md()
    return f"⚠️ '{word}' nahi mila.", load_dict_md()

def load_dict_md():
    d = load_custom_dict()
    if not d: return "📖 Dictionary khaali hai."
    return "\n".join(f"**{k}** → {v}" for k,v in d.items())

def apply_emotion(e):
    return EMOTION_PRESETS.get(e, EMOTION_PRESETS["😊 सामान्य (Normal)"])["speed"]

# ══════════════════════════════════════════════════════════════════
# १४. Modern Dark UI
# ══════════════════════════════════════════════════════════════════
CSS = """
.gradio-container{font-family:'Segoe UI','Inter',Arial,sans-serif!important;
  background:#0d1117!important;color:#e6edf3!important}
.main,.gr-panel,.gr-form,.gr-box{background:#161b22!important;
  border:1px solid #21262d!important;border-radius:12px!important}
.gr-accordion{background:#161b22!important;border:1px solid #21262d!important;border-radius:10px!important}
label span,.gr-label{color:#c9d1d9!important;font-weight:500!important;font-size:.88em!important}
textarea,input[type=text]{background:#0d1117!important;border:1px solid #30363d!important;
  color:#e6edf3!important;border-radius:8px!important}
textarea:focus,input:focus{border-color:#f7931a!important;
  box-shadow:0 0 0 2px rgba(247,147,26,.15)!important;outline:none!important}
input[type=range]{accent-color:#f7931a!important}
.gr-button-primary{background:linear-gradient(135deg,#f7931a,#e67e00)!important;
  border:none!important;color:#fff!important;font-weight:700!important;
  border-radius:10px!important;box-shadow:0 4px 16px rgba(247,147,26,.35)!important}
.gr-button-primary:hover{transform:translateY(-1px)!important;
  box-shadow:0 6px 24px rgba(247,147,26,.5)!important}
.gr-button-secondary{background:#21262d!important;border:1px solid #30363d!important;
  color:#c9d1d9!important;border-radius:8px!important}
select,.gr-dropdown{background:#0d1117!important;border:1px solid #30363d!important;
  color:#e6edf3!important;border-radius:8px!important}
.status-out textarea{background:#0d1117!important;border:1px solid #21262d!important;
  color:#7ee787!important;font-family:Consolas,monospace!important;font-size:.85em!important}
input[type=checkbox]{accent-color:#f7931a!important}
.gr-tab-item{background:#161b22!important;color:#8b949e!important;border-radius:8px 8px 0 0!important}
.gr-tab-item.selected{background:#21262d!important;color:#f7931a!important;
  border-bottom:2px solid #f7931a!important}
hr{border-color:#21262d!important}
"""

HEADER = """
<div style="background:linear-gradient(135deg,#1a1f2e,#0d1117,#1a0a00);
  border:1px solid rgba(247,147,26,.3);border-radius:16px;padding:22px 28px;
  margin-bottom:18px;text-align:center;box-shadow:0 4px 32px rgba(247,147,26,.08)">
  <div style="font-size:2em;font-weight:700;background:linear-gradient(90deg,#f7931a,#ff6b35,#f7931a);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
    background-clip:text;letter-spacing:1px">🚩 शिव AI — v3.5</div>
  <p style="color:#8b949e;font-size:.9em;margin:4px 0 12px">
    श्री राम नाग &nbsp;|&nbsp; PAISAWALA &nbsp;|&nbsp; Advanced Trilingual Voice Engine</p>
  <div style="display:flex;justify-content:center;flex-wrap:wrap;gap:7px">
    <span style="background:#21262d;border:1px solid #30363d;border-radius:20px;
      padding:2px 11px;font-size:.75em;color:#2ecc71">✅ Pitch Fix</span>
    <span style="background:#21262d;border:1px solid #30363d;border-radius:20px;
      padding:2px 11px;font-size:.75em;color:#2ecc71">✅ DeEsser</span>
    <span style="background:#21262d;border:1px solid #30363d;border-radius:20px;
      padding:2px 11px;font-size:.75em;color:#2ecc71">✅ Compressor</span>
    <span style="background:#21262d;border:1px solid #30363d;border-radius:20px;
      padding:2px 11px;font-size:.75em;color:#2ecc71">✅ Multi-Ref Voice</span>
    <span style="background:#21262d;border:1px solid #30363d;border-radius:20px;
      padding:2px 11px;font-size:.75em;color:#2ecc71">✅ Hindi+English+Sanskrit</span>
    <span style="background:#21262d;border:1px solid #30363d;border-radius:20px;
      padding:2px 11px;font-size:.75em;color:#2ecc71">✅ MP3/WAV/OGG Export</span>
  </div>
</div>
"""

with gr.Blocks(css=CSS, title="शिव AI v3.5") as demo:

    gr.HTML(HEADER)

    with gr.Tabs():

        # ═══ TAB 1: Generate ═══
        with gr.Tab("🎙️ Generate"):
            with gr.Row():

                # LEFT — Text
                with gr.Column(scale=3):
                    txt = gr.Textbox(
                        label="📝 Script — Hindi / English / Sanskrit",
                        lines=13,
                        placeholder="यहाँ script paste करें...\nExample: आज हम Success की बात करेंगे। Dharma ka path follow karo।"
                    )
                    with gr.Row():
                        wc  = gr.Markdown("शब्द: **0**")
                        cc  = gr.Markdown("अक्षर: **0**")
                    txt.change(
                        lambda x: (f"शब्द: **{len(x.split())}**", f"अक्षर: **{len(x)}**"),
                        [txt],[wc,cc]
                    )
                    with gr.Row():
                        prev_btn = gr.Button("👁 Text Preview (cleanup dekhein)", size="sm", variant="secondary")
                    text_preview = gr.Markdown(visible=False)
                    prev_btn.click(lambda t,cw: (preview_cleaned_text(t,cw), gr.update(visible=True)),
                                   [txt, gr.State("")], [text_preview, text_preview])

                # RIGHT — Controls
                with gr.Column(scale=2):

                    with gr.Group():
                        gr.Markdown("### 🎤 Reference Voice")
                        gr.Markdown("*1 clip: normal | 2-3 clips: better voice match*")
                        up1 = gr.Audio(label="Clip 1 (main — 6-20 sec)", type="filepath")
                        with gr.Row():
                            up2 = gr.Audio(label="Clip 2 (optional)", type="filepath")
                            up3 = gr.Audio(label="Clip 3 (optional)", type="filepath")
                        ref_qual = gr.Textbox(
                            label="🔍 Quality Check",
                            interactive=False, lines=4,
                            elem_classes=["status-out"]
                        )
                        up1.change(check_ref_quality, [up1], [ref_qual])
                        git_v = gr.Dropdown(
                            choices=["aideva.wav"],
                            label="Default Voice (agar upload na karo)",
                            value="aideva.wav"
                        )

                    with gr.Accordion("🎭 Emotion / Style", open=True):
                        emotion = gr.Radio(
                            choices=list(EMOTION_PRESETS.keys()),
                            value="😊 सामान्य (Normal)",
                            label="Tone"
                        )
                        spd = gr.Slider(minimum=0.8, maximum=1.4, value=0.0, step=0.05,
                                        label="Speed Override (0 = emotion preset)")
                        emotion.change(apply_emotion, [emotion], [spd])

                    with gr.Accordion("🎵 Voice Match Settings", open=True):
                        gr.Markdown("*Yahan se voice 100% match karo*")
                        pitch_en = gr.Checkbox(
                            label="🎵 Pitch Correction (librosa required)", value=False)
                        pitch_sl = gr.Slider(
                            minimum=-6.0, maximum=6.0, value=0.0, step=0.5,
                            label="Pitch Shift (semitones) — negative=neeche, positive=upar"
                        )
                        gpt_len = gr.Slider(
                            minimum=3, maximum=30, value=6, step=1,
                            label="Voice Match Quality: gpt_cond_len (zyada=better match, slower)"
                        )

                    with gr.Accordion("🎛️ EQ — Bass / Mid / Treble", open=True):
                        bass_sl   = gr.Slider(minimum=-6.0, maximum=12.0, value=5.0, step=0.5,
                                              label="🔊 Bass Boost dB (4-6 rec.)")
                        mid_sl    = gr.Slider(minimum=-6.0, maximum=6.0,  value=0.0, step=0.5,
                                              label="🎵 Mid dB")
                        treble_sl = gr.Slider(minimum=-9.0, maximum=3.0,  value=-2.0, step=0.5,
                                              label="🔆 Treble dB (−2 = less robotic)")

                    with gr.Accordion("⚙️ Post-Processing", open=False):
                        with gr.Row():
                            sln  = gr.Checkbox(label="🔇 Silence Remove", value=True)
                            norm = gr.Checkbox(label="🧹 Normalize",       value=True)
                            eq   = gr.Checkbox(label="🎛️ EQ",             value=True)
                        with gr.Row():
                            dess = gr.Checkbox(label="🎤 DeEsser (harsh 's' fix)", value=True)
                            comp = gr.Checkbox(label="🗜️ Compressor",              value=True)

                    with gr.Accordion("💾 Output Format", open=False):
                        out_fmt = gr.Radio(
                            choices=["wav","mp3","ogg"],
                            value="wav",
                            label="Format"
                        )

                    with gr.Accordion("📖 Custom Words (session)", open=False):
                        custom_raw = gr.Textbox(
                            label="WORD = उच्चारण (ek per line)",
                            placeholder="PAISAWALA = पेसावाला\nShriramnag = श्री राम नाग",
                            lines=3
                        )

                    btn = gr.Button("🚀 Generate करो", variant="primary", size="lg")

            with gr.Row():
                with gr.Column(scale=2):
                    out_audio = gr.Audio(label="🎧 Output", type="filepath", autoplay=True)
                with gr.Column(scale=1):
                    out_status = gr.Textbox(label="📊 Status", interactive=False,
                                            lines=7, elem_classes=["status-out"])

            with gr.Accordion("🔍 Chunk Preview", open=False):
                with gr.Row():
                    chunk_dd  = gr.Dropdown(label="Part", choices=[], interactive=True)
                    chunk_btn = gr.Button("▶️ Play", size="sm")
                chunk_out = gr.Audio(label="Chunk", type="filepath", autoplay=True)
                chunk_btn.click(get_chunk_audio, [chunk_dd], [chunk_out])

            btn.click(
                generate_v35,
                inputs=[txt, up1, up2, up3, git_v,
                        emotion, spd, pitch_sl, gpt_len,
                        bass_sl, mid_sl, treble_sl,
                        sln, norm, eq, dess, comp,
                        pitch_en, out_fmt, custom_raw],
                outputs=[out_audio, out_status, ref_qual, chunk_dd]
            )

        # ═══ TAB 2: Dictionary ═══
        with gr.Tab("📖 Custom Dictionary"):
            gr.Markdown("### Custom words permanently save karo")
            with gr.Row():
                with gr.Column():
                    dw = gr.Textbox(label="Word", placeholder="PAISAWALA")
                    dp = gr.Textbox(label="Pronunciation", placeholder="पेसावाला")
                    with gr.Row():
                        da = gr.Button("➕ Add", variant="primary")
                        dr = gr.Button("🗑️ Remove", variant="secondary")
                    ds = gr.Textbox(label="Status", interactive=False, lines=2)
                with gr.Column():
                    dd = gr.Markdown(load_dict_md())
            da.click(dict_add,    [dw,dp], [ds,dd])
            dr.click(dict_remove, [dw],    [ds,dd])

        # ═══ TAB 3: Guide ═══
        with gr.Tab("📚 Guide"):
            gr.Markdown("""
### 🚩 Shiv AI v3.5 — Complete Guide

---
#### 🎤 Voice 100% Match kaise karein?

**Step 1 — Acha Reference Upload karo:**
- 6-20 second ki apni awaaz record karo
- Quiet room mein bolo, background noise nahi
- **2-3 clips upload karo** — zyada data = better match

**Step 2 — gpt_cond_len badhao:**
- Default 6 → `12` ya `18` karo
- Zyada = slower but voice matching better hogi

**Step 3 — Pitch Correction:**
- librosa install karo: `pip install librosa`
- Agar generated voice thodi upar/neeche lag rahi hai
- `-1` ya `-2` semitones try karo

---
#### 🎛️ EQ Recommendations
| Problem | Fix |
|---------|-----|
| Voice thin/hollow | Bass +5 to +8 dB |
| Harsh/nasal sound | Mid -2 dB |
| Robotic/sharp | Treble -2 to -4 dB |
| Sibilance (harsh 's') | DeEsser ON karo |

---
#### 🌐 Trilingual — Best Practices
```
✅ Hindi:    आज हम बात करेंगे।
✅ English:  Chunks automatically detect honge।
✅ Sanskrit: Om, dharma, karma auto-convert।
```

---
#### 🎭 Emotion Guide
| Mode | Best For |
|------|----------|
| 🧘 Calm | Meditation, slow narration |
| 😊 Normal | YouTube videos, general |
| 🎙️ Pro | News, motivational |
| 🔥 Dramatic | Story, energetic |

---
#### ⚡ Pro Tips
- **Speed 0.95** = natural feel (1.0 thoda fast lagta hai)
- **Compressor ON** = volume consistent rahegi
- **DeEsser ON** = TTS ka artificial sharpness hatega
- **MP3 export** = YouTube upload ke liye best
            """)

demo.launch(share=True, show_error=True)
