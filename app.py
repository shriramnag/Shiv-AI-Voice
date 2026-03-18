"""
╔══════════════════════════════════════════════════════════════════╗
║         शिव AI (Shiv AI) v3.0 — श्री राम नाग                   ║
║         PAISAWALA YouTube Channel                                ║
║  ✅ Trilingual: Hindi + English + Sanskrit                       ║
║  ✅ Bass Fix + 3-Band EQ                                         ║
║  ✅ Emotion/Tone Control                                         ║
║  ✅ Reference Audio Quality Check                                ║
║  ✅ Custom Pronunciation Dictionary                              ║
║  ✅ Chunk-by-Chunk Preview                                       ║
║  ✅ Language-Aware Chunking (no more English stuttering)         ║
║  ✅ Modern Dark UI                                               ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os, torch, gradio as gr, requests, re, gc, json, math
import numpy as np
from TTS.api import TTS
from huggingface_hub import hf_hub_download
from pydub import AudioSegment, effects
from scipy.signal import butter, filtfilt, sosfilt, butter as butter_sos
import soundfile as sf

# ══════════════════════════════════════════════════════════════════
# १. सेटअप
# ══════════════════════════════════════════════════════════════════
os.environ["COQUI_TOS_AGREED"] = "1"
torch.backends.cudnn.benchmark = True
device = "cuda" if torch.cuda.is_available() else "cpu"

REPO_ID  = "Shriramnag/My-Shriram-Voice"
G_RAW    = "https://raw.githubusercontent.com/shriramnag/Aivoicebox/main/%F0%9F%93%81%20voices/"
DICT_FILE = "custom_dict.json"

print("🚩 शिव AI v3.0 — Trilingual Engine शुरू हो रहा है...")
try:
    hf_hub_download(repo_id=REPO_ID, filename="Ramai.pth")
    hf_hub_download(repo_id=REPO_ID, filename="config.json")
except:
    pass

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
print(f"✅ TTS Engine ready on {device.upper()}")

# ══════════════════════════════════════════════════════════════════
# २. Custom Dictionary (persistent JSON)
# ══════════════════════════════════════════════════════════════════
def load_custom_dict():
    if os.path.exists(DICT_FILE):
        try:
            with open(DICT_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            pass
    return {}

def save_custom_dict(d):
    with open(DICT_FILE, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)

CUSTOM_DICT = load_custom_dict()

# ══════════════════════════════════════════════════════════════════
# ३. Trilingual Dictionaries
# ══════════════════════════════════════════════════════════════════

# ── Sanskrit → Hindi phonetic (XTTS Hindi mode mein sahi bolta hai)
SANSKRIT_DICT = {
    "dharma": "धर्म", "karma": "कर्म", "yoga": "योग", "shakti": "शक्ति",
    "om": "ॐ", "aum": "ॐ", "namaste": "नमस्ते", "namaskaar": "नमस्कार",
    "guru": "गुरु", "mantra": "मंत्र", "tantra": "तंत्र", "yantra": "यंत्र",
    "atma": "आत्मा", "brahma": "ब्रह्म", "vishnu": "विष्णु", "shiva": "शिव",
    "maya": "माया", "moksha": "मोक्ष", "nirvana": "निर्वाण", "prana": "प्राण",
    "ahimsa": "अहिंसा", "satya": "सत्य", "seva": "सेवा", "bhakti": "भक्ति",
    "jnana": "ज्ञान", "gyan": "ज्ञान", "veda": "वेद", "sutra": "सूत्र",
    "shloka": "श्लोक", "stotra": "स्तोत्र", "pooja": "पूजा", "puja": "पूजा",
    "aarti": "आरती", "prasad": "प्रसाद", "tilak": "तिलक", "rudra": "रुद्र",
    "sadhana": "साधना", "tapas": "तपस", "satsang": "सत्संग",
}

# ── English → Hindi phonetic (बड़ा dictionary)
ENGLISH_TO_HINDI = {
    # Tech / Social
    "AI": "ए आई", "ML": "एम एल", "API": "ए पी आई", "GPU": "जी पी यू",
    "CPU": "सी पी यू", "URL": "यू आर एल", "HTML": "एच टी एम एल",
    "YouTube": "यूट्यूब", "Instagram": "इंस्टाग्राम", "Facebook": "फेसबुक",
    "WhatsApp": "व्हाट्सऐप", "Google": "गूगल", "Twitter": "ट्विटर",
    "Internet": "इंटरनेट", "Online": "ऑनलाइन", "Offline": "ऑफलाइन",
    "Software": "सॉफ्टवेयर", "Hardware": "हार्डवेयर", "Computer": "कंप्यूटर",
    "Mobile": "मोबाइल", "App": "ऐप", "Website": "वेबसाइट",
    "Download": "डाउनलोड", "Upload": "अपलोड", "Password": "पासवर्ड",
    "Server": "सर्वर", "Cloud": "क्लाउड", "Data": "डेटा",
    "Channel": "चैनल", "Video": "वीडियो", "Content": "कंटेंट",
    "Subscribe": "सब्सक्राइब", "Like": "लाइक", "Share": "शेयर",
    "Comment": "कमेंट", "Notification": "नोटिफिकेशन",
    # Motivation / Business
    "Life": "लाइफ", "Dream": "ड्रीम", "Mindset": "माइंडसेट",
    "Believe": "बिलीव", "Success": "सक्सेस", "Fail": "फेल",
    "Failure": "फेल्योर", "Goal": "गोल", "Focus": "फोकस",
    "Step": "स्टेप", "Fear": "फियर", "Simple": "सिंपल",
    "Practical": "प्रैक्टिकल", "Strong": "स्ट्रॉन्ग", "Turbo": "टर्बो",
    "Power": "पावर", "Energy": "एनर्जी", "Positive": "पॉजिटिव",
    "Negative": "नेगेटिव", "Challenge": "चैलेंज", "Time": "टाइम",
    "Work": "वर्क", "Hard": "हार्ड", "Smart": "स्मार्ट",
    "Money": "मनी", "Business": "बिज़नेस", "Market": "मार्केट",
    "Brand": "ब्रांड", "Profit": "प्रॉफिट", "Loss": "लॉस",
    "Investment": "इन्वेस्टमेंट", "Invest": "इन्वेस्ट",
    "Strategy": "स्ट्रेटेजी", "Plan": "प्लान", "Team": "टीम",
    "Leader": "लीडर", "Leadership": "लीडरशिप", "Skill": "स्किल",
    "Training": "ट्रेनिंग", "Course": "कोर्स", "Degree": "डिग्री",
    # Common words
    "because": "बिकॉज़", "but": "बट", "so": "सो",
    "OK": "ओके", "okay": "ओके", "hey": "हे", "hi": "हाय",
    "hello": "हेलो", "bye": "बाय", "thanks": "थैंक्स",
    "please": "प्लीज़", "sorry": "सॉरी", "welcome": "वेलकम",
}

# ── Numbers → Hindi words
NUMBER_MAP = {
    "1000": "एक हज़ार", "500": "पाँच सौ", "200": "दो सौ",
    "100": "सौ", "90": "नब्बे", "80": "अस्सी", "70": "सत्तर",
    "60": "साठ", "50": "पचास", "40": "चालीस", "30": "तीस",
    "25": "पच्चीस", "20": "बीस", "19": "उन्नीस", "18": "अठारह",
    "17": "सत्रह", "16": "सोलह", "15": "पंद्रह", "14": "चौदह",
    "13": "तेरह", "12": "बारह", "11": "ग्यारह", "10": "दस",
    "9": "नौ", "8": "आठ", "7": "सात", "6": "छह",
    "5": "पाँच", "4": "चार", "3": "तीन", "2": "दो",
    "1": "एक", "0": "शून्य",
}

# ══════════════════════════════════════════════════════════════════
# ४. Language Detection + Trilingual Text Processor
# ══════════════════════════════════════════════════════════════════
def detect_script(word):
    """Word kis script mein hai detect karo"""
    devanagari = sum(1 for c in word if '\u0900' <= c <= '\u097F')
    latin      = sum(1 for c in word if c.isascii() and c.isalpha())
    if devanagari > 0:
        return "hi"
    elif latin > 0:
        return "en"
    return "hi"

def is_sanskrit_word(word):
    """Sanskrit-origin words check karo"""
    w = word.lower()
    return w in SANSKRIT_DICT or any(w == k.lower() for k in SANSKRIT_DICT)

def apply_all_dicts(text, custom_dict):
    """Custom → Sanskrit → English → Numbers — sab dictionaries apply karo"""

    # 1. Custom dictionary (user-defined, highest priority)
    for src, tgt in custom_dict.items():
        text = re.sub(rf'(?<![a-zA-Z\u0900-\u097F]){re.escape(src)}(?![a-zA-Z\u0900-\u097F])',
                      tgt, text, flags=re.IGNORECASE)

    # 2. Sanskrit words → proper Devanagari
    for src, tgt in SANSKRIT_DICT.items():
        text = re.sub(rf'(?<![a-zA-Z]){re.escape(src)}(?![a-zA-Z])',
                      tgt, text, flags=re.IGNORECASE)

    # 3. English → Hindi phonetic
    for src, tgt in ENGLISH_TO_HINDI.items():
        text = re.sub(rf'(?<![a-zA-Z]){re.escape(src)}(?![a-zA-Z])',
                      tgt, text, flags=re.IGNORECASE)

    # 4. Numbers → Hindi words (bade pehle)
    for num in sorted(NUMBER_MAP.keys(), key=lambda x: -len(x)):
        text = re.sub(rf'\b{num}\b', NUMBER_MAP[num], text)

    return text

def clean_punctuation(text):
    """Punctuation ko XTTS-friendly pause mein badlo"""
    text = re.sub(r'[।\.](\s|$)', '। ', text)
    text = re.sub(r'[,،]\s*', ', ', text)
    text = re.sub(r'[!]\s*', '! ', text)
    text = re.sub(r'[?]\s*', '? ', text)
    text = re.sub(r'[-–—]+', ', ', text)
    text = re.sub(r'["""\'\'()\[\]{}]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def full_text_processor(text, custom_dict):
    """Complete text processing pipeline"""
    if not text:
        return ""
    text = apply_all_dicts(text, custom_dict)
    text = clean_punctuation(text)
    return text

# ══════════════════════════════════════════════════════════════════
# ५. Language-Aware Smart Chunker
#    (English atak fix — yahi sabse bada fix hai!)
# ══════════════════════════════════════════════════════════════════
def language_aware_chunker(text, max_words=45):
    """
    Text ko sentence boundary par toro, aur detect karo ki chunk
    Hindi hai ya English — taaki XTTS ka language param sahi lage.
    Returns: list of (chunk_text, language_code)
    """
    sentences = re.split(r'(?<=[।\.\!\?])\s+', text.strip())
    chunks_with_lang = []
    current_words = []
    current_count = 0

    def commit_chunk(words):
        if not words:
            return
        chunk = ' '.join(words)
        # Chunk ki language detect karo
        latin_count = sum(1 for w in words
                          if sum(1 for c in w if c.isascii() and c.isalpha()) > len(w) // 2)
        lang = "en" if latin_count > len(words) * 0.5 else "hi"
        chunks_with_lang.append((chunk, lang))

    for sentence in sentences:
        words = sentence.split()
        wc = len(words)

        if wc > max_words:
            if current_words:
                commit_chunk(current_words)
                current_words, current_count = [], 0
            # Comma par toro
            parts = re.split(r',\s*', sentence)
            tmp, tc = [], 0
            for part in parts:
                pw = part.split()
                if tc + len(pw) > max_words and tmp:
                    commit_chunk(tmp)
                    tmp, tc = pw, len(pw)
                else:
                    tmp.extend(pw)
                    tc += len(pw)
            if tmp:
                commit_chunk(tmp)

        elif current_count + wc > max_words:
            commit_chunk(current_words)
            current_words, current_count = words, wc
        else:
            current_words.extend(words)
            current_count += wc

    commit_chunk(current_words)
    return [(c.strip(), l) for c, l in chunks_with_lang if c.strip()]

# ══════════════════════════════════════════════════════════════════
# ६. Reference Audio Quality Check
# ══════════════════════════════════════════════════════════════════
def check_ref_audio_quality(filepath):
    """Reference audio ki quality check karo"""
    warnings = []
    try:
        audio = AudioSegment.from_file(filepath)
        duration_sec = len(audio) / 1000.0
        sr = audio.frame_rate
        channels = audio.channels

        if duration_sec < 6:
            warnings.append(f"⚠️ Audio bahut chhota hai ({duration_sec:.1f}s) — kam se kam 6 second chahiye. Voice match kharab hoga.")
        elif duration_sec > 30:
            warnings.append(f"ℹ️ Audio lamba hai ({duration_sec:.1f}s) — 6-20 second best hota hai.")
        else:
            warnings.append(f"✅ Duration theek hai: {duration_sec:.1f}s")

        if sr < 22050:
            warnings.append(f"⚠️ Sample rate kam hai ({sr}Hz) — 22050Hz ya zyaada chahiye.")
        else:
            warnings.append(f"✅ Sample rate: {sr}Hz")

        if channels > 1:
            warnings.append("ℹ️ Stereo audio hai — mono mein convert ho jaayega.")

        # Simple noise check (silence ratio)
        samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
        rms = np.sqrt(np.mean(samples**2))
        if rms < 500:
            warnings.append("⚠️ Audio bahut soft hai — peak volume check karein.")
        else:
            warnings.append("✅ Volume theek hai.")

    except Exception as e:
        warnings.append(f"⚠️ Quality check failed: {e}")

    return "\n".join(warnings)

# ══════════════════════════════════════════════════════════════════
# ७. Prepare Reference Audio (normalize + trim)
# ══════════════════════════════════════════════════════════════════
def prepare_reference(filepath, out_path="ref_prepared.wav"):
    """Reference audio ko XTTS ke liye optimize karo"""
    try:
        audio = AudioSegment.from_file(filepath)
        # Mono, 22050 Hz
        audio = audio.set_channels(1).set_frame_rate(22050)
        # Silence trim
        audio = effects.strip_silence(audio, silence_thresh=-45, padding=150)
        # Normalize
        audio = effects.normalize(audio)
        # Max 25 seconds
        if len(audio) > 25000:
            audio = audio[:25000]
        audio.export(out_path, format="wav")
        return out_path
    except Exception as e:
        print(f"Reference prep warning: {e}")
        return filepath

# ══════════════════════════════════════════════════════════════════
# ८. Advanced Audio Post-Processor (Bass Fix + EQ)
# ══════════════════════════════════════════════════════════════════
def apply_eq_and_enhance(audio_seg, bass_db=4.0, mid_db=0.0, treble_db=-2.0,
                          sample_rate=22050):
    """
    Bass Fix:  150-250 Hz pe boost (jo pehle cut ho raha tha)
    Mid:       500-2000 Hz
    Treble:    5000+ Hz (thoda kam karo — roboticness hatao)
    """
    try:
        audio_seg = audio_seg.set_frame_rate(sample_rate).set_channels(1)
        samples = np.array(audio_seg.get_array_of_samples(), dtype=np.float64)
        nyq = sample_rate / 2.0

        # ── Bass Boost (80-300 Hz) ──
        if abs(bass_db) > 0.1:
            b_low, a_low   = butter(2, 80  / nyq, btype='high')
            b_high, a_high = butter(2, 300 / nyq, btype='low')
            bass_band = filtfilt(b_low, a_low, samples)
            bass_band = filtfilt(b_high, a_high, bass_band)
            gain = 10 ** (bass_db / 20.0)
            samples = samples + bass_band * (gain - 1.0)

        # ── Mid adjust (500-2000 Hz) ──
        if abs(mid_db) > 0.1:
            b_l, a_l = butter(2, 500  / nyq, btype='high')
            b_h, a_h = butter(2, 2000 / nyq, btype='low')
            mid_band = filtfilt(b_l, a_l, samples)
            mid_band = filtfilt(b_h, a_h, mid_band)
            gain = 10 ** (mid_db / 20.0)
            samples = samples + mid_band * (gain - 1.0)

        # ── Treble cut (5000+ Hz) ──
        if abs(treble_db) > 0.1:
            b_t, a_t = butter(2, 5000 / nyq, btype='high')
            treble_band = filtfilt(b_t, a_t, samples)
            gain = 10 ** (treble_db / 20.0)
            samples = samples + treble_band * (gain - 1.0)

        # Clip + convert
        samples = np.clip(samples, -32768, 32767).astype(np.int16)
        enhanced = AudioSegment(
            samples.tobytes(),
            frame_rate=sample_rate,
            sample_width=2,
            channels=1
        )
        return effects.normalize(enhanced)

    except Exception as e:
        print(f"EQ error (skipping): {e}")
        return effects.normalize(audio_seg)

# ══════════════════════════════════════════════════════════════════
# ९. Crossfade Joiner
# ══════════════════════════════════════════════════════════════════
def crossfade_join(segments, crossfade_ms=60):
    if not segments:
        return AudioSegment.silent(duration=100)
    result = segments[0]
    for seg in segments[1:]:
        cf = min(crossfade_ms, len(result) // 2, len(seg) // 2)
        result = result.append(seg, crossfade=max(cf, 10))
    return result

# ══════════════════════════════════════════════════════════════════
# १०. Emotion Presets
# ══════════════════════════════════════════════════════════════════
EMOTION_PRESETS = {
    "🧘 शांत (Calm)":        {"temperature": 0.20, "rep_pen": 7.0, "speed": 0.92},
    "😊 सामान्य (Normal)":   {"temperature": 0.35, "rep_pen": 6.0, "speed": 1.00},
    "🎙️ प्रभावशाली (Pro)":  {"temperature": 0.50, "rep_pen": 5.0, "speed": 1.05},
    "🔥 नाटकीय (Dramatic)":  {"temperature": 0.68, "rep_pen": 4.0, "speed": 1.10},
}

# ══════════════════════════════════════════════════════════════════
# ११. Main Generation Engine
# ══════════════════════════════════════════════════════════════════
# Global store for chunk-by-chunk preview
_chunk_audios = []

def generate_shiv_v3(
    text, up_ref, git_ref,
    emotion_mode,
    speed_override, bass_db, mid_db, treble_db,
    use_silence, use_clean, use_enhance,
    custom_words_raw,
    progress=gr.Progress()
):
    global _chunk_audios
    _chunk_audios = []

    if not text or not text.strip():
        return None, "❌ कोई टेक्स्ट नहीं दिया।", "", gr.update(choices=[], value=None)

    # ── Emotion preset ──
    preset  = EMOTION_PRESETS.get(emotion_mode, EMOTION_PRESETS["😊 सामान्य (Normal)"])
    temperature = preset["temperature"]
    rep_pen     = preset["rep_pen"]
    speed_s     = float(speed_override) if speed_override != 0 else preset["speed"]

    # ── Custom dictionary ──
    custom_dict = load_custom_dict()
    if custom_words_raw and custom_words_raw.strip():
        for line in custom_words_raw.strip().splitlines():
            if "=" in line:
                src, tgt = line.split("=", 1)
                src, tgt = src.strip(), tgt.strip()
                if src and tgt:
                    custom_dict[src] = tgt
        save_custom_dict(custom_dict)
        CUSTOM_DICT.update(custom_dict)

    # ── Text process ──
    progress(0.02, desc="📝 Trilingual text processing...")
    p_text = full_text_processor(text, custom_dict)

    # ── Reference audio ──
    progress(0.05, desc="🎤 Reference audio prepare ho raha hai...")
    ref_path = up_ref if up_ref else "ref_raw.wav"
    if not up_ref:
        url = G_RAW + requests.utils.quote(git_ref)
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            with open(ref_path, "wb") as f:
                f.write(resp.content)
        except Exception as e:
            return None, f"❌ Reference download failed: {e}", "", gr.update(choices=[], value=None)

    ref = prepare_reference(ref_path, "ref_prepared.wav")
    ref_quality = check_ref_audio_quality(ref)

    # ── Language-aware chunking ──
    progress(0.08, desc="✂️ Smart chunking (language-aware)...")
    chunks_with_lang = language_aware_chunker(p_text, max_words=45)
    total = len(chunks_with_lang)

    if total == 0:
        return None, "❌ Text empty ho gaya process ke baad.", ref_quality, gr.update(choices=[], value=None)

    segments = []
    errors   = []
    chunk_files = []

    for i, (chunk, lang) in enumerate(chunks_with_lang):
        progress((i + 1) / total * 0.82,
                 desc=f"🎙️ Generate: Part {i+1}/{total} [{lang.upper()}]")
        name = f"chunk_{i}.wav"

        try:
            # Set model params
            try:
                tts.synthesizer.tts_config.model_args.temperature        = float(temperature)
                tts.synthesizer.tts_config.model_args.repetition_penalty = float(rep_pen)
            except:
                pass

            # Language-aware generation (MAIN FIX for English stuttering)
            tts.tts_to_file(
                text=chunk,
                speaker_wav=ref,
                language=lang,          # "hi" ya "en" — yahi fix hai!
                file_path=name,
                speed=float(speed_s),
            )

            seg = AudioSegment.from_wav(name)

            if use_silence:
                try:
                    seg = effects.strip_silence(seg, silence_thresh=-50, padding=180)
                except:
                    pass

            if len(seg) > 80:
                segments.append(seg)
                # Chunk preview ke liye save karo
                chunk_out = f"preview_chunk_{i+1}.wav"
                seg.export(chunk_out, format="wav")
                chunk_files.append(chunk_out)
                _chunk_audios.append(chunk_out)

            os.remove(name)

        except Exception as e:
            errors.append(f"Part {i+1} [{lang}]: {str(e)[:100]}")
            if os.path.exists(name):
                os.remove(name)

        torch.cuda.empty_cache()
        gc.collect()

    if not segments:
        return None, f"❌ Koi chunk generate nahi hua.\n{chr(10).join(errors[:5])}", ref_quality, gr.update(choices=[], value=None)

    # ── Join with crossfade ──
    progress(0.87, desc="🔗 Crossfade join ho raha hai...")
    combined = crossfade_join(segments, crossfade_ms=60)

    # ── Normalize ──
    if use_clean:
        progress(0.91, desc="🧹 Normalize + Clean...")
        combined = combined.set_frame_rate(22050).set_channels(1)
        combined = effects.normalize(combined)

    # ── EQ + Bass Fix ──
    if use_enhance:
        progress(0.95, desc="🎛️ EQ + Bass Enhancement...")
        combined = apply_eq_and_enhance(
            combined,
            bass_db=float(bass_db),
            mid_db=float(mid_db),
            treble_db=float(treble_db),
            sample_rate=22050
        )

    final_name = "ShivAI_v3_Output.wav"
    combined.export(final_name, format="wav", parameters=["-ar", "22050"])
    progress(1.0, desc="✅ तैयार!")

    status = f"✅ {len(segments)}/{total} parts successfully generate hue."
    if errors:
        status += f"\n⚠️ {len(errors)} error(s):\n" + "\n".join(errors[:3])
    status += f"\n⏱️ Total duration: {len(combined)/1000:.1f}s"

    # Chunk preview dropdown
    chunk_choices = [f"Part {i+1}" for i in range(len(chunk_files))]

    return final_name, status, ref_quality, gr.update(choices=chunk_choices, value=None)


def get_chunk_audio(chunk_label):
    """Chunk preview ke liye audio return karo"""
    if not chunk_label or not _chunk_audios:
        return None
    try:
        idx = int(chunk_label.split(" ")[1]) - 1
        if 0 <= idx < len(_chunk_audios) and os.path.exists(_chunk_audios[idx]):
            return _chunk_audios[idx]
    except:
        pass
    return None


def add_to_dict(word, pronunciation, status_box):
    """Custom dictionary mein word add karo"""
    if not word or not pronunciation:
        return "❌ Word aur pronunciation dono bharein.", load_dict_display()
    d = load_custom_dict()
    d[word.strip()] = pronunciation.strip()
    save_custom_dict(d)
    CUSTOM_DICT.update(d)
    return f"✅ '{word}' → '{pronunciation}' add ho gaya!", load_dict_display()


def remove_from_dict(word):
    """Dictionary se word hatao"""
    d = load_custom_dict()
    removed = d.pop(word.strip(), None)
    if removed:
        save_custom_dict(d)
        CUSTOM_DICT.clear()
        CUSTOM_DICT.update(d)
        return f"✅ '{word}' hata diya.", load_dict_display()
    return f"⚠️ '{word}' nahi mila.", load_dict_display()


def load_dict_display():
    d = load_custom_dict()
    if not d:
        return "📖 Dictionary khaali hai."
    lines = [f"**{k}** → {v}" for k, v in d.items()]
    return "\n".join(lines)


def apply_emotion_preset(emotion):
    p = EMOTION_PRESETS.get(emotion, EMOTION_PRESETS["😊 सामान्य (Normal)"])
    return p["speed"]

# ══════════════════════════════════════════════════════════════════
# १२. Modern Dark UI
# ══════════════════════════════════════════════════════════════════

# Custom CSS — dark, modern, Colab mein bhi acha dikhega
CUSTOM_CSS = """
/* ── Global ── */
.gradio-container {
    font-family: 'Segoe UI', 'Inter', Arial, sans-serif !important;
    background: #0d1117 !important;
    color: #e6edf3 !important;
}
.main { background: #0d1117 !important; }

/* ── Header banner ── */
.shiv-header {
    background: linear-gradient(135deg, #1a1f2e 0%, #0d1117 50%, #1a0a00 100%);
    border: 1px solid #f7931a40;
    border-radius: 16px;
    padding: 24px 32px;
    margin-bottom: 20px;
    text-align: center;
    box-shadow: 0 4px 32px rgba(247,147,26,0.08);
}
.shiv-title {
    font-size: 2.2em;
    font-weight: 700;
    background: linear-gradient(90deg, #f7931a, #ff6b35, #f7931a);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 6px 0;
    letter-spacing: 1px;
}
.shiv-sub {
    color: #8b949e;
    font-size: 0.95em;
    margin: 0;
}
.badge-row {
    display: flex;
    justify-content: center;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 14px;
}
.badge {
    background: #21262d;
    border: 1px solid #30363d;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.78em;
    color: #2ecc71;
}

/* ── Cards / Panels ── */
.gr-box, .gr-panel, .gr-form {
    background: #161b22 !important;
    border: 1px solid #21262d !important;
    border-radius: 12px !important;
}
.gr-accordion {
    background: #161b22 !important;
    border: 1px solid #21262d !important;
    border-radius: 10px !important;
}

/* ── Labels ── */
label span, .gr-label {
    color: #c9d1d9 !important;
    font-weight: 500 !important;
    font-size: 0.88em !important;
}

/* ── Textbox ── */
textarea, input[type="text"] {
    background: #0d1117 !important;
    border: 1px solid #30363d !important;
    color: #e6edf3 !important;
    border-radius: 8px !important;
    font-size: 0.95em !important;
}
textarea:focus, input:focus {
    border-color: #f7931a !important;
    box-shadow: 0 0 0 2px rgba(247,147,26,0.15) !important;
    outline: none !important;
}

/* ── Sliders ── */
input[type="range"] { accent-color: #f7931a !important; }

/* ── Buttons ── */
.gr-button-primary {
    background: linear-gradient(135deg, #f7931a, #e67e00) !important;
    border: none !important;
    color: #fff !important;
    font-weight: 700 !important;
    font-size: 1em !important;
    border-radius: 10px !important;
    box-shadow: 0 4px 16px rgba(247,147,26,0.35) !important;
    transition: all 0.2s !important;
}
.gr-button-primary:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 24px rgba(247,147,26,0.5) !important;
}
.gr-button-secondary {
    background: #21262d !important;
    border: 1px solid #30363d !important;
    color: #c9d1d9 !important;
    border-radius: 8px !important;
}

/* ── Dropdowns ── */
select, .gr-dropdown {
    background: #0d1117 !important;
    border: 1px solid #30363d !important;
    color: #e6edf3 !important;
    border-radius: 8px !important;
}

/* ── Status box ── */
.status-box textarea {
    background: #0d1117 !important;
    border: 1px solid #21262d !important;
    color: #7ee787 !important;
    font-family: 'Consolas', monospace !important;
    font-size: 0.85em !important;
}

/* ── Checkboxes ── */
input[type="checkbox"] { accent-color: #f7931a !important; }

/* ── Audio player ── */
.gr-audio {
    background: #161b22 !important;
    border: 1px solid #21262d !important;
    border-radius: 10px !important;
}

/* ── Tabs ── */
.gr-tab-item {
    background: #161b22 !important;
    color: #8b949e !important;
    border-radius: 8px 8px 0 0 !important;
}
.gr-tab-item.selected {
    background: #21262d !important;
    color: #f7931a !important;
    border-bottom: 2px solid #f7931a !important;
}

/* ── Word count ── */
.word-badge {
    display: inline-block;
    background: #21262d;
    color: #f7931a;
    border-radius: 6px;
    padding: 2px 10px;
    font-size: 0.82em;
    font-weight: 600;
    margin-top: 4px;
}

/* ── Divider ── */
hr { border-color: #21262d !important; }

/* ── Markdown ── */
.gr-markdown h3 { color: #f7931a !important; }
.gr-markdown p  { color: #8b949e !important; }
"""

HEADER_HTML = """
<div class="shiv-header">
  <div class="shiv-title">🚩 शिव AI — v3.0</div>
  <p class="shiv-sub">श्री राम नाग &nbsp;|&nbsp; PAISAWALA &nbsp;|&nbsp; Trilingual Voice Engine</p>
  <div class="badge-row">
    <span class="badge">✅ Hindi</span>
    <span class="badge">✅ English</span>
    <span class="badge">✅ Sanskrit</span>
    <span class="badge">✅ Bass Fix</span>
    <span class="badge">✅ No Stutter</span>
    <span class="badge">✅ Voice Match</span>
    <span class="badge">✅ EQ Panel</span>
    <span class="badge">✅ Emotion Mode</span>
  </div>
</div>
"""

# ── Build Gradio App ──
with gr.Blocks(css=CUSTOM_CSS, title="शिव AI v3.0") as demo:

    gr.HTML(HEADER_HTML)

    with gr.Tabs():

        # ════ TAB 1: Generate ════
        with gr.Tab("🎙️ Generate"):
            with gr.Row():

                # ── LEFT: Text ──
                with gr.Column(scale=3):
                    txt = gr.Textbox(
                        label="📝 Script (Hindi / English / Sanskrit)",
                        lines=14,
                        placeholder="यहाँ कोई भी भाषा में script paste करें...\nExample: आज हम Life के बारे में बात करेंगे। Dharma aur karma ka path follow karo.",
                        elem_id="main-text"
                    )
                    with gr.Row():
                        word_count = gr.Markdown("शब्द: **0**")
                        char_count = gr.Markdown("अक्षर: **0**")

                    def update_counts(x):
                        if not x:
                            return "शब्द: **0**", "अक्षर: **0**"
                        return f"शब्द: **{len(x.split())}**", f"अक्षर: **{len(x)}**"

                    txt.change(update_counts, [txt], [word_count, char_count])

                # ── RIGHT: Controls ──
                with gr.Column(scale=2):

                    with gr.Group():
                        gr.Markdown("### 🎤 Reference Voice")
                        up_v = gr.Audio(
                            label="अपनी आवाज़ Upload करें (6-20 sec recommended)",
                            type="filepath"
                        )
                        ref_quality_out = gr.Textbox(
                            label="🔍 Voice Quality Check",
                            interactive=False, lines=4,
                            elem_classes=["status-box"]
                        )
                        up_v.change(
                            check_ref_audio_quality,
                            [up_v], [ref_quality_out]
                        )
                        git_v = gr.Dropdown(
                            choices=["aideva.wav"],
                            label="Default Voice (agar upload na karo)",
                            value="aideva.wav"
                        )

                    with gr.Accordion("🎭 Emotion / Tone", open=True):
                        emotion = gr.Radio(
                            choices=list(EMOTION_PRESETS.keys()),
                            value="😊 सामान्य (Normal)",
                            label="Tone / Style"
                        )
                        spd = gr.Slider(
                            minimum=0.8, maximum=1.4, value=0.0, step=0.05,
                            label="Speed Override (0 = emotion preset ki speed use hogi)"
                        )
                        emotion.change(apply_emotion_preset, [emotion], [spd])

                    with gr.Accordion("🎛️ EQ — Bass / Mid / Treble", open=True):
                        gr.Markdown("*Bass kam tha — ab yahan se fix karo*")
                        bass_sl = gr.Slider(
                            minimum=-6.0, maximum=12.0, value=5.0, step=0.5,
                            label="🔊 Bass Boost (dB) — 4-6 recommended"
                        )
                        mid_sl = gr.Slider(
                            minimum=-6.0, maximum=6.0, value=0.0, step=0.5,
                            label="🎵 Mid (dB)"
                        )
                        treble_sl = gr.Slider(
                            minimum=-9.0, maximum=3.0, value=-2.0, step=0.5,
                            label="🔆 Treble (dB) — negative = less robotic"
                        )

                    with gr.Accordion("⚙️ Post-Processing", open=False):
                        with gr.Row():
                            sln = gr.Checkbox(label="🔇 Silence Remove", value=True)
                            cln = gr.Checkbox(label="🧹 Normalize", value=True)
                            enh = gr.Checkbox(label="✨ EQ Enhance", value=True)

                    with gr.Accordion("📖 Quick Custom Words (session)", open=False):
                        custom_raw = gr.Textbox(
                            label="Format: WORD = उच्चारण (ek per line)",
                            placeholder="PAISAWALA = पेसावाला\nShriramnag = श्री राम नाग",
                            lines=4
                        )

                    btn = gr.Button("🚀 आवाज़ Generate करो", variant="primary", size="lg")

            # ── Output ──
            with gr.Row():
                with gr.Column(scale=2):
                    out_audio = gr.Audio(
                        label="🎧 Final Output",
                        type="filepath",
                        autoplay=True
                    )
                with gr.Column(scale=1):
                    out_status = gr.Textbox(
                        label="📊 Status",
                        interactive=False, lines=6,
                        elem_classes=["status-box"]
                    )

            # ── Chunk Preview ──
            with gr.Accordion("🔍 Chunk-by-Chunk Preview (ek part dobara suno)", open=False):
                gr.Markdown("*Agar kisi ek part mein problem hai to sirf woh suno*")
                with gr.Row():
                    chunk_dd  = gr.Dropdown(label="Part chuniye", choices=[], interactive=True)
                    chunk_btn = gr.Button("▶️ Suno", size="sm")
                chunk_audio = gr.Audio(label="Chunk Audio", type="filepath", autoplay=True)
                chunk_btn.click(get_chunk_audio, [chunk_dd], [chunk_audio])

            # ── Generate button click ──
            btn.click(
                generate_shiv_v3,
                inputs=[txt, up_v, git_v, emotion, spd,
                        bass_sl, mid_sl, treble_sl,
                        sln, cln, enh, custom_raw],
                outputs=[out_audio, out_status, ref_quality_out, chunk_dd]
            )

        # ════ TAB 2: Custom Dictionary ════
        with gr.Tab("📖 Custom Dictionary"):
            gr.Markdown("### अपने Custom Words save karo — हर बार load honge")
            with gr.Row():
                with gr.Column():
                    dict_word  = gr.Textbox(label="Word (jaise: PAISAWALA)", placeholder="PAISAWALA")
                    dict_pron  = gr.Textbox(label="Pronunciation (jaise: पेसावाला)", placeholder="पेसावाला")
                    with gr.Row():
                        dict_add_btn = gr.Button("➕ Add", variant="primary")
                        dict_rem_btn = gr.Button("🗑️ Remove", variant="secondary")
                    dict_status = gr.Textbox(label="Status", interactive=False, lines=2)

                with gr.Column():
                    dict_display = gr.Markdown(load_dict_display())

            dict_add_btn.click(
                add_to_dict,
                [dict_word, dict_pron, dict_status],
                [dict_status, dict_display]
            )
            dict_rem_btn.click(
                remove_from_dict,
                [dict_word],
                [dict_status, dict_display]
            )

        # ════ TAB 3: Guide ════
        with gr.Tab("📚 Guide & Tips"):
            gr.Markdown("""
### 🚩 Shiv AI v3.0 — Poori Guide

---

#### 🎤 Reference Audio — Best Practices
- **6-20 second** ki apni awaaz record karo
- Quiet room mein record karo — background noise mat aane do
- **Mono, 22050 Hz** best hota hai (auto-convert hota hai)
- Ek hi tone mein bolo — drama mat karo reference mein

---

#### 🎭 Emotion Modes
| Mode | Temperature | Best For |
|------|------------|----------|
| 🧘 Calm | 0.20 | Meditation, slow narration |
| 😊 Normal | 0.35 | YouTube videos, general |
| 🎙️ Pro | 0.50 | Motivational, news |
| 🔥 Dramatic | 0.68 | Story, energetic content |

---

#### 🌐 Trilingual Mode (v3.0 ka naya feature)
```
Hindi:    यह एक Hindi sentence है।
English:  This English part bolega properly.
Sanskrit: Om namaste, dharma aur karma ka path.
```
Sab alag-alag generate hoga — koi stutter nahi!

---

#### 🎛️ EQ Settings — Voice Match Tips
- **Bass +4 to +6 dB** — voice natural lagegi, hollow nahi
- **Mid 0** — chhedo mat jab tak problem na ho
- **Treble -2 dB** — robotic sound hatata hai

---

#### 📖 Custom Dictionary
```
PAISAWALA = पेसावाला
Shriramnag = श्री राम नाग
TTS = टी टी एस
```

---

#### ⚡ Speed Tips
- Speed 1.0 = natural
- Speed 0.9 = thoda slow (clarity better)
- Speed 1.1+ = fast (YouTube Shorts ke liye)
            """)

demo.launch(share=True, show_error=True)
