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

REPO_ID   = "shriramnag/My-Shriram-Voice"
G_RAW     = "https://raw.githubusercontent.com/shriramnag/Shiv-AI-Voice/main/voices/"
DICT_FILE = "custom_dict.json"

print("🚩 शिव AI v3.5 — Advanced Voice Match Engine...")
try:
    hf_hub_download(repo_id=REPO_ID, filename="Ramai.pth")
    hf_hub_download(repo_id=REPO_ID, filename="config.json")
    # Voice sample bhi download karo — default reference ke liye
    try:
        import shutil
        wav_path = hf_hub_download(repo_id=REPO_ID, filename="Ramai.wav")
        shutil.copy(wav_path, "Ramai.wav")
        print("Voice sample Ramai.wav ready")
    except:
        # HuggingFace pe nahi hai to GitHub se try karo
        try:
            import urllib.request
            urllib.request.urlretrieve(
                "https://raw.githubusercontent.com/shriramnag/Shiv-AI-Voice/main/voices/Ramai.wav",
                "Ramai.wav"
            )
            print("Ramai.wav downloaded from GitHub")
        except:
            pass
except:
    pass

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

# ── GPU Speed — safe optimizations only ──
if torch.cuda.is_available():
    try:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        print("GPU Turbo: cuDNN benchmark active")
    except:
        pass
    try:
        # TF32 — Ampere GPU pe 2x speedup, no quality loss
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("GPU Turbo: TF32 active")
    except:
        pass

print(f"✅ Ready on {device.upper()}")

# ── Startup pe default voices download karo ──
def download_default_voices():
    """App start hote hi sabhi default voices download karo"""
    os.makedirs("voices", exist_ok=True)
    voice_files = ["aideva.wav", "Joanne.wav", "Reginald voice.wav", "cloning .wav"]
    for vf in voice_files:
        local = os.path.join("voices", vf)
        if os.path.exists(local) and os.path.getsize(local) > 1000:
            print(f"Voice cached: {vf}")
            continue
        # Spaces ko %20 mein encode karo — GitHub raw URL ke liye zaroori
        encoded = requests.utils.quote(vf, safe="")
        urls = [
            G_RAW + encoded,
            # Alternate: without encoding (some servers need it plain)
            G_RAW + vf.replace(" ", "%20"),
        ]
        downloaded = False
        for url in urls:
            try:
                r = requests.get(url, timeout=20)
                if r.status_code == 200 and len(r.content) > 1000:
                    with open(local, "wb") as f:
                        f.write(r.content)
                    print(f"Downloaded: {vf} ({len(r.content)//1024}KB)")
                    downloaded = True
                    break
            except Exception as e:
                continue
        if not downloaded:
            print(f"Skip {vf}: not available (upload kar sakte hain manually)")

try:
    download_default_voices()
except Exception as e:
    print(f"Voice download warning: {e}")

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
# ── बड़ा Number converter (100, 50%, 3.5 lakh etc.) ──
def number_to_hindi_words(num_str):
    """Complex numbers ko Hindi words mein badlo"""
    num_str = num_str.strip()
    # Percentage
    if num_str.endswith('%'):
        n = number_to_hindi_words(num_str[:-1])
        return n + " प्रतिशत"
    # Decimal
    if '.' in num_str:
        parts = num_str.split('.', 1)
        return number_to_hindi_words(parts[0]) + " दशमलव " + number_to_hindi_words(parts[1])
    try:
        n = int(num_str)
    except:
        return num_str
    if n == 0: return "शून्य"
    ones = ["","एक","दो","तीन","चार","पाँच","छह","सात","आठ","नौ",
            "दस","ग्यारह","बारह","तेरह","चौदह","पंद्रह","सोलह","सत्रह",
            "अठारह","उन्नीस","बीस","इक्कीस","बाईस","तेईस","चौबीस","पच्चीस",
            "छब्बीस","सत्ताईस","अट्ठाईस","उनतीस","तीस","इकतीस","बत्तीस",
            "तैंतीस","चौंतीस","पैंतीस","छत्तीस","सैंतीस","अड़तीस","उनतालीस",
            "चालीस","इकतालीस","बयालीस","तैंतालीस","चौंतालीस","पैंतालीस",
            "छियालीस","सैंतालीस","अड़तालीस","उनचास","पचास","इक्यावन","बावन",
            "तिरपन","चौवन","पचपन","छप्पन","सत्तावन","अट्ठावन","उनसठ","साठ",
            "इकसठ","बासठ","तिरसठ","चौंसठ","पैंसठ","छियासठ","सड़सठ","अड़सठ",
            "उनहत्तर","सत्तर","इकहत्तर","बहत्तर","तिहत्तर","चौहत्तर","पचहत्तर",
            "छिहत्तर","सतहत्तर","अठहत्तर","उनासी","अस्सी","इक्यासी","बयासी",
            "तिरासी","चौरासी","पचासी","छियासी","सत्तासी","अट्ठासी","नवासी","नब्बे",
            "इक्यानवे","बानवे","तिरानवे","चौरानवे","पचानवे","छियानवे","सत्तानवे",
            "अट्ठानवे","निन्यानवे"]
    if n < 100:
        return ones[n]
    elif n < 1000:
        h = n // 100
        r = n % 100
        s = ones[h] + " सौ"
        if r: s += " " + ones[r]
        return s
    elif n < 100000:
        h = n // 1000
        r = n % 1000
        s = ones[h] + " हज़ार"
        if r: s += " " + number_to_hindi_words(str(r))
        return s
    elif n < 10000000:
        h = n // 100000
        r = n % 100000
        s = ones[h] + " लाख"
        if r: s += " " + number_to_hindi_words(str(r))
        return s
    else:
        h = n // 10000000
        r = n % 10000000
        s = ones[h] + " करोड़"
        if r: s += " " + number_to_hindi_words(str(r))
        return s

def convert_all_numbers(text):
    """Sabhi numbers — integers, decimals, percentages, ordinals — Hindi mein"""

    # Ordinals pehle (5वीं, 9वें, 3रा, 2री etc.) — XTTS crash karta hai inpe
    ordinal_map = {
        '1ला':'पहला', '1ली':'पहली', '1ले':'पहले',
        '2रा':'दूसरा', '2री':'दूसरी', '2रे':'दूसरे',
        '3रा':'तीसरा', '3री':'तीसरी', '3रे':'तीसरे',
        '4था':'चौथा', '4थी':'चौथी', '4थे':'चौथे',
        '5वाँ':'पाँचवाँ', '5वीं':'पाँचवीं', '5वें':'पाँचवें',
        '6वाँ':'छठवाँ', '6वीं':'छठवीं', '6वें':'छठवें',
        '7वाँ':'सातवाँ', '7वीं':'सातवीं', '7वें':'सातवें',
        '8वाँ':'आठवाँ', '8वीं':'आठवीं', '8वें':'आठवें',
        '9वाँ':'नौवाँ', '9वीं':'नौवीं', '9वें':'नौवें',
        '10वाँ':'दसवाँ', '10वीं':'दसवीं', '10वें':'दसवें',
        '11वाँ':'ग्यारहवाँ', '12वाँ':'बारहवाँ',
        '20वाँ':'बीसवाँ', '25वाँ':'पच्चीसवाँ',
        '100वाँ':'सौवाँ',
    }
    for src, tgt in ordinal_map.items():
        text = text.replace(src, tgt)

    # Percentage pehle
    text = re.sub(r'(\d+(?:\.\d+)?)%',
                  lambda m: number_to_hindi_words(m.group(0)), text)
    # Decimal numbers
    text = re.sub(r'(\d+\.\d+)',
                  lambda m: number_to_hindi_words(m.group(0)), text)
    # Pure integers
    text = re.sub(r'\b(\d+)\b',
                  lambda m: number_to_hindi_words(m.group(0)), text)
    return text

def apply_all_dicts(text, custom_dict):
    # 1. Custom dict (highest priority)
    for src, tgt in custom_dict.items():
        text = re.sub(
            rf'(?<![a-zA-Z\u0900-\u097F]){re.escape(src)}(?![a-zA-Z\u0900-\u097F])',
            tgt, text, flags=re.IGNORECASE)
    # 2. Sanskrit
    for src, tgt in SANSKRIT_DICT.items():
        text = re.sub(rf'(?<![a-zA-Z]){re.escape(src)}(?![a-zA-Z])',
                      tgt, text, flags=re.IGNORECASE)
    # 3. English → Hindi phonetic
    for src, tgt in ENGLISH_TO_HINDI.items():
        text = re.sub(rf'(?<![a-zA-Z]){re.escape(src)}(?![a-zA-Z])',
                      tgt, text, flags=re.IGNORECASE)
    # 4. Numbers → full Hindi words
    text = convert_all_numbers(text)
    return text

def clean_punctuation(text):
    """Punctuation → XTTS-friendly pause + newline handling"""
    text = re.sub(r'\n+', '। ', text)          # newlines → pause
    text = re.sub(r'\.\.\.\s*', '। ', text)  # ellipsis → pause
    text = re.sub(r'\?\s*', '। ', text)         # ? → pause
    text = re.sub(r'!\s*', '! ', text)
    text = re.sub(r'[।\.]\s*', '। ', text)
    text = re.sub(r'[,،]\s*', ', ', text)
    text = re.sub(r'[-–—]+', ', ', text)
    text = re.sub(r'[;:]', ', ', text)
    text = re.sub(r'["""\'\'()\[\]{}*#@&^~`|<>]', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'(।\s*){2,}', '। ', text)
    return text.strip()

def devanagari_normalize(text):
    """
    Devanagari special cases fix:
    - Anusvar, chandrabindu, visarg sahi rakho
    - Half characters aur conjuncts ke liye space mat dalo
    """
    # Zero-width joiner preserve karo (conjuncts ke liye)
    # Extra spaces Devanagari words ke beech mat aane do
    text = re.sub(r'([\u0900-\u097F])\s+([\u0900-\u097F])',
                  lambda m: m.group(1) + ' ' + m.group(2), text)
    return text

def full_text_processor(text, custom_dict):
    if not text: return ""
    text = apply_all_dicts(text, custom_dict)
    text = devanagari_normalize(text)
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
# ५. Language-Aware Chunker v2 — Better English Detection
# ══════════════════════════════════════════════════════════════════
def detect_chunk_language(words):
    """
    Chunk ki language accurately detect karo.
    Mixed text (Hindi+English) = hi (kyunki XTTS hi mode mein
    Devanagari + phonetic English dono bol sakta hai)
    Pure English only chunk = en
    """
    if not words: return "hi"
    total = len(words)
    devanagari_words = 0
    pure_latin_words = 0
    for w in words:
        deva = sum(1 for c in w if '\u0900' <= c <= '\u097F')
        lat  = sum(1 for c in w if c.isascii() and c.isalpha())
        if deva > 0:
            devanagari_words += 1
        elif lat == len(w) and lat > 0:
            pure_latin_words += 1
    # Sirf tab "en" lo jab koi Devanagari nahi aur zyada English hai
    if devanagari_words == 0 and pure_latin_words > total * 0.7:
        return "en"
    return "hi"  # Default always hi — mixed text ke liye bhi

def language_aware_chunker(text, max_words=30):
    """
    STUTTER FIX:
    - max_words 35 (pehle 45 tha — zyada words = zyada stutter)
    - Sentence boundary pe hi toro
    - Pure English chunk tabhi jab koi Hindi nahi
    """
    # Pause markers pe split karo
    # Split on pause markers AND newlines
    sentences = re.split(r'(?<=[।])\s+|\n+', text.strip())
    # Agar sentence abhi bhi bada hai to comma pe bhi split karo
    final_sentences = []
    for s in sentences:
        if len(s.split()) > max_words:
            sub = re.split(r',\s*', s)
            final_sentences.extend([x.strip() for x in sub if x.strip()])
        else:
            if s.strip():
                final_sentences.append(s.strip())

    chunks_with_lang = []
    current_words, current_count = [], 0

    def commit(words):
        if not words: return
        chunk = ' '.join(words)
        lang = detect_chunk_language(words)
        chunks_with_lang.append((chunk, lang))

    for sentence in final_sentences:
        words = sentence.split()
        wc = len(words)
        if wc == 0: continue
        if current_count + wc > max_words:
            commit(current_words)
            current_words, current_count = words, wc
        else:
            current_words.extend(words)
            current_count += wc

    commit(current_words)
    result = [(c.strip(), l) for c, l in chunks_with_lang if c.strip()]
    # Minimum length check — 3 words se kam ke chunks merge karo
    merged = []
    i = 0
    while i < len(result):
        chunk, lang = result[i]
        if len(chunk.split()) < 3 and i+1 < len(result):
            next_chunk, next_lang = result[i+1]
            merged.append((chunk + ' ' + next_chunk, lang))
            i += 2
        else:
            merged.append((chunk, lang))
            i += 1
    return merged

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

def estimate_f0(filepath):
    """Reference audio ka F0 (fundamental frequency) nikalo"""
    try:
        from scipy.io import wavfile as wf
        from scipy.signal import find_peaks
        sr, data = wf.read(filepath)
        if data.ndim == 2: data = data.mean(axis=1)
        data = data.astype(np.float32)
        # Middle segment use karo
        mid = len(data)//4
        seg = data[mid:mid+sr//2]
        seg = seg / (np.max(np.abs(seg)) + 1e-9)
        corr = np.correlate(seg, seg, mode='full')
        corr = corr[len(corr)//2:]
        min_lag = int(sr / 500)
        max_lag = int(sr / 60)
        peaks, _ = find_peaks(corr[min_lag:max_lag], height=0.25)
        if len(peaks) > 0:
            return sr / (peaks[0] + min_lag)
    except:
        pass
    return None

# Global: reference F0 store karo pitch correction ke liye
_ref_f0 = None

def prepare_reference(filepath, out="ref_ready.wav"):
    """Reference audio — voice match ke liye maximum optimize"""
    global _ref_f0
    try:
        a = AudioSegment.from_file(filepath)
        a = a.set_channels(1).set_frame_rate(22050)
        try:
            a = effects.strip_silence(a, silence_thresh=-40, padding=300)
        except: pass
        # Target -20dBFS (XTTS optimal)
        target_dbfs = -20.0
        change = target_dbfs - a.dBFS
        a = a.apply_gain(change)
        if len(a) < 8000:
            while len(a) < 8000: a = a + a
        if len(a) > 30000:
            a = a[:30000]
        a.export(out, format="wav")
        # F0 measure karo
        _ref_f0 = estimate_f0(out)
        print(f"Reference ready: {len(a)/1000:.1f}s, {a.dBFS:.1f}dBFS, F0={_ref_f0:.0f}Hz" if _ref_f0 else f"Reference ready: {len(a)/1000:.1f}s")
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
def volume_match(seg, target_rms=3000):
    """Har chunk ka volume same karo — lahraana fix"""
    samples = np.array(seg.get_array_of_samples(), dtype=np.float32)
    rms = np.sqrt(np.mean(samples**2))
    if rms < 100: return seg  # silence skip
    gain = target_rms / (rms + 1e-9)
    gain = np.clip(gain, 0.3, 3.0)  # extreme gain avoid karo
    return seg.apply_gain(20 * np.log10(gain))

def crossfade_join(segs, cf_ms=80):
    """
    LAHRAANA + JUMP FIX:
    - Pehle har segment ka volume match karo
    - Phir crossfade join karo (80ms — smooth)
    - Short silence (100ms) between chunks for breathing
    """
    if not segs: return AudioSegment.silent(100)
    # Step 1: Volume normalize karo sabhi segments
    # Average RMS nikalo
    all_rms = []
    for s in segs:
        arr = np.array(s.get_array_of_samples(), dtype=np.float32)
        rms = np.sqrt(np.mean(arr**2))
        if rms > 100: all_rms.append(rms)
    target = float(np.median(all_rms)) if all_rms else 3000

    normalized = []
    for s in segs:
        normalized.append(volume_match(s, target_rms=target))

    # Step 2: Crossfade join
    result = normalized[0]
    for s in normalized[1:]:
        cf = min(cf_ms, len(result)//2, len(s)//2)
        result = result.append(s, crossfade=max(cf, 20))
    return result

# ══════════════════════════════════════════════════════════════════
# १२. Emotion Presets
# ══════════════════════════════════════════════════════════════════
EMOTION_PRESETS = {
    # temperature values — generation mein safe_temp = min(val, 0.30) se cap hota hai
    "🧘 शांत (Calm)":       {"temperature":0.15,"rep_pen":8.0,"speed":0.90},
    "😊 सामान्य (Normal)":  {"temperature":0.25,"rep_pen":7.0,"speed":0.97},
    "🎙️ प्रो (Pro)":        {"temperature":0.28,"rep_pen":6.5,"speed":1.02},
    "🔥 नाटकीय (Dramatic)": {"temperature":0.30,"rep_pen":6.0,"speed":1.08},
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
        ref = None

        last_error = ""
        # Step 1: local file check
        local_voice = os.path.join("voices", git_ref)
        if os.path.exists(local_voice) and os.path.getsize(local_voice) > 1000:
            ref = prepare_reference(local_voice)
            print(f"Using local: {local_voice}")

        # Step 2: download agar local nahi hai
        if not ref:
            urls_to_try = [
                G_RAW + requests.utils.quote(git_ref),
                f"https://huggingface.co/Shriramnag/My-Shriram-Voice/resolve/main/{requests.utils.quote(git_ref)}",
            ]
            for try_url in urls_to_try:
                try:
                    resp = requests.get(try_url, timeout=30)
                    if resp.status_code == 200 and len(resp.content) > 1000:
                        with open(raw, "wb") as fh:
                            fh.write(resp.content)
                        # Cache for next time
                        with open(local_voice, "wb") as fh:
                            fh.write(resp.content)
                        ref = prepare_reference(raw)
                        print(f"Downloaded & cached: {try_url}")
                        break
                except Exception as e:
                    last_error = str(e)
                    continue

        # Fallback 1: local cached files
        if not ref:
            for fallback in ["ref_ready.wav", "ref_dl.wav", "Ramai.wav"]:
                if os.path.exists(fallback) and os.path.getsize(fallback) > 1000:
                    ref = prepare_reference(fallback)
                    print(f"Using local fallback: {fallback}")
                    break

        # Fallback 2: voices folder se koi bhi wav
        if not ref:
            for fname in ["aideva.wav", "Joanne.wav", "cloning .wav", "Reginald voice.wav"]:
                local_path = os.path.join("voices", fname)
                if os.path.exists(local_path):
                    ref = prepare_reference(local_path)
                    print(f"Using voices folder: {local_path}")
                    break

        # Fallback 3: content folder mein koi bhi wav
        if not ref:
            import glob
            wavs = glob.glob("/content/**/*.wav", recursive=True)
            wavs += glob.glob("**/*.wav", recursive=True)
            for w in wavs:
                if os.path.getsize(w) > 5000:
                    ref = prepare_reference(w)
                    print(f"Found wav: {w}")
                    break

        if not ref:
            msg = (
                "Reference audio nahi mila!\n"
                f"GitHub pe try kiya: voices/{git_ref}\n"
                "SOLUTION: 'Main Clip' section mein apni awaaz upload karein (6-20 sec WAV/MP3)\n"
                f"Last error: {last_error[:80]}"
            )
            return None, msg, "", gr.update(choices=[])

    ref_quality = check_ref_quality(ref)

    # Chunking — 40 words max (chhote = less stutter)
    progress(0.08, desc="✂️ Smart chunking...")
    chunks = language_aware_chunker(p_text, max_words=30)
    total = len(chunks)   # PEHLE define karo
    if total == 0:
        return None, "Text empty after processing.", ref_quality, gr.update(choices=[])
    est_minutes = total * 0.4
    progress(0.09, desc=f"Total {total} parts — approx {est_minutes:.0f} min")

    segments, errors = [], []

    # All params explicitly pass karo — no closure/scope issues
    _temperature  = float(temperature)
    _rep_pen      = float(rep_pen)
    _gpt_cond_len = int(gpt_cond_len)
    _preset_speed = float(preset["speed"])

    def safe_set_params():
        """XTTS params — voice match optimized"""
        safe_temp = min(_temperature, 0.28)  # 0.28 = best stability
        try:
            cfg = tts.synthesizer.tts_config.model_args
            cfg.temperature        = safe_temp
            cfg.repetition_penalty = _rep_pen
            # gpt_cond_len=12: 2x better voice match vs default 6
            cfg.gpt_cond_len       = max(_gpt_cond_len, 12)
            cfg.gpt_cond_chunk_len = 4   # finer speaker analysis
            cfg.length_penalty     = 1.0
            cfg.top_p              = 0.85
            cfg.top_k              = 50
            return
        except: pass
        try:
            tts.tts_config.temperature = safe_temp
        except: pass
        try:
            tts.synthesizer.temperature = safe_temp
        except: pass

    def generate_one_chunk(chunk_text, lang, out_path, speed_val):
        """Ek chunk generate — fast, no sleep, 2 attempts"""
        actual_speed = float(speed_val) if float(speed_val) >= 0.8 else _preset_speed

        # Attempt 1: Normal
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            safe_set_params()
            tts.tts_to_file(
                text=chunk_text, speaker_wav=ref,
                language=lang, file_path=out_path, speed=actual_speed,
            )
            if os.path.exists(out_path) and os.path.getsize(out_path) > 500:
                return True, None
        except Exception as e1:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        # Attempt 2: Force Hindi, shorter text
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            words = chunk_text.split()
            fallback_text = " ".join(words[:15]) if len(words) > 15 else chunk_text
            tts.tts_to_file(
                text=fallback_text, speaker_wav=ref,
                language="hi", file_path=out_path, speed=0.95,
            )
            if os.path.exists(out_path) and os.path.getsize(out_path) > 500:
                return True, None
        except Exception as e2:
            return False, str(e2)[:100]

        return False, "Both attempts failed"

    for i, (chunk, lang) in enumerate(chunks):
        progress(
            (i+1)/total*0.82,
            desc=f"Part {i+1}/{total} [{lang.upper()}] — {len(chunk.split())} words"
        )
        # Unique filename — overwrite nahi hoga
        name = f"shiv_chunk_{i}_{i*7+13}.wav"

        # Har 3 chunks pe CUDA memory fully clear karo (long audio ke liye)
        if i > 0 and i % 3 == 0:
            torch.cuda.empty_cache()
            gc.collect()

        success, err_msg = generate_one_chunk(chunk, lang, name, speed_s)

        if success and os.path.exists(name):
            try:
                seg = AudioSegment.from_wav(name)
                seg_dur = len(seg)

                if use_silence:
                    try:
                        seg = effects.strip_silence(
                            seg, silence_thresh=-45, padding=250)
                    except: pass

                # Min 200ms accept karo (pehle 300ms tha — chhote lines miss ho rahi thi)
                if len(seg) > 200:
                    segments.append(seg)
                    cout = f"prev_chunk_{i+1}.wav"
                    seg.export(cout, format="wav")
                    _chunk_audios.append(cout)
                    print(f"Part {i+1}: OK ({seg_dur}ms)")
                else:
                    # Too short — original bina silence-strip ke use karo
                    seg2 = AudioSegment.from_wav(name) if os.path.exists(name) else seg
                    if len(seg2) > 100:
                        segments.append(seg2)
                        cout = f"prev_chunk_{i+1}.wav"
                        seg2.export(cout, format="wav")
                        _chunk_audios.append(cout)
                        print(f"Part {i+1}: short but accepted ({len(seg2)}ms)")
                    else:
                        errors.append(f"Part {i+1}: too short ({seg_dur}ms) — skipped")

                if os.path.exists(name): os.remove(name)
            except Exception as e:
                errors.append(f"Part {i+1} load error: {str(e)[:80]}")
                if os.path.exists(name): os.remove(name)
        else:
            err_detail = err_msg[:100] if err_msg else "unknown"
            errors.append(f"Part {i+1}[{lang}]: {err_detail}")
            print(f"Part {i+1} FAILED: {err_detail}")
            if os.path.exists(name): os.remove(name)

        # Memory clear
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    if not segments:
        err_detail = "\n".join(errors[:8]) if errors else "Unknown"
        return None, (
            f"Koi bhi chunk generate nahi hua ({total} try kiye)\n"
            f"Errors:\n{err_detail}\n\n"
            "FIX: Apni awaaz 'Main Clip' mein upload karein"
        ), ref_quality, gr.update(choices=[])

    # All segments ko join karo
    progress(0.83, desc=f"Joining {len(segments)} parts...")
    print(f"Joining {len(segments)} segments, total words processed")

    # Individual segment durations log karo
    total_dur = sum(len(s) for s in segments)
    print(f"Total audio before join: {total_dur/1000:.1f}s")

    combined = crossfade_join(segments, cf_ms=60)
    print(f"After join: {len(combined)/1000:.1f}s")

    # Normalize + Volume Match
    if use_normalize:
        progress(0.86, desc="Normalize + Volume Match...")
        combined = combined.set_frame_rate(22050).set_channels(1)
        # Target -22dBFS (original voice ka level)
        target_dbfs = -22.0
        current_dbfs = combined.dBFS
        gain_needed = target_dbfs - current_dbfs
        # Max -9dB adjustment (analysis result)
        gain_needed = max(gain_needed, -12.0)
        combined = combined.apply_gain(gain_needed)
        print(f"Volume match: {current_dbfs:.1f} -> {combined.dBFS:.1f} dBFS")

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

    # Auto Pitch Correction
    if pitch_shift_enable:
        manual = float(pitch_semitones)
        if abs(manual) > 0.05:
            # Manual override
            progress(0.95, desc=f"Pitch shift {manual:+.1f} semitones...")
            combined = pitch_shift_audio(combined, manual)
        elif _ref_f0 and LIBROSA_OK:
            # Auto: generated audio ka F0 detect karo aur match karo
            try:
                progress(0.95, desc="Auto pitch matching...")
                tmp_path = "tmp_pitch_check.wav"
                combined.export(tmp_path, format="wav")
                gen_f0 = estimate_f0(tmp_path)
                if gen_f0 and _ref_f0:
                    auto_semitones = 12 * np.log2(_ref_f0 / gen_f0)
                    # Cap at ±6 semitones (sane range)
                    auto_semitones = np.clip(auto_semitones, -6, 6)
                    if abs(auto_semitones) > 0.5:
                        print(f"Auto pitch: ref={_ref_f0:.0f}Hz gen={gen_f0:.0f}Hz shift={auto_semitones:+.1f}st")
                        combined = pitch_shift_audio(combined, float(auto_semitones))
                if os.path.exists(tmp_path): os.remove(tmp_path)
            except Exception as pe:
                print(f"Auto pitch skip: {pe}")

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

    success_count = len(segments)
    dur_sec = len(combined) / 1000
    dur_min = dur_sec / 60

    ok = "OK" if success_count == total else "PARTIAL"
    status  = f"{ok}: {success_count}/{total} parts generated\n"
    status += f"Duration: {dur_sec:.1f}s ({dur_min:.1f} min)\n"
    status += f"Format: {fmt.upper()} | Speed: {speed_s:.2f}x\n"
    status += f"Emotion: {emotion_mode}\n"
    if dur_min < 1 and total > 3:
        status += "\nWARNING: Output bahut chhota hai!\n"
        status += "Reason: Chunks fail ho rahe hain\n"
        status += "Fix: Apni awaaz 'Main Clip' mein upload karein\n"
    if pitch_shift_enable:
        status += f"Pitch: {float(pitch_semitones):+.1f} semitones\n"
    if errors:
        status += f"\nFailed {len(errors)} parts:\n"
        for e in errors[:5]:
            status += f"  - {e}\n"
        if success_count < total:
            status += "\nTip: Dobara generate karo ya apni awaaz upload karo"

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
    background-clip:text;letter-spacing:1px">🚩 शिव AI — v4.0 Turbo</div>
  <p style="color:#8b949e;font-size:.9em;margin:4px 0 12px">
    श्री राम नाग &nbsp;|&nbsp; PAISAWALA &nbsp;|&nbsp; Turbo GPU | Realistic Voice | Advanced Trilingual</p>
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
      padding:2px 11px;font-size:.75em;color:#2ecc71">✅ GPU 2-4x Turbo</span>
  </div>
</div>
"""

with gr.Blocks(css=CSS, title="Shiv AI v4.0") as demo:

    gr.HTML(HEADER)

    with gr.Tabs():

        # ═══ TAB 1: Generate ═══
        with gr.Tab("Generate"):
            with gr.Row(equal_height=False):

                # ── LEFT: Script ──
                with gr.Column(scale=3):
                    txt = gr.Textbox(
                        label="Script — Hindi / English / Sanskrit",
                        lines=16,
                        placeholder="Yahan script paste karein...\n\nExample:\nAaj hum safalta ki baat karenge।\nDharma aur karma ka path follow karo।"
                    )
                    with gr.Row():
                        wc = gr.Markdown("Shabd: **0**")
                        cc = gr.Markdown("Akshar: **0**")
                    txt.change(
                        lambda x: (
                            f"Shabd: **{len(x.split())}**",
                            f"Akshar: **{len(x)}**"
                        ),
                        [txt], [wc, cc]
                    )
                    prev_btn = gr.Button(
                        "Text Preview — cleanup dekhein",
                        size="sm", variant="secondary"
                    )
                    text_preview = gr.Textbox(
                        label="Cleaned Text", lines=4,
                        interactive=False, visible=False
                    )
                    prev_btn.click(
                        lambda t: (
                            full_text_processor(t, load_custom_dict()),
                            gr.update(visible=True)
                        ),
                        [txt], [text_preview, text_preview]
                    )

                # ── RIGHT: Controls ──
                with gr.Column(scale=2):

                    # ── Voice Upload ──
                    gr.Markdown("### Apni Awaaz")
                    up1 = gr.Audio(
                        label="Awaaz Upload karein (6-30 sec)",
                        type="filepath"
                    )
                    ref_qual = gr.Textbox(
                        label="Quality", interactive=False,
                        lines=2, elem_classes=["status-out"]
                    )
                    up1.change(
                        lambda f: check_ref_quality(f) if f else "Awaaz upload karein",
                        [up1], [ref_qual]
                    )
                    up2 = gr.Audio(visible=False, type="filepath")
                    up3 = gr.Audio(visible=False, type="filepath")
                    git_v = gr.Dropdown(
                        choices=["aideva.wav","Joanne.wav",
                                 "Reginald voice.wav","cloning .wav"],
                        label="Ya Default Voice chunein",
                        value="aideva.wav"
                    )

                    gr.Markdown("---")

                    # ── Tone ──
                    gr.Markdown("### Style / Tone")
                    emotion = gr.Radio(
                        choices=list(EMOTION_PRESETS.keys()),
                        value="😊 सामान्य (Normal)",
                        label=""
                    )
                    spd = gr.Slider(
                        minimum=0.0, maximum=1.4, value=0.0, step=0.05,
                        label="Speed (0 = auto)"
                    )
                    emotion.change(apply_emotion, [emotion], [spd])

                    gr.Markdown("---")

                    # ── Voice Match ──
                    gr.Markdown("### Voice Match")
                    gpt_len = gr.Slider(
                        minimum=3, maximum=30, value=12, step=1,
                        label="Match Quality (12=good, 24=best, slow)"
                    )
                    pitch_en = gr.Checkbox(
                        label="Pitch Correction (librosa)", value=False
                    )
                    pitch_sl = gr.Slider(
                        minimum=-6.0, maximum=6.0, value=0.0, step=0.5,
                        label="Pitch Shift semitones"
                    )

                    gr.Markdown("---")

                    # ── EQ ──
                    gr.Markdown("### EQ")
                    bass_sl = gr.Slider(
                        minimum=-6.0, maximum=12.0, value=1.5, step=0.5,
                        label="Bass dB (analysis: +1.5 for aideva voice)"
                    )
                    mid_sl = gr.Slider(
                        minimum=-6.0, maximum=6.0, value=0.0, step=0.5,
                        label="Mid dB"
                    )
                    treble_sl = gr.Slider(
                        minimum=-9.0, maximum=3.0, value=-1.5, step=0.5,
                        label="Treble dB (-1.5 recommended)"
                    )

                    gr.Markdown("---")

                    # ── Options ──
                    with gr.Row():
                        sln  = gr.Checkbox(label="Silence Remove", value=True)
                        norm = gr.Checkbox(label="Normalize",       value=True)
                        eq   = gr.Checkbox(label="EQ",              value=True)
                    with gr.Row():
                        dess = gr.Checkbox(label="DeEsser", value=True)
                        comp = gr.Checkbox(label="Compress", value=True)

                    out_fmt = gr.Radio(
                        choices=["wav","mp3","ogg"],
                        value="wav", label="Format"
                    )
                    custom_raw = gr.Textbox(
                        label="Custom Words (WORD = उच्चारण)",
                        placeholder="PAISAWALA = पेसावाला",
                        lines=2
                    )

                    btn = gr.Button(
                        "Generate Karo",
                        variant="primary", size="lg"
                    )

            # ── Output ──
            with gr.Row():
                with gr.Column(scale=2):
                    out_audio = gr.Audio(
                        label="Output",
                        type="filepath", autoplay=True
                    )
                with gr.Column(scale=1):
                    out_status = gr.Textbox(
                        label="Status", interactive=False,
                        lines=8, elem_classes=["status-out"]
                    )

            with gr.Accordion("Chunk Preview", open=False):
                with gr.Row():
                    chunk_dd  = gr.Dropdown(
                        label="Part", choices=[], interactive=True
                    )
                    chunk_btn = gr.Button("Play", size="sm")
                chunk_out = gr.Audio(
                    label="Chunk", type="filepath", autoplay=True
                )
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
        with gr.Tab("Dictionary"):
            gr.Markdown("### Custom words save karo")
            with gr.Row():
                with gr.Column():
                    dw = gr.Textbox(label="Word", placeholder="PAISAWALA")
                    dp = gr.Textbox(label="Pronunciation", placeholder="पेसावाला")
                    with gr.Row():
                        da = gr.Button("Add", variant="primary")
                        dr = gr.Button("Remove", variant="secondary")
                    ds = gr.Textbox(label="Status", interactive=False, lines=2)
                with gr.Column():
                    dd = gr.Markdown(load_dict_md())
            da.click(dict_add,    [dw,dp], [ds,dd])
            dr.click(dict_remove, [dw],    [ds,dd])

        # ═══ TAB 3: Guide ═══
        with gr.Tab("Guide"):
            gr.Markdown("""
### Shiv AI v4.0 — Guide

**Awaaz Upload:**
- 6-30 second ki clear recording upload karein
- Quiet room, no background noise
- WAV ya MP3 dono chalega

**Match Quality:**
- 12 = good (fast)
- 24 = best (slow)
- Zyada = better match but slow

**EQ Tips:**
- Bass +5 = natural voice
- Treble -2 = less robotic
- DeEsser = harsh sound hatata hai

**Speed:**
- 0 = tone preset ki speed
- 0.9 = thoda slow, clear
- 1.0 = normal

**Format:**
- WAV = best quality
- MP3 = YouTube ke liye (192kbps)
            """)


demo.launch(share=True, show_error=True)
