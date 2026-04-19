"""
Shiv AI v4.0 — Professional Voice Cloning
PAISAWALA | Shri Ram Nag
"""

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

# ─────────────────────────────────────────────
# 1. DEVICE + MODEL
# ─────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark     = True
    torch.backends.cudnn.deterministic = False
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32       = True
    except: pass

REPO_ID   = "Shriramnag/My-Shriram-Voice"
G_RAW     = "https://raw.githubusercontent.com/shriramnag/Shiv-AI-Voice/main/voices/"
DICT_FILE = "custom_dict.json"

print("Shiv AI v4.0 loading...")
try:
    hf_hub_download(repo_id=REPO_ID, filename="Ramai.pth")
    hf_hub_download(repo_id=REPO_ID, filename="config.json")
except: pass

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
print(f"Model ready: {device.upper()}")

try:
    import librosa, librosa.effects
    LIBROSA = True
    print("librosa ready")
except:
    LIBROSA = False

# ─────────────────────────────────────────────
# 2. STARTUP — download default voices
# ─────────────────────────────────────────────
os.makedirs("voices", exist_ok=True)

def _dl(name):
    p = os.path.join("voices", name)
    if os.path.exists(p) and os.path.getsize(p) > 1000:
        return
    for url in [G_RAW + name.replace(" ","%20"),
                G_RAW + requests.utils.quote(name, safe="")]:
        try:
            r = requests.get(url, timeout=15)
            if r.status_code == 200 and len(r.content) > 1000:
                with open(p,"wb") as f: f.write(r.content)
                print(f"Downloaded: {name}"); return
        except: pass

for _v in ["aideva.wav","Joanne.wav","Reginald voice.wav","cloning .wav"]:
    _dl(_v)

# ─────────────────────────────────────────────
# 3. DICTIONARY
# ─────────────────────────────────────────────
def load_dict():
    try:
        if os.path.exists(DICT_FILE):
            with open(DICT_FILE, encoding="utf-8") as f: return json.load(f)
    except: pass
    return {}

def save_dict(d):
    with open(DICT_FILE,"w",encoding="utf-8") as f: json.dump(d,f,ensure_ascii=False,indent=2)

def load_dict_md():
    d = load_dict()
    return "\n".join(f"**{k}** → {v}" for k,v in d.items()) if d else "Dictionary khaali hai."

# ─────────────────────────────────────────────
# 4. TEXT PROCESSING
# ─────────────────────────────────────────────
SANSKRIT = {"dharma":"धर्म","karma":"कर्म","yoga":"योग","shakti":"शक्ति","om":"ॐ","aum":"ॐ","namaste":"नमस्ते","guru":"गुरु","mantra":"मंत्र","atma":"आत्मा","maya":"माया","moksha":"मोक्ष","ahimsa":"अहिंसा","satya":"सत्य","seva":"सेवा","bhakti":"भक्ति","veda":"वेद","pooja":"पूजा","puja":"पूजा"}
ENGLISH  = {"AI":"ए आई","ML":"एम एल","API":"ए पी आई","GPU":"जी पी यू","YouTube":"यूट्यूब","Instagram":"इंस्टाग्राम","Facebook":"फेसबुक","WhatsApp":"व्हाट्सऐप","Google":"गूगल","Internet":"इंटरनेट","Online":"ऑनलाइन","Software":"सॉफ्टवेयर","Computer":"कंप्यूटर","Mobile":"मोबाइल","App":"ऐप","Website":"वेबसाइट","Download":"डाउनलोड","Upload":"अपलोड","Channel":"चैनल","Video":"वीडियो","Content":"कंटेंट","Subscribe":"सब्सक्राइब","Life":"लाइफ","Dream":"ड्रीम","Mindset":"माइंडसेट","Success":"सक्सेस","Fail":"फेल","Goal":"गोल","Focus":"फोकस","Power":"पावर","Money":"मनी","Business":"बिज़नेस","Market":"मार्केट","Brand":"ब्रांड","Smart":"स्मार्ट","Team":"टीम","Plan":"प्लान","Skill":"स्किल","OK":"ओके","okay":"ओके","hello":"हेलो","bye":"बाय","thanks":"थैंक्स"}
ORDINALS = {"1ला":"पहला","1ली":"पहली","2रा":"दूसरा","2री":"दूसरी","3रा":"तीसरा","4था":"चौथा","5वीं":"पाँचवीं","5वें":"पाँचवें","6वें":"छठवें","7वें":"सातवें","8वीं":"आठवीं","8वें":"आठवें","9वें":"नौवें","9वीं":"नौवीं","10वें":"दसवें"}

ONES = ["","एक","दो","तीन","चार","पाँच","छह","सात","आठ","नौ","दस","ग्यारह","बारह","तेरह","चौदह","पंद्रह","सोलह","सत्रह","अठारह","उन्नीस","बीस","इक्कीस","बाईस","तेईस","चौबीस","पच्चीस","छब्बीस","सत्ताईस","अट्ठाईस","उनतीस","तीस","इकतीस","बत्तीस","तैंतीस","चौंतीस","पैंतीस","छत्तीस","सैंतीस","अड़तीस","उनतालीस","चालीस","इकतालीस","बयालीस","तैंतालीस","चौंतालीस","पैंतालीस","छियालीस","सैंतालीस","अड़तालीस","उनचास","पचास","इक्यावन","बावन","तिरपन","चौवन","पचपन","छप्पन","सत्तावन","अट्ठावन","उनसठ","साठ","इकसठ","बासठ","तिरसठ","चौंसठ","पैंसठ","छियासठ","सड़सठ","अड़सठ","उनहत्तर","सत्तर","इकहत्तर","बहत्तर","तिहत्तर","चौहत्तर","पचहत्तर","छिहत्तर","सतहत्तर","अठहत्तर","उनासी","अस्सी","इक्यासी","बयासी","तिरासी","चौरासी","पचासी","छियासी","सत्तासी","अट्ठासी","नवासी","नब्बे","इक्यानवे","बानवे","तिरानवे","चौरानवे","पचानवे","छियानवे","सत्तानवे","अट्ठानवे","निन्यानवे"]

def n2h(n):
    if n==0: return "शून्य"
    if n<0: return "ऋण "+n2h(-n)
    if n<=99: return ONES[n]
    if n<1000:
        h,r = divmod(n,100)
        return ONES[h]+" सौ"+((" "+ONES[r]) if r else "")
    if n<100000:
        h,r = divmod(n,1000)
        return ONES[h]+" हज़ार"+((" "+n2h(r)) if r else "")
    if n<10000000:
        h,r = divmod(n,100000)
        return ONES[h]+" लाख"+((" "+n2h(r)) if r else "")
    h,r = divmod(n,10000000)
    return ONES[h]+" करोड़"+((" "+n2h(r)) if r else "")

def convert_nums(text):
    for k,v in ORDINALS.items(): text = text.replace(k,v)
    text = re.sub(r'(\d+(?:\.\d+)?)%', lambda m: n2h(int(float(m.group(1))))+" प्रतिशत", text)
    text = re.sub(r'(\d+)\.(\d+)', lambda m: n2h(int(m.group(1)))+" दशमलव "+n2h(int(m.group(2))), text)
    text = re.sub(r'\b(\d+)\b', lambda m: n2h(int(m.group(1))), text)
    return text

def apply_dicts(text, custom):
    for k,v in custom.items():
        text = re.sub(rf'(?<![a-zA-Z\u0900-\u097F]){re.escape(k)}(?![a-zA-Z\u0900-\u097F])', v, text, flags=re.IGNORECASE)
    for k,v in SANSKRIT.items():
        text = re.sub(rf'(?<![a-zA-Z]){re.escape(k)}(?![a-zA-Z])', v, text, flags=re.IGNORECASE)
    for k,v in ENGLISH.items():
        text = re.sub(rf'(?<![a-zA-Z]){re.escape(k)}(?![a-zA-Z])', v, text, flags=re.IGNORECASE)
    return text

def clean(text):
    text = re.sub(r'\n+',     '। ',  text)
    text = re.sub(r'\.\.\.+', '। ',  text)
    text = re.sub(r'\?+',     '। ',  text)
    text = re.sub(r'!+',      '! ',  text)
    text = re.sub(r'[।\.]+',  '। ',  text)
    text = re.sub(r'[,،]+',   ', ',  text)
    text = re.sub(r'[-–—]+',  ', ',  text)
    text = re.sub(r'[;:]',    ', ',  text)
    text = re.sub(r'["""\'\'()\[\]{}*#@&^~`|<>]', '', text)
    text = re.sub(r'\s+',     ' ',   text)
    text = re.sub(r'(। ){2,}','। ',  text)
    return text.strip()

def process(text, custom):
    if not text: return ""
    text = apply_dicts(text, custom)
    text = convert_nums(text)
    text = clean(text)
    return text

# ─────────────────────────────────────────────
# 5. CHUNKER
# ─────────────────────────────────────────────
def detect_lang(words):
    if not words: return "hi"
    deva = sum(1 for w in words for c in w if '\u0900'<=c<='\u097F')
    if deva == 0:
        lat = sum(1 for w in words if w.isalpha() and w.isascii())
        if lat > len(words)*0.7: return "en"
    return "hi"

def chunker(text, max_w=30):
    sents = [s.strip() for s in re.split(r'(?<=।)\s+|\n+', text.strip()) if s.strip()]
    result, buf = [], []

    def flush():
        if buf:
            result.append((" ".join(buf), detect_lang(buf)))
            buf.clear()

    for sent in sents:
        words = sent.split()
        if not words: continue
        if len(words) > max_w:
            flush()
            parts = re.split(r',\s*', sent)
            tmp = []
            for p in parts:
                pw = p.split()
                if len(tmp)+len(pw) > max_w and tmp:
                    result.append((" ".join(tmp), detect_lang(tmp)))
                    tmp = pw
                else: tmp.extend(pw)
            if tmp: result.append((" ".join(tmp), detect_lang(tmp)))
        elif len(buf)+len(words) > max_w:
            flush(); buf.extend(words)
        else: buf.extend(words)
    flush()

    # Merge micro chunks
    merged = []
    for chunk, lang in result:
        if len(chunk.split()) < 3 and merged:
            merged[-1] = (merged[-1][0]+" "+chunk, merged[-1][1])
        else: merged.append((chunk, lang))
    return merged

# ─────────────────────────────────────────────
# 6. REFERENCE AUDIO
# ─────────────────────────────────────────────
_ref_f0 = None

def get_f0(path):
    try:
        s, d = wavfile.read(path)
        if d.ndim==2: d=d.mean(axis=1)
        d=d.astype(np.float32)
        mid=len(d)//4; seg=d[mid:mid+s//2]
        seg/=(np.max(np.abs(seg))+1e-9)
        corr=np.correlate(seg,seg,mode='full')[len(seg)-1:]
        lo,hi=int(s/500),int(s/60)
        peaks,_=find_peaks(corr[lo:hi],height=0.25)
        if len(peaks): return s/(peaks[0]+lo)
    except: pass
    return None

def prep_ref(path, out="ref_ready.wav"):
    global _ref_f0
    try:
        a = AudioSegment.from_file(path)
        a = a.set_channels(1).set_frame_rate(22050)
        try: a = effects.strip_silence(a, silence_thresh=-40, padding=300)
        except: pass
        a = a.apply_gain(-18.0 - a.dBFS)
        while len(a) < 8000: a = a+a
        if len(a) > 30000: a = a[:30000]
        a.export(out, format="wav")
        _ref_f0 = get_f0(out)
        print(f"Ref: {len(a)/1000:.1f}s F0={_ref_f0:.0f}Hz" if _ref_f0 else f"Ref: {len(a)/1000:.1f}s")
        return out
    except Exception as e:
        print(f"prep_ref: {e}"); return path

def check_quality(path):
    if not path or not os.path.exists(path): return "Awaaz upload karein"
    try:
        a = AudioSegment.from_file(path)
        dur = len(a)/1000
        rms = np.sqrt(np.mean(np.array(a.get_array_of_samples(),dtype=np.float32)**2))
        dstr = "OK" if 6<=dur<=30 else ("Chhota" if dur<6 else "Lamba")
        vstr = "OK" if rms>800 else "Soft — louder record karo"
        return f"Duration: {dur:.1f}s [{dstr}]\nVolume: {rms:.0f} [{vstr}]\nRate: {a.frame_rate}Hz"
    except Exception as e: return f"Error: {e}"

def get_ref(up1, up2, up3, git_ref):
    uploads = [r for r in [up1,up2,up3] if r and os.path.exists(r)]
    if uploads:
        if len(uploads)>1:
            segs=[]
            for u in uploads:
                try:
                    a=AudioSegment.from_file(u).set_channels(1).set_frame_rate(22050)
                    try: a=effects.strip_silence(a,silence_thresh=-42,padding=100)
                    except: pass
                    segs.append(a)
                except: pass
            if segs:
                m=segs[0]; sil=AudioSegment.silent(300,frame_rate=22050)
                for s in segs[1:]: m=m+sil+s
                if len(m)>30000: m=m[:30000]
                m.export("ref_merged.wav",format="wav")
                return prep_ref("ref_merged.wav")
        return prep_ref(uploads[0])

    local = os.path.join("voices", git_ref)
    if os.path.exists(local) and os.path.getsize(local)>1000:
        return prep_ref(local)

    for url in [G_RAW+git_ref.replace(" ","%20"), G_RAW+requests.utils.quote(git_ref,safe=""),
                f"https://huggingface.co/Shriramnag/My-Shriram-Voice/resolve/main/{requests.utils.quote(git_ref)}"]:
        try:
            r=requests.get(url,timeout=20)
            if r.status_code==200 and len(r.content)>1000:
                with open("ref_dl.wav","wb") as f: f.write(r.content)
                with open(local,"wb") as f: f.write(r.content)
                return prep_ref("ref_dl.wav")
        except: pass

    for fb in ["ref_ready.wav","Ramai.wav"]+glob.glob("voices/*.wav")+glob.glob("/content/**/*.wav",recursive=True):
        if isinstance(fb,str) and os.path.exists(fb) and os.path.getsize(fb)>5000:
            return prep_ref(fb)
    return None

# ─────────────────────────────────────────────
# 7. AUDIO POST-PROCESSING
# ─────────────────────────────────────────────
def eq_audio(seg, bass=1.5, mid=0.0, treble=-1.5, sr=22050):
    try:
        seg=seg.set_frame_rate(sr).set_channels(1)
        s=np.array(seg.get_array_of_samples(),dtype=np.float64); nyq=sr/2.0
        def bpf(lo,hi):
            b1,a1=butter(2,lo/nyq,btype='high'); b2,a2=butter(2,hi/nyq,btype='low')
            return filtfilt(b2,a2,filtfilt(b1,a1,s))
        def hpf(lo):
            b,a=butter(2,lo/nyq,btype='high'); return filtfilt(b,a,s)
        if abs(bass)>0.1:   s+=bpf(80,300)   * (10**(bass/20)-1)
        if abs(mid)>0.1:    s+=bpf(500,2000)  * (10**(mid/20)-1)
        if abs(treble)>0.1: s+=hpf(5000)      * (10**(treble/20)-1)
        s=np.clip(s,-32768,32767).astype(np.int16)
        return AudioSegment(s.tobytes(),frame_rate=sr,sample_width=2,channels=1)
    except: return seg

def deess(seg, freq=6000, thr_db=-22, sr=22050):
    try:
        seg=seg.set_frame_rate(sr).set_channels(1)
        s=np.array(seg.get_array_of_samples(),dtype=np.float64); nyq=sr/2.0
        b,a=butter(2,freq/nyq,btype='high'); hi=filtfilt(b,a,s)
        thr=10**(thr_db/20)*32768
        gain=np.where(np.abs(hi)>thr,thr/(np.abs(hi)+1e-9),1.0)
        from scipy.ndimage import uniform_filter1d
        gain=uniform_filter1d(np.clip(gain,0.2,1.0),size=int(sr*0.005))
        s2=np.clip(s-hi+hi*gain,-32768,32767).astype(np.int16)
        return AudioSegment(s2.tobytes(),frame_rate=sr,sample_width=2,channels=1)
    except: return seg

def compress_a(seg, thr_db=-18, ratio=3.0, sr=22050):
    try:
        seg=seg.set_frame_rate(sr).set_channels(1)
        s=np.array(seg.get_array_of_samples(),dtype=np.float64)
        thr=10**(thr_db/20)*32768
        gain=np.where(np.abs(s)>thr,thr/np.abs(s)*(np.abs(s)/thr)**(1/ratio),1.0)
        from scipy.ndimage import uniform_filter1d
        gain=uniform_filter1d(np.clip(gain,0.1,1.0),size=int(sr*0.01))
        s2=np.clip(s*gain,-32768,32767).astype(np.int16)
        return effects.normalize(AudioSegment(s2.tobytes(),frame_rate=sr,sample_width=2,channels=1))
    except: return seg

def pitch_shift_fn(seg, st, sr=22050):
    if not LIBROSA or abs(st)<0.1: return seg
    try:
        s=np.array(seg.get_array_of_samples(),dtype=np.float32)/32768.0
        if seg.channels==2: s=s.reshape(-1,2).mean(axis=1)
        sh=librosa.effects.pitch_shift(s,sr=sr,n_steps=float(st),bins_per_octave=24)
        sh=np.clip(sh*32768,-32768,32767).astype(np.int16)
        return AudioSegment(sh.tobytes(),frame_rate=sr,sample_width=2,channels=1)
    except: return seg

def join_segs(segs, cf=60):
    if not segs: return AudioSegment.silent(100)
    rms_list=[np.sqrt(np.mean(np.array(s.get_array_of_samples(),dtype=np.float32)**2)) for s in segs]
    rms_list=[r for r in rms_list if r>100]
    tgt=float(np.median(rms_list)) if rms_list else 3000
    normed=[]
    for s in segs:
        arr=np.array(s.get_array_of_samples(),dtype=np.float32)
        rms=np.sqrt(np.mean(arr**2))
        if rms>100: s=s.apply_gain(20*np.log10(np.clip(tgt/(rms+1e-9),0.3,3.0)))
        normed.append(s)
    out=normed[0]
    for s in normed[1:]:
        c=min(cf,len(out)//2,len(s)//2); out=out.append(s,crossfade=max(c,10))
    return out

# ─────────────────────────────────────────────
# 8. TTS PARAMS
# ─────────────────────────────────────────────
PRESETS = {
    "Calm":     {"temp":0.15,"rep":8.0,"speed":0.90},
    "Normal":   {"temp":0.20,"rep":7.0,"speed":0.95},
    "Pro":      {"temp":0.25,"rep":6.5,"speed":1.00},
    "Dramatic": {"temp":0.28,"rep":6.0,"speed":1.05},
}

def set_params(temp, rep, gpt_len):
    t=min(float(temp),0.28)
    try:
        cfg=tts.synthesizer.tts_config.model_args
        cfg.temperature=t; cfg.repetition_penalty=float(rep)
        cfg.gpt_cond_len=max(int(gpt_len),12); cfg.gpt_cond_chunk_len=4
        cfg.top_p=0.85; cfg.top_k=50; return
    except: pass
    try: tts.tts_config.temperature=t
    except: pass

# ─────────────────────────────────────────────
# 9. GENERATE ONE CHUNK
# ─────────────────────────────────────────────
def gen_chunk(text, lang, ref, out, spd, temp, rep, gpt_len, preset_spd):
    speed = float(spd) if float(spd)>=0.8 else preset_spd

    def try_gen(t, l, s):
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        gc.collect()
        set_params(temp, rep, gpt_len)
        tts.tts_to_file(text=t, speaker_wav=ref, language=l, file_path=out, speed=s)
        return os.path.exists(out) and os.path.getsize(out)>500

    try:
        if try_gen(text, lang, speed): return True, None
    except Exception as e1:
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        gc.collect()

    try:
        words=text.split()
        t2=" ".join(words[:20]) if len(words)>20 else text
        if try_gen(t2,"hi",0.95): return True, "short"
    except Exception as e2:
        return False, str(e2)[:80]

    return False, "both failed"

# ─────────────────────────────────────────────
# 10. MAIN GENERATE
# ─────────────────────────────────────────────
_previews = []

def generate(text, up1, up2, up3, git_ref, preset_name, spd_ovr,
             pitch_en, pitch_manual, gpt_len,
             bass, mid, treble,
             use_sil, use_norm, use_eq, use_des, use_cmp,
             out_fmt, cwords, progress=gr.Progress()):

    global _previews
    _previews = []

    if not text or not text.strip():
        return None,"Text khaali hai.","",gr.update(choices=[])

    preset=PRESETS.get(preset_name,PRESETS["Normal"])
    spd=float(spd_ovr) if float(spd_ovr)>=0.8 else preset["speed"]

    custom=load_dict()
    if cwords:
        for line in cwords.strip().splitlines():
            if "=" in line:
                k,v=line.split("=",1)
                if k.strip() and v.strip(): custom[k.strip()]=v.strip()
        save_dict(custom)

    progress(0.02, desc="Text processing...")
    cleaned=process(text, custom)

    progress(0.05, desc="Reference audio...")
    ref=get_ref(up1,up2,up3,git_ref)
    if not ref:
        return None,(
            "Reference audio nahi mila!\n"
            "Apni awaaz upload karein (6-30 sec WAV/MP3)\n"
            "Ya GitHub se voice select karein."
        ),"",gr.update(choices=[])

    ref_info=check_quality(ref)

    progress(0.08, desc="Chunking...")
    chunks=chunker(cleaned, max_w=30)
    total=len(chunks)
    if total==0:
        return None,"Text khaali ho gaya.",ref_info,gr.update(choices=[])

    progress(0.09, desc=f"{total} parts — ~{total*0.35:.0f} min")
    segs,errors=[],[]

    for i,(chunk,lang) in enumerate(chunks):
        progress(0.10+(i/total)*0.75, desc=f"Part {i+1}/{total} [{lang.upper()}]")
        name=f"_c{i}.wav"

        if i>0 and i%5==0:
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            gc.collect()

        ok,err=gen_chunk(chunk,lang,ref,name,spd,preset["temp"],preset["rep"],gpt_len,preset["speed"])

        if ok and os.path.exists(name):
            try:
                seg=AudioSegment.from_wav(name)
                if use_sil:
                    try: seg=effects.strip_silence(seg,silence_thresh=-45,padding=250)
                    except: pass
                if len(seg)>150:
                    segs.append(seg)
                    pv=f"_p{i+1}.wav"; seg.export(pv,format="wav"); _previews.append(pv)
                    print(f"Part {i+1}: {len(seg)}ms")
                else: errors.append(f"Part {i+1}: short ({len(seg)}ms)")
            except Exception as e: errors.append(f"Part {i+1}: {str(e)[:50]}")
            if os.path.exists(name): os.remove(name)
        else:
            errors.append(f"Part {i+1}[{lang}]: {err}")
            if os.path.exists(name): os.remove(name)

        if torch.cuda.is_available(): torch.cuda.empty_cache()
        gc.collect()

    if not segs:
        return None,(
            f"Koi part nahi bana ({total} try).\n"+"\n".join(errors[:6])+
            "\n\nFix: Apni awaaz upload karein"
        ),ref_info,gr.update(choices=[])

    progress(0.87, desc=f"Joining {len(segs)}...")
    combined=join_segs(segs)
    print(f"Joined: {len(combined)/1000:.1f}s")

    if use_norm:
        progress(0.89, desc="Volume match...")
        combined=combined.set_frame_rate(22050).set_channels(1)
        combined=combined.apply_gain(max(-12.0,-22.0-combined.dBFS))

    if use_eq:
        progress(0.91, desc="EQ...")
        combined=eq_audio(combined,float(bass),float(mid),float(treble))

    if use_des:
        progress(0.92, desc="DeEsser...")
        combined=deess(combined)

    if use_cmp:
        progress(0.93, desc="Compressor...")
        combined=compress_a(combined)

    if pitch_en and LIBROSA:
        manual=float(pitch_manual)
        if abs(manual)>0.1:
            progress(0.95, desc=f"Pitch {manual:+.1f}st...")
            combined=pitch_shift_fn(combined,manual)
        elif _ref_f0:
            try:
                tmp="_pt.wav"; combined.export(tmp,format="wav")
                gf0=get_f0(tmp)
                if gf0 and _ref_f0 and gf0>0:
                    auto=float(np.clip(12*np.log2(_ref_f0/gf0),-6,6))
                    if abs(auto)>0.5:
                        progress(0.95,desc=f"Auto pitch {auto:+.1f}st...")
                        combined=pitch_shift_fn(combined,auto)
                        print(f"Auto pitch: ref={_ref_f0:.0f} gen={gf0:.0f} shift={auto:+.1f}st")
                if os.path.exists(tmp): os.remove(tmp)
            except: pass

    fmt=out_fmt.lower(); fname=f"ShivAI_v4.{fmt}"
    progress(0.97, desc=f"Exporting {fmt.upper()}...")
    if fmt=="mp3": combined.export(fname,format="mp3",bitrate="192k")
    elif fmt=="ogg": combined.export(fname,format="ogg")
    else: combined.export(fname,format="wav",parameters=["-ar","22050"])

    dur=len(combined)/1000
    status=(
        f"{'OK' if len(segs)==total else 'PARTIAL'}: {len(segs)}/{total} parts\n"
        f"Duration: {dur:.1f}s ({dur/60:.1f} min)\n"
        f"Speed: {spd:.2f}x | Style: {preset_name}"
    )
    if errors: status+=f"\n\nFailed ({len(errors)}):\n"+"\n".join(errors[:4])

    return fname, status, ref_info, gr.update(choices=[f"Part {i+1}" for i in range(len(_previews))], value=None)

def get_preview(label):
    if not label or not _previews: return None
    try:
        i=int(label.split()[1])-1
        if 0<=i<len(_previews) and os.path.exists(_previews[i]): return _previews[i]
    except: pass
    return None

def dict_add(w,p):
    if not w or not p: return "Dono bharo.",load_dict_md()
    d=load_dict(); d[w.strip()]=p.strip(); save_dict(d)
    return f"Saved: {w} → {p}",load_dict_md()

def dict_del(w):
    d=load_dict()
    if w.strip() in d: del d[w.strip()]; save_dict(d); return f"Removed: {w}",load_dict_md()
    return f"Not found: {w}",load_dict_md()

# ─────────────────────────────────────────────
# 11. UI
# ─────────────────────────────────────────────
CSS = """
.gradio-container{font-family:'Segoe UI','Inter',Arial,sans-serif!important;background:#0d1117!important;color:#e6edf3!important}
.main,.gr-panel,.gr-form,.gr-box,.gr-group{background:#161b22!important;border:1px solid #21262d!important;border-radius:12px!important}
label span,.gr-label{color:#c9d1d9!important;font-weight:500!important;font-size:.9em!important}
textarea,input[type=text]{background:#0d1117!important;border:1px solid #30363d!important;color:#e6edf3!important;border-radius:8px!important}
textarea:focus,input:focus{border-color:#f7931a!important;box-shadow:0 0 0 2px rgba(247,147,26,.2)!important;outline:none!important}
input[type=range]{accent-color:#f7931a!important}
.gr-button-primary{background:linear-gradient(135deg,#f7931a,#e06d00)!important;border:none!important;color:#fff!important;font-weight:700!important;font-size:1em!important;border-radius:10px!important;padding:12px 24px!important;box-shadow:0 4px 16px rgba(247,147,26,.4)!important}
.gr-button-primary:hover{transform:translateY(-1px)!important;box-shadow:0 6px 24px rgba(247,147,26,.55)!important}
.gr-button-secondary{background:#21262d!important;border:1px solid #30363d!important;color:#c9d1d9!important;border-radius:8px!important}
select,.gr-dropdown select{background:#0d1117!important;border:1px solid #30363d!important;color:#e6edf3!important;border-radius:8px!important}
.mono textarea{background:#0d1117!important;color:#7ee787!important;font-family:Consolas,monospace!important;font-size:.85em!important}
input[type=checkbox]{accent-color:#f7931a!important}
hr{border:none!important;border-top:1px solid #21262d!important;margin:12px 0!important}
.gr-tab-item{background:#161b22!important;color:#8b949e!important;border-radius:8px 8px 0 0!important}
.gr-tab-item.selected{background:#21262d!important;color:#f7931a!important;border-bottom:2px solid #f7931a!important}
"""

HEADER = """
<div style="background:linear-gradient(135deg,#1a1f2e,#0d1117,#1a0800);border:1px solid rgba(247,147,26,.25);border-radius:16px;padding:20px 28px;margin-bottom:16px;text-align:center">
  <div style="font-size:1.9em;font-weight:700;letter-spacing:.5px;background:linear-gradient(90deg,#f7931a,#ffb347,#f7931a);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text">Shiv AI — v4.0</div>
  <div style="color:#8b949e;font-size:.85em;margin:4px 0 10px">Shri Ram Nag &nbsp;|&nbsp; PAISAWALA &nbsp;|&nbsp; Professional Hindi Voice Cloning</div>
  <div style="display:flex;justify-content:center;flex-wrap:wrap;gap:6px">
    <span style="background:#1c2a1c;border:1px solid #2d5a2d;color:#7ee787;border-radius:20px;padding:2px 10px;font-size:.75em">Voice Match</span>
    <span style="background:#1c2a1c;border:1px solid #2d5a2d;color:#7ee787;border-radius:20px;padding:2px 10px;font-size:.75em">Auto Pitch</span>
    <span style="background:#1c2a1c;border:1px solid #2d5a2d;color:#7ee787;border-radius:20px;padding:2px 10px;font-size:.75em">Trilingual</span>
    <span style="background:#1c2a1c;border:1px solid #2d5a2d;color:#7ee787;border-radius:20px;padding:2px 10px;font-size:.75em">Long Audio</span>
    <span style="background:#1c2a1c;border:1px solid #2d5a2d;color:#7ee787;border-radius:20px;padding:2px 10px;font-size:.75em">EQ + DeEss + Compress</span>
  </div>
</div>
"""

with gr.Blocks(css=CSS, title="Shiv AI v4.0") as demo:
    gr.HTML(HEADER)
    with gr.Tabs():

        with gr.Tab("Generate"):
            with gr.Row(equal_height=False):

                with gr.Column(scale=3):
                    txt = gr.Textbox(label="Script — Hindi / English / Sanskrit", lines=16,
                                     placeholder="Yahan script paste karein...\n\nHindi, English aur Sanskrit teeno chal sakti hain.")
                    with gr.Row():
                        wc = gr.Markdown("Shabd: **0**")
                        cc = gr.Markdown("Akshar: **0**")
                    txt.change(lambda x:(f"Shabd: **{len(x.split())}**",f"Akshar: **{len(x)}**"),[txt],[wc,cc])
                    with gr.Row():
                        prev_btn = gr.Button("Text Preview", size="sm", variant="secondary")
                        gen_btn  = gr.Button("Generate Karo", variant="primary", size="lg")
                    preview_box = gr.Textbox(label="Cleaned Text", lines=4, interactive=False, visible=False)
                    prev_btn.click(lambda t:(process(t,load_dict()),gr.update(visible=True)),[txt],[preview_box,preview_box])

                with gr.Column(scale=2):
                    gr.Markdown("### Awaaz Upload")
                    up1 = gr.Audio(label="Apni awaaz (6-30 sec WAV/MP3)", type="filepath")
                    ref_info = gr.Textbox(label="Quality", interactive=False, lines=3, elem_classes=["mono"])
                    up1.change(lambda f: check_quality(f) if f else "Awaaz upload karein",[up1],[ref_info])
                    up2 = gr.Audio(visible=False, type="filepath")
                    up3 = gr.Audio(visible=False, type="filepath")
                    git_v = gr.Dropdown(choices=["aideva.wav","Joanne.wav","Reginald voice.wav","cloning .wav"],
                                        label="Ya default voice chunein", value="aideva.wav")

                    gr.HTML("<hr>")
                    gr.Markdown("### Style")
                    preset = gr.Radio(choices=list(PRESETS.keys()), value="Normal", label="")
                    spd = gr.Slider(0.0, 1.4, value=0.0, step=0.05, label="Speed (0 = auto)")
                    preset.change(lambda p: PRESETS.get(p,PRESETS["Normal"])["speed"],[preset],[spd])

                    gr.HTML("<hr>")
                    gr.Markdown("### Voice Match")
                    gpt_len = gr.Slider(3, 30, value=12, step=1, label="Match Quality (12=fast, 24=best)")
                    pitch_en = gr.Checkbox(label="Pitch Correction (librosa)", value=True)
                    pitch_sl = gr.Slider(-6.0, 6.0, value=0.0, step=0.5, label="Manual Pitch semitones (0 = auto)")

                    gr.HTML("<hr>")
                    gr.Markdown("### EQ")
                    bass_sl   = gr.Slider(-6.0, 12.0, value=1.5,  step=0.5, label="Bass dB")
                    mid_sl    = gr.Slider(-6.0,  6.0, value=0.0,  step=0.5, label="Mid dB")
                    treble_sl = gr.Slider(-9.0,  3.0, value=-1.5, step=0.5, label="Treble dB")

                    gr.HTML("<hr>")
                    with gr.Row():
                        use_sil = gr.Checkbox(label="Silence",  value=True)
                        use_norm= gr.Checkbox(label="Normalize", value=True)
                        use_eq  = gr.Checkbox(label="EQ",        value=True)
                    with gr.Row():
                        use_des = gr.Checkbox(label="DeEss",    value=True)
                        use_cmp = gr.Checkbox(label="Compress", value=True)
                    out_fmt = gr.Radio(["wav","mp3","ogg"], value="wav", label="Format")
                    cwords = gr.Textbox(label="Custom Words (WORD = उच्चारण)", placeholder="PAISAWALA = पेसावाला", lines=2)

            gr.HTML("<hr>")
            with gr.Row():
                with gr.Column(scale=2):
                    out_audio = gr.Audio(label="Output", type="filepath", autoplay=True)
                with gr.Column(scale=1):
                    out_status = gr.Textbox(label="Status", interactive=False, lines=9, elem_classes=["mono"])

            with gr.Accordion("Chunk Preview", open=False):
                with gr.Row():
                    ch_dd  = gr.Dropdown(label="Part", choices=[], interactive=True)
                    ch_btn = gr.Button("Play", size="sm")
                ch_out = gr.Audio(label="Chunk", type="filepath", autoplay=True)
                ch_btn.click(get_preview,[ch_dd],[ch_out])

            gen_btn.click(generate,
                inputs=[txt,up1,up2,up3,git_v,preset,spd,pitch_en,pitch_sl,gpt_len,
                        bass_sl,mid_sl,treble_sl,use_sil,use_norm,use_eq,use_des,use_cmp,out_fmt,cwords],
                outputs=[out_audio,out_status,ref_info,ch_dd])

        with gr.Tab("Dictionary"):
            gr.Markdown("### Custom Words — permanently save karo")
            with gr.Row():
                with gr.Column():
                    dw = gr.Textbox(label="Word", placeholder="PAISAWALA")
                    dp = gr.Textbox(label="Pronunciation", placeholder="पेसावाला")
                    with gr.Row():
                        da = gr.Button("Add", variant="primary")
                        dd_btn = gr.Button("Remove", variant="secondary")
                    ds = gr.Textbox(label="Status", interactive=False, lines=2)
                with gr.Column():
                    dm = gr.Markdown(load_dict_md())
            da.click(dict_add,[dw,dp],[ds,dm])
            dd_btn.click(dict_del,[dw],[ds,dm])

        with gr.Tab("Guide"):
            gr.Markdown("""
### Shiv AI v4.0 — Guide

**Best Voice Match:**
1. 6-30 sec ki saaf awaaz upload karein
2. Quiet room, koi background noise nahi
3. Pitch Correction ON rakhein
4. Match Quality 12-24

**Long Audio (30-40 min):**
- Poori script paste karein
- GPU T4 Colab runtime use karein
- 1000 words ≈ 8-10 min audio

**EQ Tips:**
- Bass +1.5 = natural warmth
- Treble -1.5 = less robotic
- DeEss = harsh sounds fix
- Compress = consistent volume

**Styles:**
- Calm: meditation, slow
- Normal: YouTube, general
- Pro: news, professional
- Dramatic: story, energetic
            """)

demo.launch(share=True, show_error=True)
