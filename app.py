# Shiv AI v5.0 — Dual Engine Voice Cloning
# XTTS v2 (Hindi/Sanskrit/Long audio) + Voice (Realistic/Emotions/600+ languages)
# PAISAWALA | Shri Ram Nag

import os, re, gc, json, glob
import numpy as np
import torch
import gradio as gr
import requests
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

REPO  = "Shriramnag/My-Shriram-Voice"
GRAW  = "https://raw.githubusercontent.com/shriramnag/Shiv-AI-Voice/main/voices/"
DFILE = "custom_dict.json"

# ════════════════════════════════════════════════════════════════════
# ENGINE 1: XTTS v2 — Hindi/Long audio/Voice clone
# ════════════════════════════════════════════════════════════════════
print("Shiv AI v5.0 — Loading XTTS engine...")
try:
    from TTS.api import TTS
    from huggingface_hub import hf_hub_download
    try:
        hf_hub_download(repo_id=REPO, filename="Ramai.pth")
        hf_hub_download(repo_id=REPO, filename="config.json")
    except: pass
    xtts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    XTTS_OK = True
    print(f"XTTS ready on {device.upper()}")
except Exception as e:
    xtts_model = None
    XTTS_OK = False
    print(f"XTTS load failed: {e}")

# ════════════════════════════════════════════════════════════════════
# ENGINE 2: OmniVoice — Realistic/Emotions/600 languages
# ════════════════════════════════════════════════════════════════════
print("Loading OmniVoice engine...")
try:
    from omnivoice import OmniVoice, OmniVoiceGenerationConfig
    from omnivoice.utils.lang_map import LANG_NAMES, lang_display_name
    omni_model = OmniVoice.from_pretrained(
        "k2-fsa/OmniVoice",
        device_map=device,
        dtype=torch.float16 if device == "cuda" else torch.float32,
        load_asr=False,
    )
    OMNI_SR = omni_model.sampling_rate
    OMNI_OK = True
    _ALL_LANGS = ["Auto"] + sorted(lang_display_name(n) for n in LANG_NAMES)
    print("OmniVoice ready!")
except Exception as e:
    omni_model = None
    OMNI_OK = False
    OMNI_SR  = 24000
    _ALL_LANGS = ["Auto", "Hindi", "English", "Sanskrit"]
    print(f"OmniVoice load failed (optional): {e}")

# Librosa
try:
    import librosa, librosa.effects
    HAS_LIBROSA = True
except:
    HAS_LIBROSA = False

# ── Voices download ──────────────────────────────────────────────────
os.makedirs("voices", exist_ok=True)
for vf in ["aideva.wav","Joanne.wav","Reginald voice.wav","cloning .wav"]:
    lp = os.path.join("voices", vf)
    if os.path.exists(lp) and os.path.getsize(lp) > 1000: continue
    for url in [GRAW+vf.replace(" ","%20"), GRAW+requests.utils.quote(vf,safe="")]:
        try:
            r = requests.get(url, timeout=15)
            if r.status_code == 200 and len(r.content) > 1000:
                open(lp,"wb").write(r.content)
                print(f"Got: {vf}"); break
        except: pass

# ── Dictionary ──────────────────────────────────────────────────────
def load_dict():
    try:
        if os.path.exists(DFILE): return json.load(open(DFILE,encoding="utf-8"))
    except: pass
    return {}

def save_dict(d):
    json.dump(d, open(DFILE,"w",encoding="utf-8"), ensure_ascii=False, indent=2)

def dict_md():
    d = load_dict()
    return "\n".join(f"**{k}** → {v}" for k,v in d.items()) if d else "Khaali hai."

# ── Number → Hindi ───────────────────────────────────────────────────
_H=["","एक","दो","तीन","चार","पाँच","छह","सात","आठ","नौ","दस","ग्यारह","बारह","तेरह","चौदह","पंद्रह","सोलह","सत्रह","अठारह","उन्नीस","बीस","इक्कीस","बाईस","तेईस","चौबीस","पच्चीस","छब्बीस","सत्ताईस","अट्ठाईस","उनतीस","तीस","इकतीस","बत्तीस","तैंतीस","चौंतीस","पैंतीस","छत्तीस","सैंतीस","अड़तीस","उनतालीस","चालीस","इकतालीस","बयालीस","तैंतालीस","चौंतालीस","पैंतालीस","छियालीस","सैंतालीस","अड़तालीस","उनचास","पचास","इक्यावन","बावन","तिरपन","चौवन","पचपन","छप्पन","सत्तावन","अट्ठावन","उनसठ","साठ","इकसठ","बासठ","तिरसठ","चौंसठ","पैंसठ","छियासठ","सड़सठ","अड़सठ","उनहत्तर","सत्तर","इकहत्तर","बहत्तर","तिहत्तर","चौहत्तर","पचहत्तर","छिहत्तर","सतहत्तर","अठहत्तर","उनासी","अस्सी","इक्यासी","बयासी","तिरासी","चौरासी","पचासी","छियासी","सत्तासी","अट्ठासी","नवासी","नब्बे","इक्यानवे","बानवे","तिरानवे","चौरानवे","पचानवे","छियानवे","सत्तानवे","अट्ठानवे","निन्यानवे"]
def n2h(n):
    if n==0: return "शून्य"
    if n<0: return "ऋण "+n2h(-n)
    if n<=99: return _H[n]
    if n<1000: h,r=divmod(n,100); return _H[h]+" सौ"+(" "+_H[r] if r else "")
    if n<100000: h,r=divmod(n,1000); return _H[h]+" हज़ार"+(" "+n2h(r) if r else "")
    if n<10000000: h,r=divmod(n,100000); return _H[h]+" लाख"+(" "+n2h(r) if r else "")
    h,r=divmod(n,10000000); return _H[h]+" करोड़"+(" "+n2h(r) if r else "")

# ── Text Processing (XTTS mode) ──────────────────────────────────────
_SK={"dharma":"धर्म","karma":"कर्म","yoga":"योग","shakti":"शक्ति","om":"ॐ","namaste":"नमस्ते","guru":"गुरु","mantra":"मंत्र","atma":"आत्मा","maya":"माया","moksha":"मोक्ष","ahimsa":"अहिंसा","satya":"सत्य","seva":"सेवा","bhakti":"भक्ति","veda":"वेद","puja":"पूजा"}
_EN={"AI":"ए आई","ML":"एम एल","API":"ए पी आई","GPU":"जी पी यू","YouTube":"यूट्यूब","Instagram":"इंस्टाग्राम","Facebook":"फेसबुक","WhatsApp":"व्हाट्सऐप","Google":"गूगल","Internet":"इंटरनेट","Online":"ऑनलाइन","Software":"सॉफ्टवेयर","Computer":"कंप्यूटर","Mobile":"मोबाइल","App":"ऐप","Website":"वेबसाइट","Download":"डाउनलोड","Upload":"अपलोड","Channel":"चैनल","Video":"वीडियो","Content":"कंटेंट","Subscribe":"सब्सक्राइब","Life":"लाइफ","Dream":"ड्रीम","Mindset":"माइंडसेट","Success":"सक्सेस","Fail":"फेल","Goal":"गोल","Focus":"फोकस","Power":"पावर","Money":"मनी","Business":"बिज़नेस","Smart":"स्मार्ट","Team":"टीम","Skill":"स्किल","OK":"ओके","hello":"हेलो","thanks":"थैंक्स"}
_ORD={"1ला":"पहला","1ली":"पहली","2रा":"दूसरा","3रा":"तीसरा","4था":"चौथा","5वीं":"पाँचवीं","5वें":"पाँचवें","6वें":"छठवें","7वें":"सातवें","8वीं":"आठवीं","8वें":"आठवें","9वें":"नौवें","10वें":"दसवें"}

def process_text(text, custom):
    if not text: return ""
    for k,v in custom.items(): text=re.sub(rf'(?<![a-zA-Z\u0900-\u097F]){re.escape(k)}(?![a-zA-Z\u0900-\u097F])',v,text,flags=re.IGNORECASE)
    for k,v in _SK.items(): text=re.sub(rf'(?<![a-zA-Z]){re.escape(k)}(?![a-zA-Z])',v,text,flags=re.IGNORECASE)
    for k,v in _EN.items(): text=re.sub(rf'(?<![a-zA-Z]){re.escape(k)}(?![a-zA-Z])',v,text,flags=re.IGNORECASE)
    for k,v in _ORD.items(): text=text.replace(k,v)
    text=re.sub(r'(\d+(?:\.\d+)?)%',lambda m:n2h(int(float(m.group(1))))+" प्रतिशत",text)
    text=re.sub(r'(\d+)\.(\d+)',lambda m:n2h(int(m.group(1)))+" दशमलव "+n2h(int(m.group(2))),text)
    text=re.sub(r'\b(\d+)\b',lambda m:n2h(int(m.group(1))),text)
    text=re.sub(r'\n+','। ',text); text=re.sub(r'\.\.\.+','। ',text)
    text=re.sub(r'\?+','। ',text); text=re.sub(r'!+','! ',text)
    text=re.sub(r'[।\.]+','। ',text); text=re.sub(r'[,،]+',', ',text)
    text=re.sub(r'[-–—]+',', ',text); text=re.sub(r'[;:]+',', ',text)
    text=re.sub(r'["""\'\'()\[\]{}*#@&^~`|<>]','',text)
    text=re.sub(r'\s+',' ',text); text=re.sub(r'(। ){2,}','। ',text)
    return text.strip()

# ── Chunker (XTTS) ───────────────────────────────────────────────────
def get_lang(words):
    if not words: return "hi"
    deva=sum(1 for w in words for c in w if '\u0900'<=c<='\u097F')
    if deva==0 and sum(1 for w in words if w.isalpha() and w.isascii())>len(words)*0.7: return "en"
    return "hi"

def make_chunks(text, max_w=20):
    sents=[s.strip() for s in re.split(r'(?<=।)\s+|\n+',text) if s.strip()]
    result,buf=[],[]
    def flush():
        if buf: result.append((" ".join(buf),get_lang(buf))); buf.clear()
    for sent in sents:
        words=sent.split()
        if not words: continue
        if len(words)>max_w:
            flush()
            tmp=[]
            for part in re.split(r',\s*',sent):
                pw=part.split()
                if len(tmp)+len(pw)>max_w and tmp: result.append((" ".join(tmp),get_lang(tmp))); tmp=pw
                else: tmp.extend(pw)
            if tmp: result.append((" ".join(tmp),get_lang(tmp)))
        elif len(buf)+len(words)>max_w: flush(); buf.extend(words)
        else: buf.extend(words)
    flush()
    out=[]
    for chunk,lang in result:
        if len(chunk.split())<3 and out: out[-1]=(out[-1][0]+" "+chunk,out[-1][1])
        else: out.append((chunk,lang))
    return out

# ── Reference Audio ──────────────────────────────────────────────────
_REF_F0=None

def measure_f0(path):
    try:
        sr,d=wavfile.read(path)
        if d.ndim==2: d=d.mean(axis=1)
        d=d.astype(np.float32)
        seg=d[len(d)//4:len(d)//4+sr//2]; seg/=(np.max(np.abs(seg))+1e-9)
        corr=np.correlate(seg,seg,'full')[len(seg)-1:]
        lo,hi=int(sr/500),int(sr/60)
        peaks,_=find_peaks(corr[lo:hi],height=0.2)
        if len(peaks): return sr/(peaks[0]+lo)
    except: pass
    return None

def prep_ref(path, out="ref.wav"):
    global _REF_F0
    try:
        a=AudioSegment.from_file(path)
        a=a.set_channels(1).set_frame_rate(22050)
        try: a=effects.strip_silence(a,silence_thresh=-40,padding=300)
        except: pass
        a=a.apply_gain(-18.0-a.dBFS)
        while len(a)<8000: a=a+a
        if len(a)>30000: a=a[:30000]
        a.export(out,format="wav")
        _REF_F0=measure_f0(out)
        print(f"Ref: {len(a)/1000:.1f}s F0={_REF_F0:.0f}Hz" if _REF_F0 else f"Ref: {len(a)/1000:.1f}s")
        return out
    except Exception as e: print(f"prep_ref: {e}"); return path

def check_quality(path):
    if not path or not os.path.exists(path): return "Awaaz upload karein (6-30 sec)"
    try:
        a=AudioSegment.from_file(path)
        dur=len(a)/1000
        rms=np.sqrt(np.mean(np.array(a.get_array_of_samples(),dtype=np.float32)**2))
        ds="OK" if 6<=dur<=30 else ("Chhota" if dur<6 else "Lamba")
        vs="OK" if rms>800 else "Soft — louder record karein"
        return f"Duration: {dur:.1f}s  {ds}\nVolume: {rms:.0f}  {vs}\nRate: {a.frame_rate}Hz"
    except Exception as e: return f"Error: {e}"

def find_ref(up, git_ref):
    if up and os.path.exists(up): return prep_ref(up)
    lp=os.path.join("voices",git_ref)
    if os.path.exists(lp) and os.path.getsize(lp)>1000: return prep_ref(lp)
    for url in [GRAW+git_ref.replace(" ","%20"),GRAW+requests.utils.quote(git_ref,safe=""),
                f"https://huggingface.co/Shriramnag/My-Shriram-Voice/resolve/main/{requests.utils.quote(git_ref)}"]:
        try:
            r=requests.get(url,timeout=20)
            if r.status_code==200 and len(r.content)>1000:
                open("ref_dl.wav","wb").write(r.content); open(lp,"wb").write(r.content)
                return prep_ref("ref_dl.wav")
        except: pass
    for fb in ["ref.wav"]+glob.glob("voices/*.wav")+glob.glob("/content/**/*.wav",recursive=True):
        if isinstance(fb,str) and os.path.exists(fb) and os.path.getsize(fb)>5000: return prep_ref(fb)
    return None

# ── Audio Post-Processing ─────────────────────────────────────────────
def apply_eq(seg, bass=0.0, mid=1.5, treble=-1.5, sr=22050):
    try:
        seg=seg.set_frame_rate(sr).set_channels(1)
        s=np.array(seg.get_array_of_samples(),dtype=np.float64); nyq=sr/2.0
        def bp(lo,hi):
            b1,a1=butter(2,lo/nyq,btype='high'); b2,a2=butter(2,hi/nyq,btype='low')
            return filtfilt(b2,a2,filtfilt(b1,a1,s))
        def hp(lo): b,a=butter(2,lo/nyq,btype='high'); return filtfilt(b,a,s)
        if abs(bass)>0.1: s+=bp(80,250)*(10**(bass/20)-1)
        if abs(mid)>0.1: s+=bp(250,800)*(10**(mid/20)-1)
        s+=bp(1000,3000)*(10**(0.5/20)-1)  # presence fixed
        if abs(treble)>0.1: s+=hp(5000)*(10**(treble/20)-1)
        s=np.clip(s,-32768,32767).astype(np.int16)
        return AudioSegment(s.tobytes(),frame_rate=sr,sample_width=2,channels=1)
    except: return seg

def apply_deess(seg, sr=22050):
    try:
        seg=seg.set_frame_rate(sr).set_channels(1)
        s=np.array(seg.get_array_of_samples(),dtype=np.float64); nyq=sr/2.0
        b,a=butter(2,6000/nyq,btype='high'); hi=filtfilt(b,a,s)
        thr=10**(-22/20)*32768
        gain=np.where(np.abs(hi)>thr,thr/(np.abs(hi)+1e-9),1.0)
        from scipy.ndimage import uniform_filter1d
        gain=uniform_filter1d(np.clip(gain,0.2,1.0),size=int(sr*0.005))
        s2=np.clip(s-hi+hi*gain,-32768,32767).astype(np.int16)
        return AudioSegment(s2.tobytes(),frame_rate=sr,sample_width=2,channels=1)
    except: return seg

def apply_compress(seg, sr=22050):
    try:
        seg=seg.set_frame_rate(sr).set_channels(1)
        s=np.array(seg.get_array_of_samples(),dtype=np.float64)
        thr=10**(-12/20)*32768
        gain=np.where(np.abs(s)>thr,thr/np.abs(s)*(np.abs(s)/thr)**(1/1.5),1.0)
        from scipy.ndimage import uniform_filter1d
        gain=uniform_filter1d(np.clip(gain,0.3,1.0),size=int(sr*0.02))
        s2=np.clip(s*gain,-32768,32767).astype(np.int16)
        return AudioSegment(s2.tobytes(),frame_rate=sr,sample_width=2,channels=1)
    except: return seg

def apply_pitch(seg, st, sr=22050):
    if not HAS_LIBROSA or abs(st)<0.05: return seg
    try:
        s=np.array(seg.get_array_of_samples(),dtype=np.float32)/32768.0
        if seg.channels==2: s=s.reshape(-1,2).mean(axis=1)
        sh=librosa.effects.pitch_shift(s,sr=sr,n_steps=float(st),bins_per_octave=24)
        sh=np.clip(sh*32768,-32768,32767).astype(np.int16)
        return AudioSegment(sh.tobytes(),frame_rate=sr,sample_width=2,channels=1)
    except: return seg

def smart_join(segs, cf=80):
    if not segs: return AudioSegment.silent(100)
    rms_v=[np.sqrt(np.mean(np.array(s.get_array_of_samples(),dtype=np.float32)**2)) for s in segs]
    rms_v=[r for r in rms_v if r>200]
    tgt=float(np.median(rms_v)) if rms_v else 3000
    leveled=[]
    for s in segs:
        rms=np.sqrt(np.mean(np.array(s.get_array_of_samples(),dtype=np.float32)**2))
        if rms>200: s=s.apply_gain(20*np.log10(np.clip(tgt/(rms+1e-9),0.5,2.0)))
        leveled.append(s)
    out=leveled[0]
    for s in leveled[1:]:
        c=min(cf,len(out)//2,len(s)//2); out=out.append(s,crossfade=max(c,20))
    return out

def post_process(audio_seg, bass, mid, treble, do_norm, do_eq, do_deess, do_comp,
                 pitch_on, pitch_manual, sr=22050):
    out = audio_seg
    if do_norm:
        out=out.set_frame_rate(sr).set_channels(1)
        out=out.apply_gain(max(-6.0,-22.7-out.dBFS))
    if do_eq:
        try:
            arr=np.array(out.get_array_of_samples(),dtype=np.float32)
            from scipy.signal import welch as _w
            f,p=_w(arr,fs=sr,nperseg=4096)
            def _be(lo,hi): m=(f>=lo)&(f<=hi); return np.mean(p[m]) if m.any() else 1
            bc=-2.0 if _be(80,300)/_be(800,2500)>120 else 0.0
        except: bc=0.0
        out=apply_eq(out,float(bass)+bc,float(mid),float(treble))
    if do_deess: out=apply_deess(out)
    if do_comp:  out=apply_compress(out)
    if pitch_on and HAS_LIBROSA:
        manual=float(pitch_manual)
        if abs(manual)>0.1: out=apply_pitch(out,manual)
        elif _REF_F0:
            try:
                out.export("_pt.wav",format="wav"); gf0=measure_f0("_pt.wav")
                if gf0 and _REF_F0 and gf0>0:
                    auto=float(np.clip(12*np.log2(_REF_F0/gf0),-3,3))
                    if abs(auto)>0.4: out=apply_pitch(out,auto)
                if os.path.exists("_pt.wav"): os.remove("_pt.wav")
            except: pass
    return out

# ── XTTS params ──────────────────────────────────────────────────────
XTTS_STYLES={
    "Calm":     {"temp":0.10,"rep":9.0,"speed":0.88},
    "Normal":   {"temp":0.15,"rep":8.0,"speed":0.92},
    "Pro":      {"temp":0.20,"rep":7.5,"speed":0.97},
    "Dramatic": {"temp":0.25,"rep":7.0,"speed":1.02},
}

def set_xtts(temp, rep, gpt_len):
    t=min(float(temp),0.28)
    try:
        cfg=xtts_model.synthesizer.tts_config.model_args
        cfg.temperature=t; cfg.repetition_penalty=float(rep)
        cfg.gpt_cond_len=max(int(gpt_len),14); cfg.gpt_cond_chunk_len=4
        cfg.length_penalty=1.0; cfg.top_p=0.80; cfg.top_k=40; return
    except: pass
    try: xtts_model.tts_config.temperature=t
    except: pass

def xtts_one(text, lang, ref, out, spd, temp, rep, gpt_len, fb_spd):
    speed=float(spd) if float(spd)>=0.8 else fb_spd
    def try_it(t,l,s):
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        gc.collect(); set_xtts(temp,rep,gpt_len)
        xtts_model.tts_to_file(text=t,speaker_wav=ref,language=l,file_path=out,speed=s)
        return os.path.exists(out) and os.path.getsize(out)>500
    try:
        if try_it(text,lang,speed): return True,None
    except:
        if torch.cuda.is_available(): torch.cuda.empty_cache(); gc.collect()
    try:
        w=text.split(); t2=" ".join(w[:18]) if len(w)>18 else text
        if try_it(t2,"hi",0.95): return True,"short"
    except Exception as e: return False,str(e)[:80]
    return False,"both failed"

# ════════════════════════════════════════════════════════════════════
# GENERATE: ENGINE 1 — XTTS (Hindi/Long/Clone)
# ════════════════════════════════════════════════════════════════════
_xtts_previews=[]

def generate_xtts(text, upload, git_ref, style_name, spd_ovr,
                  pitch_on, pitch_manual, gpt_len,
                  bass, mid, treble,
                  do_norm, do_eq, do_deess, do_comp,
                  out_fmt, custom_raw, progress=gr.Progress()):
    global _xtts_previews
    _xtts_previews=[]

    if not XTTS_OK:
        return None,"XTTS engine load nahi hua. Colab restart karein.","",gr.update(choices=[])
    if not text or not text.strip():
        return None,"Text khaali hai.","",gr.update(choices=[])

    style=XTTS_STYLES.get(style_name,XTTS_STYLES["Normal"])
    spd=float(spd_ovr) if float(spd_ovr)>=0.8 else style["speed"]
    custom=load_dict()
    if custom_raw:
        for line in custom_raw.strip().splitlines():
            if "=" in line:
                k,v=line.split("=",1)
                if k.strip() and v.strip(): custom[k.strip()]=v.strip()
        save_dict(custom)

    progress(0.02, desc="Text processing...")
    cleaned=process_text(text,custom)

    progress(0.05, desc="Reference audio...")
    ref=find_ref(upload,git_ref)
    if not ref:
        return None,"Reference audio nahi mila.\nApni awaaz upload karein (6-30 sec WAV/MP3).","",gr.update(choices=[])
    ref_info=check_quality(ref)

    progress(0.08, desc="Chunking...")
    chunks=make_chunks(cleaned,max_w=20)
    total=len(chunks)
    if total==0: return None,"Text khaali ho gaya.",ref_info,gr.update(choices=[])

    progress(0.09,desc=f"{total} parts — ~{total*0.35:.0f} min")
    segs,errors=[],[]

    for i,(chunk,lang) in enumerate(chunks):
        progress(0.10+(i/total)*0.74, desc=f"Part {i+1}/{total}")
        tmp=f"_c{i}.wav"
        if i>0 and i%5==0:
            if torch.cuda.is_available(): torch.cuda.empty_cache(); gc.collect()
        ok,err=xtts_one(chunk,lang,ref,tmp,spd,style["temp"],style["rep"],gpt_len,style["speed"])
        if ok and os.path.exists(tmp):
            try:
                seg=AudioSegment.from_wav(tmp)
                try: seg=effects.strip_silence(seg,silence_thresh=-48,padding=300)
                except: pass
                if len(seg)>400:
                    segs.append(seg); pf=f"_p{i+1}.wav"
                    seg.export(pf,format="wav"); _xtts_previews.append(pf)
                else: errors.append(f"Part {i+1}: short {len(seg)}ms")
            except Exception as e: errors.append(f"Part {i+1}: {str(e)[:50]}")
            if os.path.exists(tmp): os.remove(tmp)
        else:
            errors.append(f"Part {i+1}: {err}")
            if os.path.exists(tmp): os.remove(tmp)
        if torch.cuda.is_available(): torch.cuda.empty_cache(); gc.collect()

    if not segs:
        return None,(f"Koi part nahi bana.\n"+"\n".join(errors[:6])+"\nFix: Apni awaaz upload karein."),ref_info,gr.update(choices=[])

    progress(0.85, desc=f"Joining {len(segs)}...")
    out=smart_join(segs)

    progress(0.90, desc="Post-processing...")
    out=post_process(out,bass,mid,treble,do_norm,do_eq,do_deess,do_comp,pitch_on,pitch_manual)

    fmt=out_fmt.lower(); fname=f"ShivAI_v5_XTTS.{fmt}"
    progress(0.97, desc="Saving...")
    if fmt=="mp3": out.export(fname,format="mp3",bitrate="192k")
    elif fmt=="ogg": out.export(fname,format="ogg")
    else: out.export(fname,format="wav",parameters=["-ar","22050"])

    dur=len(out)/1000
    status=(f"{'Done' if len(segs)==total else 'Partial'}: {len(segs)}/{total} parts\n"
            f"Duration: {dur:.1f}s ({dur/60:.1f} min)\nStyle: {style_name}")
    if errors: status+=f"\nFailed ({len(errors)}):\n"+"\n".join(errors[:3])
    return fname,status,ref_info,gr.update(choices=[f"Part {i+1}" for i in range(len(_xtts_previews))],value=None)

def play_xtts_chunk(label):
    if not label or not _xtts_previews: return None
    try:
        i=int(label.split()[1])-1
        if 0<=i<len(_xtts_previews) and os.path.exists(_xtts_previews[i]): return _xtts_previews[i]
    except: pass
    return None

# ════════════════════════════════════════════════════════════════════
# GENERATE: ENGINE 2 — OmniVoice (Realistic/Emotions/600 langs)
# ════════════════════════════════════════════════════════════════════

# Emotion tags from OmniVoice
EMOTION_TAGS=[
    "[laughter]","[sigh]","[confirmation-en]","[question-en]",
    "[question-ah]","[surprise-ah]","[surprise-oh]","[dissatisfaction-hnn]"
]

INSERT_TAG_JS="""
(tag_val, current_text) => {
    const textarea = document.querySelector('#omni_text textarea');
    if (!textarea) return (current_text||'') + ' ' + tag_val;
    const start = textarea.selectionStart;
    const end   = textarea.selectionEnd;
    const t = current_text || '';
    const pre = (start>0 && t[start-1]!==' ') ? ' ' : '';
    const suf = (end<t.length && t[end]!==' ')  ? ' ' : '';
    return t.slice(0,start)+pre+tag_val+suf+t.slice(end);
}
"""

def generate_omni(text, ref_audio, language, speed, num_steps, guidance,
                  progress=gr.Progress()):
    if not OMNI_OK:
        return None, "OmniVoice engine load nahi hua. pip install omnivoice karein."
    if not text or not text.strip():
        return None, "Text khaali hai."
    if not ref_audio:
        return None, "Reference audio zaroori hai OmniVoice ke liye."

    progress(0.1, desc="OmniVoice generating...")
    try:
        from omnivoice import OmniVoiceGenerationConfig
        gen_cfg = OmniVoiceGenerationConfig(
            num_step=int(num_steps),
            guidance_scale=float(guidance),
            denoise=True,
            preprocess_prompt=True,
            postprocess_output=True,
        )
        lang = language if (language and language != "Auto") else None
        kw = dict(text=text.strip(), language=lang, generation_config=gen_cfg)
        if float(speed) != 1.0: kw["speed"] = float(speed)
        kw["voice_clone_prompt"] = omni_model.create_voice_clone_prompt(ref_audio=ref_audio)
        audio = omni_model.generate(**kw)
        waveform = (audio[0] * 32767).astype(np.int16)
        fname = "ShivAI_v5_OmniVoice.wav"
        import scipy.io.wavfile as swf
        swf.write(fname, OMNI_SR, waveform)
        progress(1.0, desc="Done!")
        return (OMNI_SR, waveform), f"Done! Duration: {len(waveform)/OMNI_SR:.1f}s"
    except Exception as e:
        return None, f"Error: {type(e).__name__}: {e}"

# ── Dictionary helpers ────────────────────────────────────────────────
def dict_add(w,p):
    if not w or not p: return "Dono bharo.",dict_md()
    d=load_dict(); d[w.strip()]=p.strip(); save_dict(d)
    return f"Saved: {w} → {p}",dict_md()

def dict_del(w):
    d=load_dict()
    if w.strip() in d: del d[w.strip()]; save_dict(d); return f"Removed: {w}",dict_md()
    return f"Not found: {w}",dict_md()

# ════════════════════════════════════════════════════════════════════
# UI
# ════════════════════════════════════════════════════════════════════
CSS = """
* { box-sizing: border-box; }
.gradio-container {
    font-family: 'Segoe UI', Inter, Arial, sans-serif !important;
    background: #0d1117 !important; color: #e6edf3 !important;
    max-width: 1100px !important; margin: 0 auto !important;
}
.gr-panel,.gr-form,.gr-box,.gr-group,.main {
    background: #161b22 !important; border: 1px solid #21262d !important; border-radius: 10px !important;
}
label span { color: #8b949e !important; font-size: 0.85em !important; font-weight: 500 !important; }
textarea, input[type=text] {
    background: #0d1117 !important; border: 1px solid #30363d !important;
    color: #e6edf3 !important; border-radius: 8px !important; font-size: 0.95em !important;
}
textarea:focus, input:focus {
    border-color: #f7931a !important; outline: none !important;
    box-shadow: 0 0 0 2px rgba(247,147,26,0.15) !important;
}
input[type=range] { accent-color: #f7931a !important; }
.gr-button-primary {
    background: linear-gradient(135deg,#f7931a,#e06d00) !important; border: none !important;
    color: #fff !important; font-weight: 700 !important; font-size: 1em !important;
    border-radius: 10px !important; padding: 12px 32px !important;
    box-shadow: 0 4px 20px rgba(247,147,26,0.35) !important; transition: all 0.15s !important;
}
.gr-button-primary:hover { transform: translateY(-1px) !important; box-shadow: 0 6px 28px rgba(247,147,26,0.5) !important; }
.gr-button-secondary { background: #21262d !important; border: 1px solid #30363d !important; color: #c9d1d9 !important; border-radius: 8px !important; }
select { background: #0d1117 !important; border: 1px solid #30363d !important; color: #e6edf3 !important; border-radius: 8px !important; }
.mono textarea { background: #0d1117 !important; color: #7ee787 !important; font-family: Consolas, monospace !important; font-size: 0.82em !important; }
input[type=checkbox] { accent-color: #f7931a !important; }
.gr-tab-item { background: #161b22 !important; color: #8b949e !important; font-size: 0.9em !important; }
.gr-tab-item.selected { color: #f7931a !important; border-bottom: 2px solid #f7931a !important; background: #1c2128 !important; }
.sec { color: #8b949e; font-size: 0.78em; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; margin: 10px 0 4px; }
.tag-btn { min-width: fit-content !important; height: 28px !important; font-size: 12px !important;
    background: #1c2a3a !important; border: 1px solid #1f4068 !important; color: #58a6ff !important;
    border-radius: 6px !important; padding: 0 8px !important; }
.tag-btn:hover { background: #1f4068 !important; }
"""

HEADER = """
<div style="text-align:center;padding:16px 0 12px;border-bottom:1px solid #21262d;margin-bottom:16px">
  <div style="font-size:1.7em;font-weight:700;background:linear-gradient(90deg,#f7931a,#ffb347);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text">
    Shiv AI — v5.0
  </div>
  <div style="color:#8b949e;font-size:.82em;margin-top:3px">
    Shri Ram Nag · PAISAWALA · XTTS + OmniVoice Dual Engine
  </div>
  <div style="display:flex;justify-content:center;gap:6px;margin-top:8px;flex-wrap:wrap">
    <span style="background:#1c2128;border:1px solid #2d5a2d;color:#7ee787;padding:2px 10px;border-radius:12px;font-size:.72em">XTTS Hindi Clone</span>
    <span style="background:#1c2128;border:1px solid #2d5a2d;color:#7ee787;padding:2px 10px;border-radius:12px;font-size:.72em">OmniVoice Realistic</span>
    <span style="background:#1c2128;border:1px solid #2d5a2d;color:#7ee787;padding:2px 10px;border-radius:12px;font-size:.72em">Emotion Tags</span>
    <span style="background:#1c2128;border:1px solid #2d5a2d;color:#7ee787;padding:2px 10px;border-radius:12px;font-size:.72em">600+ Languages</span>
    <span style="background:#1c2128;border:1px solid #2d5a2d;color:#7ee787;padding:2px 10px;border-radius:12px;font-size:.72em">Long Audio</span>
    <span style="background:#1c2128;border:1px solid #2d5a2d;color:#7ee787;padding:2px 10px;border-radius:12px;font-size:.72em">EQ · DeEss · Compress</span>
  </div>
</div>
"""

with gr.Blocks(css=CSS, title="Shiv AI v5.0") as demo:
    gr.HTML(HEADER)

    with gr.Tabs():

        # ══ TAB 1: XTTS — Hindi / Long Audio ══════════════════════════
        with gr.Tab("XTTS — Hindi / Long Audio"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=5):
                    t1_txt = gr.Textbox(label="Script — Hindi / English / Sanskrit",
                                        placeholder="Yahan script paste karein...", lines=16)
                    with gr.Row():
                        t1_wc = gr.Markdown("Words: **0**")
                        t1_cc = gr.Markdown("Chars: **0**")
                    t1_txt.change(
                        lambda x:(f"Words: **{len(x.split())}**",f"Chars: **{len(x)}**"),
                        [t1_txt],[t1_wc,t1_cc])
                    with gr.Row():
                        t1_prev = gr.Button("Text Preview",size="sm",variant="secondary")
                        t1_gen  = gr.Button("Generate (XTTS)",variant="primary",size="lg")
                    t1_prevbox = gr.Textbox(label="Cleaned Text",lines=4,interactive=False,
                                            visible=False,elem_classes=["mono"])
                    t1_prev.click(
                        lambda t:(process_text(t,load_dict()),gr.update(visible=True)),
                        [t1_txt],[t1_prevbox,t1_prevbox])

                with gr.Column(scale=3):
                    gr.HTML('<div class="sec">Voice Upload</div>')
                    t1_up   = gr.Audio(label="Apni awaaz (6-30 sec)",type="filepath")
                    t1_qual = gr.Textbox(label="Quality",interactive=False,lines=3,elem_classes=["mono"])
                    t1_up.change(lambda f:check_quality(f) if f else "Upload karein",[t1_up],[t1_qual])
                    t1_gitv = gr.Dropdown(
                        choices=["aideva.wav","Joanne.wav","Reginald voice.wav","cloning .wav"],
                        label="Ya default voice",value="aideva.wav")

                    gr.HTML('<div class="sec">Style</div>')
                    t1_style = gr.Radio(choices=list(XTTS_STYLES.keys()),value="Normal",label="")
                    t1_spd   = gr.Slider(0.0,1.4,0.0,step=0.05,label="Speed (0=auto)")
                    t1_style.change(lambda s:XTTS_STYLES.get(s,XTTS_STYLES["Normal"])["speed"],[t1_style],[t1_spd])

                    gr.HTML('<div class="sec">Voice Match</div>')
                    t1_gpt     = gr.Slider(6,24,14,step=1,label="Match quality (14=fast · 20=best)")
                    t1_pitchen = gr.Checkbox(label="Pitch correction",value=False)
                    t1_pitchsl = gr.Slider(-3.0,3.0,0.0,step=0.5,label="Manual pitch (0=auto)")

                    gr.HTML('<div class="sec">EQ</div>')
                    t1_bass   = gr.Slider(-3.0,3.0, 0.0,step=0.5,label="Bass dB")
                    t1_mid    = gr.Slider(-3.0,4.0, 1.5,step=0.5,label="Mid dB (+1.5=deeper)")
                    t1_treble = gr.Slider(-4.0,2.0,-1.5,step=0.5,label="Treble dB")

                    gr.HTML('<div class="sec">Options</div>')
                    with gr.Row():
                        t1_norm  = gr.Checkbox(label="Normalize",value=True)
                        t1_eq    = gr.Checkbox(label="EQ",value=True)
                        t1_deess = gr.Checkbox(label="DeEss",value=True)
                        t1_comp  = gr.Checkbox(label="Compress",value=True)
                    t1_fmt   = gr.Radio(["wav","mp3","ogg"],value="wav",label="Format")
                    t1_cw    = gr.Textbox(label="Custom words (WORD = उच्चारण)",
                                          placeholder="PAISAWALA = पेसावाला",lines=2)

            with gr.Row():
                with gr.Column(scale=3):
                    t1_out = gr.Audio(label="Output",type="filepath",autoplay=True)
                with gr.Column(scale=2):
                    t1_status = gr.Textbox(label="Status",interactive=False,lines=8,elem_classes=["mono"])

            with gr.Accordion("Chunk Preview",open=False):
                with gr.Row():
                    t1_chdd  = gr.Dropdown(label="Part",choices=[],interactive=True)
                    t1_chbtn = gr.Button("Play",size="sm")
                t1_chout = gr.Audio(label="",type="filepath",autoplay=True)
                t1_chbtn.click(play_xtts_chunk,[t1_chdd],[t1_chout])

            t1_gen.click(
                generate_xtts,
                inputs=[t1_txt,t1_up,t1_gitv,t1_style,t1_spd,
                        t1_pitchen,t1_pitchsl,t1_gpt,
                        t1_bass,t1_mid,t1_treble,
                        t1_norm,t1_eq,t1_deess,t1_comp,
                        t1_fmt,t1_cw],
                outputs=[t1_out,t1_status,t1_qual,t1_chdd])

        # ══ TAB 2: OmniVoice — Realistic / Emotions ════════════════════
        with gr.Tab("OmniVoice — Realistic / Emotions"):
            if not OMNI_OK:
                gr.Markdown("""
> **OmniVoice load nahi hua.**
> Install: `pip install omnivoice faster-whisper`
> Phir Colab restart karein aur app.py dobara chalayein.
                """)

            with gr.Row(equal_height=False):
                with gr.Column(scale=5):
                    omni_txt = gr.Textbox(
                        label="Script (Emotion tags bhi daal sakte hain)",
                        placeholder='Example:\nAaj hum [laughter] ek naye topic ke baare mein baat karenge [sigh]',
                        lines=14, elem_id="omni_text"
                    )
                    # Emotion tag buttons
                    gr.HTML('<div class="sec">Emotion Tags</div>')
                    with gr.Row():
                        for tag in EMOTION_TAGS:
                            btn = gr.Button(tag, elem_classes=["tag-btn"])
                            btn.click(fn=None, inputs=[btn, omni_txt],
                                      outputs=omni_txt, js=INSERT_TAG_JS)

                    omni_gen = gr.Button("Generate (OmniVoice)", variant="primary", size="lg")

                with gr.Column(scale=3):
                    gr.HTML('<div class="sec">Reference Voice</div>')
                    omni_ref  = gr.Audio(label="Apni awaaz (3-15 sec)",type="filepath")
                    omni_qual = gr.Textbox(label="Quality",interactive=False,lines=2,elem_classes=["mono"])
                    omni_ref.change(lambda f:check_quality(f) if f else "Upload karein",[omni_ref],[omni_qual])

                    gr.HTML('<div class="sec">Language</div>')
                    omni_lang = gr.Dropdown(choices=_ALL_LANGS,value="Auto",label="Language (600+ supported)")

                    gr.HTML('<div class="sec">Quality Settings</div>')
                    omni_spd  = gr.Slider(0.5,1.5,1.0,step=0.05,label="Speed")
                    omni_step = gr.Slider(4,64,32,step=1,label="Inference Steps (32=balanced · 48=best)")
                    omni_gs   = gr.Slider(0.0,4.0,2.0,step=0.1,label="Guidance Scale (2.0=natural)")

                    gr.Markdown("""
**OmniVoice ke fayde:**
- Realistic, human-like voice
- Emotion tags: `[laughter]` `[sigh]` `[question-en]`
- 600+ languages
- Short scripts ke liye best (30 sec - 2 min)

**XTTS ke fayde:**
- Long audio (30-40 min)
- Hindi/Sanskrit exact match
- Custom chunking aur EQ
                    """)

            with gr.Row():
                with gr.Column(scale=3):
                    omni_out    = gr.Audio(label="Output",type="numpy",autoplay=True)
                with gr.Column(scale=2):
                    omni_status = gr.Textbox(label="Status",interactive=False,lines=4,elem_classes=["mono"])

            omni_gen.click(
                generate_omni,
                inputs=[omni_txt,omni_ref,omni_lang,omni_spd,omni_step,omni_gs],
                outputs=[omni_out,omni_status])

        # ══ TAB 3: Dictionary ══════════════════════════════════════════
        with gr.Tab("Dictionary"):
            gr.Markdown("### Custom words — permanently save karo")
            with gr.Row():
                with gr.Column():
                    dw = gr.Textbox(label="Word",placeholder="PAISAWALA")
                    dp = gr.Textbox(label="Pronunciation",placeholder="पेसावाला")
                    with gr.Row():
                        da_btn = gr.Button("Add",variant="primary")
                        dd_btn = gr.Button("Remove",variant="secondary")
                    ds = gr.Textbox(label="Status",interactive=False,lines=2)
                with gr.Column():
                    dm = gr.Markdown(dict_md())
            da_btn.click(dict_add,[dw,dp],[ds,dm])
            dd_btn.click(dict_del,[dw],[ds,dm])

        # ══ TAB 4: Guide ═══════════════════════════════════════════════
        with gr.Tab("Guide"):
            gr.Markdown("""
### Shiv AI v5.0 — Kab kya use karein?

| Feature | XTTS Tab | OmniVoice Tab |
|---------|----------|----------------|
| Hindi long script | ✅ Best | ⚠️ Limited |
| 30-40 min audio | ✅ Yes | ❌ No |
| Realistic sound | ⚠️ Good | ✅ Best |
| Emotion tags | ❌ No | ✅ Yes |
| 600+ languages | ❌ Hindi/En | ✅ Yes |
| Voice match | ✅ High | ✅ High |
| Speed | Medium | Faster |

---

### OmniVoice Emotion Tags
```
[laughter]         — haste hue bolega
[sigh]             — saans lete hue
[question-en]      — question tone
[surprise-ah]      — surprised tone
[dissatisfaction-hnn] — unhappy tone
```

---

### XTTS Best Settings
- Style: **Normal**
- Bass: **0**, Mid: **+1.5**, Treble: **-1.5**
- Pitch: **OFF** (auto ON karo agar needed)
- Match quality: **14-20**

---

### Long Audio Tips
- XTTS tab use karein
- GPU T4 Colab runtime zaroori
- ~1000 words = 8-10 min audio
- Ek baar mein poori script paste karein
            """)

demo.launch(share=True, show_error=True)
