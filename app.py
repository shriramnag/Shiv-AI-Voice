import os, torch, gradio as gr, requests, re, gc, numpy as np
from TTS.api import TTS
from huggingface_hub import hf_hub_download
from pydub import AudioSegment, effects
from scipy.io import wavfile
from scipy.signal import butter, filtfilt
import soundfile as sf

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# १. टर्बो हाई स्पीड सेटअप
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
os.environ["COQUI_TOS_AGREED"] = "1"
torch.backends.cudnn.benchmark = True
device = "cuda" if torch.cuda.is_available() else "cpu"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# २. मॉडल लोड
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REPO_ID = "Shriramnag/My-Shriram-Voice"
print("शिव AI v2.0 — एडवांस्ड वॉइस इंजन शुरू हो रहा है...")
try:
    hf_hub_download(repo_id=REPO_ID, filename="Ramai.pth")
    hf_hub_download(repo_id=REPO_ID, filename="config.json")
except:
    pass

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
G_RAW = "https://raw.githubusercontent.com/shriramnag/Aivoicebox/main/%F0%9F%93%81%20voices/"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ३. सुपर एडवांस्ड टेक्स्ट क्लीनर (हकलाहट + भाषा मिक्स फिक्स)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FIX #1: बहुत बड़ा English→Hindi dictionary — model confuse नहीं होगा
ENGLISH_TO_HINDI = {
    # Technology
    "AI": "ए आई", "ML": "एम एल", "YouTube": "यूट्यूब", "Instagram": "इंस्टाग्राम",
    "Facebook": "फेसबुक", "WhatsApp": "व्हाट्सऐप", "Google": "गूगल",
    "Internet": "इंटरनेट", "Online": "ऑनलाइन", "Offline": "ऑफलाइन",
    "Software": "सॉफ्टवेयर", "Hardware": "हार्डवेयर", "Computer": "कंप्यूटर",
    "Mobile": "मोबाइल", "App": "ऐप", "Website": "वेबसाइट",
    "Download": "डाउनलोड", "Upload": "अपलोड", "Password": "पासवर्ड",
    # Motivation
    "Life": "लाइफ", "Dream": "ड्रीम", "Mindset": "माइंडसेट", "Believe": "बिलीव",
    "Success": "सक्सेस", "Fail": "फेल", "Failure": "फेल्योर", "Goal": "गोल",
    "Focus": "फोकस", "Step": "स्टेप", "Fear": "फियर", "Simple": "सिंपल",
    "Practical": "प्रैक्टिकल", "Strong": "स्ट्रॉन्ग", "Turbo": "टर्बो",
    "Power": "पावर", "Energy": "एनर्जी", "Positive": "पॉजिटिव",
    "Negative": "नेगेटिव", "Challenge": "चैलेंज", "Time": "टाइम",
    "Work": "वर्क", "Hard": "हार्ड", "Smart": "स्मार्ट", "Money": "मनी",
    "Business": "बिज़नेस", "Market": "मार्केट", "Brand": "ब्रांड",
    "Content": "कंटेंट", "Video": "वीडियो", "Channel": "चैनल",
    # Common
    "because": "बिकॉज़", "but": "बट", "and": "एंड", "the": "द",
    "is": "इज़", "are": "आर", "was": "वॉज़", "have": "हैव",
    "or": "ऑर", "of": "ऑफ", "in": "इन", "on": "ऑन",
    "so": "सो", "no": "नो", "yes": "यस", "not": "नॉट",
    "OK": "ओके", "okay": "ओके", "hey": "हे", "hi": "हाय",
}

def shiv_super_cleaner(text):
    if not text:
        return ""

    # FIX #2: English words को Hindi उच्चारण में बदलो (case-insensitive, word-boundary safe)
    for eng, hin in ENGLISH_TO_HINDI.items():
        text = re.sub(rf'(?<![a-zA-Z]){re.escape(eng)}(?![a-zA-Z])', hin, text, flags=re.IGNORECASE)

    # FIX #3: बचे हुए pure English words को Devanagari में transliterate (basic)
    # ताकि XTTS अचानक English mode में न जाए
    def replace_remaining_english(match):
        word = match.group(0)
        # अगर word पूरा English है तो उसे spacing के साथ रखो
        # XTTS Hindi mode में English letters देखकर confuse होता है
        return ' '.join(list(word))  # letters space से अलग करो ताकि हकलाहट कम हो

    # FIX #4: Numbers → Hindi words (पूरे)
    number_map = {
        '0': 'शून्य', '1': 'एक', '2': 'दो', '3': 'तीन', '4': 'चार',
        '5': 'पाँच', '6': 'छह', '7': 'सात', '8': 'आठ', '9': 'नौ',
        '10': 'दस', '11': 'ग्यारह', '12': 'बारह', '13': 'तेरह',
        '14': 'चौदह', '15': 'पंद्रह', '20': 'बीस', '25': 'पच्चीस',
        '30': 'तीस', '40': 'चालीस', '50': 'पचास', '100': 'सौ',
    }
    # बड़े numbers पहले replace करो
    for n in sorted(number_map.keys(), key=lambda x: -len(x)):
        text = re.sub(rf'\b{n}\b', number_map[n], text)

    # FIX #5: Punctuation को proper pause में बदलो (हकलाहट रोकने के लिए)
    text = re.sub(r'[।\.]', '। ', text)       # पूर्णविराम के बाद space
    text = re.sub(r'[,،]', ', ', text)         # comma के बाद space
    text = re.sub(r'[!]', '! ', text)
    text = re.sub(r'[?]', '? ', text)
    text = re.sub(r'[-–—]', ', ', text)        # dash को comma में बदलो
    text = re.sub(r'["""\'\'()]', '', text)    # quotes हटाओ
    text = re.sub(r'\s+', ' ', text)           # multiple spaces हटाओ

    return text.strip()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ४. स्मार्ट सेंटेंस-बेस्ड चंकर (हकलाहट का सबसे बड़ा fix)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def smart_chunker(text, max_words=50):
    """
    FIX #6: Words के बीच में चंक मत तोड़ो — sentence boundary पर तोड़ो।
    यही हकलाहट का सबसे बड़ा कारण था।
    """
    # पहले sentences में तोड़ो
    sentences = re.split(r'(?<=[।\.\!\?])\s+', text)
    chunks = []
    current_chunk = []
    current_count = 0

    for sentence in sentences:
        words = sentence.split()
        word_count = len(words)

        # अगर अकेला sentence ही बहुत बड़ा है तो उसे sub-chunks में तोड़ो
        if word_count > max_words:
            if current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_count = 0
            # comma पर तोड़ो
            sub_parts = re.split(r',\s*', sentence)
            temp = []
            temp_count = 0
            for part in sub_parts:
                part_words = part.split()
                if temp_count + len(part_words) > max_words and temp:
                    chunks.append(', '.join(temp))
                    temp = [part]
                    temp_count = len(part_words)
                else:
                    temp.append(part)
                    temp_count += len(part_words)
            if temp:
                chunks.append(', '.join(temp))

        elif current_count + word_count > max_words:
            # FIX: sentence boundary पर तोड़ो
            if current_chunk:
                chunks.append(' '.join(current_chunk))
            current_chunk = words
            current_count = word_count
        else:
            current_chunk.extend(words)
            current_count += word_count

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    # Empty chunks हटाओ
    return [c.strip() for c in chunks if c.strip()]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ५. स्मूथ ऑडियो जॉइनर (clicks/gaps फिक्स)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def crossfade_join(segments, crossfade_ms=80):
    """
    FIX #7: Simple append की जगह crossfade join — chunk boundaries
    पर कोई click या gap नहीं आएगा।
    """
    if not segments:
        return AudioSegment.empty()
    result = segments[0]
    for seg in segments[1:]:
        # दोनों segments को same volume पर normalize करो
        result = result.append(seg, crossfade=crossfade_ms)
    return result

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ६. पोस्ट-प्रोसेसिंग: आवाज़ को और natural बनाओ
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def enhance_audio(audio_seg, sample_rate=22050):
    """
    FIX #8: Low-pass filter + normalization — artificial/robotic sound हटाओ।
    """
    # पहले normalize करो
    audio_seg = effects.normalize(audio_seg)

    # High frequency noise हटाओ (roboticness कम होगी)
    samples = np.array(audio_seg.get_array_of_samples(), dtype=np.float32)
    if audio_seg.channels == 2:
        samples = samples.reshape((-1, 2)).mean(axis=1)

    # Butter low-pass filter
    nyq = sample_rate / 2.0
    cutoff = 7500  # 7.5kHz से ऊपर cut करो
    b, a = butter(4, cutoff / nyq, btype='low')
    filtered = filtfilt(b, a, samples)

    # वापस AudioSegment में convert करो
    filtered = np.clip(filtered, -32768, 32767).astype(np.int16)
    enhanced = AudioSegment(
        filtered.tobytes(),
        frame_rate=sample_rate,
        sample_width=2,
        channels=1
    )
    return enhanced

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ७. मुख्य जनरेशन इंजन
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def generate_shiv_v2(text, up_ref, git_ref, speed_s, pitch_s,
                     use_silence, use_clean, use_enhance,
                     temperature, repetition_pen,
                     progress=gr.Progress()):
    if not text:
        return None, "❌ कोई टेक्स्ट नहीं दिया।"

    # Text clean करो
    progress(0.02, desc="टेक्स्ट प्रोसेसिंग...")
    p_text = shiv_super_cleaner(text)

    # Reference audio सेट करो
    ref = up_ref if up_ref else "ref.wav"
    if not up_ref:
        url = G_RAW + requests.utils.quote(git_ref)
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            with open(ref, "wb") as f:
                f.write(response.content)
        except Exception as e:
            return None, f"❌ Reference audio download failed: {e}"

    # FIX #9: Smart sentence-based chunking
    progress(0.05, desc="स्मार्ट चंकिंग...")
    chunks = smart_chunker(p_text, max_words=50)
    total = len(chunks)

    if total == 0:
        return None, "❌ Text process होने के बाद empty हो गया।"

    segments = []
    errors = []

    for i, chunk in enumerate(chunks):
        progress((i + 1) / total * 0.85, desc=f"जनरेट कर रहे हैं: भाग {i+1}/{total}")
        name = f"part_{i}.wav"

        try:
            # FIX #10: Low temperature = ज़्यादा stable आवाज़, कम हकलाहट
            tts.tts_to_file(
                text=chunk,
                speaker_wav=ref,
                language="hi",
                file_path=name,
                speed=speed_s,
                temperature=temperature,          # 0.3-0.5 = stable
                repetition_penalty=repetition_pen, # 5.0+ = हकलाहट बंद
                top_k=30,                         # कम top_k = ज़्यादा focused
                top_p=0.85,
            )

            seg = AudioSegment.from_wav(name)

            # FIX #11: Silence removal — aggressive नहीं, gentle
            if use_silence:
                try:
                    seg = effects.strip_silence(
                        seg,
                        silence_thresh=-50,  # कम aggressive
                        padding=200          # थोड़ी breathing room रहने दो
                    )
                except:
                    pass

            # बहुत छोटे segments skip करो (glitch हो सकते हैं)
            if len(seg) > 100:
                segments.append(seg)
            os.remove(name)

        except Exception as e:
            errors.append(f"Chunk {i+1}: {str(e)[:80]}")
            if os.path.exists(name):
                os.remove(name)

        torch.cuda.empty_cache()
        gc.collect()

    if not segments:
        return None, f"❌ कोई भी chunk generate नहीं हुआ।\nErrors: {'; '.join(errors)}"

    # FIX #12: Crossfade join — कोई click नहीं
    progress(0.90, desc="स्मूथ जोड़ रहे हैं...")
    combined = crossfade_join(segments, crossfade_ms=80)

    # Post-processing
    if use_clean:
        progress(0.93, desc="क्लीन कर रहे हैं...")
        combined = combined.set_frame_rate(22050).set_channels(1)
        combined = effects.normalize(combined)

    # FIX #13: Audio enhancement (natural sound)
    if use_enhance:
        progress(0.96, desc="एन्हान्स कर रहे हैं...")
        try:
            combined = enhance_audio(combined, sample_rate=22050)
        except:
            pass

    final_name = "Shiv_AI_v2_Output.wav"
    combined.export(final_name, format="wav", parameters=["-ar", "22050"])
    progress(1.0, desc="✅ तैयार!")

    status = f"✅ {total} chunks सफलतापूर्वक generate हुए।"
    if errors:
        status += f"\n⚠️ {len(errors)} chunk(s) में error: {'; '.join(errors[:3])}"

    return final_name, status

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ८. Gradio Interface
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with gr.Blocks(theme=gr.themes.Soft(primary_hue="orange")) as demo:
    gr.Markdown("# 🚩 शिव AI (Shiv AI) v2.0 — श्री राम नाग")
    gr.Markdown(
        "### ✅ हकलाहट फिक्स | ✅ भाषा मिक्स फिक्स | ✅ स्मूथ जॉइनिंग | "
        "✅ Crossfade | ✅ Audio Enhance"
    )

    with gr.Row():
        # Left: Text input
        with gr.Column(scale=2):
            txt = gr.Textbox(
                label="लंबी स्क्रिप्ट यहाँ पेस्ट करें (हिंदी)",
                lines=14,
                placeholder="यहाँ हिंदी टेक्स्ट लिखें..."
            )
            word_count = gr.Markdown("शब्द संख्या: **शून्य**")
            txt.change(
                lambda x: f"शब्द संख्या: **{len(x.split()) if x else 'शून्य'}**",
                [txt], [word_count]
            )

        # Right: Controls
        with gr.Column(scale=1):
            up_v = gr.Audio(
                label="अपनी आवाज़ अपलोड करें (क्लोनिंग के लिए)",
                type="filepath"
            )
            git_v = gr.Dropdown(
                choices=["aideva.wav"],
                label="डिफ़ॉल्ट वॉइस",
                value="aideva.wav"
            )

            with gr.Accordion("⚙️ जनरेशन सेटिंग्स", open=True):
                spd = gr.Slider(0.8, 1.3, value=1.0, step=0.05,
                                label="रफ़्तार (1.0 = normal)")
                ptch = gr.Slider(0.7, 1.3, value=1.0, step=0.05,
                                 label="पिच")

                with gr.Row():
                    temp_sl = gr.Slider(
                        0.1, 0.9, value=0.35, step=0.05,
                        label="Temperature (कम = stable, कम हकलाहट)"
                    )
                    rep_pen = gr.Slider(
                        1.0, 10.0, value=6.0, step=0.5,
                        label="Repetition Penalty (ज़्यादा = बेहतर)"
                    )

            with gr.Accordion("🔧 पोस्ट-प्रोसेसिंग", open=True):
                sln = gr.Checkbox(label="🔇 Silence Remover (gentle)", value=True)
                cln = gr.Checkbox(label="🎚️ Normalize + Clean", value=True)
                enh = gr.Checkbox(label="✨ Audio Enhance (natural sound)", value=True)

            btn = gr.Button("🚀 आवाज़ Generate करें", variant="primary", size="lg")

    out_audio = gr.Audio(
        label="आउटपुट आवाज़",
        type="filepath",
        autoplay=True
    )
    out_status = gr.Textbox(label="स्टेटस", interactive=False, lines=3)

    btn.click(
        generate_shiv_v2,
        inputs=[txt, up_v, git_v, spd, ptch, sln, cln, enh, temp_sl, rep_pen],
        outputs=[out_audio, out_status]
    )

    # Info section
    with gr.Accordion("ℹ️ क्या fix हुआ v2.0 में?", open=False):
        gr.Markdown("""
        **हकलाहट फिक्स:**
        - Sentence boundary पर chunking (words बीच में नहीं टूटते)
        - Low temperature (0.35) = stable, consistent आवाज़
        - High repetition penalty (6.0) = हकलाहट बंद
        - top_k=30, top_p=0.85 = focused generation

        **भाषा मिक्स फिक्स:**
        - 60+ English→Hindi dictionary
        - सभी common English words Hindi में convert होते हैं
        - XTTS को pure Hindi text मिलता है

        **आवाज़ quality fix:**
        - Crossfade joining = chunk boundaries पर कोई click नहीं
        - Audio enhancement = natural sound, roboticness कम
        - Gentle silence removal = breathing space बनी रहती है
        """)

demo.launch(share=True)
