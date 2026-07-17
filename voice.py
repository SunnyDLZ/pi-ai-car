"""
voice.py - 语音输入输出模块

输出: TTS 通过喇叭 (3.5mm 音频口)
输入: USB 麦克风语音识别 (可选)
"""

import os
import subprocess
import threading


class VoiceOutput:
    """语音输出 (TTS 通过喇叭播放)"""

    def __init__(self):
        self._speaking = False

    def init(self):
        """检查 TTS 引擎"""
        self._tts_engine = None
        # 检查 espeak 是否可用
        try:
            result = subprocess.run(["espeak", "--version"],
                                    capture_output=True, timeout=5)
            if result.returncode == 0:
                self._tts_engine = "espeak"
                print("[Voice] TTS 引擎: espeak")
            else:
                print("[Voice] espeak 存在但返回异常，TTS 不可用")
        except FileNotFoundError:
            # 尝试安装 espeak
            print("[Voice] espeak 未安装，尝试安装...")
            ret = os.system("sudo apt install -y espeak espeak-data 2>/dev/null")
            # 验证安装是否成功
            try:
                result = subprocess.run(["espeak", "--version"],
                                        capture_output=True, timeout=5)
                if result.returncode == 0:
                    self._tts_engine = "espeak"
                    print("[Voice] espeak 安装成功")
                else:
                    print("[Voice] espeak 安装失败，TTS 不可用")
            except FileNotFoundError:
                print("[Voice] espeak 安装失败，TTS 不可用")

        return self._tts_engine is not None

    def say(self, text, lang="zh"):
        """TTS 朗读文本

        Args:
            text: 要朗读的文字
            lang: 语言 (zh=中文, en=英文)
        """
        if not self._tts_engine:
            return

        self._speaking = True

        if self._tts_engine == "espeak":
            if lang == "zh":
                # espeak 中文需要指定 zh 语种
                subprocess.Popen(
                    ["espeak", "-v", "zh+f3", "-s", "140", "-p", "50", text],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            else:
                subprocess.Popen(
                    ["espeak", "-v", "en+f3", "-s", "150", "-p", "50", text],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )

        # 等待播报完成
        def _wait_done():
            import time
            time.sleep(len(text) * 0.1 + 0.5)  # 粗估时长
            self._speaking = False

        threading.Thread(target=_wait_done, daemon=True).start()

    def say_wait(self, text, lang="zh"):
        """朗读并等待完成"""
        if not self._tts_engine:
            return
        self.say(text, lang)
        import time
        time.sleep(len(text) * 0.1 + 1)

    def is_speaking(self):
        return self._speaking


class VoiceInput:
    """语音输入 (USB 麦克风 → 语音识别)

    需要安装: sudo apt install -y python3-speechd
    pip install SpeechRecognition
    """

    def __init__(self):
        self._recognizer = None
        self._available = False
        self._energy_threshold = 300

    def init(self):
        """初始化语音识别"""
        try:
            import speech_recognition as sr
            self._recognizer = sr.Recognizer()
            self._recognizer.energy_threshold = self._energy_threshold
            self._recognizer.dynamic_energy_threshold = True
            self._available = True
            print("[VoiceInput] 语音识别初始化完成")
            return True
        except ImportError:
            print("[VoiceInput] SpeechRecognition 未安装，语音输入不可用")
            print("  安装: pip install SpeechRecognition")
            return False

    def listen_once(self, timeout=5, phrase_timeout=3, lang="zh-CN"):
        """一次语音识别

        需要联网 (使用 Google Speech API)

        Args:
            timeout: 总等待时间 (秒)
            phrase_timeout: 静默判定时间 (秒)
            lang: 语言 (zh-CN=中文, en-US=英文)

        Returns:
            str: 识别文本, 失败返回 None
        """
        if not self._available or not self._recognizer:
            return None

        import speech_recognition as sr
        try:
            with sr.Microphone() as source:
                print("[VoiceInput] 请说话...")
                self._recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = self._recognizer.listen(
                    source, timeout=timeout, phrase_time_limit=phrase_timeout
                )
        except sr.WaitTimeoutError:
            print("[VoiceInput] 未检测到语音")
            return None
        except OSError as e:
            print(f"[VoiceInput] 麦克风错误: {e}")
            return None

        try:
            text = self._recognizer.recognize_google(audio, language=lang)
            print(f"[VoiceInput] 识别结果: {text}")
            return text
        except sr.UnknownValueError:
            print("[VoiceInput] 无法识别语音")
            return None
        except sr.RequestError as e:
            print(f"[VoiceInput] 语音服务请求失败: {e}")
            return None
