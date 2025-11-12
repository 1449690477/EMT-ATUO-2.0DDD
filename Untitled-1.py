# -*- coding: utf-8 -*-
"""
苏苏多功能自动化工具
- Tab1：赛琪大烟花（武器突破材料本 60 级）
- Tab2：探险无尽血清 - 人物碎片自动刷取
"""

import os
import sys
import json
import time
import threading
import traceback
import copy
import queue
import random
import importlib.util
import ctypes
import math
import itertools
from ctypes import wintypes
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from collections import deque
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

# ---------- 路径 ----------
if getattr(sys, "frozen", False):
    APP_DIR = os.path.dirname(sys.executable)
    DATA_DIR = getattr(sys, "_MEIPASS", APP_DIR)
else:
    APP_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = APP_DIR


def ensure_directory(path: str):
    try:
        os.makedirs(path, exist_ok=True)
    except OSError:
        # 可能是只读目录（如 _MEIPASS），忽略异常
        pass


def resolve_preferred_directory(preferred_path: str, fallback_path: str) -> str:
    """Return the preferred runtime directory when available.

    Packaged builds may expose writable folders next to the executable. For
    those cases we prefer the sibling directory (``preferred_path``) so custom
    assets persist across updates. When only the bundled resources exist we
    fall back to the internal path shipped with the program. If neither is
    present we create the preferred location to keep behaviour consistent with
    prior releases.
    """

    if os.path.isdir(preferred_path):
        return preferred_path
    if os.path.isdir(fallback_path):
        return fallback_path

    ensure_directory(preferred_path)
    return preferred_path


BASE_DIR = DATA_DIR
TEMPLATE_DIR = os.path.join(DATA_DIR, "templates")
SCRIPTS_DIR = os.path.join(DATA_DIR, "scripts")
CONFIG_PATH = os.path.join(APP_DIR, "config.json")
SP_DIR = os.path.join(DATA_DIR, "SP")
UID_DIR = os.path.join(DATA_DIR, "UID")
MOD_DIR = os.path.join(DATA_DIR, "mod")
WEAPON_BLUEPRINT_DIR = os.path.join(DATA_DIR, "weapon_blueprint")
WQ_DIR = os.path.join(DATA_DIR, "WQ")
HS_DIR = os.path.join(DATA_DIR, "HS")
GAME_DIR = os.path.join(DATA_DIR, "Game")
GAME_SQ_DIR = os.path.join(DATA_DIR, "GAME-sq")
XP50_DIR = os.path.join(DATA_DIR, "50XP")

IS_WINDOWS = sys.platform.startswith("win")
GAME_WINDOW_KEYWORD = "二重螺旋"
INTERNATIONAL_WINDOW_KEYWORD = "Duet Night Abyss"

MOD_DIR = resolve_preferred_directory(os.path.join(APP_DIR, "mod"), MOD_DIR)
WEAPON_BLUEPRINT_DIR = resolve_preferred_directory(
    os.path.join(APP_DIR, "weapon_blueprint"), WEAPON_BLUEPRINT_DIR
)
WQ_DIR = resolve_preferred_directory(os.path.join(APP_DIR, "WQ"), WQ_DIR)
HS_DIR = resolve_preferred_directory(os.path.join(APP_DIR, "HS"), HS_DIR)
GAME_DIR = resolve_preferred_directory(os.path.join(APP_DIR, "Game"), GAME_DIR)
GAME_SQ_DIR = resolve_preferred_directory(os.path.join(APP_DIR, "GAME-sq"), GAME_SQ_DIR)
XP50_DIR = resolve_preferred_directory(os.path.join(APP_DIR, "50XP"), XP50_DIR)

# 新项目：人物密函图片 / 掉落物图片
TEMPLATE_LETTERS_DIR = os.path.join(DATA_DIR, "templates_letters")
TEMPLATE_DROPS_DIR = os.path.join(DATA_DIR, "templates_drops")

for d in (
    TEMPLATE_DIR,
    SCRIPTS_DIR,
    TEMPLATE_LETTERS_DIR,
    TEMPLATE_DROPS_DIR,
    SP_DIR,
    UID_DIR,
    MOD_DIR,
    WEAPON_BLUEPRINT_DIR,
    WQ_DIR,
    HS_DIR,
    GAME_DIR,
    GAME_SQ_DIR,
    XP50_DIR,
):
    ensure_directory(d)

# ---------- 第三方库 ----------
try:
    import pyautogui
    pyautogui.FAILSAFE = False
except Exception:
    pyautogui = None

try:
    import cv2
    import numpy as np
except Exception:
    cv2 = None
    np = None

try:
    import keyboard
except Exception:
    keyboard = None

try:
    import pygetwindow as gw
except Exception:
    gw = None

_pil_spec = importlib.util.find_spec("PIL")
if _pil_spec is not None:
    from PIL import Image, ImageOps, ImageTk
else:
    Image = None
    ImageOps = None
    ImageTk = None

# ---------- 全局 ----------
DEFAULT_CONFIG = {
    "hotkey": "1",
    "support_international": True,
    "wait_seconds": 8.0,
    "macro_a_path": "",
    "macro_b_path": "",
    "auto_loop": False,
    "firework_no_trick": False,
    "guard_settings": {
        "waves": 10,
        "timeout": 160,
        "hotkey": "",
        "no_trick_decrypt": False,
        "auto_e_enabled": True,
        "auto_e_interval": 5.0,
        "auto_q_enabled": False,
        "auto_q_interval": 5.0,
    },
    "expel_settings": {
        "waves": 10,
        "timeout": 160,
        "hotkey": "",
        "auto_e_enabled": True,
        "auto_e_interval": 5.0,
        "auto_q_enabled": False,
        "auto_q_interval": 5.0,
    },
    "mod_guard_settings": {
        "waves": 10,
        "timeout": 160,
        "hotkey": "",
        "no_trick_decrypt": False,
        "auto_e_enabled": True,
        "auto_e_interval": 5.0,
        "auto_q_enabled": False,
        "auto_q_interval": 5.0,
    },
    "mod_expel_settings": {
        "waves": 10,
        "timeout": 160,
        "hotkey": "",
        "auto_e_enabled": True,
        "auto_e_interval": 5.0,
        "auto_q_enabled": False,
        "auto_q_interval": 5.0,
    },
    "weapon_blueprint_guard_settings": {
        "waves": 10,
        "timeout": 160,
        "hotkey": "",
        "no_trick_decrypt": False,
        "auto_e_enabled": True,
        "auto_e_interval": 5.0,
        "auto_q_enabled": False,
        "auto_q_interval": 5.0,
    },
    "weapon_blueprint_expel_settings": {
        "waves": 10,
        "timeout": 160,
        "hotkey": "",
        "auto_e_enabled": True,
        "auto_e_interval": 5.0,
        "auto_q_enabled": False,
        "auto_q_interval": 5.0,
    },
    "xp50_settings": {
        "hotkey": "",
        "wait_seconds": 120.0,
        "loop_count": 0,
        "auto_loop": True,
        "no_trick_decrypt": True,
    },
    "hs70_settings": {
        "hotkey": "",
        "loop_count": 0,
        "auto_loop": True,
        "no_trick_decrypt": True,
    },
}

GAME_REGION = None
worker_stop = threading.Event()
round_running_lock = threading.Lock()
hotkey_handle = None

app = None             # 赛琪大烟花 GUI 实例
xp50_app = None        # 50 经验副本 GUI 实例
hs70_app = None        # 70 红珠副本 GUI 实例
fragment_apps = []     # 人物碎片 GUI 实例列表

# 手动常驻解密日志抑制
log_context = threading.local()

# 独立无巧手解密（常驻）控制
manual_firework_service = None
manual_line_service = None
manual_firework_var = None
manual_line_var = None
manual_collapse_active = False
manual_original_geometry = None
manual_expand_button = None
root_window = None
toolbar_frame = None
manual_previous_minsize = None
uid_mask_manager = None
international_support_var = None
config_data = None

INTERNATIONAL_SUPPORT_ENABLED = True

tk_call_queue = queue.Queue()
ACTIVE_FRAGMENT_GUI = None


def post_to_main_thread(func, *args, **kwargs):
    if func is None:
        return
    tk_call_queue.put((func, args, kwargs))


def set_international_support_enabled(enabled: bool):
    global INTERNATIONAL_SUPPORT_ENABLED
    INTERNATIONAL_SUPPORT_ENABLED = bool(enabled)


def get_active_window_keywords() -> Tuple[str, ...]:
    if INTERNATIONAL_SUPPORT_ENABLED:
        return (GAME_WINDOW_KEYWORD, INTERNATIONAL_WINDOW_KEYWORD)
    return (GAME_WINDOW_KEYWORD,)


def get_window_name_hint() -> str:
    if INTERNATIONAL_SUPPORT_ENABLED:
        return "『二重螺旋』或『Duet Night Abyss』"
    return "『二重螺旋』"


def start_ui_dispatch_loop(root, interval_ms: int = 30):
    def _drain_queue():
        while True:
            try:
                func, args, kwargs = tk_call_queue.get_nowait()
            except queue.Empty:
                break
            try:
                func(*args, **kwargs)
            except Exception:
                traceback.print_exc()
        root.after(interval_ms, _drain_queue)

    _drain_queue()


def set_active_fragment_gui(gui):
    global ACTIVE_FRAGMENT_GUI
    ACTIVE_FRAGMENT_GUI = gui


def get_active_fragment_gui():
    return ACTIVE_FRAGMENT_GUI


# 人物碎片：通用按钮名（放在 templates/）
BTN_OPEN_LETTER = "选择密函.png"
BTN_CONFIRM_LETTER = "确认选择.png"
BTN_RETREAT_START = "撤退.png"
BTN_EXPEL_NEXT_WAVE = "再次进行.png"
BTN_CONTINUE_CHALLENGE = "继续挑战.png"

CLUE_MACRO_A = "人物碎片刷取A.json"
CLUE_MACRO_B = "人物碎片刷取B.json"
CLUE_MACRO_30 = "30级.json"
CLUE_MAP_30_TEMPLATE = "30map.png"
CLUE_LEVEL_10_TEMPLATE = "10.png"
CLUE_LEVEL_30_TEMPLATE = "30.png"
CLUE_FIRE_TEMPLATE = "火.png"
CLUE_SETTINGS_TEMPLATE = "设置.png"
CLUE_EXIT_ENTRUST_TEMPLATE = "退出委托.png"
CLUE_TRAINING_TEMPLATE = "历练.png"
CLUE_ENTRUST_TEMPLATE = "委托.png"
CLUE_ADVENTURE_TEMPLATE = "探险.png"
CLUE_AUTO_SWITCH_MIN_SECONDS = 60.0

AUTO_REVIVE_TEMPLATE = "x.png"
AUTO_REVIVE_THRESHOLD = 0.8
AUTO_REVIVE_CHECK_INTERVAL = 10.0
AUTO_REVIVE_HOLD_SECONDS = 6.0

LETTER_MATCH_THRESHOLD = 0.8
LETTER_IMAGE_SIZE = 128

UID_MASK_ALPHA = 0.92
UID_MASK_CELL = 10
UID_MASK_COLORS = ("#2e2f3a", "#4a4c5e", "#5c6075", "#3c3e4e")
UID_FIXED_MASKS = (
    # (relative_x, relative_y, width, height)
    (830, 1090, 260, 24),   # HUD 底部 UID
    (60, 1090, 260, 24),    # 左下角载入界面 UID，保持与 HUD 一致
)
UID_WINDOW_MISS_LIMIT = 60


def get_template_name(name: str, default: str) -> str:
    """Gracefully fall back to default if a global template name is missing."""
    return globals().get(name, default)


# ---------- 小工具 ----------
def format_hms(sec: float) -> str:
    sec = int(max(0, sec))
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


# ---------- 日志 / 进度 ----------

LOG_CATEGORY_STYLES = {
    "error": {"icon": "❌", "color": "#ff4d4f"},
    "warning": {"icon": "⚠️", "color": "#fa8c16"},
    "stop": {"icon": "⏹️", "color": "#1890ff"},
    "start": {"icon": "▶️", "color": "#1890ff"},
    "detect": {"icon": "🔍", "color": "#52c41a"},
    "success": {"icon": "✅", "color": "#52c41a"},
    "info": {"icon": "•", "color": "#4a4a4a"},
}

LOG_KEYWORD_RULES = (
    ("error", ["错误", "异常", "失败", "崩溃", "缺少", "未找到", "卡死", "未能", "未成功", "报错"]),
    ("warning", ["警告", "超时", "未识别", "未匹配", "未检测", "风险", "注意", "重试", "无法识别"]),
    ("stop", ["停止", "终止", "结束监听", "结束", "退出", "中止"]),
    ("detect", ["识别", "匹配", "检测", "发现", "回放进度", "命中"]),
    ("start", ["开始", "启动", "准备执行", "等待开始", "进入", "监听"],),
    ("success", ["完成", "成功", "已保存", "已更新", "已加载", "准备就绪", "解密完成"]),
)


def categorize_log_message(message: str) -> str:
    """Return a category key based on keywords contained in the message."""

    if not message:
        return "info"

    for category, keywords in LOG_KEYWORD_RULES:
        for kw in keywords:
            if kw in message:
                return category
    return "info"


def get_log_icon(category: str) -> str:
    meta = LOG_CATEGORY_STYLES.get(category) or LOG_CATEGORY_STYLES["info"]
    return meta.get("icon", "•")


def ensure_log_widget_styles(widget):
    if widget is None:
        return
    if getattr(widget, "_log_styles_ready", False):
        return
    try:
        widget.configure(wrap="word", font=("Microsoft YaHei", 10))
    except Exception:
        try:
            widget.configure(wrap="word")
        except Exception:
            pass
    widget.tag_configure("timestamp", foreground="#888888")
    for key, meta in LOG_CATEGORY_STYLES.items():
        widget.tag_configure(key, foreground=meta.get("color", "#333333"))
    widget._log_styles_ready = True


def append_formatted_log(widget, message: str):
    if widget is None:
        return
    ensure_log_widget_styles(widget)
    category = categorize_log_message(message)
    icon = get_log_icon(category)
    timestamp = time.strftime("[%H:%M:%S] ")
    try:
        widget.insert("end", timestamp, ("timestamp",))
        widget.insert("end", f"{icon} ", (category,))
        widget.insert("end", message + "\n", (category,))
        widget.see("end")
    except Exception:
        # As a fallback avoid crashing the caller if the widget is unavailable.
        pass


def register_fragment_app(gui):
    if gui not in fragment_apps:
        fragment_apps.append(gui)


def log(msg: str):
    if getattr(log_context, "suppress", False):
        return
    category = categorize_log_message(msg)
    icon = get_log_icon(category)
    ts = time.strftime("[%H:%M:%S] ")
    print(f"{ts}{icon} {msg}")
    if app is not None:
        app.log(msg)
    if xp50_app is not None:
        try:
            xp50_app.log(msg)
        except Exception:
            pass
    if hs70_app is not None:
        try:
            hs70_app.log(msg)
        except Exception:
            pass
    for gui in fragment_apps:
        try:
            gui.log(msg)
        except Exception:
            pass


def report_progress(p: float):
    if app is not None:
        app.set_progress(p)
    if xp50_app is not None:
        try:
            xp50_app.on_global_progress(p)
        except Exception:
            pass


GOAL_STYLE_INITIALIZED = False


def ensure_goal_progress_style():
    global GOAL_STYLE_INITIALIZED
    if GOAL_STYLE_INITIALIZED:
        return
    try:
        style = ttk.Style()
        style.configure(
            "Goal.Horizontal.TProgressbar",
            troughcolor="#ffe6fa",
            background="#5aa9ff",
            bordercolor="#f8b5dd",
            lightcolor="#8fc5ff",
            darkcolor="#ff92cf",
        )
        GOAL_STYLE_INITIALIZED = True
    except Exception:
        pass


class CollapsibleLogPanel(tk.Frame):
    """Small helper that shows a toggle button and a collapsible log area."""

    def __init__(self, parent, title: str, text_height: int = 10):
        super().__init__(parent)
        self.title = title
        self._opened = tk.BooleanVar(value=False)

        header = tk.Frame(self)
        header.pack(fill="x")

        self.toggle_btn = ttk.Checkbutton(
            header,
            text="展开日志",
            variable=self._opened,
            command=self._update_visibility,
        )
        self.toggle_btn.pack(side="left", anchor="w")
        try:
            self.toggle_btn.configure(takefocus=False)
        except Exception:
            pass

        self.body = tk.LabelFrame(self, text=title)

        self.text = tk.Text(self.body, height=text_height, wrap="word")
        self.text.pack(side="left", fill="both", expand=True)

        scrollbar = tk.Scrollbar(self.body, command=self.text.yview)
        scrollbar.pack(side="right", fill="y")
        self.text.config(yscrollcommand=scrollbar.set)

        ensure_log_widget_styles(self.text)

        self._update_visibility()

    def _update_visibility(self):
        if self._opened.get():
            self.toggle_btn.config(text="折叠日志")
            self.body.pack(fill="both", expand=True, pady=(2, 0))
        else:
            self.toggle_btn.config(text="展开日志")
            self.body.pack_forget()

    def append(self, message: str):
        append_formatted_log(self.text, message)

    def clear(self):
        self.text.delete("1.0", "end")

class StandaloneNoTrickStub:
    """Minimal GUI stub used by 常驻无巧手解密服务."""

    def __init__(self, log_prefix: str):
        self.log_prefix = log_prefix
        self.suppress_log = True
        self._finished = threading.Event()
        self.last_session = None

    def reset(self):
        self._finished.clear()
        self.last_session = None

    def on_no_trick_unavailable(self, reason: str):
        pass

    def on_no_trick_no_templates(self, game_dir: str):
        pass

    def on_no_trick_monitor_started(self, templates):
        pass

    def on_no_trick_detected(self, entry, score: float):
        pass

    def on_no_trick_macro_start(self, entry, score: float):
        pass

    def on_no_trick_progress(self, progress: float):
        pass

    def on_no_trick_macro_complete(self, entry):
        pass

    def on_no_trick_macro_missing(self, entry):
        pass

    def on_no_trick_session_finished(self, triggered: bool, macro_executed: bool, macro_missing: bool):
        self.last_session = (triggered, macro_executed, macro_missing)
        self._finished.set()

    def on_no_trick_idle(self, remaining: float):
        pass

    def on_no_trick_idle_complete(self):
        pass

    @property
    def finished_event(self):
        return self._finished


class StandaloneDecryptService:
    """Run a decrypt controller in the background until explicitly stopped."""

    def __init__(self, controller_cls, game_dir: str, log_prefix: str, cooldown_seconds: float = 0.0):
        self.controller_cls = controller_cls
        self.game_dir = game_dir
        self.log_prefix = log_prefix
        self._thread = None
        self._stop_event = threading.Event()
        self._cooldown_seconds = max(0.0, float(cooldown_seconds or 0.0))
        self._cooldown_until = 0.0

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            try:
                self._thread.join(timeout=1.5)
            except Exception:
                pass
        self._thread = None

    def _run(self):
        log_context.suppress = True
        try:
            while not self._stop_event.is_set():
                if self._cooldown_seconds > 0.0:
                    remaining = self._cooldown_until - time.time()
                    if remaining > 0.0:
                        if self._stop_event.wait(min(0.5, remaining)):
                            break
                        continue
                if GAME_REGION is None and not init_game_region():
                    if self._stop_event.wait(3.0):
                        break
                    continue
                stub = StandaloneNoTrickStub(self.log_prefix)
                stub.reset()
                controller = self.controller_cls(stub, self.game_dir)
                try:
                    if not controller.start():
                        controller.stop()
                        controller.finish_session()
                        if self._stop_event.wait(3.0):
                            break
                        continue
                except Exception:
                    if self._stop_event.wait(3.0):
                        break
                    continue

                keyboard_state = KeyboardPlaybackState()

                try:
                    while not self._stop_event.is_set() and controller.session_started:
                        try:
                            pause = controller.run_decrypt_if_needed(
                                keyboard_state=keyboard_state
                            )
                        except TypeError:
                            pause = controller.run_decrypt_if_needed()
                        if pause and pause > 0:
                            time.sleep(min(0.05, pause))
                        else:
                            time.sleep(0.05)
                except Exception:
                    pass
                finally:
                    try:
                        controller.stop()
                    except Exception:
                        pass
                    try:
                        controller.finish_session()
                    except Exception:
                        pass

                session = getattr(stub, "last_session", None)
                if (
                    self._cooldown_seconds > 0.0
                    and session
                    and session[1]
                ):
                    self._cooldown_until = time.time() + self._cooldown_seconds

                if self._stop_event.wait(0.2):
                    break
        finally:
            try:
                log_context.suppress = False
            except Exception:
                pass


def load_preview_image(path: str, max_size: int = 72):
    if not path or not os.path.exists(path):
        return None
    try:
        img = tk.PhotoImage(file=path)
        w = max(img.width(), 1)
        h = max(img.height(), 1)
        scale = max(1, (max(w, h) + max_size - 1) // max_size)
        if scale > 1:
            img = img.subsample(scale, scale)
        return img
    except Exception:
        return None


# ---------- 配置 ----------
def load_config():
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg.update(json.load(f))
        except Exception as e:
            log(f"读取配置失败：{e}")
    return cfg


def save_config(cfg: dict):
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
        log("配置已保存。")
    except Exception as e:
        log(f"保存配置失败：{e}")


# ---------- 游戏窗口 / 截图 ----------
if IS_WINDOWS:
    try:
        _user32 = ctypes.windll.user32
    except (AttributeError, OSError):
        _user32 = None
    try:
        _shcore = ctypes.windll.shcore
    except (AttributeError, OSError):
        _shcore = None
else:
    _user32 = None
    _shcore = None

_dpi_awareness_applied = False


def ensure_windows_dpi_awareness():
    global _dpi_awareness_applied
    if _dpi_awareness_applied or not IS_WINDOWS or _user32 is None:
        return
    _dpi_awareness_applied = True

    try:
        DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 = ctypes.c_void_p(-4)
        _user32.SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2)
        return
    except Exception:
        pass

    if _shcore is not None:
        try:
            _shcore.SetProcessDpiAwareness(2)
            return
        except Exception:
            pass

    try:
        _user32.SetProcessDPIAware()
    except Exception:
        pass


def _enum_windows_by_title(keywords):
    if _user32 is None:
        return []

    if isinstance(keywords, str):
        keywords = [keywords]
    keywords = [k for k in (keywords or []) if k]
    if not keywords:
        return []

    handles = []

    @ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
    def _enum_proc(hwnd, lparam):
        if not _user32.IsWindowVisible(hwnd):
            return True
        length = _user32.GetWindowTextLengthW(hwnd)
        if length <= 0:
            return True
        buffer = ctypes.create_unicode_buffer(length + 1)
        _user32.GetWindowTextW(hwnd, buffer, length + 1)
        title = buffer.value
        if title:
            for keyword in keywords:
                if keyword in title:
                    handles.append(hwnd)
                    return False
        return True

    _user32.EnumWindows(_enum_proc, 0)
    return handles


def get_game_client_rect(title_keywords=None):
    if _user32 is None:
        return None

    if title_keywords is None:
        title_keywords = get_active_window_keywords()

    handles = _enum_windows_by_title(title_keywords)
    if not handles:
        return None

    hwnd = handles[0]
    rect = wintypes.RECT()
    if not _user32.GetClientRect(hwnd, ctypes.byref(rect)):
        return None

    origin = wintypes.POINT(0, 0)
    if not _user32.ClientToScreen(hwnd, ctypes.byref(origin)):
        return None

    width = rect.right - rect.left
    height = rect.bottom - rect.top
    if width <= 0 or height <= 0:
        return None

    return hwnd, origin.x, origin.y, width, height


def focus_game_window(hwnd):
    if _user32 is None or not hwnd:
        return
    try:
        _user32.ShowWindow(hwnd, 9)  # SW_RESTORE
    except Exception:
        pass
    try:
        _user32.SetForegroundWindow(hwnd)
    except Exception:
        pass


def find_game_window():
    if gw is None:
        log("未安装 pygetwindow，无法定位游戏窗口。")
        return None
    try:
        wins = gw.getAllWindows()
    except Exception as e:
        log(f"获取窗口列表失败：{e}")
        return None
    keywords = get_active_window_keywords()
    for w in wins:
        title = (w.title or "")
        if any(keyword in title for keyword in keywords) and w.width > 400 and w.height > 300:
            return w
    log(f"未找到标题包含{get_window_name_hint()}的窗口。")
    return None


def init_game_region():
    """以窗口中心 1920x1080 作为识别区域"""
    global GAME_REGION
    if pyautogui is None:
        log("未安装 pyautogui，无法截图。")
        return False
    win = find_game_window()
    if not win:
        return False
    cx = win.left + win.width // 2
    cy = win.top + win.height // 2
    GAME_REGION = (cx - 960, cy - 540, 1920, 1080)
    log(
        f"使用窗口中心区域：left={GAME_REGION[0]}, "
        f"top={GAME_REGION[1]}, w={GAME_REGION[2]}, h={GAME_REGION[3]}"
    )
    return True


def screenshot_game():
    if GAME_REGION is None:
        raise RuntimeError("GAME_REGION 未初始化")
    if pyautogui is None:
        raise RuntimeError("未安装 pyautogui")
    img = pyautogui.screenshot(region=GAME_REGION)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


# ---------- 模板匹配（templates/） ----------
def load_template(name: str):
    if cv2 is None or np is None:
        log("缺少 opencv/numpy，无法图像识别。")
        return None
    path = os.path.join(TEMPLATE_DIR, name)
    if not os.path.exists(path):
        log(f"模板不存在：{path}")
        return None
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if img is None:
        log(f"无法读取模板：{path}")
    return img


def match_template(name: str):
    tpl = load_template(name)
    if tpl is None:
        return 0.0, None, None
    img = screenshot_game()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    res = cv2.matchTemplate(gray, tpl, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    th, tw = tpl.shape[:2]
    x = GAME_REGION[0] + max_loc[0] + tw // 2
    y = GAME_REGION[1] + max_loc[1] + th // 2
    return max_val, x, y


def wait_for_template(name, step_name, timeout=20.0, threshold=0.5):
    start = time.time()
    while time.time() - start < timeout and not worker_stop.is_set():
        score, _, _ = match_template(name)
        log(f"{step_name} 匹配度 {score:.3f}")
        if score >= threshold:
            log(f"{step_name} 匹配成功。")
            return True
        time.sleep(0.5)
    return False


def perform_click(
    x: Optional[float] = None,
    y: Optional[float] = None,
    *,
    button: str = "left",
    clicks: int = 1,
    interval: float = 0.0,
) -> bool:
    """执行一次或多次稳定的鼠标点击动作。"""

    if pyautogui is None:
        log("pyautogui 不可用，无法执行鼠标点击。")
        return False

    try:
        target_x = None if x is None else int(round(x))
        target_y = None if y is None else int(round(y))

        if target_x is not None and target_y is not None:
            pyautogui.moveTo(target_x, target_y)
            time.sleep(0.02)

        try:
            total_clicks = int(clicks)
        except (TypeError, ValueError):
            total_clicks = 1
        total_clicks = max(1, total_clicks)

        try:
            interval_val = float(interval)
        except (TypeError, ValueError):
            interval_val = 0.0
        interval_val = max(0.0, interval_val)

        for idx in range(total_clicks):
            if target_x is not None and target_y is not None:
                pyautogui.click(target_x, target_y, button=button)
            else:
                pyautogui.click(button=button)
            if idx < total_clicks - 1 and interval_val > 0:
                time.sleep(interval_val)
        return True
    except Exception as exc:
        log(f"执行鼠标点击失败：{exc}")
        return False


def wait_and_click_template(name, step_name, timeout=15.0, threshold=0.8):
    start = time.time()
    while time.time() - start < timeout and not worker_stop.is_set():
        score, x, y = match_template(name)
        log(f"{step_name} 匹配度 {score:.3f}")
        if score >= threshold and x is not None:
            if perform_click(x, y):
                log(f"{step_name} 点击 ({x},{y})")
                return True
            log(f"{step_name} 点击 ({x},{y}) 失败，重试。")
        time.sleep(0.5)
    return False


def click_template(name, step_name, threshold=0.7):
    score, x, y = match_template(name)
    if score >= threshold and x is not None:
        if perform_click(x, y):
            log(f"{step_name} 点击 ({x},{y}) 匹配度 {score:.3f}")
            return True
        log(f"{step_name} 点击 ({x},{y}) 失败，匹配度 {score:.3f}")
    log(f"{step_name} 匹配度 {score:.3f}，未点击。")
    return False


def is_exit_ui_visible(threshold=0.8) -> bool:
    """检测退图界面（exit_step1/exit_step2 任一）"""
    for nm in ("exit_step1.png", "exit_step2.png"):
        score, _, _ = match_template(nm)
        if score >= threshold:
            log(f"检测到退图界面：{nm} 匹配度 {score:.3f}")
            return True
    return False


# ---------- 模板匹配（任意路径：人物密函 / 掉落物） ----------
def load_template_from_path(path: str):
    if cv2 is None or np is None:
        log("缺少 opencv/numpy，无法图像识别。")
        return None
    if not os.path.exists(path):
        log(f"模板不存在：{path}")
        return None
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if img is None:
        log(f"无法读取模板：{path}")
    return img


def match_template_from_path(path: str):
    tpl = load_template_from_path(path)
    if tpl is None:
        return 0.0, None, None
    img = screenshot_game()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    res = cv2.matchTemplate(gray, tpl, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    th, tw = tpl.shape[:2]
    x = GAME_REGION[0] + max_loc[0] + tw // 2
    y = GAME_REGION[1] + max_loc[1] + th // 2
    return max_val, x, y


def wait_and_click_template_from_path(
    path: str,
    step_name: str,
    timeout: float = 15.0,
    threshold: float = LETTER_MATCH_THRESHOLD,
) -> bool:
    start = time.time()
    while time.time() - start < timeout and not worker_stop.is_set():
        score, x, y = match_template_from_path(path)
        log(f"{step_name} 匹配度 {score:.3f}")
        if score >= threshold and x is not None:
            if perform_click(x, y):
                log(f"{step_name} 点击 ({x},{y})")
                return True
            log(f"{step_name} 点击 ({x},{y}) 失败，重试。")
        time.sleep(0.5)
    return False


def click_template_from_path(
    path: str,
    step_name: str,
    threshold: float = LETTER_MATCH_THRESHOLD,
) -> bool:
    score, x, y = match_template_from_path(path)
    if score >= threshold and x is not None:
        if perform_click(x, y):
            log(f"{step_name} 点击 ({x},{y}) 匹配度 {score:.3f}")
            return True
        log(f"{step_name} 点击 ({x},{y}) 失败，匹配度 {score:.3f}")
    log(f"{step_name} 匹配度 {score:.3f}，未点击。")
    return False


LETTER_SCROLL_TEMPLATE = "不使用.png"
LETTER_SCROLL_ATTEMPTS = 20
LETTER_SCROLL_AMOUNT = -120
# 将滚动重试的等待时间进一步缩短，约为此前的三十倍，
# 以最快速度在滚动之间重新尝试识别。
LETTER_SCROLL_DELAY = 0.00016
LETTER_SCROLL_INITIAL_WAIT = 1.0


def _scroll_letter_list_and_retry(
    path: str,
    step_name: str,
    threshold: float = LETTER_MATCH_THRESHOLD,
    anchor_threshold: float = 0.5,
) -> bool:
    if pyautogui is None:
        log(f"{step_name}：pyautogui 不可用，无法滚动列表。")
        return False

    anchor_template = get_template_name("LETTER_SCROLL_TEMPLATE", LETTER_SCROLL_TEMPLATE)
    score, anchor_x, anchor_y = match_template(anchor_template)
    log(f"{step_name}：定位 {anchor_template} 匹配度 {score:.3f}")
    if score < anchor_threshold or anchor_x is None:
        log(f"{step_name}：未找到 {anchor_template}，无法滚动查找。")
        return False

    pyautogui.moveTo(anchor_x, anchor_y)

    for attempt in range(LETTER_SCROLL_ATTEMPTS):
        if worker_stop.is_set():
            return False
        pyautogui.scroll(LETTER_SCROLL_AMOUNT, x=anchor_x, y=anchor_y)
        log(f"{step_name}：第 {attempt + 1} 次滚动后重新识别…")
        time.sleep(LETTER_SCROLL_DELAY)
        score, x, y = match_template_from_path(path)
        log(f"{step_name} 滚动后匹配度 {score:.3f}")
        if score >= threshold and x is not None:
            if perform_click(x, y):
                log(f"{step_name} 点击 ({x},{y})（滚动第 {attempt + 1} 次）")
                return True
            log(
                f"{step_name} 点击 ({x},{y}) 失败（滚动第 {attempt + 1} 次），继续尝试。"
            )

    log(
        f"{step_name}：滚动 {LETTER_SCROLL_ATTEMPTS} 次后仍未找到目标密函，停止尝试。"
    )
    return False


def click_letter_template(
    path: str,
    step_name: str,
    timeout: float = 20.0,
    threshold: float = LETTER_MATCH_THRESHOLD,
) -> bool:
    initial_timeout = min(timeout, LETTER_SCROLL_INITIAL_WAIT)
    if initial_timeout > 0:
        if wait_and_click_template_from_path(
            path, step_name, initial_timeout, threshold
        ):
            return True

    log(f"{step_name}：初次匹配失败，尝试滚动列表寻找目标密函。")
    if pyautogui is None:
        log(f"{step_name}：pyautogui 不可用，无法滚动或再次匹配。")
        return False
    if _scroll_letter_list_and_retry(path, step_name, threshold):
        return True

    remaining_timeout = max(0.0, timeout - initial_timeout)
    if remaining_timeout > 0:
        end_time = time.time() + remaining_timeout
        while time.time() < end_time and not worker_stop.is_set():
            score, x, y = match_template_from_path(path)
            if score >= threshold and x is not None:
                if perform_click(x, y):
                    log(f"{step_name} 点击 ({x},{y})（滚动后等待匹配）")
                    return True
                log(
                    f"{step_name} 点击 ({x},{y}) 失败（滚动后等待匹配），继续等待。"
                )
            time.sleep(0.1)

    return False


def load_uniform_letter_image(path: str, box_size: int = LETTER_IMAGE_SIZE):
    if Image is not None and ImageTk is not None:
        try:
            with Image.open(path) as pil_img:
                pil_img = pil_img.convert("RGBA")
                fitted = ImageOps.contain(pil_img, (box_size, box_size))
                background = Image.new("RGBA", (box_size, box_size), (0, 0, 0, 0))
                offset = (
                    (box_size - fitted.width) // 2,
                    (box_size - fitted.height) // 2,
                )
                background.paste(fitted, offset, fitted)
                return ImageTk.PhotoImage(background)
        except Exception as exc:
            log(f"加载图片失败：{path}，{exc}")

    try:
        tk_img = tk.PhotoImage(file=path)
    except Exception as exc:
        log(f"加载图片失败：{path}，{exc}")
        return None

    max_side = max(tk_img.width(), tk_img.height())
    if max_side > box_size:
        scale = max(1, (max_side + box_size - 1) // box_size)
        tk_img = tk_img.subsample(scale, scale)

    canvas = tk.PhotoImage(width=box_size, height=box_size)
    offset_x = max((box_size - tk_img.width()) // 2, 0)
    offset_y = max((box_size - tk_img.height()) // 2, 0)
    canvas.tk.call(
        canvas,
        "copy",
        tk_img,
        "-from",
        0,
        0,
        tk_img.width(),
        tk_img.height(),
        "-to",
        offset_x,
        offset_y,
    )
    return canvas


# ---------- 自定义脚本：上下文与模块基础 ----------


@dataclass
class ScriptNode:
    """Represent a node on the custom script canvas."""

    node_id: int
    module_type: str
    module: "CustomModuleDefinition"
    config: Dict[str, Any]


class CustomScriptContext:
    """Runtime context shared across custom script modules."""

    def __init__(self, gui=None):
        self.gui = gui
        self.loop_enabled = False
        self.loop_limit = 0
        self.completed_loops = 0
        self.terminated = False
        self.state: Dict[str, Any] = {}
        self.last_result = True
        self.log_prefix = "[自定义脚本]"

    # ---- 状态 ----
    def should_stop(self) -> bool:
        return worker_stop.is_set() or self.terminated

    def fail(self, message: Optional[str] = None):
        if message:
            self.log(message)
        self.last_result = False

    def reset_last_result(self):
        self.last_result = True

    # ---- 循环控制 ----
    def request_loop(self, enabled: bool, limit: int = 0):
        self.loop_enabled = bool(enabled)
        self.loop_limit = max(0, int(limit or 0))
        self.completed_loops = 0

    def advance_loop(self) -> bool:
        self.completed_loops += 1
        if self.loop_limit and self.completed_loops >= self.loop_limit:
            self.loop_enabled = False
            return False
        return self.loop_enabled

    def terminate(self):
        self.terminated = True

    # ---- 日志 ----
    def log(self, message: str):
        prefix = self.log_prefix
        text = f"{prefix} {message}" if prefix else message
        if self.gui is not None:
            try:
                self.gui.queue_log(text)
            except Exception:
                pass
        log(text)


class CustomModuleDefinition:
    """Base class for modules reusable by the custom script editor."""

    module_type = "base"
    display_name = "基础模块"
    description = ""
    allow_manual_add = True

    def default_config(self) -> Dict[str, Any]:
        return {}

    def summary(self, config: Dict[str, Any]) -> str:
        return self.display_name

    def execute(self, context: CustomScriptContext, config: Dict[str, Any]) -> bool:
        """Execute module logic. Return ``True`` when successful."""

        return True


CUSTOM_MODULE_REGISTRY: Dict[str, CustomModuleDefinition] = {}
SCRIPT_NODE_COUNTER = itertools.count(1)


def register_custom_module(module_cls):
    inst = module_cls()
    CUSTOM_MODULE_REGISTRY[inst.module_type] = inst
    return inst


def create_script_node(module_type: str) -> ScriptNode:
    module = CUSTOM_MODULE_REGISTRY[module_type]
    config = copy.deepcopy(module.default_config())
    node_id = next(SCRIPT_NODE_COUNTER)
    return ScriptNode(node_id=node_id, module_type=module_type, module=module, config=config)


def sleep_with_stop(seconds: float):
    end_time = time.time() + max(0.0, float(seconds))
    while time.time() < end_time and not worker_stop.is_set():
        remaining = end_time - time.time()
        time.sleep(min(0.2, max(0.01, remaining)))


def _resolve_template_source(template: str, source: str) -> str:
    if source == "custom" or (template and os.path.isfile(template)):
        return template
    return get_template_name(template, template)


def _wait_and_click_generic(
    template: str,
    step_name: str,
    timeout: float,
    threshold: float,
    source: str = "global",
) -> bool:
    resolved = _resolve_template_source(template, source)
    if not resolved:
        return False
    if source == "custom" or os.path.isfile(resolved):
        return wait_and_click_template_from_path(resolved, step_name, timeout, threshold)
    return wait_and_click_template(resolved, step_name, timeout, threshold)


def _wait_for_generic(
    template: str,
    step_name: str,
    timeout: float,
    threshold: float,
    source: str = "global",
) -> bool:
    resolved = _resolve_template_source(template, source)
    if not resolved:
        return False
    if source == "custom" or os.path.isfile(resolved):
        end_time = time.time() + timeout
        while time.time() < end_time and not worker_stop.is_set():
            score, _, _ = match_template_from_path(resolved)
            log(f"{step_name} 匹配度 {score:.3f}")
            if score >= threshold:
                log(f"{step_name} 匹配成功。")
                return True
            time.sleep(0.3)
        return False
    return wait_for_template(resolved, step_name, timeout, threshold)


def _click_generic(
    template: str,
    step_name: str,
    threshold: float,
    source: str = "global",
) -> bool:
    resolved = _resolve_template_source(template, source)
    if not resolved:
        return False
    if source == "custom" or os.path.isfile(resolved):
        return click_template_from_path(resolved, step_name, threshold)
    return click_template(resolved, step_name, threshold)


def _play_macro_path(
    path: str,
    label: str,
    context: CustomScriptContext,
    allow_segment: bool = True,
) -> bool:
    if not path:
        context.log(f"{label}：未配置宏文件路径。")
        return False
    if not os.path.exists(path):
        context.log(f"{label}：未找到宏文件 {path}")
        return False

    played = False

    def progress_cb(p):
        try:
            context.gui.update_progress(p * 100.0)
        except Exception:
            pass

    if allow_segment and macro_has_segments(path):
        result = play_segment_macro(path, label, progress_callback=progress_cb)
        played = bool(result)
        if result is None:
            context.log(f"{label}：分段轨迹宏回放提前结束。")
    if not played:
        played = bool(
            play_macro(
                path,
                label,
                0.0,
                0.0,
                interrupt_on_exit=False,
                progress_callback=progress_cb,
            )
        )
    if played:
        context.log(f"{label}：宏执行完成。")
    else:
        context.log(f"{label}：宏执行失败或被中断。")
    return played


def _press_key_once(key: str):
    if not key:
        return False
    try:
        if keyboard is not None:
            keyboard.press_and_release(key)
        else:
            pyautogui.press(key)
    except Exception as exc:
        log(f"按键 {key} 失败：{exc}")
        return False
    return True


def _press_keys(keys: List[str], interval: float = 0.05):
    ok = True
    for idx, key in enumerate(keys):
        ok = _press_key_once(key) and ok
        if interval > 0 and idx < len(keys) - 1:
            sleep_with_stop(interval)
            if worker_stop.is_set():
                break
    return ok


class TemplateSequenceModule(CustomModuleDefinition):
    module_type = "template_sequence"
    display_name = "模板操作序列"
    description = (
        "按顺序执行一组模板动作，支持点击、等待、按键及宏回放，可由用户自定义每一步。"
    )

    def default_config(self) -> Dict[str, Any]:
        return {
            "halt_on_fail": True,
            "steps": [
                {
                    "label": "点击开始挑战",
                    "template": "开始挑战.png",
                    "source": "global",
                    "action": "wait_click",
                    "timeout": 20.0,
                    "threshold": 0.8,
                    "post_delay": 0.5,
                }
            ],
        }

    def summary(self, config: Dict[str, Any]) -> str:
        steps = config.get("steps") or []
        labels = [step.get("label") or step.get("template", "") for step in steps]
        if not labels:
            return "未配置步骤"
        if len(labels) == 1:
            return labels[0]
        return " → ".join(labels[:3]) + ("…" if len(labels) > 3 else "")

    def _resolve_template_value(self, step: Dict[str, Any], context: CustomScriptContext) -> str:
        template = step.get("template", "")
        context_key = step.get("context_key")
        if context_key:
            template = context.state.get(context_key, template)
        return template

    def _execute_action(
        self,
        step: Dict[str, Any],
        context: CustomScriptContext,
        label: str,
    ) -> bool:
        action = step.get("action", "wait_click")
        timeout = float(step.get("timeout", 20.0) or 0.0)
        threshold = float(step.get("threshold", 0.8) or 0.0)
        source = step.get("source", "global")
        template = self._resolve_template_value(step, context)

        if action == "wait_click":
            return _wait_and_click_generic(template, label, timeout, threshold, source)
        if action == "click":
            return _click_generic(template, label, threshold, source)
        if action == "wait_for":
            return _wait_for_generic(template, label, timeout, threshold, source)
        if action == "letter_click":
            if not template:
                context.log(f"{label}：未配置密函图片路径。")
                return False
            return click_letter_template(template, label, timeout, threshold)
        if action == "delay":
            sleep_with_stop(timeout)
            return not worker_stop.is_set()
        if action == "press_key":
            key = step.get("key")
            if not key:
                context.log(f"{label}：未配置按键。")
                return False
            return _press_key_once(key)
        if action == "press_keys":
            keys = step.get("keys") or []
            if not keys:
                context.log(f"{label}：未配置按键列表。")
                return False
            interval = float(step.get("interval", 0.05) or 0.0)
            return _press_keys([str(k) for k in keys], interval)
        if action == "macro":
            path = step.get("macro_path") or template
            label_name = step.get("macro_label") or label
            return _play_macro_path(path, label_name, context, allow_segment=False)
        if action == "segment_macro":
            path = step.get("macro_path") or template
            label_name = step.get("macro_label") or label
            return _play_macro_path(path, label_name, context, allow_segment=True)
        if action == "set_state":
            key = step.get("state_key")
            value = step.get("state_value")
            if key:
                context.state[key] = value
                context.log(f"{label}：已记录状态 {key} = {value}")
                return True
            context.log(f"{label}：未提供 state_key。")
            return False
        if action == "copy_to_state":
            key = step.get("state_key")
            if key:
                context.state[key] = template
                context.log(f"{label}：已保存模板到状态 {key}。")
                return True
            context.log(f"{label}：未提供 state_key。")
            return False

        context.log(f"{label}：未知动作 {action}，已忽略。")
        return True

    def execute(self, context: CustomScriptContext, config: Dict[str, Any]) -> bool:
        steps = config.get("steps") or []
        halt_on_fail = bool(config.get("halt_on_fail", True))
        context.reset_last_result()

        for idx, step in enumerate(steps, 1):
            if context.should_stop():
                return False
            label = step.get("label") or f"步骤 {idx}"
            success = self._execute_action(step, context, label)
            if not success:
                context.fail(f"{label} 执行失败。")
                if halt_on_fail:
                    return False
            delay = float(step.get("post_delay", 0.0) or 0.0)
            if delay > 0:
                sleep_with_stop(delay)
                if context.should_stop():
                    return False

        return True


class LetterSelectionModule(CustomModuleDefinition):
    module_type = "letter_selection"
    display_name = "选择密函"
    description = "点击『选择密函』按钮、选择图片并确认，可将结果写入上下文供后续模块复用。"

    def default_config(self) -> Dict[str, Any]:
        return {
            "button_template": BTN_OPEN_LETTER,
            "button_source": "global",
            "letter_path": "",
            "confirm_template": BTN_CONFIRM_LETTER,
            "confirm_source": "global",
            "timeout": 25.0,
            "letter_timeout": 20.0,
            "threshold": 0.8,
            "store_key": "selected_letter",
            "post_delay": 0.5,
        }

    def summary(self, config: Dict[str, Any]) -> str:
        path = config.get("letter_path") or "未配置密函"
        return str(path)

    def execute(self, context: CustomScriptContext, config: Dict[str, Any]) -> bool:
        if context.should_stop():
            return False

        button_tpl = config.get("button_template") or BTN_OPEN_LETTER
        button_source = config.get("button_source", "global")
        timeout = float(config.get("timeout", 25.0) or 0.0)
        threshold = float(config.get("threshold", LETTER_MATCH_THRESHOLD) or 0.0)

        if not _wait_and_click_generic(button_tpl, "选择密函按钮", timeout, threshold, button_source):
            context.fail("未能点击选择密函按钮。")
            return False

        letter_path = config.get("letter_path")
        if not letter_path:
            context.fail("未配置密函图片路径。")
            return False
        if not os.path.exists(letter_path):
            context.fail(f"密函图片不存在：{letter_path}")
            return False

        letter_timeout = float(config.get("letter_timeout", 20.0) or 0.0)
        if not click_letter_template(
            letter_path,
            "点击密函",
            timeout=letter_timeout,
            threshold=threshold,
        ):
            context.fail("未能点击密函图片。")
            return False

        confirm_tpl = config.get("confirm_template")
        if confirm_tpl:
            confirm_source = config.get("confirm_source", "global")
            if not _wait_and_click_generic(
                confirm_tpl,
                "确认密函",
                timeout,
                threshold,
                confirm_source,
            ):
                context.fail("未能点击确认选择按钮。")
                return False

        store_key = config.get("store_key") or "selected_letter"
        context.state[store_key] = letter_path
        context.log(f"已记录密函路径到 {store_key}。")

        delay = float(config.get("post_delay", 0.0) or 0.0)
        if delay > 0:
            sleep_with_stop(delay)

        return True


class ImageTriggerModule(CustomModuleDefinition):
    module_type = "image_trigger"
    display_name = "图片识别触发器"
    description = "识别自定义图片并执行动作，可配置多条触发规则。"

    def default_config(self) -> Dict[str, Any]:
        return {
            "mode": "first",
            "halt_on_fail": True,
            "triggers": [
                {
                    "label": "示例：点击并按键",
                    "template": "",
                    "source": "custom",
                    "threshold": 0.8,
                    "timeout": 20.0,
                    "click": True,
                    "action": {"type": "press_key", "key": "f"},
                    "post_delay": 0.5,
                }
            ],
        }

    def summary(self, config: Dict[str, Any]) -> str:
        triggers = config.get("triggers") or []
        if not triggers:
            return "未配置触发规则"
        labels = [t.get("label") or t.get("template", "") for t in triggers]
        preview = "，".join(labels[:3])
        if len(labels) > 3:
            preview += "…"
        return preview

    def _execute_follow_action(
        self,
        action: Dict[str, Any],
        context: CustomScriptContext,
        label: str,
    ) -> bool:
        if not action:
            return True
        typ = action.get("type", "none")
        if typ == "none":
            return True
        if typ == "press_key":
            return _press_key_once(action.get("key"))
        if typ == "press_keys":
            keys = action.get("keys") or []
            interval = float(action.get("interval", 0.05) or 0.0)
            return _press_keys([str(k) for k in keys], interval)
        if typ == "macro":
            label_name = action.get("label") or label
            allow_segment = bool(action.get("allow_segment", False))
            return _play_macro_path(action.get("path"), label_name, context, allow_segment)
        if typ == "segment_macro":
            label_name = action.get("label") or label
            return _play_macro_path(action.get("path"), label_name, context, True)
        if typ == "template_sequence":
            seq_steps = action.get("steps") or []
            module = CUSTOM_MODULE_REGISTRY.get("template_sequence", TemplateSequenceModule())
            temp_cfg = {
                "halt_on_fail": bool(action.get("halt_on_fail", True)),
                "steps": seq_steps,
            }
            return module.execute(context, temp_cfg)
        if typ == "set_state":
            key = action.get("key")
            value = action.get("value")
            if key:
                context.state[key] = value
                context.log(f"{label}：已写入状态 {key} = {value}")
                return True
            context.log(f"{label}：未提供状态键。")
            return False

        context.log(f"{label}：不支持的动作类型 {typ}。")
        return False

    def execute(self, context: CustomScriptContext, config: Dict[str, Any]) -> bool:
        triggers = config.get("triggers") or []
        mode = config.get("mode", "first")
        halt_on_fail = bool(config.get("halt_on_fail", True))

        any_success = False

        for trig in triggers:
            if context.should_stop():
                break
            label = trig.get("label") or "图片触发"
            template = trig.get("template", "")
            source = trig.get("source", "global")
            timeout = float(trig.get("timeout", 20.0) or 0.0)
            threshold = float(trig.get("threshold", 0.8) or 0.0)
            click = bool(trig.get("click", True))

            if click:
                success = _wait_and_click_generic(template, label, timeout, threshold, source)
            else:
                success = _wait_for_generic(template, label, timeout, threshold, source)

            if not success:
                context.log(f"{label}：未匹配到目标图片。")
                if halt_on_fail:
                    context.fail(f"{label} 执行失败。")
                    return False
                continue

            any_success = True

            action = trig.get("action")
            if not self._execute_follow_action(action, context, label):
                if halt_on_fail:
                    context.fail(f"{label} 后续动作失败。")
                    return False

            delay = float(trig.get("post_delay", 0.0) or 0.0)
            if delay > 0:
                sleep_with_stop(delay)
                if context.should_stop():
                    break

            if mode == "first":
                break

        return any_success or not halt_on_fail


class MacroPlaybackModule(CustomModuleDefinition):
    module_type = "macro_playback"
    display_name = "执行键盘宏"
    description = "回放一个键盘 JSON 宏，支持可选的鼠标段判断。"

    def default_config(self) -> Dict[str, Any]:
        return {
            "macro_path": "",
            "label": "自定义宏",
            "allow_segment": True,
        }

    def summary(self, config: Dict[str, Any]) -> str:
        path = config.get("macro_path") or "未配置宏"
        return str(path)

    def execute(self, context: CustomScriptContext, config: Dict[str, Any]) -> bool:
        label = config.get("label") or "自定义宏"
        allow_segment = bool(config.get("allow_segment", True))
        return _play_macro_path(config.get("macro_path"), label, context, allow_segment)


class DelayModule(CustomModuleDefinition):
    module_type = "delay"
    display_name = "等待"
    description = "简单地等待指定秒数，可用于节奏控制。"

    def default_config(self) -> Dict[str, Any]:
        return {"seconds": 5.0}

    def summary(self, config: Dict[str, Any]) -> str:
        sec = float(config.get("seconds", 0.0) or 0.0)
        return f"等待 {sec:.1f} 秒"

    def execute(self, context: CustomScriptContext, config: Dict[str, Any]) -> bool:
        seconds = float(config.get("seconds", 0.0) or 0.0)
        sleep_with_stop(seconds)
        return not context.should_stop()


class PressKeyModule(CustomModuleDefinition):
    module_type = "press_key"
    display_name = "按键操作"
    description = "按照自定义间隔循环按下指定按键，可用于技能施放。"

    def default_config(self) -> Dict[str, Any]:
        return {
            "keys": ["e"],
            "count": 1,
            "interval": 0.2,
        }

    def summary(self, config: Dict[str, Any]) -> str:
        keys = config.get("keys") or []
        count = int(config.get("count", 1) or 1)
        return f"按键 {','.join(str(k) for k in keys)} × {count}"

    def execute(self, context: CustomScriptContext, config: Dict[str, Any]) -> bool:
        keys = config.get("keys") or []
        count = max(1, int(config.get("count", 1) or 1))
        interval = float(config.get("interval", 0.2) or 0.0)
        if not keys:
            context.fail("未配置按键。")
            return False
        for _ in range(count):
            if context.should_stop():
                return False
            if not _press_keys([str(k) for k in keys], interval):
                context.fail("按键失败。")
                return False
        return True


class AutoLoopModule(CustomModuleDefinition):
    module_type = "auto_loop"
    display_name = "自动循环控制"
    description = "开启或关闭脚本自动循环，可自定义循环次数（0 表示无限）。"

    def default_config(self) -> Dict[str, Any]:
        return {"enabled": True, "loop_count": 0}

    def summary(self, config: Dict[str, Any]) -> str:
        enabled = bool(config.get("enabled", True))
        limit = int(config.get("loop_count", 0) or 0)
        if not enabled:
            return "关闭循环"
        if limit <= 0:
            return "开启循环：无限次"
        return f"开启循环：{limit} 次"

    def execute(self, context: CustomScriptContext, config: Dict[str, Any]) -> bool:
        enabled = bool(config.get("enabled", True))
        limit = int(config.get("loop_count", 0) or 0)
        context.request_loop(enabled, limit)
        if enabled:
            context.log(
                "已开启循环" + ("（无限次）" if limit <= 0 else f"，最多 {limit} 次"),
            )
        else:
            context.log("已关闭自动循环。")
        return True


class EndScriptModule(CustomModuleDefinition):
    module_type = "end_script"
    display_name = "结束执行"
    description = "立即结束脚本运行，可用于流程最后一步。"

    def execute(self, context: CustomScriptContext, config: Dict[str, Any]) -> bool:
        context.terminate()
        context.log("已结束脚本执行。")
        return True


class _BaseManualNoTrickAdapter:
    def __init__(self, context: CustomScriptContext, prefix: str):
        self.context = context
        self.log_prefix = prefix
        self.suppress_log = True
        self.detected_event = threading.Event()
        self.completed_event = threading.Event()
        self.macro_executed = False
        self.macro_missing = False

    # 通用回调
    def on_no_trick_unavailable(self, reason: str):
        self.context.fail(f"{self.log_prefix}：功能不可用（{reason}）。")

    def on_no_trick_no_templates(self, directory: str):
        self.context.fail(f"{self.log_prefix}：未在 {directory} 中找到模板。")

    def on_no_trick_monitor_started(self, templates):
        self.context.log(f"{self.log_prefix}：加载 {len(templates)} 张模板，等待触发…")

    def on_no_trick_detected(self, entry, score: float):
        name = entry.get("name") if entry else ""
        self.context.log(f"{self.log_prefix}：检测到 {name}（匹配度 {score:.2f}）。")
        self.detected_event.set()

    def on_no_trick_macro_start(self, entry, score: float):
        base = entry.get("base_name") if entry else ""
        self.context.log(f"{self.log_prefix}：开始回放 {base}.json（匹配度 {score:.2f}）。")

    def on_no_trick_progress(self, value: float):
        try:
            self.context.gui.update_progress(value * 100.0)
        except Exception:
            pass

    def on_no_trick_macro_complete(self, entry):
        base = entry.get("base_name") if entry else ""
        self.context.log(f"{self.log_prefix}：{base}.json 回放完成。")
        self.macro_executed = True

    def on_no_trick_macro_missing(self, entry):
        name = entry.get("name") if entry else ""
        self.context.fail(f"{self.log_prefix}：缺少宏文件 {name}.json。")
        self.macro_missing = True

    def on_no_trick_session_finished(self, **kwargs):
        self.completed_event.set()

    # 兼容赛琪接口
    def on_no_trick_idle(self, remaining: float):
        self.context.log(f"{self.log_prefix}：等待剩余 {remaining:.1f} 秒…")

    def on_no_trick_idle_complete(self):
        self.context.log(f"{self.log_prefix}：检测完成。")


def run_line_decrypt_once(game_dir: str, context: CustomScriptContext, timeout: float = 120.0) -> bool:
    adapter = _BaseManualNoTrickAdapter(context, "划线无巧手解密")
    controller = NoTrickDecryptController(adapter, game_dir)
    if not controller.start():
        return False

    keyboard_state = KeyboardPlaybackState()
    deadline = time.time() + max(5.0, float(timeout)) if timeout else None

    try:
        while not context.should_stop():
            controller.run_decrypt_if_needed(keyboard_state)
            if adapter.completed_event.wait(0.05):
                break
            if deadline and time.time() > deadline:
                context.fail("划线无巧手解密：等待超时。")
                break
    finally:
        controller.stop()
        controller.finish_session()

    return adapter.macro_executed and not adapter.macro_missing


def run_firework_decrypt_once(game_dir: str, context: CustomScriptContext, timeout: float = 180.0) -> bool:
    adapter = _BaseManualNoTrickAdapter(context, "转盘无巧手解密")
    controller = FireworkNoTrickController(adapter, game_dir)
    if not controller.start():
        return False

    deadline = time.time() + max(5.0, float(timeout)) if timeout else None

    try:
        while not context.should_stop():
            controller.run_decrypt_if_needed()
            if adapter.completed_event.wait(0.05):
                break
            if deadline and time.time() > deadline:
                context.fail("转盘无巧手解密：等待超时。")
                break
    finally:
        controller.stop()
        controller.finish_session()

    return adapter.macro_executed and not adapter.macro_missing


class LineDecryptModule(CustomModuleDefinition):
    module_type = "line_no_trick"
    display_name = "划线无巧手解密"
    description = "调用 Game 目录的划线解密宏，适用于探险无尽血清、50XP 等界面。"

    def default_config(self) -> Dict[str, Any]:
        return {"game_dir": GAME_DIR, "timeout": 120.0}

    def summary(self, config: Dict[str, Any]) -> str:
        path = config.get("game_dir") or GAME_DIR
        return f"目录：{path}"

    def execute(self, context: CustomScriptContext, config: Dict[str, Any]) -> bool:
        directory = config.get("game_dir") or GAME_DIR
        timeout = float(config.get("timeout", 120.0) or 0.0)
        return run_line_decrypt_once(directory, context, timeout)


class FireworkDecryptModule(CustomModuleDefinition):
    module_type = "firework_no_trick"
    display_name = "转盘无巧手解密"
    description = "调用 GAME-sq 目录的赛琪无巧手解密宏。"

    def default_config(self) -> Dict[str, Any]:
        return {"game_dir": GAME_SQ_DIR, "timeout": 180.0}

    def summary(self, config: Dict[str, Any]) -> str:
        path = config.get("game_dir") or GAME_SQ_DIR
        return f"目录：{path}"

    def execute(self, context: CustomScriptContext, config: Dict[str, Any]) -> bool:
        directory = config.get("game_dir") or GAME_SQ_DIR
        timeout = float(config.get("timeout", 180.0) or 0.0)
        return run_firework_decrypt_once(directory, context, timeout)


for _module_cls in (
    TemplateSequenceModule,
    LetterSelectionModule,
    ImageTriggerModule,
    MacroPlaybackModule,
    DelayModule,
    PressKeyModule,
    AutoLoopModule,
    EndScriptModule,
    LineDecryptModule,
    FireworkDecryptModule,
):
    register_custom_module(_module_cls)


class UIDMaskManager:
    """Manage UID mosaic overlays that follow the game window."""

    def __init__(self, root):
        self.root = root
        self.active = False
        self.overlays = []
        self.monitor_thread = None
        self.stop_event = threading.Event()
        self._lock = threading.Lock()
        self.mask_rects = UID_FIXED_MASKS

    def start(self):
        if self.active:
            messagebox.showinfo("UID遮挡", "UID遮挡已经开启。")
            return
        if not self.mask_rects:
            messagebox.showwarning("UID遮挡", "未配置任何遮挡区域。")
            return
        win = find_game_window()
        if win is None:
            messagebox.showwarning(
                "UID遮挡", f"未找到{get_window_name_hint()}窗口。"
            )
            return
        self.stop_event.clear()
        self.active = True
        self._create_overlays(win)
        self.monitor_thread = threading.Thread(target=self._follow_window, daemon=True)
        self.monitor_thread.start()
        log("UID 遮挡：已开启。")

    def stop(self, manual: bool = True, silent: bool = False):
        if not self.active:
            if manual and not silent:
                messagebox.showinfo("UID遮挡", "UID遮挡当前未开启。")
            return
        self.stop_event.set()
        self._destroy_overlays()
        self.monitor_thread = None
        self.active = False
        if not silent:
            log(f"UID 遮挡：{'手动' if manual else '自动'}关闭。")

    def _create_overlays(self, win):
        self._destroy_overlays()
        for idx, rect in enumerate(self.mask_rects):
            rel_x, rel_y, width, height = rect
            left = int(win.left + rel_x)
            top = int(win.top + rel_y)
            overlay = tk.Toplevel(self.root)
            overlay.withdraw()
            overlay.overrideredirect(True)
            overlay.attributes("-topmost", True)
            overlay.attributes("-alpha", UID_MASK_ALPHA)
            base_color = UID_MASK_COLORS[idx % len(UID_MASK_COLORS)]
            overlay.configure(bg=base_color)
            canvas = tk.Canvas(
                overlay,
                width=width,
                height=height,
                highlightthickness=0,
                bd=0,
                bg=base_color,
            )
            canvas.pack(fill="both", expand=True)
            self._draw_mosaic(canvas, idx, width, height)
            overlay.geometry(f"{width}x{height}+{left}+{top}")
            overlay.deiconify()
            data = {
                "window": overlay,
                "offset_x": rel_x,
                "offset_y": rel_y,
                "width": width,
                "height": height,
            }
            with self._lock:
                self.overlays.append(data)

    def _draw_mosaic(self, canvas, seed: int, width: int, height: int):
        rnd = random.Random(1000 + seed * 131)
        for x in range(0, width, UID_MASK_CELL):
            for y in range(0, height, UID_MASK_CELL):
                color = rnd.choice(UID_MASK_COLORS)
                canvas.create_rectangle(
                    x,
                    y,
                    min(x + UID_MASK_CELL, width),
                    min(y + UID_MASK_CELL, height),
                    fill=color,
                    outline=color,
                )

    def _destroy_overlays(self):
        with self._lock:
            overlays = self.overlays
            self.overlays = []
        for data in overlays:
            win = data.get("window")
            try:
                win.destroy()
            except Exception:
                pass

    def _follow_window(self):
        miss_count = 0
        while not self.stop_event.is_set():
            win = find_game_window()
            if win is None:
                miss_count += 1
                if miss_count >= UID_WINDOW_MISS_LIMIT:
                    self.stop_event.set()
                    post_to_main_thread(
                        lambda: self._handle_auto_stop(
                            f"未检测到{get_window_name_hint()}窗口，UID遮挡已自动关闭。"
                        )
                    )
                    break
            else:
                miss_count = 0
                left = win.left
                top = win.top
                with self._lock:
                    overlays = list(self.overlays)
                for data in overlays:
                    self._move_overlay(data, left, top)
            time.sleep(0.05)

    def _move_overlay(self, data, left: int, top: int):
        win = data.get("window")
        if win is None:
            return
        width = int(data.get("width", 0))
        height = int(data.get("height", 0))
        x = int(left + data.get("offset_x", 0))
        y = int(top + data.get("offset_y", 0))
        geom = f"{width}x{height}+{x}+{y}"
        try:
            win.geometry(geom)
        except Exception:
            pass

    def _handle_auto_stop(self, message: str):
        self.stop(manual=False, silent=True)
        if message:
            messagebox.showwarning("UID遮挡", message)
# ---------- 宏回放（EMT 风格高精度） ----------
def load_actions(path: str):
    if not path or not os.path.exists(path):
        log(f"宏文件不存在：{path}")
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        log(f"加载宏失败：{e}")
        return []
    acts = data.get("actions", [])
    if not isinstance(acts, list) or not acts:
        log(f"宏文件中没有有效动作：{path}")
        return []
    acts.sort(key=lambda a: a.get("time", 0.0))
    return acts


MOUSE_ACTION_TYPES = {
    "mouse_move",
    "mouse_move_relative",
    "mouse_click",
    "mouse_down",
    "mouse_up",
    "mouse_scroll",
    "mouse_rotation",
    "mouse_drag",
    "mouse_drag_relative",
}


class KeyboardPlaybackState:
    """Track pressed keys during macro playback.

    The state helps us temporarily release modifiers before running nested
    decrypt macros so they don't combine with replayed keys to trigger system
    shortcuts (例如 Win+数字 打开计算器)。
    """

    def __init__(self):
        self._active = []

    def press(self, key: str) -> bool:
        if keyboard is None or not key:
            return False
        try:
            keyboard.press(key)
            self._active.append(key)
            return True
        except Exception:
            return False

    def release(self, key: str) -> bool:
        if keyboard is None or not key:
            return False
        try:
            keyboard.release(key)
        except Exception:
            return False
        for idx in range(len(self._active) - 1, -1, -1):
            if self._active[idx] == key:
                del self._active[idx]
                break
        return True

    def suspend(self):
        """Release all currently pressed keys and return them for restoration."""

        if not self._active or keyboard is None:
            keys = list(self._active)
            self._active.clear()
            return keys

        keys = list(self._active)
        for key in reversed(keys):
            if not key:
                continue
            try:
                keyboard.release(key)
            except Exception:
                pass
        self._active.clear()
        return keys

    def resume(self, keys):
        if keyboard is None or not keys:
            return
        blocked_tokens = ("win", "windows", "cmd", "gui")
        for key in keys:
            if not key:
                continue
            lower = str(key).lower()
            if any(token in lower for token in blocked_tokens):
                continue
            try:
                keyboard.press(key)
                self._active.append(key)
            except Exception:
                pass

    def active_keys(self):
        """Return a snapshot of currently pressed keys."""

        return list(self._active)

    def release_all(self):
        if not self._active or keyboard is None:
            self._active.clear()
            return
        for key in reversed(self._active):
            if not key:
                continue
            try:
                keyboard.release(key)
            except Exception:
                pass
        self._active.clear()


# ---------- 自动 70 红珠资源 ----------
HS_START_TEMPLATE = "开始挑战.png"
HS_RETRY_TEMPLATE = "再次进行.png"
HS_INITIAL_MAP_TEMPLATE = "初始map.png"
HS_SETTINGS_TEMPLATE = "设置.png"
HS_MORE_TEMPLATE = "更多.png"
HS_RESET_TEMPLATE = "复位.png"
HS_RESET_CONFIRM_TEMPLATE = "Q.png"
HS_BRANCH_TEMPLATE = "分支A.png"
HS_TARGET_TEMPLATE = "目标.png"
HS_WARNING_TEMPLATE = "警告.png"
HS_BRANCH_OPTIONS = {
    "2-3": {
        "templates": ["2-3.png"],
        "macro": "mapa-开锁2-3.json",
        "calibrate": "mapa-开锁2-3-校准.json",
    },
    "2-4": {
        "templates": ["2-4.png", "分支2-4-1.png"],
        "macro": "mapa-开锁2-4.json",
        "calibrate": "mapa-开锁2-4-校准.json",
    },
}
HS_MAIN_MACROS = [
    "mapa.json",
    "mapa-开锁1.json",
    "mapa-开锁2.json",
]
HS_CALIBRATION_MACROS = {
    "mapa-开锁1.json": "mapa-开锁1-校准.json",
    "mapa-开锁2.json": "mapa-开锁2-校准.json",
}
HS_SUBMAP_TEMPLATES = {
    "A": "A类地图.png",
    "B": "B类地图.png",
    "C": "C类地图.png",
}
HS_FINAL_MACROS = {
    "A": "A类复位撤离.json",
    "B": "B类复位撤离.json",
    "C": "C类复位撤离.json",
}
HS_COMPENSATE_MACRO = "补偿.json"
HS_FINE_TUNE_MACRO = "微调.json"
HS_TIP_IMAGE = "提示.png"
HS_ASSET_CACHE = {}
HS_CLICK_THRESHOLD = 0.72


def hs_template_path(name: str, allow_templates: bool = False) -> str:
    if allow_templates:
        template_path = os.path.join(TEMPLATE_DIR, name)
        if os.path.exists(template_path):
            return template_path
    return os.path.join(HS_DIR, name)


def hs_reset_asset_cache():
    HS_ASSET_CACHE.clear()


def hs_find_asset(name: str, allow_templates: bool = False) -> Optional[str]:
    key = (name, bool(allow_templates))
    if key in HS_ASSET_CACHE:
        cached = HS_ASSET_CACHE[key]
        if cached and os.path.exists(cached):
            return cached
        if cached:
            HS_ASSET_CACHE.pop(key, None)
        else:
            return None

    primary = os.path.join(HS_DIR, name)
    if os.path.exists(primary):
        HS_ASSET_CACHE[key] = primary
        return primary

    for root, _, files in os.walk(HS_DIR):
        if name in files:
            found = os.path.join(root, name)
            HS_ASSET_CACHE[key] = found
            return found

    if allow_templates:
        fallback = os.path.join(TEMPLATE_DIR, name)
        if os.path.exists(fallback):
            HS_ASSET_CACHE[key] = fallback
            return fallback

    HS_ASSET_CACHE[key] = None
    return None


def hs_wait_and_click_template(
    name: str,
    step_name: str,
    timeout: float = 20.0,
    threshold: float = HS_CLICK_THRESHOLD,
) -> bool:
    path = hs_find_asset(name, allow_templates=True)
    if not path:
        log(f"{HS70AutoGUI.LOG_PREFIX if 'HS70AutoGUI' in globals() else '[70HS]'} 缺少 {name}，请放置于 HS 或 templates 目录。")
        return False
    return wait_and_click_template_from_path(path, step_name, timeout=timeout, threshold=threshold)


# ---------- 全自动 50 经验副本资源 ----------
XP50_START_TEMPLATE = "开始挑战.png"
XP50_RETRY_TEMPLATE = "再次进行.png"
XP50_SERUM_TEMPLATE = "血清完成.png"
XP50_MAP_TEMPLATES = {"A": "mapa.png", "B": "mapb.png"}
XP50_MACRO_SEQUENCE = {
    "A": ["mapa-1.json", "mapa-2.json", "mapa-3撤离.json"],
    "B": ["mapb-1.json", "mapb-2.json", "mapb-3撤离.json"],
}
XP50_CLICK_THRESHOLD = 0.75
XP50_MAP_THRESHOLD = 0.7
XP50_SERUM_THRESHOLD = 0.75
XP50_ASSET_CACHE = {}


def xp50_template_path(name: str) -> str:
    return os.path.join(XP50_DIR, name)


def xp50_reset_asset_cache():
    XP50_ASSET_CACHE.clear()


def xp50_find_asset(name: str, allow_templates: bool = False) -> Optional[str]:
    """Locate a 50XP asset even when stored in nested folders."""

    key = (name, bool(allow_templates))
    if key in XP50_ASSET_CACHE:
        cached = XP50_ASSET_CACHE[key]
        if cached and os.path.exists(cached):
            return cached
        if cached:
            # 缓存的路径已不存在，清理后重新搜索
            XP50_ASSET_CACHE.pop(key, None)
        else:
            return None

    primary = os.path.join(XP50_DIR, name)
    if os.path.exists(primary):
        XP50_ASSET_CACHE[key] = primary
        return primary

    for root, _, files in os.walk(XP50_DIR):
        if name in files:
            found = os.path.join(root, name)
            XP50_ASSET_CACHE[key] = found
            return found

    if allow_templates:
        fallback = os.path.join(TEMPLATE_DIR, name)
        if os.path.exists(fallback):
            XP50_ASSET_CACHE[key] = fallback
            return fallback
        for root, _, files in os.walk(TEMPLATE_DIR):
            if name in files:
                found = os.path.join(root, name)
                XP50_ASSET_CACHE[key] = found
                return found

    XP50_ASSET_CACHE[key] = None
    return None


def xp50_wait_and_click(name: str, step_name: str, timeout: float = 20.0, threshold: float = XP50_CLICK_THRESHOLD) -> bool:
    path = xp50_find_asset(name, allow_templates=True)
    if not path:
        log(
            "50XP 模板缺失：{}；已尝试在 {} 及 templates 子目录中查找".format(
                xp50_template_path(name), XP50_DIR
            )
        )
        return False
    return wait_and_click_template_from_path(path, step_name, timeout, threshold)


def macro_has_segments(path: str) -> bool:
    """Return True when the JSON macro contains segment playback data."""

    if not path or not os.path.exists(path):
        return False

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return False

    segments = data.get("segments")
    return isinstance(segments, list) and len(segments) > 0


def play_segment_macro(path: str, label: str, progress_callback=None):
    """回放自定义鼠标轨迹段宏。

    轨迹文件格式：
    {
        "segments": [{"from": [x, y], "to": [x, y]}, ...],
        "recorded_w": 1920,
        "recorded_h": 1080,
    }
    """

    if pyautogui is None:
        log(f"{label}：未安装 pyautogui 模块，无法回放鼠标轨迹宏。")
        return None

    if not path or not os.path.exists(path):
        log(f"{label}：宏文件不存在：{path}")
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        log(f"{label}：加载轨迹宏失败：{e}")
        return None

    segments = data.get("segments")
    if not isinstance(segments, list) or not segments:
        return False

    ensure_windows_dpi_awareness()

    window_info = get_game_client_rect()
    if window_info is None:
        log(
            f"{label}：未找到{get_window_name_hint()}窗口，无法回放鼠标轨迹宏。"
        )
        return None

    hwnd, origin_x, origin_y, client_w, client_h = window_info
    focus_game_window(hwnd)

    try:
        recorded_w = float(data.get("recorded_w", 1920))
    except (TypeError, ValueError):
        recorded_w = 1920.0
    if recorded_w <= 0:
        recorded_w = 1920.0

    try:
        recorded_h = float(data.get("recorded_h", 1080))
    except (TypeError, ValueError):
        recorded_h = 1080.0
    if recorded_h <= 0:
        recorded_h = 1080.0

    scale_x = client_w / recorded_w if recorded_w else 1.0
    scale_y = client_h / recorded_h if recorded_h else 1.0

    def _parse_point(value):
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            try:
                return float(value[0]), float(value[1])
            except (TypeError, ValueError):
                return None
        if isinstance(value, dict):
            try:
                return float(value.get("x")), float(value.get("y"))
            except (TypeError, ValueError):
                return None
        return None

    start_point = _parse_point(segments[0].get("from"))
    if start_point is None:
        log(f"{label}：轨迹宏缺少起点坐标，已跳过。")
        return False

    start_x = origin_x + start_point[0] * scale_x
    start_y = origin_y + start_point[1] * scale_y

    total_segments = len(segments)
    log(f"{label}：共 {total_segments} 段轨迹，按当前分辨率缩放后开始回放。")

    if progress_callback is not None:
        try:
            progress_callback(0.0)
        except Exception:
            pass

    executed_segments = 0
    last_percent = 0
    start_time = time.perf_counter()
    mouse_held = False
    result = None
    current_x = start_x
    current_y = start_y

    try:
        pyautogui.moveTo(int(round(start_x)), int(round(start_y)))
        time.sleep(0.05)
        pyautogui.mouseDown(button="left")
        mouse_held = True

        for idx, seg in enumerate(segments):
            if worker_stop.is_set():
                log(f"{label}：检测到停止信号，中断轨迹回放。")
                break

            target = _parse_point(seg.get("to"))
            if target is None:
                log(f"{label}：第 {idx + 1} 段缺少终点坐标，停止回放。")
                break

            tx = origin_x + target[0] * scale_x
            ty = origin_y + target[1] * scale_y

            try:
                requested_duration = float(seg.get("duration", 0.0) or 0.0)
            except (TypeError, ValueError):
                requested_duration = 0.0

            distance = max(abs(tx - current_x), abs(ty - current_y))
            if requested_duration <= 0:
                if distance <= 1:
                    duration = 0.0
                else:
                    duration = max(0.0, min(0.008, distance / 120000.0))
            else:
                duration = max(0.0, min(requested_duration, 0.15))

            try:
                pyautogui.moveTo(int(round(tx)), int(round(ty)), duration=duration)
            except Exception as e:
                log(f"{label}：移动到第 {idx + 1} 段终点失败：{e}")
                break

            current_x = tx
            current_y = ty

            executed_segments += 1

            progress = executed_segments / total_segments
            if progress_callback is not None:
                try:
                    progress_callback(progress)
                except Exception:
                    pass

            percent = int(progress * 100)
            if percent - last_percent >= 10:
                log(f"{label} 回放进度：{percent}%（鼠标段:{executed_segments}）")
                last_percent = percent

            if worker_stop.is_set():
                break

            time.sleep(0.002)

        else:
            # 循环未被 break，确保进度到 100%
            if progress_callback is not None:
                try:
                    progress_callback(1.0)
                except Exception:
                    pass

        elapsed = time.perf_counter() - start_time

        if executed_segments >= total_segments:
            log(f"{label} 执行完成：")
            log(f"  实际耗时：{elapsed:.3f} 秒")
            log(f"  执行段数：{executed_segments}/{total_segments}（鼠标:{executed_segments}）")
            result = True
        else:
            log(
                f"{label}：轨迹回放提前结束（已执行 {executed_segments}/{total_segments} 段）。"
            )
            result = None

    finally:
        if mouse_held:
            try:
                pyautogui.mouseUp(button="left")
            except Exception:
                pass

    return result


def wait_after_decrypt_delay(delay: float = 1.0):
    """解密宏结束后短暂等待，避免角色仍处于僵直状态。"""

    if delay <= 0:
        return

    end_time = time.perf_counter() + delay
    while time.perf_counter() < end_time and not worker_stop.is_set():
        time.sleep(0.05)


def _execute_mouse_action(action: dict, label: str) -> bool:
    if pyautogui is None:
        return False

    ttype = action.get("type")
    try:
        duration = float(action.get("duration", 0.0) or 0.0)
    except Exception:
        duration = 0.0

    if ttype == "mouse_move":
        try:
            x = float(action.get("x"))
            y = float(action.get("y"))
        except (TypeError, ValueError):
            log(f"{label}：鼠标移动动作缺少坐标，已跳过。")
            return False
        pyautogui.moveTo(int(round(x)), int(round(y)), duration=max(0.0, duration))
        return True

    if ttype == "mouse_move_relative":
        try:
            dx = float(action.get("dx", action.get("x")))
            dy = float(action.get("dy", action.get("y")))
        except (TypeError, ValueError):
            log(f"{label}：相对鼠标移动动作缺少位移，已跳过。")
            return False
        pyautogui.moveRel(int(round(dx)), int(round(dy)), duration=max(0.0, duration))
        return True

    if ttype == "mouse_click":
        button = action.get("button", "left") or "left"
        clicks = action.get("clicks", 1)
        interval = action.get("interval", 0)
        try:
            clicks = int(clicks)
        except (TypeError, ValueError):
            clicks = 1
        try:
            interval = float(interval)
        except (TypeError, ValueError):
            interval = 0.0
        return perform_click(
            button=button,
            clicks=max(1, clicks),
            interval=max(0.0, interval),
        )

    if ttype == "mouse_down":
        button = action.get("button", "left") or "left"
        pyautogui.mouseDown(button=button)
        return True

    if ttype == "mouse_up":
        button = action.get("button", "left") or "left"
        pyautogui.mouseUp(button=button)
        return True

    if ttype == "mouse_scroll":
        amount = action.get("amount", action.get("clicks"))
        try:
            amount = int(amount)
        except (TypeError, ValueError):
            log(f"{label}：鼠标滚轮动作缺少数量，已跳过。")
            return False
        x = action.get("x")
        y = action.get("y")
        try:
            x = None if x is None else int(round(float(x)))
            y = None if y is None else int(round(float(y)))
        except (TypeError, ValueError):
            x = None
            y = None
        pyautogui.scroll(amount, x=x, y=y)
        return True

    if ttype in {"mouse_drag", "mouse_drag_relative"}:
        try:
            dx = float(action.get("dx", action.get("x")))
            dy = float(action.get("dy", action.get("y")))
        except (TypeError, ValueError):
            log(f"{label}：鼠标拖拽动作缺少位移，已跳过。")
            return False
        button = action.get("button", "left") or "left"
        pyautogui.dragRel(int(round(dx)), int(round(dy)), duration=max(0.0, duration), button=button)
        return True

    if ttype == "mouse_rotation":
        direction = str(action.get("direction", "")).lower()
        try:
            angle = float(action.get("angle", 0.0) or 0.0)
        except (TypeError, ValueError):
            angle = 0.0
        try:
            sensitivity = float(action.get("sensitivity", 1.0) or 1.0)
        except (TypeError, ValueError):
            sensitivity = 1.0
        magnitude = angle * sensitivity
        dx = dy = 0.0
        if direction in ("left", "right"):
            dx = magnitude if direction == "right" else -magnitude
        elif direction in ("up", "down"):
            dy = -magnitude if direction == "up" else magnitude
        else:
            log(f"{label}：鼠标旋转方向未知（{direction}），已跳过。")
            return False
        dx_i = int(round(dx))
        dy_i = int(round(dy))
        if dx_i == 0 and dy_i == 0:
            return True
        pyautogui.moveRel(dx_i, dy_i, duration=max(0.0, duration))
        return True

    return False


def play_macro(
    path: str,
    label: str,
    p1: float,
    p2: float,
    interrupt_on_exit: bool = False,
    interrupter=None,
    progress_callback=None,
):
    """
    EMT 风格高精度回放：
    - 按 actions 里的 time 字段作为绝对时间轴
    - time.perf_counter + 自旋保证时间精度
    - interrupt_on_exit=True 时，会周期性检测退图界面，发现就提前结束宏
    """
    actions = load_actions(path)
    if not actions:
        return False

    requires_keyboard = any(act.get("type") in {"key_down", "key_up"} for act in actions)
    requires_mouse = any(act.get("type") in MOUSE_ACTION_TYPES for act in actions)

    if requires_keyboard and keyboard is None:
        log("未安装 keyboard 模块，无法回放包含按键的宏。")
        return

    if requires_mouse and pyautogui is None:
        log("未安装 pyautogui 模块，无法回放包含鼠标动作的宏。")
        return

    if not label:
        label = "宏"

    total_time = float(actions[-1].get("time", 0.0))
    total_actions = len(actions)
    log(f"{label}：共 {total_actions} 个动作，时长约 {total_time:.2f} 秒。")

    start_time = time.perf_counter()
    executed_count = 0
    keyboard_count = 0
    mouse_count = 0
    last_progress_percent = 0
    keyboard_state = KeyboardPlaybackState() if requires_keyboard else None

    try:
        for i, action in enumerate(actions):
            if worker_stop.is_set():
                log(f"{label}：检测到停止信号，中断宏回放。")
                break

            if interrupt_on_exit and i % 5 == 0 and is_exit_ui_visible():
                log(f"{label}：检测到退图界面，提前结束宏。")
                break

            if interrupter is not None:
                pause_time = interrupter.run_decrypt_if_needed(keyboard_state)
                if pause_time:
                    start_time += pause_time

            target_time = float(action.get("time", 0.0))
            if interrupter is None:
                elapsed = time.perf_counter() - start_time
                sleep_time = target_time - elapsed
                if sleep_time > 0:
                    if sleep_time > 0.001:
                        time.sleep(max(0, sleep_time - 0.0005))
                    while time.perf_counter() - start_time < target_time:
                        pass
            else:
                while True:
                    elapsed = time.perf_counter() - start_time
                    sleep_time = target_time - elapsed
                    if sleep_time <= 0:
                        break
                    chunk = min(0.05, max(sleep_time - 0.0005, 0.0))
                    if chunk > 0:
                        time.sleep(chunk)
                    pause_time = interrupter.run_decrypt_if_needed(keyboard_state)
                    if pause_time:
                        start_time += pause_time
                while True:
                    pause_time = interrupter.run_decrypt_if_needed(keyboard_state)
                    if pause_time:
                        start_time += pause_time
                        continue
                    if time.perf_counter() - start_time >= target_time:
                        break

            executed = False
            try:
                ttype = action.get("type", "key_down")
                key = action.get("key")
                if ttype == "key_down" and key and keyboard is not None:
                    if keyboard_state is not None:
                        if keyboard_state.press(key):
                            keyboard_count += 1
                            executed = True
                    else:
                        try:
                            keyboard.press(key)
                            keyboard_count += 1
                            executed = True
                        except Exception:
                            pass
                elif ttype == "key_up" and key and keyboard is not None:
                    if keyboard_state is not None:
                        executed = keyboard_state.release(key)
                    else:
                        try:
                            keyboard.release(key)
                            executed = True
                        except Exception:
                            pass
                elif ttype in MOUSE_ACTION_TYPES:
                    executed = _execute_mouse_action(action, label)
                    if executed:
                        mouse_count += 1
                elif ttype == "sleep":
                    try:
                        delay = float(action.get("duration", action.get("time", 0.0)))
                    except (TypeError, ValueError):
                        delay = 0.0
                    if delay > 0:
                        time.sleep(delay)
                    executed = True
                else:
                    log(f"{label}：未知动作类型 {ttype}，已跳过。")
            except Exception as e:
                log(f"{label}：动作 {i} 发送失败：{e}")
                continue

            if executed:
                executed_count += 1

            local_progress = (i + 1) / total_actions
            global_p = p1 + local_progress * (p2 - p1)
            report_progress(global_p)
            if progress_callback is not None:
                try:
                    progress_callback(local_progress)
                except Exception:
                    pass

            percent = int(local_progress * 100)
            if percent - last_progress_percent >= 10:
                stats = [f"键盘:{keyboard_count}"]
                if mouse_count:
                    stats.append(f"鼠标:{mouse_count}")
                log(f"{label} 回放进度：{percent}%（{'，'.join(stats)}）")
                last_progress_percent = percent

        actual_elapsed = time.perf_counter() - start_time
        time_diff = actual_elapsed - total_time
        accuracy = (1 - abs(time_diff) / total_time) * 100 if total_time > 0 else 100
        log(f"{label} 执行完成：")
        log(f"  预期时长：{total_time:.3f} 秒")
        log(f"  实际耗时：{actual_elapsed:.3f} 秒")
        log(f"  时间偏差：{time_diff * 1000:.1f} 毫秒")
        log(f"  时间轴还原精度：{accuracy:.2f}%")
        stats = [f"键盘:{keyboard_count}"]
        if mouse_count:
            stats.append(f"鼠标:{mouse_count}")
        log(f"  执行动作：{executed_count}/{total_actions}（{'，'.join(stats)}）")

    finally:
        if interrupter is not None:
            pause_time = interrupter.run_decrypt_if_needed(keyboard_state)
            if pause_time:
                start_time += pause_time
        if keyboard_state is not None:
            active = keyboard_state.active_keys()
            if active:
                log(f"{label}：释放未松开的按键：{', '.join(k for k in active if k)}")
            keyboard_state.release_all()

    return executed_count > 0


class NoTrickDecryptController:
    MATCH_THRESHOLD = 0.7
    CHECK_INTERVAL = 0.4

    def __init__(self, gui, game_dir: str):
        self.gui = gui
        self.game_dir = game_dir
        self.stop_event = threading.Event()
        self.trigger_lock = threading.Lock()
        self.detected_entry = None
        self.detected_score = 0.0
        self.trigger_consumed = False
        self.macro_executed = False
        self.macro_missing = False
        self.executed_macros = 0
        self.templates = []
        self.thread = None
        self.session_started = False
        self.macro_done_event = threading.Event()

    def _log(self, message: str):
        if getattr(self.gui, "suppress_log", False):
            return
        log(message)

    def start(self) -> bool:
        if cv2 is None or np is None:
            self._log("缺少 opencv/numpy，无法开启无巧手解密监控。")
            try:
                self.gui.on_no_trick_unavailable("缺少 opencv/numpy")
            except Exception:
                pass
            return False
        if GAME_REGION is None:
            if not init_game_region():
                self._log("初始化游戏区域失败，无法开启无巧手解密监控。")
                try:
                    self.gui.on_no_trick_unavailable("未定位游戏窗口")
                except Exception:
                    pass
                return False
        self.templates = self._load_templates()
        if not self.templates:
            self.gui.on_no_trick_no_templates(self.game_dir)
            return False
        self.stop_event.clear()
        self.detected_entry = None
        self.detected_score = 0.0
        self.trigger_consumed = False
        self.macro_executed = False
        self.macro_missing = False
        self.executed_macros = 0
        self.session_started = True
        self.macro_done_event.clear()
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        self.gui.on_no_trick_monitor_started(self.templates)
        return True

    def stop(self):
        self.session_completed = True
        self.stop_event.set()
        self.session_started = False
        self.macro_done_event.set()

    def finish_session(self):
        if self.thread and self.thread.is_alive():
            self.stop_event.set()
            try:
                self.thread.join(timeout=0.5)
            except Exception:
                pass
        self.session_started = False
        self.macro_done_event.set()
        self.gui.on_no_trick_session_finished(
            triggered=self.detected_entry is not None,
            macro_executed=self.macro_executed,
            macro_missing=self.macro_missing,
        )

    def run_decrypt_if_needed(self, keyboard_state=None) -> float:
        if worker_stop.is_set():
            return 0.0
        with self.trigger_lock:
            entry = self.detected_entry
            score = self.detected_score
            consumed = self.trigger_consumed
        if entry is None or consumed:
            return 0.0

        with self.trigger_lock:
            if self.trigger_consumed:
                return 0.0
            self.trigger_consumed = True

        macro_path = entry.get("json_path")
        if not macro_path or not os.path.exists(macro_path):
            self._log(f"{self.gui.log_prefix} 无巧手解密：缺少对应宏文件 {macro_path}")
            self.macro_missing = True
            self.gui.on_no_trick_macro_missing(entry)
            self.macro_done_event.set()
            return 0.0

        restore_keys = None
        if keyboard_state is not None:
            restore_keys = keyboard_state.suspend()

        base_name = entry.get("base_name") or os.path.splitext(entry.get("name", ""))[0]
        macro_label = f"{self.gui.log_prefix} 无巧手解密 {base_name}.json"
        self._log(f"{self.gui.log_prefix} 无巧手解密：回放 {base_name}.json 宏。")

        # 重置执行标记，之前因为复用上一轮的 True 状态，会让等待环节误以为解密已经完成。
        # 这里清零后，HS70 的解密判定会一直阻塞到当前宏真正播放完毕。
        self.macro_executed = False
        self.macro_done_event.clear()

        start = time.perf_counter()
        self.gui.on_no_trick_macro_start(entry, score)

        def progress_cb(p):
            self.gui.on_no_trick_progress(p)

        executed = False
        use_segment_macro = bool(entry.get("has_segments"))

        try:
            if use_segment_macro:
                played = play_segment_macro(
                    macro_path,
                    macro_label,
                    progress_callback=progress_cb,
                )
                if played:
                    executed = True
                else:
                    use_segment_macro = False
                    try:
                        progress_cb(0.0)
                    except Exception:
                        pass

            if not use_segment_macro:
                executed = play_macro(
                    macro_path,
                    macro_label,
                    0.0,
                    0.0,
                    interrupt_on_exit=False,
                    progress_callback=progress_cb,
                )
        finally:
            if keyboard_state is not None:
                keyboard_state.resume(restore_keys)

        end = time.perf_counter()

        if executed:
            self.macro_executed = True
            self.executed_macros += 1
            self.gui.on_no_trick_macro_complete(entry)
            wait_after_decrypt_delay()
            end = time.perf_counter()
        else:
            end = time.perf_counter()

        with self.trigger_lock:
            if self.detected_entry is entry:
                self.detected_entry = None
                self.detected_score = 0.0

        self.macro_done_event.set()
        if executed:
            return max(0.0, end - start)
        return 0.0

    def wait_for_completion(self, timeout: Optional[float] = None) -> bool:
        return self.macro_done_event.wait(timeout)

    def _monitor_loop(self):
        while not self.stop_event.is_set() and not worker_stop.is_set():
            try:
                img = screenshot_game()
            except Exception as e:
                self._log(f"无巧手解密：截图失败 {e}")
                time.sleep(self.CHECK_INTERVAL)
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            for entry in self.templates:
                tpl = entry.get("template")
                if tpl is None:
                    continue
                try:
                    res = cv2.matchTemplate(gray, tpl, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, _ = cv2.minMaxLoc(res)
                except Exception as e:
                    self._log(f"无巧手解密：匹配 {entry.get('name')} 失败：{e}")
                    continue
                if max_val >= self.MATCH_THRESHOLD:
                    with self.trigger_lock:
                        self.detected_entry = entry
                        self.detected_score = max_val
                        # allow subsequent detections to be consumed by the
                        # playback loop without requiring the controller to be
                        # restarted after the first trigger
                        self.trigger_consumed = False
                    self.gui.on_no_trick_detected(entry, max_val)
                    break

            time.sleep(self.CHECK_INTERVAL)

    def _load_templates(self):
        templates = []
        if not os.path.isdir(self.game_dir):
            return templates
        try:
            candidates = [
                f
                for f in os.listdir(self.game_dir)
                if f.lower().endswith(".png")
            ]
        except Exception as e:
            self._log(f"读取 Game 目录失败：{e}")
            return templates

        def sort_key(name):
            base = os.path.splitext(name)[0]
            try:
                return int(base)
            except ValueError:
                return base

        for name in sorted(candidates, key=sort_key):
            base_name = os.path.splitext(name)[0]
            png_path = os.path.join(self.game_dir, name)
            json_path = os.path.join(self.game_dir, base_name + ".json")
            has_segments = macro_has_segments(json_path)
            try:
                data = np.fromfile(png_path, dtype=np.uint8)
                tpl = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
            except Exception as e:
                self._log(f"无巧手解密：读取模板 {png_path} 失败：{e}")
                tpl = None
            templates.append(
                {
                    "name": name,
                    "png_path": png_path,
                    "json_path": json_path,
                    "base_name": base_name,
                    "has_segments": has_segments,
                    "template": tpl,
                }
            )
        return templates


class FireworkNoTrickController:
    MATCH_THRESHOLD = 0.7
    CHECK_INTERVAL = 0.4
    COMPLETE_TIMEOUT = 3.0
    DUPLICATE_COOLDOWN = 1.0

    def __init__(self, gui, game_dir: str):
        self.gui = gui
        self.game_dir = game_dir
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.templates = []
        self.pending = deque()
        self.pending_names = set()
        self.recent_hits = {}
        self.last_detect_time = 0.0
        self.last_wait_notify = 0.0
        self.session_started = False
        self.session_completed = False
        self.trigger_count = 0
        self.executed_macros = 0
        self.macro_missing = False
        self.active = False
        self.thread = None
        self.verifying_completion = False
        self.macro_done_event = threading.Event()

    def _log(self, message: str):
        if getattr(self.gui, "suppress_log", False):
            return
        log(message)

    def start(self) -> bool:
        if cv2 is None or np is None:
            self._log("缺少 opencv/numpy，无法开启无巧手解密监控。")
            try:
                self.gui.on_no_trick_unavailable("缺少 opencv/numpy")
            except Exception:
                pass
            return False
        if GAME_REGION is None:
            if not init_game_region():
                self._log("初始化游戏区域失败，无法开启无巧手解密监控。")
                try:
                    self.gui.on_no_trick_unavailable("未定位游戏窗口")
                except Exception:
                    pass
                return False
        self.templates = self._load_templates()
        if not self.templates:
            try:
                self.gui.on_no_trick_no_templates(self.game_dir)
            except Exception:
                pass
            return False
        self.stop_event.clear()
        self.pending.clear()
        self.pending_names.clear()
        self.recent_hits.clear()
        self.last_detect_time = 0.0
        self.last_wait_notify = 0.0
        self.trigger_count = 0
        self.executed_macros = 0
        self.macro_missing = False
        self.active = False
        self.session_completed = False
        self.session_started = True
        self.verifying_completion = False
        self.macro_done_event.clear()
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        try:
            self.gui.on_no_trick_monitor_started(self.templates)
        except Exception:
            pass
        return True

    def stop(self):
        self.session_completed = True
        self.stop_event.set()
        self.session_started = False
        self.macro_done_event.set()

    def finish_session(self):
        if self.thread and self.thread.is_alive():
            self.stop_event.set()
            try:
                self.thread.join(timeout=0.5)
            except Exception:
                pass
        self.session_started = False
        try:
            self.gui.on_no_trick_session_finished(
                triggered=self.trigger_count > 0,
                macro_executed=self.executed_macros > 0,
                macro_missing=self.macro_missing,
            )
        except Exception:
            pass

    def run_decrypt_if_needed(self, keyboard_state=None) -> float:
        if worker_stop.is_set() or not self.session_started:
            return 0.0

        task = None
        with self.lock:
            if self.pending:
                task = self.pending.popleft()
                entry = task[0]
                if entry:
                    self.pending_names.discard(entry.get("name"))
            active = self.active
            last_time = self.last_detect_time

        if task is not None:
            entry, score = task
            return self._execute_entry(entry, score, keyboard_state)

        if active:
            now = time.time()
            elapsed = now - last_time if last_time else 0.0
            remaining = self.COMPLETE_TIMEOUT - elapsed
            if remaining > 0:
                if now - self.last_wait_notify >= 0.3:
                    self.last_wait_notify = now
                    if hasattr(self.gui, "on_no_trick_idle"):
                        try:
                            self.gui.on_no_trick_idle(max(0.0, remaining))
                        except Exception:
                            pass
                sleep_time = min(self.CHECK_INTERVAL, max(0.05, remaining))
                time.sleep(sleep_time)
                return sleep_time
            finalize = False
            with self.lock:
                self.active = False
                if self.verifying_completion and not self.pending:
                    finalize = True
                    self.verifying_completion = False
            if hasattr(self.gui, "on_no_trick_idle_complete"):
                try:
                    self.gui.on_no_trick_idle_complete()
                except Exception:
                    pass
            if finalize:
                with self.lock:
                    self.active = False
        return 0.0

    def _execute_entry(self, entry, score: float, keyboard_state=None) -> float:
        if entry is None:
            return 0.0
        name = entry.get("name", "")
        base_name = entry.get("base_name") or os.path.splitext(name)[0]
        macro_path = entry.get("json_path")
        with self.lock:
            self.active = True
            self.last_detect_time = time.time()
            self.last_wait_notify = 0.0
        if not macro_path or not os.path.exists(macro_path):
            self.macro_missing = True
            try:
                self.gui.on_no_trick_macro_missing(entry)
            except Exception:
                pass
            self.macro_done_event.set()
            return 0.0

        restore_keys = None
        if keyboard_state is not None:
            restore_keys = keyboard_state.suspend()

        macro_label = f"赛琪无巧手解密 {base_name}.json"
        self._log(f"赛琪无巧手解密：回放 {base_name}.json 宏。")

        try:
            self.gui.on_no_trick_macro_start(entry, score)
        except Exception:
            pass

        start = time.perf_counter()
        self.macro_done_event.clear()

        def progress_cb(p):
            try:
                self.gui.on_no_trick_progress(p)
            except Exception:
                pass

        try:
            try:
                play_macro(
                    macro_path,
                    macro_label,
                    0.0,
                    0.0,
                    interrupt_on_exit=False,
                    progress_callback=progress_cb,
                )
            finally:
                with self.lock:
                    self.last_detect_time = time.time()
        finally:
            if keyboard_state is not None:
                keyboard_state.resume(restore_keys)
            wait_after_decrypt_delay()

        self.executed_macros += 1

        with self.lock:
            self.last_detect_time = time.time()
            self.last_wait_notify = 0.0
            self.active = False
            self.verifying_completion = False

        # 一次解密宏执行完成后立即结束本次会话，确保外部冷却逻辑生效。
        self.stop()
        self.finish_session()

        try:
            self.gui.on_no_trick_macro_complete(entry)
        except Exception:
            pass

        self.macro_done_event.set()

        end = time.perf_counter()
        return max(0.0, end - start)

    def _monitor_loop(self):
        while not self.stop_event.is_set() and not worker_stop.is_set():
            try:
                img = screenshot_game()
            except Exception as e:
                self._log(f"赛琪无巧手解密：截图失败 {e}")
                time.sleep(self.CHECK_INTERVAL)
                continue

            try:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            except Exception as e:
                self._log(f"赛琪无巧手解密：转灰度失败 {e}")
                time.sleep(self.CHECK_INTERVAL)
                continue

            detected = False
            for entry in self.templates:
                tpl = entry.get("template")
                if tpl is None:
                    continue
                try:
                    res = cv2.matchTemplate(gray, tpl, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, _ = cv2.minMaxLoc(res)
                except Exception as e:
                    self._log(f"赛琪无巧手解密：匹配 {entry.get('name')} 失败：{e}")
                    continue
                if max_val >= self.MATCH_THRESHOLD:
                    self._queue_detection(entry, max_val)
                    detected = True
            if not detected:
                time.sleep(self.CHECK_INTERVAL)

    def _queue_detection(self, entry, score: float):
        now = time.time()
        name = entry.get("name")
        with self.lock:
            if self.session_completed:
                return
            last_hit = self.recent_hits.get(name, 0.0)
            if name in self.pending_names and now - last_hit < self.DUPLICATE_COOLDOWN:
                return
            if now - last_hit < self.DUPLICATE_COOLDOWN:
                return
            self.pending.append((entry, score))
            if name is not None:
                self.pending_names.add(name)
                self.recent_hits[name] = now
            self.last_detect_time = now
            self.active = True
            self.last_wait_notify = 0.0
        self.trigger_count += 1
        try:
            self.gui.on_no_trick_detected(entry, score)
        except Exception:
            pass

    def _load_templates(self):
        templates = []
        if not os.path.isdir(self.game_dir):
            return templates
        try:
            candidates = [
                f
                for f in os.listdir(self.game_dir)
                if f.lower().endswith(".png")
            ]
        except Exception as e:
            self._log(f"读取 {self.game_dir} 目录失败：{e}")
            return templates

        for name in sorted(candidates):
            base_name = os.path.splitext(name)[0]
            png_path = os.path.join(self.game_dir, name)
            json_path = os.path.join(self.game_dir, base_name + ".json")
            try:
                data = np.fromfile(png_path, dtype=np.uint8)
                tpl = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
            except Exception as e:
                self._log(f"赛琪无巧手解密：读取模板 {png_path} 失败：{e}")
                tpl = None
            templates.append(
                {
                    "name": name,
                    "png_path": png_path,
                    "json_path": json_path,
                    "base_name": base_name,
                    "template": tpl,
                }
            )
        return templates

    def was_stuck(self) -> bool:
        return False

# ======================================================================
#  赛琪大烟花（老项目）
# ======================================================================
def do_enter_buttons_first_round() -> bool:
    """第一轮需要 enter_step1 / enter_step2"""
    if not wait_and_click_template("enter_step1.png", "进入 步骤1", 20.0, 0.85):
        log("进入 步骤1 失败，本轮放弃。")
        return False
    if not wait_and_click_template("enter_step2.png", "进入 步骤2", 15.0, 0.85):
        log("进入 步骤2 失败，本轮放弃。")
        return False
    return True


def check_map_by_map1() -> bool:
    """只看 map1，阈值沿用 0.5"""
    if not wait_for_template("map1.png", "地图确认（map1）", 30.0, 0.5):
        log("地图匹配失败（map1 匹配度始终低于 0.5），本轮放弃。")
        return False
    return True


def do_exit_dungeon():
    wait_and_click_template("exit_step1.png", "退图 步骤1", 20.0, 0.8)
    wait_and_click_template("exit_step2.png", "退图 步骤2", 15.0, 0.8)


def emergency_recover():
    log("执行防卡死退图：ESC → G → Q → 退图")
    try:
        if keyboard is not None:
            keyboard.press_and_release("esc")
        else:
            pyautogui.press("esc")
    except Exception as e:
        log(f"发送 ESC 失败：{e}")
    time.sleep(1.0)
    click_template("G.png", "点击 G.png", 0.6)
    time.sleep(1.0)
    click_template("Q.png", "点击 Q.png", 0.6)
    time.sleep(1.0)
    do_exit_dungeon()


def run_one_round(wait_interval: float,
                  macro_a: str,
                  macro_b: str,
                  skip_enter_buttons: bool,
                  gui=None):
    log("===== 赛琪大烟花：新一轮开始 =====")
    report_progress(0.0)

    if not init_game_region():
        log("初始化游戏区域失败，本轮结束。")
        return

    if not skip_enter_buttons:
        if not do_enter_buttons_first_round():
            return

    if not check_map_by_map1():
        return

    log("地图确认成功，等待 2 秒让画面稳定…")
    t0 = time.time()
    while time.time() - t0 < 2.0 and not worker_stop.is_set():
        time.sleep(0.1)
    report_progress(0.3)

    controller = gui._start_firework_no_trick_monitor() if gui is not None else None
    try:
        play_macro(
            macro_a,
            "A 阶段（靠近大烟花）",
            0.3,
            0.6,
            interrupt_on_exit=True,
            interrupter=controller,
        )
    finally:
        if controller is not None and gui is not None:
            stuck = controller.was_stuck()
            controller.stop()
            controller.finish_session()
            gui._clear_firework_no_trick_controller(controller)
            if stuck:
                log("赛琪无巧手解密：连续解密失败，执行防卡死流程。")
                emergency_recover()
                return
            controller = None
    if worker_stop.is_set():
        return

    if wait_interval > 0:
        log(f"等待大烟花爆炸 {wait_interval:.1f} 秒…")
        t0 = time.time()
        while time.time() - t0 < wait_interval and not worker_stop.is_set():
            time.sleep(0.1)

    play_macro(macro_b, "B 阶段（撤退）", 0.7, 0.95, interrupt_on_exit=True)
    if worker_stop.is_set():
        return

    if is_exit_ui_visible():
        log("检测到退图按钮，执行正常退图。")
        do_exit_dungeon()
    else:
        emergency_recover()

    report_progress(1.0)
    log("赛琪大烟花：本轮完成。")


def worker_loop(wait_interval: float,
                macro_a: str,
                macro_b: str,
                auto_loop: bool,
                gui=None):
    try:
        first_round = True
        while not worker_stop.is_set():
            skip_enter = (auto_loop and not first_round)
            if skip_enter:
                log("自动循环：本轮跳过 enter_step1/2，只从地图确认(map1)开始。")
            run_one_round(wait_interval, macro_a, macro_b, skip_enter, gui=gui)
            first_round = False
            if worker_stop.is_set() or not auto_loop:
                break
            log("本轮结束，3 秒后继续下一轮…")
            time.sleep(3.0)
    except Exception as e:
        log(f"后台线程异常：{e}")
        traceback.print_exc()
    finally:
        report_progress(0.0)
        log("后台线程结束。")


# ---------- GUI：赛琪大烟花 ----------
class MainGUI:
    def __init__(self, root, cfg):
        self.root = root

        self.hotkey_var = tk.StringVar(value=cfg.get("hotkey", "1"))
        self.wait_var = tk.StringVar(value=str(cfg.get("wait_seconds", 8.0)))
        self.macro_a_var = tk.StringVar(value=cfg.get("macro_a_path", ""))
        self.macro_b_var = tk.StringVar(value=cfg.get("macro_b_path", ""))
        self.auto_loop_var = tk.BooleanVar(value=cfg.get("auto_loop", False))
        self.progress_var = tk.DoubleVar(value=0.0)
        self.no_trick_var = tk.BooleanVar(value=cfg.get("firework_no_trick", False))
        self.no_trick_status_var = tk.StringVar(value="未启用")
        self.no_trick_progress_var = tk.DoubleVar(value=0.0)
        self.no_trick_controller = None
        self.no_trick_image_ref = None
        self._last_idle_remaining = None

        self._build_ui()

    def _build_ui(self):
        self.content_frame = tk.Frame(self.root)
        self.content_frame.pack(fill="both", expand=True)

        self.left_panel = tk.Frame(self.content_frame)
        self.left_panel.pack(side="left", fill="both", expand=True)

        self.right_panel = tk.Frame(self.content_frame)
        self.right_panel.pack(side="right", fill="y", padx=(5, 10), pady=5)

        top = tk.Frame(self.left_panel)
        top.pack(fill="x", padx=10, pady=5)

        tk.Label(top, text="热键:").grid(row=0, column=0, sticky="e")
        tk.Entry(top, textvariable=self.hotkey_var, width=15).grid(row=0, column=1, sticky="w")
        ttk.Button(top, text="录制热键", command=self.capture_hotkey).grid(row=0, column=2, padx=3)
        ttk.Button(top, text="保存配置", command=self.save_cfg).grid(row=0, column=3, padx=3)

        tk.Label(top, text="烟花等待(秒):").grid(row=1, column=0, sticky="e")
        tk.Entry(top, textvariable=self.wait_var, width=8).grid(row=1, column=1, sticky="w")
        tk.Checkbutton(top, text="自动循环", variable=self.auto_loop_var).grid(row=1, column=2, sticky="w")

        toggle = tk.Frame(self.left_panel)
        toggle.pack(fill="x", padx=10, pady=(0, 5))
        tk.Checkbutton(
            toggle,
            text="开启无巧手解密",
            variable=self.no_trick_var,
            command=self._on_no_trick_toggle,
        ).pack(anchor="w")

        self.log_panel = CollapsibleLogPanel(self.left_panel, "日志")
        self.log_panel.pack(fill="both", padx=10, pady=(5, 5))
        self.log_text = self.log_panel.text

        progress_wrap = tk.LabelFrame(self.left_panel, text="执行进度")
        progress_wrap.pack(fill="x", padx=10, pady=(0, 5))
        self.progress = ttk.Progressbar(
            progress_wrap,
            variable=self.progress_var,
            maximum=100.0,
            mode="determinate",
        )
        self.progress.pack(fill="x", padx=10, pady=5)

        frm2 = tk.LabelFrame(self.left_panel, text="宏设置")
        frm2.pack(fill="x", padx=10, pady=5)

        tk.Label(frm2, text="A 宏（靠近大烟花）:").grid(row=0, column=0, sticky="e")
        tk.Entry(frm2, textvariable=self.macro_a_var, width=60).grid(row=0, column=1, sticky="w")
        ttk.Button(frm2, text="浏览…", command=self.choose_a).grid(row=0, column=2, padx=3)

        tk.Label(frm2, text="B 宏（撤退 / 退图前）:").grid(row=1, column=0, sticky="e")
        tk.Entry(frm2, textvariable=self.macro_b_var, width=60).grid(row=1, column=1, sticky="w")
        ttk.Button(frm2, text="浏览…", command=self.choose_b).grid(row=1, column=2, padx=3)

        frm3 = tk.Frame(self.left_panel)
        frm3.pack(padx=10, pady=5)

        ttk.Button(
            frm3,
            text="开始执行",
            command=lambda: self.start_worker(self.auto_loop_var.get()),
        ).grid(row=0, column=0, padx=3)
        ttk.Button(frm3, text="开始监听热键", command=self.start_listen).grid(row=0, column=1, padx=3)
        ttk.Button(frm3, text="停止", command=self.stop_listen).grid(row=0, column=2, padx=3)
        ttk.Button(frm3, text="只执行一轮", command=self.run_once).grid(row=0, column=3, padx=3)

        self.no_trick_status_frame = tk.LabelFrame(self.right_panel, text="无巧手解密状态")
        self.no_trick_status_frame.pack(fill="both", expand=True, padx=5, pady=5)

        status_inner = tk.Frame(self.no_trick_status_frame)
        status_inner.pack(fill="x", padx=5, pady=5)

        self.no_trick_status_label = tk.Label(
            status_inner,
            textvariable=self.no_trick_status_var,
            anchor="w",
            justify="left",
        )
        self.no_trick_status_label.pack(fill="x", anchor="w")

        self.no_trick_image_label = tk.Label(
            self.no_trick_status_frame,
            relief="sunken",
            bd=1,
            bg="#f8f8f8",
        )
        self.no_trick_image_label.pack(fill="both", expand=True, padx=10, pady=(0, 5))

        self.no_trick_progress = ttk.Progressbar(
            self.no_trick_status_frame,
            variable=self.no_trick_progress_var,
            maximum=100.0,
            mode="determinate",
        )
        self.no_trick_progress.pack(fill="x", padx=10, pady=(0, 8))

        self._update_no_trick_ui()

    def _on_no_trick_toggle(self):
        if not self.no_trick_var.get():
            self._stop_firework_no_trick_monitor()
        self._update_no_trick_ui()

    def _update_no_trick_ui(self):
        if self.no_trick_var.get():
            self._set_no_trick_status("等待刷图时识别解密图像…")
            self._set_no_trick_progress(0.0)
            self._set_no_trick_image(None)
        else:
            self._set_no_trick_status("未启用")
            self._set_no_trick_progress(0.0)
            self._set_no_trick_image(None)

    def _set_no_trick_status(self, text: str):
        self.no_trick_status_var.set(text)

    def _set_no_trick_progress(self, percent: float):
        self.no_trick_progress_var.set(max(0.0, min(100.0, percent)))

    def _set_no_trick_image(self, photo):
        if photo is None:
            self.no_trick_image_label.config(image="")
        else:
            self.no_trick_image_label.config(image=photo)
        self.no_trick_image_ref = photo

    def _load_no_trick_preview(self, path: str, max_size: int = 240):
        if not path or not os.path.exists(path):
            return None
        if Image is not None and ImageTk is not None:
            try:
                with Image.open(path) as pil_img:
                    pil_img = pil_img.convert("RGBA")
                    w, h = pil_img.size
                    scale = 1.0
                    if max(w, h) > max_size:
                        scale = max_size / max(w, h)
                        pil_img = pil_img.resize(
                            (
                                max(1, int(w * scale)),
                                max(1, int(h * scale)),
                            ),
                            Image.LANCZOS,
                        )
                    return ImageTk.PhotoImage(pil_img)
            except Exception:
                pass
        try:
            img = tk.PhotoImage(file=path)
        except Exception:
            return None
        w = max(img.width(), 1)
        h = max(img.height(), 1)
        factor = max(1, (max(w, h) + max_size - 1) // max_size)
        if factor > 1:
            img = img.subsample(factor, factor)
        return img

    def _start_firework_no_trick_monitor(self):
        if not self.no_trick_var.get():
            return None
        if self.no_trick_controller is not None:
            return self.no_trick_controller
        controller = FireworkNoTrickController(self, GAME_SQ_DIR)
        if controller.start():
            self.no_trick_controller = controller
            self._last_idle_remaining = None
            return controller
        return None

    def _stop_firework_no_trick_monitor(self):
        controller = self.no_trick_controller
        if controller is None:
            return
        controller.stop()
        controller.finish_session()
        self.no_trick_controller = None

    def _clear_firework_no_trick_controller(self, controller):
        if self.no_trick_controller is controller:
            self.no_trick_controller = None

    # ---- 无巧手解密回调 ----
    def on_no_trick_unavailable(self, reason: str):
        if not self.no_trick_var.get():
            return

        def _():
            if not self.no_trick_var.get():
                return
            self._set_no_trick_status(f"无巧手解密不可用：{reason}。")
            self._set_no_trick_progress(0.0)
            self._set_no_trick_image(None)

        post_to_main_thread(_)

    def on_no_trick_no_templates(self, game_dir: str):
        if not self.no_trick_var.get():
            return

        def _():
            if not self.no_trick_var.get():
                return
            self._set_no_trick_status(
                "GAME-sq 文件夹中未找到解密图像，请放置 PNG 和对应 JSON。"
            )
            self._set_no_trick_progress(0.0)
            self._set_no_trick_image(None)

        post_to_main_thread(_)

    def on_no_trick_monitor_started(self, templates):
        if not self.no_trick_var.get():
            return
        total = len(templates)
        valid = sum(1 for t in templates if t.get("template") is not None)

        def _():
            if not self.no_trick_var.get():
                return
            if valid <= 0:
                self._set_no_trick_status("无有效模板，无法识别解密图像。")
            else:
                self._set_no_trick_status(f"等待识别解密图像（共 {total} 张模板）…")
            self._set_no_trick_progress(0.0)
            self._set_no_trick_image(None)
            self._last_idle_remaining = None

        post_to_main_thread(_)

    def on_no_trick_detected(self, entry, score: float):
        if not self.no_trick_var.get():
            return

        def _():
            if not self.no_trick_var.get():
                return
            name = entry.get("name", "")
            self._set_no_trick_status(f"已识别解密图像：{name}，开始执行宏…")
            self._set_no_trick_progress(0.0)
            photo = self._load_no_trick_preview(entry.get("png_path"))
            self._set_no_trick_image(photo)
            self._last_idle_remaining = None

        post_to_main_thread(_)

    def on_no_trick_macro_start(self, entry, score: float):
        if not self.no_trick_var.get():
            return

        def _():
            if not self.no_trick_var.get():
                return
            self._set_no_trick_progress(0.0)

        post_to_main_thread(_)

    def on_no_trick_progress(self, progress: float):
        if not self.no_trick_var.get():
            return

        def _():
            if not self.no_trick_var.get():
                return
            self._set_no_trick_progress(progress * 100.0)

        post_to_main_thread(_)

    def on_no_trick_macro_complete(self, entry):
        if not self.no_trick_var.get():
            return

        def _():
            if not self.no_trick_var.get():
                return
            name = entry.get("name", "")
            self._set_no_trick_status(f"{name} 解密完成。")
            self._set_no_trick_progress(100.0)
            self._last_idle_remaining = None

        post_to_main_thread(_)

    def on_no_trick_retry(self, entry, attempt_no: int):
        if not self.no_trick_var.get():
            return

        def _():
            if not self.no_trick_var.get():
                return
            base = os.path.splitext(entry.get("name", ""))[0]
            self._set_no_trick_status(
                f"{base} 解密失败，准备第 {attempt_no} 次重试…"
            )
            self._set_no_trick_progress(0.0)
            self._last_idle_remaining = None

        post_to_main_thread(_)

    def on_no_trick_stuck(self, entry):
        if not self.no_trick_var.get():
            return

        def _():
            if not self.no_trick_var.get():
                return
            base = os.path.splitext(entry.get("name", ""))[0]
            self._set_no_trick_status(f"{base} 解密失败，执行防卡死…")
            self._set_no_trick_progress(0.0)
            self._last_idle_remaining = None

        post_to_main_thread(_)

    def on_no_trick_macro_missing(self, entry):
        if not self.no_trick_var.get():
            return

        def _():
            if not self.no_trick_var.get():
                return
            base = os.path.splitext(entry.get("name", ""))[0]
            self._set_no_trick_status(f"未找到 {base}.json，跳过本次解密。")
            self._set_no_trick_progress(0.0)
            self._set_no_trick_image(None)

        post_to_main_thread(_)

    def on_no_trick_idle(self, remaining: float):
        if not self.no_trick_var.get():
            return

        def _():
            if not self.no_trick_var.get():
                return
            if self._last_idle_remaining is not None and abs(self._last_idle_remaining - remaining) < 0.1:
                return
            self._last_idle_remaining = remaining
            self._set_no_trick_status(
                f"等待下一张解密图像…（约 {remaining:.1f} 秒）"
            )

        post_to_main_thread(_)

    def on_no_trick_idle_complete(self):
        if not self.no_trick_var.get():
            return

        def _():
            if not self.no_trick_var.get():
                return
            self._set_no_trick_status("解密流程结束，恢复原宏执行。")
            self._set_no_trick_progress(100.0)
            self._last_idle_remaining = None

        post_to_main_thread(_)

    def on_no_trick_session_finished(self, triggered: bool, macro_executed: bool, macro_missing: bool):
        if not self.no_trick_var.get():
            return

        def _():
            if not self.no_trick_var.get():
                return
            if not triggered:
                self._set_no_trick_status("本轮未识别到解密图像。")
                self._set_no_trick_progress(0.0)
                self._set_no_trick_image(None)
            elif macro_executed:
                self._set_no_trick_status("解密流程完成，继续执行原宏。")
                self._set_no_trick_progress(100.0)
            elif macro_missing:
                # 状态已在缺失回调中更新
                pass

        post_to_main_thread(_)

    def log(self, msg: str):
        widget = getattr(self, "log_text", None)
        if widget is None:
            return

        def _append():
            append_formatted_log(widget, msg)

        post_to_main_thread(_append)

    def set_progress(self, p: float):
        self.progress_var.set(max(0.0, min(1.0, p)) * 100.0)

    # 事件
    def choose_a(self):
        p = filedialog.askopenfilename(
            title="选择 A 宏 JSON",
            initialdir=SCRIPTS_DIR,
            filetypes=[("JSON 文件", "*.json"), ("所有文件", "*.*")],
        )
        if p:
            self.macro_a_var.set(p)

    def choose_b(self):
        p = filedialog.askopenfilename(
            title="选择 B 宏 JSON",
            initialdir=SCRIPTS_DIR,
            filetypes=[("JSON 文件", "*.json"), ("所有文件", "*.*")],
        )
        if p:
            self.macro_b_var.set(p)

    def capture_hotkey(self):
        if keyboard is None:
            messagebox.showerror("错误", "未安装 keyboard，无法录制热键。")
            return
        log("请按下你想要的热键组合…")

        def worker():
            try:
                hk = keyboard.read_hotkey(suppress=False)
                self.hotkey_var.set(hk)
                log(f"捕获热键：{hk}")
            except Exception as e:
                log(f"录制热键失败：{e}")
        threading.Thread(target=worker, daemon=True).start()

    def save_cfg(self):
        try:
            cfg = {
                "hotkey": self.hotkey_var.get().strip(),
                "wait_seconds": float(self.wait_var.get()),
                "macro_a_path": self.macro_a_var.get(),
                "macro_b_path": self.macro_b_var.get(),
                "auto_loop": self.auto_loop_var.get(),
                "firework_no_trick": bool(self.no_trick_var.get()),
            }
            save_config(cfg)
        except Exception as e:
            messagebox.showerror("错误", f"保存配置失败：{e}")

    def ensure_macros(self) -> bool:
        if not self.macro_a_var.get() or not self.macro_b_var.get():
            messagebox.showwarning("提示", "请同时设置 A 宏和 B 宏。")
            return False
        return True

    def start_listen(self):
        global hotkey_handle
        if keyboard is None:
            messagebox.showerror("错误", "未安装 keyboard，无法使用热键监听。")
            return
        if not self.ensure_macros():
            return
        hk = self.hotkey_var.get().strip()
        if not hk:
            messagebox.showwarning("提示", "请先设置一个热键。")
            return

        worker_stop.clear()
        if hotkey_handle is not None:
            try:
                keyboard.remove_hotkey(hotkey_handle)
            except Exception:
                pass

        def on_hotkey():
            log("检测到热键，开始执行一轮。")
            self.start_worker(self.auto_loop_var.get())

        try:
            hotkey_handle = keyboard.add_hotkey(hk, on_hotkey)
        except Exception as e:
            messagebox.showerror("错误", f"注册热键失败：{e}")
            return
        log(f"开始监听热键：{hk}")

    def stop_listen(self):
        global hotkey_handle
        worker_stop.set()
        if keyboard is not None and hotkey_handle is not None:
            try:
                keyboard.remove_hotkey(hotkey_handle)
            except Exception:
                pass
        hotkey_handle = None
        log("已停止监听，当前轮结束后退出。")

    def start_worker(self, auto_loop: bool):
        if not self.ensure_macros():
            return
        if not round_running_lock.acquire(blocking=False):
            log("已有一轮在运行，本次忽略。")
            return
        worker_stop.clear()
        wait_sec = float(self.wait_var.get())
        macro_a = self.macro_a_var.get()
        macro_b = self.macro_b_var.get()

        def worker():
            try:
                worker_loop(wait_sec, macro_a, macro_b, auto_loop, gui=self)
            finally:
                round_running_lock.release()
        threading.Thread(target=worker, daemon=True).start()

    def run_once(self):
        self.start_worker(auto_loop=False)


# ======================================================================
#  多人物/多mod 密函选择辅助
# ======================================================================


class MultiLetterSelectionMixin:
    """Mixin providing optional multi-letter selection for guard/expel UIs."""

    multi_toggle_text: Optional[str] = None
    multi_list_title: str = "刷取序列"
    multi_item_prefix: str = "刷取"
    multi_config_enabled_key: str = "multi_letter_enabled"
    multi_config_files_key: str = "multi_letter_files"

    def _init_multi_selection_feature(self, cfg_section: dict):
        self.multi_enabled_var: Optional[tk.BooleanVar] = None
        self.multi_letter_names: List[str] = []
        self.multi_letter_index_map: Dict[str, int] = {}
        self.multi_selection_panel: Optional[tk.Frame] = None
        self.multi_selection_canvas: Optional[tk.Canvas] = None
        self.multi_selection_canvas_window: Optional[int] = None
        self.multi_selection_items: Optional[tk.Frame] = None
        self.multi_selection_listbox: Optional[tk.Widget] = None
        self.multi_save_button: Optional[ttk.Button] = None
        self.multi_clear_button: Optional[ttk.Button] = None
        self.multi_order_labels: List[tk.Label] = []
        self.multi_toggle_frame: Optional[tk.Frame] = None
        self.letters_outer: Optional[tk.Frame] = None
        self.letters_grid_container: Optional[tk.Frame] = None
        self.multi_preview_refs: List[tk.PhotoImage] = []
        self.multi_runtime_queue: List[Dict[str, Any]] = []
        self.multi_runtime_index: int = 0
        self.multi_runtime_status: Dict[str, bool] = {}
        self.current_multi_entry: Optional[Dict[str, Any]] = None

        if not self.multi_toggle_text:
            return

        self.multi_enabled_var = tk.BooleanVar(
            value=bool(cfg_section.get(self.multi_config_enabled_key, False))
        )

        stored = cfg_section.get(self.multi_config_files_key, [])
        if isinstance(stored, list):
            self.multi_letter_names = [str(x) for x in stored if isinstance(x, str)]
        else:
            self.multi_letter_names = []

        self._rebuild_multi_letter_index()

    # ------------------------------------------------------------------
    # 基础状态查询
    # ------------------------------------------------------------------
    def _multi_feature_available(self) -> bool:
        return bool(self.multi_toggle_text)

    def _multi_mode_active(self) -> bool:
        return bool(self.multi_enabled_var and self.multi_enabled_var.get())

    def _rebuild_multi_letter_index(self):
        self.multi_letter_index_map = {
            name: idx + 1 for idx, name in enumerate(self.multi_letter_names)
        }

    # ------------------------------------------------------------------
    # UI 构建与更新
    # ------------------------------------------------------------------
    def _build_multi_toggle(self, container: tk.Widget):
        if not self._multi_feature_available():
            return
        self.multi_toggle_frame = tk.Frame(container)
        self.multi_toggle_frame.pack(fill="x", padx=10, pady=(0, 5))
        tk.Checkbutton(
            self.multi_toggle_frame,
            text=self.multi_toggle_text,
            variable=self.multi_enabled_var,
            command=self._on_multi_toggle,
        ).pack(anchor="w")

    def _build_letter_layout(self, frame_letters: tk.LabelFrame) -> tk.Frame:
        if not self._multi_feature_available():
            self.letters_grid_container = tk.Frame(frame_letters)
            self.letters_grid_container.pack(fill="both", expand=True, padx=5, pady=5)
            return self.letters_grid_container

        self.letters_outer = tk.Frame(frame_letters)
        self.letters_outer.pack(fill="both", expand=True, padx=5, pady=5)

        self.multi_selection_panel = tk.Frame(
            self.letters_outer, width=190, bd=1, relief="groove"
        )
        self.multi_selection_panel.pack_propagate(False)

        title = tk.Label(
            self.multi_selection_panel,
            text=self.multi_list_title,
            fg="#cc0000",
            font=("Microsoft YaHei", 10, "bold"),
        )
        title.pack(fill="x", padx=6, pady=(6, 3))

        canvas_box = tk.Frame(self.multi_selection_panel)
        canvas_box.pack(fill="both", expand=True, padx=0, pady=(0, 6))

        self.multi_selection_canvas = tk.Canvas(
            canvas_box,
            width=160,
            highlightthickness=0,
            borderwidth=0,
        )
        self.multi_selection_canvas.pack(side="left", fill="both", expand=True, padx=(6, 0))
        scrollbar = ttk.Scrollbar(
            canvas_box, orient="vertical", command=self.multi_selection_canvas.yview
        )
        scrollbar.pack(side="right", fill="y", padx=(0, 6))
        self.multi_selection_canvas.configure(yscrollcommand=scrollbar.set)

        self.multi_selection_items = tk.Frame(self.multi_selection_canvas)
        self.multi_selection_canvas_window = self.multi_selection_canvas.create_window(
            (0, 0), window=self.multi_selection_items, anchor="nw"
        )

        def _sync_scrollregion(event):
            self.multi_selection_canvas.configure(
                scrollregion=self.multi_selection_canvas.bbox("all")
            )

        self.multi_selection_items.bind("<Configure>", _sync_scrollregion)
        self.multi_selection_canvas.bind(
            "<Configure>",
            lambda e: self.multi_selection_canvas.itemconfigure(
                self.multi_selection_canvas_window, width=e.width
            ),
        )
        self.multi_selection_listbox = self.multi_selection_items

        btn_box = tk.Frame(self.multi_selection_panel)
        btn_box.pack(fill="x", padx=6, pady=(0, 6))
        self.multi_save_button = ttk.Button(
            btn_box, text="保存刷取配置", command=self._save_multi_letter_config
        )
        self.multi_save_button.pack(fill="x")
        self.multi_clear_button = ttk.Button(
            btn_box, text="清空选择", command=self._clear_multi_selection
        )
        self.multi_clear_button.pack(fill="x", pady=(6, 0))

        self.letters_grid_container = tk.Frame(self.letters_outer)
        self.letters_grid_container.pack(side="left", fill="both", expand=True)

        self._update_multi_panel_visibility()
        self._update_multi_listbox()
        return self.letters_grid_container

    def _update_multi_panel_visibility(self):
        if not self._multi_feature_available() or self.multi_selection_panel is None:
            return
        if self._multi_mode_active():
            if not self.multi_selection_panel.winfo_manager():
                self.multi_selection_panel.pack(side="left", fill="y", padx=(0, 6))
        else:
            if self.multi_selection_panel.winfo_manager():
                self.multi_selection_panel.pack_forget()

    def _on_multi_toggle(self):
        self._update_multi_panel_visibility()
        self._update_multi_selection_display()
        self._update_letter_order_labels()
        self._highlight_button(None)

    def _update_multi_selection_display(self):
        if not hasattr(self, "selected_label_var"):
            return
        if self._multi_mode_active():
            count = len(self.multi_letter_names)
            if count:
                self.selected_letter_path = self._letter_name_to_path(
                    self.multi_letter_names[0]
                )
                self.selected_label_var.set(
                    f"多刷模式：已选择 {count} 个{self.letter_label}"
                )
                self.stat_name_var.set(self.multi_letter_names[0])
                img = load_uniform_letter_image(self.selected_letter_path)
                if img is not None:
                    self.stat_image = img
                    self.stat_image_label.config(image=self.stat_image)
            else:
                self.selected_letter_path = None
                self.selected_label_var.set(
                    f"多刷模式：请点击选择{self.letter_label}"
                )
                self.stat_name_var.set("（未选择）")
                self.stat_image_label.config(image="")
        else:
            if self.selected_letter_path:
                base = os.path.basename(self.selected_letter_path)
                self.selected_label_var.set(f"当前选择{self.letter_label}：{base}")
                self.stat_name_var.set(base)
                img = load_uniform_letter_image(self.selected_letter_path)
                if img is not None:
                    self.stat_image = img
                    self.stat_image_label.config(image=self.stat_image)
            else:
                self.selected_label_var.set(f"当前未选择{self.letter_label}")
                self.stat_name_var.set("（未选择）")
                self.stat_image_label.config(image="")
        self._update_multi_listbox()

    def _update_multi_listbox(self):
        if self.multi_selection_listbox is None:
            return
        for child in self.multi_selection_listbox.winfo_children():
            child.destroy()
        self.multi_preview_refs = []
        prefix = self.multi_item_prefix
        for idx, name in enumerate(self.multi_letter_names, 1):
            item = tk.Frame(self.multi_selection_listbox, bd=1, relief="groove")
            item.pack(fill="x", padx=6, pady=4)

            header = tk.Frame(item)
            header.pack(fill="x", padx=4, pady=(4, 2))
            tk.Label(
                header,
                text=f"{prefix}{idx}",
                fg="#cc0000",
                font=("Microsoft YaHei", 10, "bold"),
            ).pack(side="left")

            status_text = ""
            if self.multi_runtime_status.get(name):
                status_text = "完成"
            elif self.current_multi_entry and self.current_multi_entry.get("name") == name:
                status_text = "当前"
            if status_text:
                tk.Label(
                    header,
                    text=status_text,
                    fg="#008000" if status_text == "完成" else "#ff6600",
                    font=("Microsoft YaHei", 9, "bold"),
                ).pack(side="right")

            path = self._letter_name_to_path(name)
            img = None
            if path and os.path.exists(path):
                img = load_uniform_letter_image(path, box_size=max(72, LETTER_IMAGE_SIZE // 2))
            preview = tk.Label(item)
            preview.pack(padx=4, pady=(0, 4))
            if img is not None:
                self.multi_preview_refs.append(img)
                preview.config(image=img)
            else:
                preview.config(text="(图片缺失)", fg="#666666")

    def _update_letter_order_labels(self):
        if not getattr(self, "multi_order_labels", None):
            return
        show_numbers = self._multi_mode_active()
        mapping = self.multi_letter_index_map if show_numbers else {}
        for idx, label in enumerate(self.multi_order_labels):
            if idx >= len(getattr(self, "visible_letter_files", [])):
                continue
            name = self.visible_letter_files[idx]
            order = mapping.get(name)
            if order:
                label.config(text=str(order))
            else:
                label.config(text="")

    # ------------------------------------------------------------------
    # 多选逻辑
    # ------------------------------------------------------------------
    def _multi_handle_letter_click(self, path: str, idx: int):
        name = os.path.basename(path)
        if name in self.multi_letter_names:
            self.multi_letter_names.remove(name)
        else:
            self.multi_letter_names.append(name)
        self._rebuild_multi_letter_index()
        self._update_multi_selection_display()
        self._update_letter_order_labels()
        self._highlight_button(None)

    def _save_multi_letter_config(self):
        if not self._multi_feature_available():
            return
        section = self.cfg.setdefault(self.cfg_key, {})
        section[self.multi_config_enabled_key] = bool(
            self.multi_enabled_var.get() if self.multi_enabled_var else False
        )
        section[self.multi_config_files_key] = list(self.multi_letter_names)
        save_config(self.cfg)
        messagebox.showinfo("提示", "刷取配置已保存。")

    def _clear_multi_selection(self):
        if not self._multi_feature_available():
            return
        self.multi_letter_names.clear()
        self._rebuild_multi_letter_index()
        self._update_multi_selection_display()
        self._update_letter_order_labels()
        self._highlight_button(None)

    def _letter_name_to_path(self, name: str) -> Optional[str]:
        if not name:
            return None
        if os.path.isabs(name):
            return name
        return os.path.join(self.letters_dir, name)

    def _prepare_multi_runtime_cycle(self) -> bool:
        if not self._multi_mode_active():
            self.multi_runtime_queue = []
            self.multi_runtime_index = 0
            self.multi_runtime_status = {}
            self.current_multi_entry = None
            return True
        if not self.multi_letter_names:
            messagebox.showwarning("提示", "多刷模式下请至少选择一个密函。")
            return False
        self.multi_runtime_queue = [
            {
                "name": name,
                "path": self._letter_name_to_path(name),
                "completed": False,
            }
            for name in self.multi_letter_names
        ]
        self.multi_runtime_index = 0
        self.multi_runtime_status = {name: False for name in self.multi_letter_names}
        self.current_multi_entry = None
        self._update_multi_listbox()
        return True

    def _get_active_multi_entry(self) -> Optional[Dict[str, Any]]:
        if not self.multi_runtime_queue:
            return None
        while self.multi_runtime_index < len(self.multi_runtime_queue):
            entry = self.multi_runtime_queue[self.multi_runtime_index]
            if not entry.get("completed"):
                return entry
            self.multi_runtime_index += 1
        return None

    def _mark_multi_entry_completed(self, entry: Dict[str, Any], reason: str):
        entry["completed"] = True
        self.multi_runtime_status[entry["name"]] = True
        self.current_multi_entry = None
        if hasattr(self, "log"):
            self.log(f"{self.log_prefix} {entry['name']} 已完成（{reason}）。")
        self._update_multi_listbox()

    def _on_multi_letter_success(self, entry: Dict[str, Any]):
        self.current_multi_entry = entry
        self.selected_letter_path = entry.get("path")
        name = entry.get("name", "")
        if name:
            self.stat_name_var.set(name)
        img = None
        if self.selected_letter_path:
            img = load_uniform_letter_image(self.selected_letter_path)
        if img is not None:
            self.stat_image = img
            self.stat_image_label.config(image=self.stat_image)
        self._update_multi_listbox()

    # ------------------------------------------------------------------
    # 选择流程（供子类调用）
    # ------------------------------------------------------------------
    def _select_letter_sequence(self, prefix: str, need_open_button: bool) -> bool:
        if need_open_button:
            btn_open_letter = get_template_name("BTN_OPEN_LETTER", "选择密函.png")
            if not wait_and_click_template(
                btn_open_letter,
                f"{prefix}：选择密函按钮",
                20.0,
                0.8,
            ):
                self.log(f"{prefix}：未能点击 选择密函.png。")
                return False

        if self._multi_mode_active():
            remaining: List[Tuple[int, Dict[str, Any]]] = [
                (idx, entry)
                for idx, entry in enumerate(self.multi_runtime_queue)
                if not entry.get("completed")
            ]
            if not remaining:
                messagebox.showinfo("提示", "多刷序列已全部完成，没有可执行的密函。")
                return False

            quick_timeout = 1.0
            missed: List[Tuple[int, Dict[str, Any], Optional[str]]] = []

            for idx, entry in remaining:
                path = entry.get("path")
                if not path or not os.path.exists(path):
                    missed.append((idx, entry, "文件缺失"))
                    continue

                if wait_and_click_template_from_path(
                    path,
                    f"{prefix}：点击{self.letter_label}",
                    quick_timeout,
                    LETTER_MATCH_THRESHOLD,
                ):
                    for m_idx, m_entry, reason in missed:
                        self.multi_runtime_index = m_idx
                        if reason:
                            self._mark_multi_entry_completed(m_entry, reason)
                        else:
                            self._mark_multi_entry_completed(
                                m_entry, "未再识别到密函"
                            )
                    self.multi_runtime_index = idx
                    self._on_multi_letter_success(entry)
                    return wait_and_click_template(
                        BTN_CONFIRM_LETTER,
                        f"{prefix}：确认选择",
                        20.0,
                        LETTER_MATCH_THRESHOLD,
                    )

                missed.append((idx, entry, None))

            fallback_success = False
            fallback_idx: Optional[int] = None
            for idx, entry, reason in missed:
                if reason is not None:
                    continue
                path = entry.get("path")
                if not path or not os.path.exists(path):
                    continue
                if click_letter_template(
                    path,
                    f"{prefix}：点击{self.letter_label}",
                    5.0,
                    LETTER_MATCH_THRESHOLD,
                ):
                    self.multi_runtime_index = idx
                    self._on_multi_letter_success(entry)
                    fallback_success = True
                    fallback_idx = idx
                    break

            if fallback_success:
                if fallback_idx is not None:
                    for m_idx, m_entry, reason in missed:
                        if m_idx == fallback_idx:
                            break
                        self.multi_runtime_index = m_idx
                        if reason:
                            self._mark_multi_entry_completed(m_entry, reason)
                        else:
                            self._mark_multi_entry_completed(
                                m_entry, "未再识别到密函"
                            )
                    self.multi_runtime_index = fallback_idx
                return wait_and_click_template(
                    BTN_CONFIRM_LETTER,
                    f"{prefix}：确认选择",
                    20.0,
                    LETTER_MATCH_THRESHOLD,
                )

            for idx, entry, reason in missed:
                self.multi_runtime_index = idx
                if reason:
                    self._mark_multi_entry_completed(entry, reason)
                else:
                    self._mark_multi_entry_completed(entry, "未再识别到密函")

            messagebox.showinfo("提示", "多刷序列已全部完成，没有可执行的密函。")
            return False

        # 单刷逻辑
        if not self.selected_letter_path:
            messagebox.showwarning("提示", f"请先选择一个{self.letter_label}。")
            return False
        if not click_letter_template(
            self.selected_letter_path,
            f"{prefix}：点击{self.letter_label}",
            20.0,
            LETTER_MATCH_THRESHOLD,
        ):
            self.log(f"{prefix}：未能点击{self.letter_label}。")
            return False
        return wait_and_click_template(
            BTN_CONFIRM_LETTER,
            f"{prefix}：确认选择",
            20.0,
            LETTER_MATCH_THRESHOLD,
        )


# ======================================================================
#  探险无尽血清 - 人物碎片自动刷取
# ======================================================================
class FragmentFarmGUI(MultiLetterSelectionMixin):
    MAX_LETTERS = 20
    multi_toggle_text = "同时刷取多个人物碎片"
    multi_list_title = "人物刷取顺序"
    multi_item_prefix = "刷取"

    def __init__(self, parent, cfg, enable_no_trick_decrypt: bool = False):
        self.parent = parent
        self.cfg = cfg
        self.cfg_key = getattr(self, "cfg_key", "guard_settings")
        self.letter_label = getattr(self, "letter_label", "人物密函")
        self.product_label = getattr(self, "product_label", "人物碎片")
        self.product_short_label = getattr(self, "product_short_label", "碎片")
        self.entity_label = getattr(self, "entity_label", "人物")
        self.letters_dir = getattr(self, "letters_dir", TEMPLATE_LETTERS_DIR)
        self.letters_dir_hint = getattr(self, "letters_dir_hint", "templates_letters")
        self.templates_dir_hint = getattr(self, "templates_dir_hint", "templates")
        self.preview_dir_hint = getattr(self, "preview_dir_hint", "SP")
        self.log_prefix = getattr(self, "log_prefix", "[碎片]")
        guard_cfg = cfg.get(self.cfg_key, {})
        self._init_multi_selection_feature(guard_cfg)

        self.enable_no_trick_decrypt = enable_no_trick_decrypt

        def _positive_float(value, default):
            try:
                val = float(value)
                if val > 0:
                    return val
            except (TypeError, ValueError):
                pass
            return default

        self.wave_var = tk.StringVar(value=str(guard_cfg.get("waves", 10)))
        self.timeout_var = tk.StringVar(value=str(guard_cfg.get("timeout", 160)))
        self.auto_loop_var = tk.BooleanVar(value=True)
        self.hotkey_var = tk.StringVar(value=guard_cfg.get("hotkey", ""))

        self.auto_e_interval_seconds = _positive_float(
            guard_cfg.get("auto_e_interval", 5.0), 5.0
        )
        self.auto_q_interval_seconds = _positive_float(
            guard_cfg.get("auto_q_interval", 5.0), 5.0
        )
        self.auto_e_enabled_var = tk.BooleanVar(
            value=bool(guard_cfg.get("auto_e_enabled", True))
        )
        self.auto_e_interval_var = tk.StringVar(
            value=f"{self.auto_e_interval_seconds:g}"
        )
        self.auto_q_enabled_var = tk.BooleanVar(
            value=bool(guard_cfg.get("auto_q_enabled", False))
        )
        self.auto_q_interval_var = tk.StringVar(
            value=f"{self.auto_q_interval_seconds:g}"
        )

        self.selected_letter_path = None
        self.macro_a_var = tk.StringVar(value="")
        self.macro_b_var = tk.StringVar(value="")
        self.hotkey_handle = None
        self._bound_hotkey_key = None
        self.hotkey_label = self.log_prefix

        if self.enable_no_trick_decrypt:
            self.no_trick_var = tk.BooleanVar(
                value=bool(guard_cfg.get("no_trick_decrypt", False))
            )
            self.no_trick_status_var = tk.StringVar(value="未启用")
            self.no_trick_progress_var = tk.DoubleVar(value=0.0)
            self.no_trick_image_ref = None
            self.no_trick_controller = None
            self.no_trick_status_frame = None
            self.no_trick_status_label = None
            self.no_trick_image_label = None
            self.no_trick_progress = None
        else:
            self.no_trick_var = None
            self.no_trick_status_var = None
            self.no_trick_progress_var = None
            self.no_trick_image_ref = None
            self.no_trick_controller = None
            self.no_trick_status_frame = None
            self.no_trick_status_label = None
            self.no_trick_image_label = None
            self.no_trick_progress = None

        self.enable_letter_paging = bool(getattr(self, "enable_letter_paging", True))
        self.letter_nav_position = getattr(self, "letter_nav_position", "top")
        self.letter_page_size = max(1, int(getattr(self, "letter_page_size", self.MAX_LETTERS)))
        self.letter_page = 0
        self.total_letter_pages = 0
        self.all_letter_files = []
        self.visible_letter_files = []
        self.letter_nav_frame = None
        self.prev_letter_btn = None
        self.next_letter_btn = None
        self.letter_page_info_var = None

        self.letter_images = []
        self.letter_buttons = []

        self.fragment_count = 0
        self.fragment_count_var = tk.StringVar(value="0")
        self.stat_name_var = tk.StringVar(value="（未选择）")
        self.stat_image = None
        self.finished_waves = 0

        self.run_start_time = None
        self.is_farming = False
        self.time_str_var = tk.StringVar(value="00:00:00")
        self.rate_str_var = tk.StringVar(value=f"0.00 {self.product_short_label}/波")
        self.eff_str_var = tk.StringVar(value=f"0.00 {self.product_short_label}/小时")
        self.wave_progress_total = 0
        self.wave_progress_count = 0
        self.wave_progress_var = tk.DoubleVar(value=0.0)
        self.wave_progress_label_var = tk.StringVar(value="轮次进度：0/0")

        self.content_frame = None
        self.left_panel = None
        self.right_panel = None

        self._build_ui()
        self._load_letters()
        self._update_wave_progress_ui()
        self._bind_hotkey()
        if self.enable_no_trick_decrypt:
            self._update_no_trick_ui()

    # ---- UI ----
    def _build_ui(self):
        tip_top = tk.Label(
            self.parent,
            text="只能刷『探险无尽血清』，请使用高练度的大范围水母角色！",
            fg="red",
            font=("Microsoft YaHei", 10, "bold"),
        )
        tip_top.pack(fill="x", padx=10, pady=3)

        self.content_frame = tk.Frame(self.parent)
        self.content_frame.pack(fill="both", expand=True)

        self.left_panel = tk.Frame(self.content_frame)
        self.left_panel.pack(side="left", fill="both", expand=True)

        self.log_panel = CollapsibleLogPanel(
            self.left_panel, f"{self.product_label}日志"
        )
        self.log_panel.pack(fill="both", padx=10, pady=(8, 5))
        self.log_text = self.log_panel.text

        ensure_goal_progress_style()
        self.wave_progress_box = tk.LabelFrame(self.left_panel, text="轮次进度")
        self.wave_progress_box.pack(fill="x", padx=10, pady=(0, 5))
        ttk.Progressbar(
            self.wave_progress_box,
            variable=self.wave_progress_var,
            maximum=100.0,
            style="Goal.Horizontal.TProgressbar",
        ).pack(fill="x", padx=10, pady=5)
        tk.Label(
            self.wave_progress_box,
            textvariable=self.wave_progress_label_var,
            anchor="e",
        ).pack(fill="x", padx=10, pady=(0, 5))

        if self.enable_no_trick_decrypt:
            self.right_panel = tk.Frame(self.content_frame)
            self.right_panel.pack(side="right", fill="y", padx=(5, 10), pady=5)
        else:
            self.right_panel = None

        top = tk.Frame(self.left_panel)
        top.pack(fill="x", padx=10, pady=5)

        tk.Label(top, text="总波数:").grid(row=0, column=0, sticky="e")
        tk.Entry(top, textvariable=self.wave_var, width=6).grid(row=0, column=1, sticky="w", padx=3)
        tk.Label(top, text="（默认 10 波）").grid(row=0, column=2, sticky="w")

        tk.Label(top, text="局内超时(秒):").grid(row=0, column=3, sticky="e")
        tk.Entry(top, textvariable=self.timeout_var, width=6).grid(row=0, column=4, sticky="w", padx=3)
        tk.Label(top, text="（防卡死判定）").grid(row=0, column=5, sticky="w")

        tk.Checkbutton(
            top,
            text="开启循环",
            variable=self.auto_loop_var,
        ).grid(row=0, column=6, sticky="w", padx=10)

        hotkey_frame = tk.Frame(self.left_panel)
        hotkey_frame.pack(fill="x", padx=10, pady=5)
        self.hotkey_frame = hotkey_frame
        self.hotkey_label_widget = tk.Label(
            hotkey_frame, text=f"刷{self.product_short_label}热键:"
        )
        self.hotkey_label_widget.pack(side="left")
        tk.Entry(hotkey_frame, textvariable=self.hotkey_var, width=20).pack(side="left", padx=5)
        ttk.Button(hotkey_frame, text="录制热键", command=self._capture_hotkey).pack(side="left", padx=3)
        ttk.Button(hotkey_frame, text="保存设置", command=self._save_settings).pack(side="left", padx=3)

        if self.enable_no_trick_decrypt:
            toggle_frame = tk.Frame(self.left_panel)
            toggle_frame.pack(fill="x", padx=10, pady=(0, 5))
            tk.Checkbutton(
                toggle_frame,
                text="开启无巧手解密",
                variable=self.no_trick_var,
                command=self._on_no_trick_toggle,
            ).pack(anchor="w")

        frame_macros = tk.LabelFrame(self.left_panel, text="地图宏脚本（mapA / mapB）")
        frame_macros.pack(fill="x", padx=10, pady=5)
        frame_macros.grid_columnconfigure(1, weight=1)
        self.frame_macros = frame_macros

        tk.Label(frame_macros, text="mapA 宏:").grid(row=0, column=0, sticky="e")
        self.macro_a_entry = tk.Entry(frame_macros, textvariable=self.macro_a_var, width=50)
        self.macro_a_entry.grid(row=0, column=1, sticky="w", padx=3)
        self.macro_a_button = ttk.Button(frame_macros, text="浏览…", command=self._choose_macro_a)
        self.macro_a_button.grid(row=0, column=2, padx=3)

        tk.Label(frame_macros, text="mapB 宏:").grid(row=1, column=0, sticky="e")
        self.macro_b_entry = tk.Entry(frame_macros, textvariable=self.macro_b_var, width=50)
        self.macro_b_entry.grid(row=1, column=1, sticky="w", padx=3)
        self.macro_b_button = ttk.Button(frame_macros, text="浏览…", command=self._choose_macro_b)
        self.macro_b_button.grid(row=1, column=2, padx=3)

        battle_frame = tk.LabelFrame(self.left_panel, text="战斗挂机设置")
        battle_frame.pack(fill="x", padx=10, pady=5)
        self.battle_frame = battle_frame

        e_row = tk.Frame(battle_frame)
        e_row.pack(fill="x", padx=5, pady=2)
        self.auto_e_check = tk.Checkbutton(
            e_row,
            text="自动释放 E 技能",
            variable=self.auto_e_enabled_var,
            command=self._update_auto_skill_states,
        )
        self.auto_e_check.pack(side="left")
        tk.Label(e_row, text="间隔(秒)：").pack(side="left", padx=(10, 2))
        self.auto_e_interval_entry = tk.Entry(
            e_row, textvariable=self.auto_e_interval_var, width=6
        )
        self.auto_e_interval_entry.pack(side="left")

        q_row = tk.Frame(battle_frame)
        q_row.pack(fill="x", padx=5, pady=2)
        self.auto_q_check = tk.Checkbutton(
            q_row,
            text="自动释放 Q 技能",
            variable=self.auto_q_enabled_var,
            command=self._update_auto_skill_states,
        )
        self.auto_q_check.pack(side="left")
        tk.Label(q_row, text="间隔(秒)：").pack(side="left", padx=(10, 2))
        self.auto_q_interval_entry = tk.Entry(
            q_row, textvariable=self.auto_q_interval_var, width=6
        )
        self.auto_q_interval_entry.pack(side="left")

        ctrl = tk.Frame(self.left_panel)
        ctrl.pack(fill="x", padx=10, pady=5)
        self.start_btn = ttk.Button(
            ctrl, text=f"开始刷{self.product_short_label}", command=lambda: self.start_farming()
        )
        self.start_btn.pack(side="left", padx=3)
        self.stop_btn = ttk.Button(ctrl, text="停止", command=lambda: self.stop_farming())
        self.stop_btn.pack(side="left", padx=3)

        if self._multi_feature_available():
            self._build_multi_toggle(self.left_panel)

        self.frame_letters = tk.LabelFrame(
            self.left_panel,
            text=f"{self.letter_label}选择（来自 {self.letters_dir_hint}/）",
        )
        self.frame_letters.pack(fill="both", expand=True, padx=10, pady=5)

        letters_container = self._build_letter_layout(self.frame_letters)

        self.letters_grid = tk.Frame(letters_container)
        self.letters_grid.pack(fill="both", expand=True, padx=5, pady=5)

        self.selected_label_var = tk.StringVar(value=f"当前未选择{self.letter_label}")
        self.selected_label_widget = tk.Label(
            self.frame_letters, textvariable=self.selected_label_var, fg="#0080ff"
        )
        self.selected_label_widget.pack(anchor="w", padx=5, pady=3)

        if self.enable_letter_paging:
            nav_parent = letters_container if letters_container is not self.frame_letters else self.frame_letters
            nav = tk.Frame(nav_parent)
            if getattr(self, "letter_nav_position", "bottom") == "top":
                nav.pack(fill="x", padx=5, pady=(0, 3), before=self.letters_grid)
            else:
                nav.pack(fill="x", padx=5, pady=(0, 3))
            self.letter_nav_frame = nav
            self.prev_letter_btn = ttk.Button(
                nav, text="上一页", width=8, command=self._prev_letter_page
            )
            self.prev_letter_btn.pack(side="left")
            self.letter_page_info_var = tk.StringVar(value="第 0/0 页（共 0 张）")
            tk.Label(nav, textvariable=self.letter_page_info_var).pack(
                side="left", expand=True, padx=5
            )
            self.next_letter_btn = ttk.Button(
                nav, text="下一页", width=8, command=self._next_letter_page
            )
            self.next_letter_btn.pack(side="right")

        if self.enable_no_trick_decrypt:
            self.no_trick_status_frame = tk.LabelFrame(self.right_panel, text="无巧手解密状态")
            status_inner = tk.Frame(self.no_trick_status_frame)
            status_inner.pack(fill="x", padx=5, pady=5)

            self.no_trick_status_label = tk.Label(
                status_inner,
                textvariable=self.no_trick_status_var,
                anchor="w",
                justify="left",
            )
            self.no_trick_status_label.pack(fill="x", anchor="w")

            self.no_trick_image_label = tk.Label(
                self.no_trick_status_frame,
                relief="sunken",
                bd=1,
                bg="#f8f8f8",
            )
            self.no_trick_image_label.pack(fill="both", expand=True, padx=10, pady=(0, 5))

            self.no_trick_progress = ttk.Progressbar(
                self.no_trick_status_frame,
                variable=self.no_trick_progress_var,
                maximum=100.0,
                mode="determinate",
            )
            self.no_trick_progress.pack(fill="x", padx=10, pady=(0, 8))

        self.stats_frame = tk.LabelFrame(
            self.left_panel, text=f"{self.product_label}统计（实时）"
        )
        self.stats_frame.pack(fill="x", padx=10, pady=5)

        self.stat_image_label = tk.Label(self.stats_frame, relief="sunken")
        self.stat_image_label.grid(row=0, column=0, rowspan=3, padx=5, pady=5)

        self.current_entity_label = tk.Label(
            self.stats_frame, text=f"当前{self.entity_label}："
        )
        self.current_entity_label.grid(row=0, column=1, sticky="e")
        tk.Label(self.stats_frame, textvariable=self.stat_name_var).grid(row=0, column=2, sticky="w")

        self.total_product_label = tk.Label(
            self.stats_frame, text=f"累计{self.product_label}："
        )
        self.total_product_label.grid(row=1, column=1, sticky="e")
        tk.Label(
            self.stats_frame,
            textvariable=self.fragment_count_var,
            font=("Microsoft YaHei", 12, "bold"),
            fg="#ff6600",
        ).grid(row=1, column=2, sticky="w")

        tk.Label(self.stats_frame, text="运行时间：").grid(row=0, column=3, sticky="e")
        tk.Label(self.stats_frame, textvariable=self.time_str_var).grid(row=0, column=4, sticky="w")

        tk.Label(self.stats_frame, text="平均掉落：").grid(row=1, column=3, sticky="e")
        tk.Label(self.stats_frame, textvariable=self.rate_str_var).grid(row=1, column=4, sticky="w")

        tk.Label(self.stats_frame, text="效率：").grid(row=2, column=3, sticky="e")
        tk.Label(self.stats_frame, textvariable=self.eff_str_var).grid(row=2, column=4, sticky="w")

        if self.enable_letter_paging:
            letter_tip = f"1. {self.letter_label}图片放入 {self.letters_dir_hint}/ 目录，数量不限，本界面支持分页浏览全部图片。\n"
        else:
            letter_tip = (
                f"1. {self.letter_label}图片放入 {self.letters_dir_hint}/ 目录，数量不限，本界面最多显示前 {self.MAX_LETTERS} 张。\n"
            )
        tip_text = (
            "提示：\n"
            + letter_tip
            + f"2. 若需要展示{self.product_label}预览，可在 {self.preview_dir_hint}/ 目录放入与{self.letter_label}同名的 1.png / 2.png 等图片。\n"
            + f"3. 按钮图（继续挑战/确认选择/撤退/mapa/mapb/G/Q/exit_step1）放在 {self.templates_dir_hint}/ 目录。\n"
        )
        self.tip_label = tk.Label(
            self.parent,
            text=tip_text,
            fg="#666666",
            anchor="w",
            justify="left",
        )
        self.tip_label.pack(fill="x", padx=10, pady=(0, 8))

        self._update_auto_skill_states()
        self._update_multi_selection_display()

    # ---- 日志 ----
    def log(self, msg: str):
        widget = getattr(self, "log_text", None)
        if widget is None:
            return

        def _append():
            append_formatted_log(widget, msg)

        post_to_main_thread(_append)

    # ---- 人物密函 ----
    def _load_letters(self):
        for b in self.letter_buttons:
            parent = b.master
            b.destroy()
            if parent not in (None, self.letters_grid):
                try:
                    parent.destroy()
                except Exception:
                    pass
        self.letter_buttons.clear()
        self.letter_images.clear()
        self.multi_order_labels = []

        files = []
        for name in os.listdir(self.letters_dir):
            low = name.lower()
            if low.endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp")):
                files.append(name)
        files.sort()
        self.all_letter_files = files

        if self.enable_letter_paging:
            total = len(files)
            if total == 0:
                self.total_letter_pages = 0
                self.letter_page = 0
                display_files = []
            else:
                page_size = self.letter_page_size
                self.total_letter_pages = math.ceil(total / page_size)
                if self.letter_page >= self.total_letter_pages:
                    self.letter_page = self.total_letter_pages - 1
                start = self.letter_page * page_size
                end = start + page_size
                display_files = files[start:end]
            self.visible_letter_files = display_files
        else:
            display_files = files[: self.MAX_LETTERS]
            self.visible_letter_files = display_files
            self.total_letter_pages = 1 if display_files else 0
            self.letter_page = 0

        if not display_files:
            if not files:
                self.selected_label_var.set(
                    f"当前未选择{self.letter_label}（{self.letters_dir_hint}/ 目录为空）"
                )
                self.selected_letter_path = None
            self._highlight_button(None)
            self._update_letter_paging_controls()
            return

        max_per_row = 5
        for col in range(max_per_row):
            self.letters_grid.grid_columnconfigure(col, weight=1, uniform="letters")
        for idx, name in enumerate(display_files):
            full_path = os.path.join(self.letters_dir, name)
            img = load_uniform_letter_image(full_path)
            if img is None:
                continue
            self.letter_images.append(img)
            r = idx // max_per_row
            c = idx % max_per_row
            cell = tk.Frame(
                self.letters_grid,
                width=LETTER_IMAGE_SIZE + 8,
                height=LETTER_IMAGE_SIZE + 8,
                bd=0,
                highlightthickness=0,
            )
            cell.grid(row=r, column=c, padx=4, pady=4, sticky="nsew")
            cell.grid_propagate(False)
            btn = tk.Button(
                cell,
                image=img,
                relief="raised",
                borderwidth=2,
                command=lambda p=full_path, b_idx=idx: self._on_letter_clicked(p, b_idx),
            )
            btn.pack(expand=True, fill="both")
            self.letter_buttons.append(btn)

            order_label = tk.Label(
                cell,
                text="",
                fg="white",
                bg="#ff6600",
                font=("Microsoft YaHei", 9, "bold"),
                width=2,
            )
            order_label.place(x=2, y=2)
            self.multi_order_labels.append(order_label)

        highlight_idx = None
        if self.selected_letter_path:
            base = os.path.basename(self.selected_letter_path)
            if base in display_files:
                highlight_idx = display_files.index(base)
            elif not os.path.exists(self.selected_letter_path):
                self.selected_letter_path = None
                self.selected_label_var.set(f"当前未选择{self.letter_label}")

        self._highlight_button(highlight_idx)
        self._update_letter_order_labels()
        self._update_letter_paging_controls()

    def _on_letter_clicked(self, path: str, idx: int):
        if self._multi_mode_active():
            self._multi_handle_letter_click(path, idx)
        else:
            self.selected_letter_path = path
            base = os.path.basename(path)
            self.selected_label_var.set(f"当前选择{self.letter_label}：{base}")
            self._highlight_button(idx)
            self.stat_name_var.set(base)
            self.stat_image = self.letter_images[idx]
            self.stat_image_label.config(image=self.stat_image)

    def _highlight_button(self, idx: Optional[int]):
        if self._multi_mode_active():
            selected = set(self.multi_letter_names)
            for btn, name in zip(self.letter_buttons, self.visible_letter_files):
                if name in selected:
                    btn.config(relief="sunken", bg="#a0cfff")
                else:
                    btn.config(relief="raised", bg="#f0f0f0")
            return

        for i, btn in enumerate(self.letter_buttons):
            if idx is not None and i == idx:
                btn.config(relief="sunken", bg="#a0cfff")
            else:
                btn.config(relief="raised", bg="#f0f0f0")

    def _update_auto_skill_states(self):
        state_e = tk.NORMAL if self.auto_e_enabled_var.get() else tk.DISABLED
        state_q = tk.NORMAL if self.auto_q_enabled_var.get() else tk.DISABLED
        try:
            self.auto_e_interval_entry.config(state=state_e)
            self.auto_q_interval_entry.config(state=state_q)
        except Exception:
            pass

    def _validate_auto_skill_settings(self) -> bool:
        try:
            e_interval = float(self.auto_e_interval_var.get().strip())
            if e_interval <= 0:
                raise ValueError
        except (ValueError, AttributeError):
            messagebox.showwarning("提示", "E 键间隔请输入大于 0 的数字秒数。")
            return False

        try:
            q_interval = float(self.auto_q_interval_var.get().strip())
            if q_interval <= 0:
                raise ValueError
        except (ValueError, AttributeError):
            messagebox.showwarning("提示", "Q 键间隔请输入大于 0 的数字秒数。")
            return False

        self.auto_e_interval_seconds = e_interval
        self.auto_q_interval_seconds = q_interval
        return True

    def _update_letter_paging_controls(self):
        if not self.enable_letter_paging or self.letter_page_info_var is None:
            return

        total = len(self.all_letter_files)
        if total == 0:
            self.total_letter_pages = 0
            self.letter_page = 0
            self.letter_page_info_var.set("暂无图片")
            if self.prev_letter_btn is not None:
                self.prev_letter_btn.config(state="disabled")
            if self.next_letter_btn is not None:
                self.next_letter_btn.config(state="disabled")
            return

        page_size = self.letter_page_size
        total_pages = max(1, math.ceil(total / page_size))
        if self.letter_page >= total_pages:
            self.letter_page = total_pages - 1
        self.total_letter_pages = total_pages
        self.letter_page_info_var.set(
            f"第 {self.letter_page + 1}/{total_pages} 页（共 {total} 张）"
        )
        if self.prev_letter_btn is not None:
            self.prev_letter_btn.config(state="normal" if self.letter_page > 0 else "disabled")
        if self.next_letter_btn is not None:
            self.next_letter_btn.config(
                state="normal" if self.letter_page < total_pages - 1 else "disabled"
            )

    def _prev_letter_page(self):
        if not self.enable_letter_paging:
            return
        if self.letter_page > 0:
            self.letter_page -= 1
            self._load_letters()

    def _next_letter_page(self):
        if not self.enable_letter_paging:
            return
        if self.total_letter_pages and self.letter_page < self.total_letter_pages - 1:
            self.letter_page += 1
            self._load_letters()

    # ---- 热键与设置 ----
    def _capture_hotkey(self):
        if keyboard is None:
            messagebox.showerror("错误", "当前环境未安装 keyboard，无法录制热键。")
            return

        def worker():
            try:
                hk = keyboard.read_hotkey(suppress=False)
            except Exception as e:
                log(f"{self.log_prefix} 录制热键失败：{e}")
                return
            post_to_main_thread(lambda: self._set_hotkey(hk))

        threading.Thread(target=worker, daemon=True).start()

    def _set_hotkey(self, hotkey: str):
        self.hotkey_var.set(hotkey or "")
        self._bind_hotkey(show_popup=False)

    def _release_hotkey(self):
        if self.hotkey_handle is None or keyboard is None:
            return
        try:
            keyboard.remove_hotkey(self.hotkey_handle)
        except Exception:
            pass
        self.hotkey_handle = None
        self._bound_hotkey_key = None

    def _bind_hotkey(self, show_popup: bool = True):
        if keyboard is None:
            return
        self._release_hotkey()
        key = self.hotkey_var.get().strip()
        if not key:
            return
        try:
            handle = keyboard.add_hotkey(
                key,
                self._on_hotkey_trigger,
            )
        except Exception as e:
            log(f"{self.log_prefix} 绑定热键失败：{e}")
            messagebox.showerror("错误", f"绑定热键失败：{e}")
            return

        self.hotkey_handle = handle
        self._bound_hotkey_key = key
        log(f"{self.log_prefix} 已绑定热键：{key}")

    def _on_hotkey_trigger(self):
        post_to_main_thread(self._handle_hotkey_if_active)

    def _handle_hotkey_if_active(self):
        active = get_active_fragment_gui()
        if active is not self and not self.is_farming:
            return
        self._toggle_by_hotkey()

    def _toggle_by_hotkey(self):
        if self.is_farming:
            log(f"{self.log_prefix} 热键触发：请求停止刷{self.product_short_label}。")
            self.stop_farming(from_hotkey=True)
        else:
            log(f"{self.log_prefix} 热键触发：开始刷{self.product_short_label}。")
            self.start_farming(from_hotkey=True)

    # ---- 无巧手解密 ----
    def _on_no_trick_toggle(self):
        if not self.enable_no_trick_decrypt:
            return
        if not self.no_trick_var.get():
            self._stop_no_trick_monitor()
        self._update_no_trick_ui()

    def _update_no_trick_ui(self):
        if not self.enable_no_trick_decrypt:
            return
        if self.no_trick_var.get():
            self._ensure_no_trick_frame_visible()
            if self.no_trick_controller is None:
                self._set_no_trick_status_direct("等待刷图时识别解密图像…")
                self._set_no_trick_progress_value(0.0)
                self._set_no_trick_image(None)
        else:
            self._hide_no_trick_frame()
            self._set_no_trick_status_direct("未启用")
            self._set_no_trick_progress_value(0.0)
            self._set_no_trick_image(None)

    def _ensure_no_trick_frame_visible(self):
        if (
            not self.enable_no_trick_decrypt
            or self.no_trick_status_frame is None
            or self.no_trick_var is None
        ):
            return
        if not self.no_trick_status_frame.winfo_ismapped():
            self.no_trick_status_frame.pack(fill="both", expand=True, padx=5, pady=5)

    def _hide_no_trick_frame(self):
        if not self.enable_no_trick_decrypt or self.no_trick_status_frame is None:
            return
        if self.no_trick_status_frame.winfo_manager():
            self.no_trick_status_frame.pack_forget()

    def _set_no_trick_status_direct(self, text: str):
        if self.no_trick_status_var is not None:
            self.no_trick_status_var.set(text)

    def _set_no_trick_progress_value(self, percent: float):
        if self.no_trick_progress_var is not None:
            self.no_trick_progress_var.set(max(0.0, min(100.0, percent)))

    def _set_no_trick_image(self, photo):
        if not self.enable_no_trick_decrypt or self.no_trick_image_label is None:
            return
        if photo is None:
            self.no_trick_image_label.config(image="")
        else:
            self.no_trick_image_label.config(image=photo)
        self.no_trick_image_ref = photo

    def _load_no_trick_preview(self, path: str, max_size: int = 240):
        if not path or not os.path.exists(path):
            return None
        if Image is not None and ImageTk is not None:
            try:
                with Image.open(path) as pil_img:
                    pil_img = pil_img.convert("RGBA")
                    w, h = pil_img.size
                    scale = 1.0
                    if max(w, h) > max_size:
                        scale = max_size / max(w, h)
                        pil_img = pil_img.resize(
                            (
                                max(1, int(w * scale)),
                                max(1, int(h * scale)),
                            ),
                            Image.LANCZOS,
                        )
                    return ImageTk.PhotoImage(pil_img)
            except Exception:
                pass
        try:
            img = tk.PhotoImage(file=path)
        except Exception:
            return None
        w = max(img.width(), 1)
        h = max(img.height(), 1)
        factor = max(1, (max(w, h) + max_size - 1) // max_size)
        if factor > 1:
            img = img.subsample(factor, factor)
        return img

    def _start_no_trick_monitor(self):
        if not self.enable_no_trick_decrypt or not self.no_trick_var.get():
            return None
        controller = NoTrickDecryptController(self, GAME_DIR)
        if controller.start():
            self.no_trick_controller = controller
            return controller
        return None

    def _stop_no_trick_monitor(self):
        if not self.enable_no_trick_decrypt:
            return
        controller = self.no_trick_controller
        if controller is not None:
            controller.stop()
            controller.finish_session()
            self.no_trick_controller = None

    def on_no_trick_unavailable(self, reason: str):
        if not self.enable_no_trick_decrypt:
            return

        def _():
            if not self.no_trick_var.get():
                return
            self._ensure_no_trick_frame_visible()
            self._set_no_trick_status_direct(f"无巧手解密不可用：{reason}。")
            self._set_no_trick_progress_value(0.0)
            self._set_no_trick_image(None)

        post_to_main_thread(_)

    def on_no_trick_no_templates(self, game_dir: str):
        if not self.enable_no_trick_decrypt:
            return

        def _():
            if not self.no_trick_var.get():
                return
            self._ensure_no_trick_frame_visible()
            self._set_no_trick_status_direct("Game 文件夹中未找到解密图像模板，请放置 1.png 等文件。")
            self._set_no_trick_progress_value(0.0)
            self._set_no_trick_image(None)

        post_to_main_thread(_)

    def on_no_trick_monitor_started(self, templates):
        if not self.enable_no_trick_decrypt:
            return
        total = len(templates)
        valid = sum(1 for t in templates if t.get("template") is not None)

        def _():
            if not self.no_trick_var.get():
                return
            self._ensure_no_trick_frame_visible()
            if valid <= 0:
                self._set_no_trick_status_direct("Game 模板加载失败，无法识别解密图像。")
            else:
                self._set_no_trick_status_direct(
                    f"等待识别解密图像（共 {total} 张模板）…"
                )
            self._set_no_trick_progress_value(0.0)
            self._set_no_trick_image(None)

        post_to_main_thread(_)

    def on_no_trick_detected(self, entry, score: float):
        if not self.enable_no_trick_decrypt:
            return

        def _():
            if not self.no_trick_var.get():
                return
            self._ensure_no_trick_frame_visible()
            name = entry.get("name", "")
            self._set_no_trick_status_direct(
                f"已经识别到解密图像 - {name}，正在解密…"
            )
            photo = self._load_no_trick_preview(entry.get("png_path"))
            self._set_no_trick_image(photo)
            self._set_no_trick_progress_value(0.0)

        post_to_main_thread(_)

    def on_no_trick_macro_start(self, entry, score: float):
        if not self.enable_no_trick_decrypt:
            return

        def _():
            if not self.no_trick_var.get():
                return
            self._set_no_trick_progress_value(0.0)

        post_to_main_thread(_)

    def on_no_trick_progress(self, progress: float):
        if not self.enable_no_trick_decrypt:
            return

        def _():
            if not self.no_trick_var.get():
                return
            self._set_no_trick_progress_value(progress * 100.0)

        post_to_main_thread(_)

    def on_no_trick_macro_complete(self, entry):
        if not self.enable_no_trick_decrypt:
            return

        def _():
            if not self.no_trick_var.get():
                return
            self._set_no_trick_status_direct("解密完成，恢复原宏执行。")
            self._set_no_trick_progress_value(100.0)

        post_to_main_thread(_)

    def on_no_trick_macro_missing(self, entry):
        if not self.enable_no_trick_decrypt:
            return

        def _():
            if not self.no_trick_var.get():
                return
            base = os.path.splitext(entry.get("name", ""))[0]
            self._set_no_trick_status_direct(
                f"未找到 {base}.json，跳过无巧手解密。"
            )
            self._set_no_trick_progress_value(0.0)
            self._set_no_trick_image(None)

        post_to_main_thread(_)

    def on_no_trick_session_finished(self, triggered: bool, macro_executed: bool, macro_missing: bool):
        if not self.enable_no_trick_decrypt:
            return

        def _():
            if not self.no_trick_var.get():
                return
            if not triggered:
                self._set_no_trick_status_direct("本次未识别到解密图像。")
                self._set_no_trick_progress_value(0.0)
                self._set_no_trick_image(None)
            elif macro_executed:
                self._set_no_trick_status_direct("解密完成，继续执行原宏。")
                self._set_no_trick_progress_value(100.0)
            elif macro_missing:
                # 状态已在 on_no_trick_macro_missing 中更新
                pass

        post_to_main_thread(_)

    def _save_settings(self):
        try:
            waves = int(self.wave_var.get().strip())
            if waves <= 0:
                raise ValueError
        except ValueError:
            messagebox.showwarning("提示", "总波数请输入大于 0 的整数。")
            return
        try:
            timeout = float(self.timeout_var.get().strip())
            if timeout <= 0:
                raise ValueError
        except ValueError:
            messagebox.showwarning("提示", "局内超时请输入大于 0 的数字秒数。")
            return
        if not self._validate_auto_skill_settings():
            return
        section = self.cfg.setdefault(self.cfg_key, {})
        section["waves"] = waves
        section["timeout"] = timeout
        section["hotkey"] = self.hotkey_var.get().strip()
        if self.enable_no_trick_decrypt and self.no_trick_var is not None:
            section["no_trick_decrypt"] = bool(self.no_trick_var.get())
        section["auto_e_enabled"] = bool(self.auto_e_enabled_var.get())
        section["auto_e_interval"] = self.auto_e_interval_seconds
        section["auto_q_enabled"] = bool(self.auto_q_enabled_var.get())
        section["auto_q_interval"] = self.auto_q_interval_seconds
        if self._multi_feature_available():
            section[self.multi_config_enabled_key] = bool(
                self.multi_enabled_var.get() if self.multi_enabled_var else False
            )
            section[self.multi_config_files_key] = list(self.multi_letter_names)
        self._bind_hotkey()
        save_config(self.cfg)
        messagebox.showinfo("提示", "设置已保存。")

    # ---- 宏选择 ----
    def _choose_macro_a(self):
        p = filedialog.askopenfilename(
            title="选择 mapA 宏 JSON",
            initialdir=SCRIPTS_DIR,
            filetypes=[("JSON 文件", "*.json"), ("所有文件", "*.*")],
        )
        if p:
            self.macro_a_var.set(p)

    def _choose_macro_b(self):
        p = filedialog.askopenfilename(
            title="选择 mapB 宏 JSON",
            initialdir=SCRIPTS_DIR,
            filetypes=[("JSON 文件", "*.json"), ("所有文件", "*.*")],
        )
        if p:
            self.macro_b_var.set(p)

    # ---- 控制 ----
    def start_farming(self, from_hotkey: bool = False):
        if self._multi_mode_active():
            if not self.multi_letter_names:
                messagebox.showwarning("提示", "多刷模式下请至少选择一个密函。")
                return
        elif not self.selected_letter_path:
            messagebox.showwarning("提示", f"请先选择一个{self.letter_label}。")
            return

        try:
            total_waves = int(self.wave_var.get().strip())
            if total_waves <= 0:
                raise ValueError
        except ValueError:
            messagebox.showwarning("提示", "总波数请输入大于 0 的整数。")
            return

        try:
            self.timeout_seconds = float(self.timeout_var.get().strip())
            if self.timeout_seconds <= 0:
                raise ValueError
        except ValueError:
            messagebox.showwarning("提示", "局内超时请输入大于 0 的数字秒数。")
            return

        if not self._validate_auto_skill_settings():
            return

        if not self.macro_a_var.get() or not self.macro_b_var.get():
            messagebox.showwarning("提示", "请设置 mapA 与 mapB 的宏 JSON。")
            return

        if pyautogui is None or cv2 is None or np is None:
            messagebox.showerror("错误", "缺少 pyautogui 或 opencv/numpy，无法刷碎片。")
            return
        if keyboard is None:
            messagebox.showerror("错误", "未安装 keyboard 模块，无法发送按键。")
            return

        if not self._prepare_multi_runtime_cycle():
            return

        if not round_running_lock.acquire(blocking=False):
            messagebox.showwarning("提示", "当前已有其它任务在运行，请先停止后再试。")
            return

        self.fragment_count = 0
        self.fragment_count_var.set("0")
        self.finished_waves = 0
        self.run_start_time = time.time()
        self.is_farming = True
        self._update_stats_ui()
        self.parent.after(1000, self._stats_timer)
        self._reset_wave_progress(total_waves)

        worker_stop.clear()
        self.start_btn.config(state="disabled")

        t = threading.Thread(target=self._farm_worker, args=(total_waves,), daemon=True)
        t.start()

        if from_hotkey:
            log(f"{self.log_prefix} 热键启动刷{self.product_short_label}成功。")

    def stop_farming(self, from_hotkey: bool = False):
        worker_stop.set()
        if not from_hotkey:
            messagebox.showinfo(
                "提示",
                f"已请求停止刷{self.product_short_label}，本波结束后将自动退出。",
            )
        else:
            log(f"{self.log_prefix} 热键停止请求已发送，等待当前波结束。")

    # ---- 统计 ----
    def _add_fragments(self, delta: int):
        if delta <= 0:
            return
        self.fragment_count += delta
        val = self.fragment_count

        def _update():
            self.fragment_count_var.set(str(val))
        post_to_main_thread(_update)

    def _update_stats_ui(self):
        if self.run_start_time is None:
            elapsed = 0
        else:
            elapsed = time.time() - self.run_start_time
        self.time_str_var.set(format_hms(elapsed))
        if self.finished_waves > 0:
            rate = self.fragment_count / self.finished_waves
        else:
            rate = 0.0
        self.rate_str_var.set(f"{rate:.2f} {self.product_short_label}/波")
        if elapsed > 0:
            eff = self.fragment_count / (elapsed / 3600.0)
        else:
            eff = 0.0
        self.eff_str_var.set(f"{eff:.2f} {self.product_short_label}/小时")

    def _stats_timer(self):
        if not self.is_farming:
            return
        self._update_stats_ui()
        self.parent.after(1000, self._stats_timer)

    def _reset_wave_progress(self, total_waves: int):
        self.wave_progress_total = max(0, total_waves)
        self.wave_progress_count = 0
        self._update_wave_progress_ui()

    def _increment_wave_progress(self):
        if self.wave_progress_total <= 0:
            return
        if self.wave_progress_count < self.wave_progress_total:
            self.wave_progress_count += 1
            self._update_wave_progress_ui()

    def _force_wave_progress_complete(self):
        if self.wave_progress_total <= 0:
            return
        if self.wave_progress_count != self.wave_progress_total:
            self.wave_progress_count = self.wave_progress_total
            self._update_wave_progress_ui()

    def _update_wave_progress_ui(self):
        total = max(1, self.wave_progress_total)
        if self.wave_progress_total <= 0:
            percent = 0.0
            label = "轮次进度：0/0"
        else:
            percent = (self.wave_progress_count / total) * 100.0
            remaining = max(0, self.wave_progress_total - self.wave_progress_count)
            label = f"轮次进度：{self.wave_progress_count}/{self.wave_progress_total}（剩余 {remaining}）"
        self.wave_progress_var.set(percent)
        self.wave_progress_label_var.set(label)

    # ---- 核心刷本流程 ----
    def _farm_worker(self, total_waves: int):
        try:
            log(f"===== {self.product_label}刷取 开始 =====")
            if not init_game_region():
                messagebox.showerror(
                    "错误",
                    f"未找到{get_window_name_hint()}窗口，无法开始刷{self.product_short_label}。",
                )
                return

            first_session = True
            session_index = 0

            while not worker_stop.is_set():
                auto_loop = self.auto_loop_var.get()
                session_index += 1
                self._reset_wave_progress(total_waves)
                log(f"{self.log_prefix} === 开始第 {session_index} 趟无尽 ===")

                if first_session:
                    if not self._enter_first_wave_and_setup():
                        return
                    first_session = False
                else:
                    if not self._restart_from_lobby_after_retreat():
                        log(f"{self.log_prefix} 循环重开失败，结束刷取。")
                        break

                current_wave = 1
                need_next_session = False

                while current_wave <= total_waves and not worker_stop.is_set():
                    log(f"{self.log_prefix} 开始第 {current_wave} 波战斗挂机…")
                    result = self._battle_and_loot(max_wait=self.timeout_seconds)
                    if worker_stop.is_set():
                        break

                    if result == "timeout":
                        log(
                            f"{self.log_prefix} 第 {current_wave} 波判定卡死，执行防卡死逻辑…"
                        )
                        if not self._anti_stuck_and_reset():
                            log(f"{self.log_prefix} 防卡死失败，结束刷取。")
                            need_next_session = False
                            break
                        # 防卡死后会重新地图识别+宏，继续当前波
                        continue

                    elif result == "ok":
                        self.finished_waves += 1
                        log(f"{self.log_prefix} 第 {current_wave} 波战斗完成。")

                        if self._check_auto_switch():
                            log(f"{self.log_prefix} 已执行模式切换，准备重新开局。")
                            need_next_session = True
                            first_session = True
                            break

                        if current_wave == total_waves:
                            if auto_loop:
                                self._force_wave_progress_complete()
                                log(
                                    f"{self.log_prefix} 波数已满，已开启循环，撤退并准备下一趟。"
                                )
                                self._retreat_only()
                                need_next_session = True
                                break
                            else:
                                self._force_wave_progress_complete()
                                log(
                                    f"{self.log_prefix} 波数已满，未开启循环，撤退并结束。"
                                )
                                self._retreat_only()
                                need_next_session = False
                                worker_stop.set()
                                break
                        else:
                            if not self._enter_next_wave_without_map():
                                log(f"{self.log_prefix} 进入下一波失败，结束刷取。")
                                need_next_session = False
                                worker_stop.set()
                                break
                            current_wave += 1
                            continue

                    else:
                        need_next_session = False
                        break

                if worker_stop.is_set():
                    break
                if not auto_loop or not need_next_session:
                    break

            log(f"===== {self.product_label}刷取 结束 =====")

        except Exception as e:
            log(f"{self.log_prefix} 后台线程异常：{e}")
            traceback.print_exc()
        finally:
            worker_stop.clear()
            round_running_lock.release()
            if self.enable_no_trick_decrypt:
                self._stop_no_trick_monitor()
            self.is_farming = False
            self._update_stats_ui()

            def restore():
                try:
                    self.start_btn.config(state="normal")
                except Exception:
                    pass
            post_to_main_thread(restore)

            if self.run_start_time is not None:
                elapsed = time.time() - self.run_start_time
                time_str = format_hms(elapsed)
                if self.finished_waves > 0:
                    rate = self.fragment_count / self.finished_waves
                else:
                    rate = 0.0
                if elapsed > 0:
                    eff = self.fragment_count / (elapsed / 3600.0)
                else:
                    eff = 0.0
                msg = (
                    f"{self.product_label}刷取已结束。\n\n"
                    f"总运行时间：{time_str}\n"
                    f"完成波数：{self.finished_waves}\n"
                    f"累计{self.product_label}：{self.fragment_count}\n"
                    f"平均掉落：{rate:.2f} {self.product_short_label}/波\n"
                    f"效率：{eff:.2f} {self.product_short_label}/小时\n"
                )
                post_to_main_thread(
                    lambda: messagebox.showinfo(
                        f"刷{self.product_short_label}完成", msg
                    )
                )

    # ---- 首次进图 / 循环重开 ----
    def _enter_first_wave_and_setup(self) -> bool:
        log(
            f"{self.log_prefix} 首次进图：选择密函按钮 → {self.letter_label} → 确认选择 → 地图AB识别 + 宏"
        )
        if not self._select_letter_sequence(f"{self.log_prefix} 首次", need_open_button=True):
            return False
        self._increment_wave_progress()
        return self._map_detect_and_run_macros()

    def _restart_from_lobby_after_retreat(self) -> bool:
        log(
            f"{self.log_prefix} 循环重开：再次进行 → {self.letter_label} → 确认选择 → 地图AB + 宏"
        )
        if not wait_and_click_template(
            BTN_EXPEL_NEXT_WAVE,
            f"{self.log_prefix} 循环重开：再次进行按钮",
            20.0,
            0.8,
        ):
            log(f"{self.log_prefix} 循环重开：未能点击 再次进行.png。")
            return False
        if not self._select_letter_sequence(
            f"{self.log_prefix} 循环重开", need_open_button=False
        ):
            return False
        self._increment_wave_progress()
        return self._map_detect_and_run_macros()

    def _execute_map_macro(self, macro_path: str, label: str):
        controller = self._start_no_trick_monitor()
        try:
            play_macro(
                macro_path,
                f"{self.log_prefix} {label}",
                0.0,
                0.3,
                interrupt_on_exit=False,
                interrupter=controller,
            )
        finally:
            if controller is not None:
                controller.stop()
                controller.finish_session()
                if self.no_trick_controller is controller:
                    self.no_trick_controller = None

    def _map_detect_and_run_macros(self) -> bool:
        """
        确认密函后，持续匹配 mapa / mapb：
        - 最多 12 秒
        - 任意一张匹配度 >= 0.7 就认定地图
        - 然后再等待 2 秒，最后执行对应宏
        """
        log(f"{self.log_prefix} 开始持续识别地图 A/B（最长 12 秒）…")

        deadline = time.time() + 12.0
        chosen = None
        score_a = 0.0
        score_b = 0.0

        while time.time() < deadline and not worker_stop.is_set():
            score_a, _, _ = match_template("mapa.png")
            score_b, _, _ = match_template("mapb.png")
            log(
                f"{self.log_prefix} mapa 匹配度 {score_a:.3f}，mapb 匹配度 {score_b:.3f}"
            )

            best = max(score_a, score_b)
            if best >= 0.7:
                chosen = "A" if score_a >= score_b else "B"
                break

            time.sleep(0.4)

        if chosen is None:
            log(f"{self.log_prefix} 12 秒内地图匹配度始终低于 0.7，本趟放弃。")
            return False

        if chosen == "A":
            macro_path = self.macro_a_var.get()
            label = "mapA 宏"
        else:
            macro_path = self.macro_b_var.get()
            label = "mapB 宏"

        if not macro_path or not os.path.exists(macro_path):
            log(f"{self.log_prefix} {label} 文件不存在：{macro_path}")
            return False

        log(
            f"{self.log_prefix} 识别为 {label}（mapa={score_a:.3f}, mapb={score_b:.3f}），"
            "再等待 2 秒后执行宏…"
        )

        t0 = time.time()
        while time.time() - t0 < 2.0 and not worker_stop.is_set():
            time.sleep(0.1)

        self._execute_map_macro(macro_path, label)
        return True

    # ---- 掉落界面检测 & 掉落识别 ----
    def _is_drop_ui_visible(self, log_detail: bool = False, threshold: float = 0.7) -> bool:
        """
        判断当前是否已经进入『物品掉落选择界面』：
        用确认按钮『确认选择.png』做判定，匹配度 >= threshold 才算界面出现。
        """
        score, _, _ = match_template(BTN_CONFIRM_LETTER)
        if log_detail:
            log(f"{self.log_prefix} 掉落界面检查：确认选择 匹配度 {score:.3f}")
        return score >= threshold

    def _detect_and_pick_drop(self, threshold=0.8) -> bool:
        """
        已经确认『物品掉落界面』出现之后调用：

        现在不再识别具体掉落物，直接点击『确认选择』进入下一步。
        """
        if click_template(
            BTN_CONFIRM_LETTER,
            f"{self.log_prefix} 掉落确认：确认选择",
            threshold=0.7,
        ):
            time.sleep(1.0)
            return True
        return False

    def _auto_revive_if_needed(self) -> bool:
        template_path = os.path.join(TEMPLATE_DIR, AUTO_REVIVE_TEMPLATE)
        if not os.path.exists(template_path):
            return False
        score, _, _ = match_template(AUTO_REVIVE_TEMPLATE)
        if score >= AUTO_REVIVE_THRESHOLD:
            log(
                f"{self.log_prefix} 检测到角色死亡（{AUTO_REVIVE_TEMPLATE} 匹配度 {score:.3f}），执行长按 X 复苏。"
            )
            if not self._press_and_hold_key("x", AUTO_REVIVE_HOLD_SECONDS):
                log(f"{self.log_prefix} 长按 X 失败，无法执行自动复苏。")
                return False
            log(f"{self.log_prefix} 自动复苏完成，继续战斗挂机。")
            return True
        return False

    def _press_and_hold_key(self, key: str, duration: float) -> bool:
        if keyboard is None and pyautogui is None:
            return False
        pressed = False
        try:
            if keyboard is not None:
                keyboard.press(key)
            else:
                pyautogui.keyDown(key)
            pressed = True
            time.sleep(duration)
            return True
        except Exception as e:
            log(f"{self.log_prefix} 长按 {key} 失败：{e}")
            return False
        finally:
            if pressed:
                try:
                    if keyboard is not None:
                        keyboard.release(key)
                    else:
                        pyautogui.keyUp(key)
                except Exception:
                    pass

    def _battle_and_loot(self, max_wait: float = 160.0) -> str:
        """
        战斗挂机 + 掉落判断，严格遵守 max_wait（例如 160 秒）：

        - 宏执行完之后调用本函数
        - 每 5 秒按一次 E
        - 在 [0, max_wait] 内循环：
            1) 先判断『物品掉落界面』是否出现（确认选择.png 匹配度 >= 0.7）
            2) 只有界面出现以后，才去识别掉落物并选择
        - 如果在 max_wait 秒内成功选到了掉落物 → 返回 'ok'
        - 如果超过 max_wait 仍然没检测到掉落界面/没选到 → 返回 'timeout'
        """
        if keyboard is None and pyautogui is None:
            log(f"{self.log_prefix} 无法发送按键。")
            return "stopped"

        auto_e_enabled = bool(self.auto_e_enabled_var.get())
        auto_q_enabled = bool(self.auto_q_enabled_var.get())
        e_interval = getattr(self, "auto_e_interval_seconds", 5.0)
        q_interval = getattr(self, "auto_q_interval_seconds", 5.0)

        desc_parts = []
        if auto_e_enabled:
            desc_parts.append(f"E 每 {e_interval:g} 秒")
        if auto_q_enabled:
            desc_parts.append(f"Q 每 {q_interval:g} 秒")
        if not desc_parts:
            desc = "不自动释放技能"
        else:
            desc = "，".join(desc_parts)

        log(f"{self.log_prefix} 开始战斗挂机（{desc}，超时 {max_wait:.1f} 秒）。")
        start = time.time()
        last_e = start
        last_q = start
        last_revive_check = start

        min_drop_check_time = 10.0
        drop_ui_visible = False
        last_ui_log = 0.0

        while not worker_stop.is_set():
            now = time.time()

            if now - last_revive_check >= AUTO_REVIVE_CHECK_INTERVAL:
                last_revive_check = now
                self._auto_revive_if_needed()

            if auto_e_enabled and now - last_e >= e_interval:
                try:
                    if keyboard is not None:
                        keyboard.press_and_release("e")
                    else:
                        pyautogui.press("e")
                except Exception as e:
                    log(f"{self.log_prefix} 发送 E 失败：{e}")
                last_e = now

            if auto_q_enabled and now - last_q >= q_interval:
                try:
                    if keyboard is not None:
                        keyboard.press_and_release("q")
                    else:
                        pyautogui.press("q")
                except Exception as e:
                    log(f"{self.log_prefix} 发送 Q 失败：{e}")
                last_q = now

            if now - start >= min_drop_check_time:
                if not drop_ui_visible:
                    if self._is_drop_ui_visible():
                        drop_ui_visible = True
                        log(f"{self.log_prefix} 检测到物品掉落界面，开始识别掉落物。")
                    else:
                        if now - last_ui_log > 3.0:
                            self._is_drop_ui_visible(log_detail=True)
                            last_ui_log = now
                else:
                    if self._detect_and_pick_drop():
                        log(f"{self.log_prefix} 本波掉落已选择。")
                        return "ok"

            if now - start > max_wait:
                log(f"{self.log_prefix} 超过 {max_wait:.1f} 秒未检测到掉落，判定卡死。")
                return "timeout"

            time.sleep(0.5)

        return "stopped"

    # ---- 正常进入下一波（不做地图识别） ----
    def _enter_next_wave_without_map(self) -> bool:
        log(
            f"{self.log_prefix} 进入下一波：再次进行 → {self.letter_label} → 确认选择"
        )
        if not wait_and_click_template(
            BTN_CONTINUE_CHALLENGE,
            f"{self.log_prefix} 下一波：继续挑战按钮",
            20.0,
            0.8,
        ):
            log(f"{self.log_prefix} 下一波：未能点击 继续挑战.png。")
            return False
        self._increment_wave_progress()
        if not self._select_letter_sequence(
            f"{self.log_prefix} 下一波", need_open_button=False
        ):
            return False
        time.sleep(2.0)
        return True

    # ---- 防卡死 ----
    def _anti_stuck_and_reset(self) -> bool:
        """
        防卡死：Esc → G → Q → 再次进行 → 人物密函 → 确认 → 地图识别
        """
        try:
            if keyboard is not None:
                keyboard.press_and_release("esc")
            else:
                pyautogui.press("esc")
        except Exception as e:
            log(f"{self.log_prefix} 发送 ESC 失败：{e}")
        time.sleep(1.0)
        click_template("G.png", f"{self.log_prefix} 防卡死：点击 G.png", 0.6)
        time.sleep(1.0)
        click_template("Q.png", f"{self.log_prefix} 防卡死：点击 Q.png", 0.6)
        time.sleep(1.0)

        if not wait_and_click_template(
            BTN_EXPEL_NEXT_WAVE,
            f"{self.log_prefix} 防卡死：再次进行按钮",
            20.0,
            0.8,
        ):
            log(f"{self.log_prefix} 防卡死：未能点击 再次进行.png。")
            return False
        if not self._select_letter_sequence(
            f"{self.log_prefix} 防卡死", need_open_button=False
        ):
            return False

        return self._map_detect_and_run_macros()

    # ---- 撤退 ----
    def _retreat_only(self):
        wait_and_click_template(
            BTN_RETREAT_START,
            f"{self.log_prefix} 撤退按钮",
            20.0,
            0.8,
        )


class ExpelFragmentGUI(MultiLetterSelectionMixin):
    MAX_LETTERS = 20
    multi_toggle_text = "同时刷取多个人物碎片"
    multi_list_title = "驱离刷取顺序"
    multi_item_prefix = "刷取"

    def __init__(self, parent, cfg):
        self.parent = parent
        self.cfg = cfg
        self.cfg_key = getattr(self, "cfg_key", "expel_settings")
        self.letter_label = getattr(self, "letter_label", "人物密函")
        self.product_label = getattr(self, "product_label", "人物碎片")
        self.product_short_label = getattr(self, "product_short_label", "碎片")
        self.entity_label = getattr(self, "entity_label", "人物")
        self.letters_dir = getattr(self, "letters_dir", TEMPLATE_LETTERS_DIR)
        self.letters_dir_hint = getattr(self, "letters_dir_hint", "templates_letters")
        self.templates_dir_hint = getattr(self, "templates_dir_hint", "templates")
        self.preview_dir_hint = getattr(self, "preview_dir_hint", "SP")
        self.log_prefix = getattr(self, "log_prefix", "[驱离]")
        expel_cfg = cfg.get(self.cfg_key, {})
        self._init_multi_selection_feature(expel_cfg)

        def _positive_float(value, default):
            try:
                val = float(value)
                if val > 0:
                    return val
            except (TypeError, ValueError):
                pass
            return default

        self.wave_var = tk.StringVar(value=str(expel_cfg.get("waves", 10)))
        self.timeout_var = tk.StringVar(value=str(expel_cfg.get("timeout", 160)))
        self.auto_loop_var = tk.BooleanVar(value=True)
        self.hotkey_var = tk.StringVar(value=expel_cfg.get("hotkey", ""))

        self.auto_e_interval_seconds = _positive_float(
            expel_cfg.get("auto_e_interval", 5.0), 5.0
        )
        self.auto_q_interval_seconds = _positive_float(
            expel_cfg.get("auto_q_interval", 5.0), 5.0
        )
        self.auto_e_enabled_var = tk.BooleanVar(
            value=bool(expel_cfg.get("auto_e_enabled", True))
        )
        self.auto_e_interval_var = tk.StringVar(
            value=f"{self.auto_e_interval_seconds:g}"
        )
        self.auto_q_enabled_var = tk.BooleanVar(
            value=bool(expel_cfg.get("auto_q_enabled", False))
        )
        self.auto_q_interval_var = tk.StringVar(
            value=f"{self.auto_q_interval_seconds:g}"
        )

        self.selected_letter_path = None

        self.letter_images = []
        self.letter_buttons = []

        self.fragment_count = 0
        self.fragment_count_var = tk.StringVar(value="0")
        self.stat_name_var = tk.StringVar(value="（未选择）")
        self.stat_image = None
        self.finished_waves = 0

        self.run_start_time = None
        self.is_farming = False
        self.time_str_var = tk.StringVar(value="00:00:00")
        self.rate_str_var = tk.StringVar(value=f"0.00 {self.product_short_label}/波")
        self.eff_str_var = tk.StringVar(value=f"0.00 {self.product_short_label}/小时")
        self.hotkey_handle = None
        self._bound_hotkey_key = None
        self.hotkey_label = self.log_prefix

        self.enable_letter_paging = bool(getattr(self, "enable_letter_paging", True))
        self.letter_nav_position = getattr(self, "letter_nav_position", "top")
        self.letter_page_size = max(1, int(getattr(self, "letter_page_size", self.MAX_LETTERS)))
        self.letter_page = 0
        self.total_letter_pages = 0
        self.all_letter_files = []
        self.visible_letter_files = []
        self.letter_nav_frame = None
        self.prev_letter_btn = None
        self.next_letter_btn = None
        self.letter_page_info_var = None

        self.auto_e_interval_entry = None
        self.auto_q_interval_entry = None

        self._build_ui()
        self._load_letters()
        self._bind_hotkey()
        self._update_auto_skill_states()
        self._update_multi_selection_display()

    def _build_ui(self):
        tip_top = tk.Label(
            self.parent,
            text=(
                f"驱离模式：选择{self.letter_label}后自动等待 7 秒进入地图 → W 键前进 10 秒 → 随机 WASD + 每 5 秒按一次 E。"
            ),
            fg="red",
            font=("Microsoft YaHei", 10, "bold"),
        )
        tip_top.pack(fill="x", padx=10, pady=3)

        self.log_panel = CollapsibleLogPanel(
            self.parent, f"{self.product_label}日志"
        )
        self.log_panel.pack(fill="both", padx=10, pady=(5, 5))
        self.log_text = self.log_panel.text

        top = tk.Frame(self.parent)
        top.pack(fill="x", padx=10, pady=5)

        tk.Label(top, text="总波数:").grid(row=0, column=0, sticky="e")
        tk.Entry(top, textvariable=self.wave_var, width=6).grid(row=0, column=1, sticky="w", padx=3)
        tk.Label(top, text="（默认 10 波）").grid(row=0, column=2, sticky="w")

        tk.Label(top, text="局内超时(秒):").grid(row=0, column=3, sticky="e")
        tk.Entry(top, textvariable=self.timeout_var, width=6).grid(row=0, column=4, sticky="w", padx=3)
        tk.Label(top, text="（防卡死判定）").grid(row=0, column=5, sticky="w")

        tk.Checkbutton(
            top,
            text="开启循环",
            variable=self.auto_loop_var,
        ).grid(row=0, column=6, sticky="w", padx=10)

        hotkey_frame = tk.Frame(self.parent)
        hotkey_frame.pack(fill="x", padx=10, pady=5)
        self.hotkey_label_widget = tk.Label(
            hotkey_frame, text=f"刷{self.product_short_label}热键:"
        )
        self.hotkey_label_widget.pack(side="left")
        tk.Entry(hotkey_frame, textvariable=self.hotkey_var, width=20).pack(side="left", padx=5)
        ttk.Button(hotkey_frame, text="录制热键", command=self._capture_hotkey).pack(side="left", padx=3)
        ttk.Button(hotkey_frame, text="保存设置", command=self._save_settings).pack(side="left", padx=3)

        battle_frame = tk.LabelFrame(self.parent, text="战斗挂机设置")
        battle_frame.pack(fill="x", padx=10, pady=5)

        e_row = tk.Frame(battle_frame)
        e_row.pack(fill="x", padx=5, pady=2)
        self.auto_e_check = tk.Checkbutton(
            e_row,
            text="自动释放 E 技能",
            variable=self.auto_e_enabled_var,
            command=self._update_auto_skill_states,
        )
        self.auto_e_check.pack(side="left")
        tk.Label(e_row, text="间隔(秒)：").pack(side="left", padx=(10, 2))
        self.auto_e_interval_entry = tk.Entry(
            e_row, textvariable=self.auto_e_interval_var, width=6
        )
        self.auto_e_interval_entry.pack(side="left")

        q_row = tk.Frame(battle_frame)
        q_row.pack(fill="x", padx=5, pady=2)
        self.auto_q_check = tk.Checkbutton(
            q_row,
            text="自动释放 Q 技能",
            variable=self.auto_q_enabled_var,
            command=self._update_auto_skill_states,
        )
        self.auto_q_check.pack(side="left")
        tk.Label(q_row, text="间隔(秒)：").pack(side="left", padx=(10, 2))
        self.auto_q_interval_entry = tk.Entry(
            q_row, textvariable=self.auto_q_interval_var, width=6
        )
        self.auto_q_interval_entry.pack(side="left")

        ctrl = tk.Frame(self.parent)
        ctrl.pack(fill="x", padx=10, pady=5)
        self.start_btn = ttk.Button(
            ctrl, text=f"开始刷{self.product_short_label}", command=lambda: self.start_farming()
        )
        self.start_btn.pack(side="left", padx=3)
        self.stop_btn = ttk.Button(ctrl, text="停止", command=lambda: self.stop_farming())
        self.stop_btn.pack(side="left", padx=3)

        if self._multi_feature_available():
            self._build_multi_toggle(self.parent)

        self.frame_letters = tk.LabelFrame(
            self.parent,
            text=f"{self.letter_label}选择（来自 {self.letters_dir_hint}/）",
        )
        self.frame_letters.pack(fill="both", expand=True, padx=10, pady=5)

        letters_container = self._build_letter_layout(self.frame_letters)

        self.letters_grid = tk.Frame(letters_container)
        self.letters_grid.pack(fill="both", expand=True, padx=5, pady=5)

        self.selected_label_var = tk.StringVar(value=f"当前未选择{self.letter_label}")
        self.selected_label_widget = tk.Label(
            self.frame_letters, textvariable=self.selected_label_var, fg="#0080ff"
        )
        self.selected_label_widget.pack(anchor="w", padx=5, pady=3)

        if self.enable_letter_paging:
            nav_parent = letters_container if letters_container is not self.frame_letters else self.frame_letters
            nav = tk.Frame(nav_parent)
            if getattr(self, "letter_nav_position", "bottom") == "top":
                nav.pack(fill="x", padx=5, pady=(0, 3), before=self.letters_grid)
            else:
                nav.pack(fill="x", padx=5, pady=(0, 3))
            self.letter_nav_frame = nav
            self.prev_letter_btn = ttk.Button(
                nav, text="上一页", width=8, command=self._prev_letter_page
            )
            self.prev_letter_btn.pack(side="left")
            self.letter_page_info_var = tk.StringVar(value="第 0/0 页（共 0 张）")
            tk.Label(nav, textvariable=self.letter_page_info_var).pack(
                side="left", expand=True, padx=5
            )
            self.next_letter_btn = ttk.Button(
                nav, text="下一页", width=8, command=self._next_letter_page
            )
            self.next_letter_btn.pack(side="right")

        self.stats_frame = tk.LabelFrame(
            self.parent, text=f"{self.product_label}统计（实时）"
        )
        self.stats_frame.pack(fill="x", padx=10, pady=5)

        self.stat_image_label = tk.Label(self.stats_frame, relief="sunken")
        self.stat_image_label.grid(row=0, column=0, rowspan=3, padx=5, pady=5)

        self.current_entity_label = tk.Label(
            self.stats_frame, text=f"当前{self.entity_label}："
        )
        self.current_entity_label.grid(row=0, column=1, sticky="e")
        tk.Label(self.stats_frame, textvariable=self.stat_name_var).grid(row=0, column=2, sticky="w")

        self.total_product_label = tk.Label(
            self.stats_frame, text=f"累计{self.product_label}："
        )
        self.total_product_label.grid(row=1, column=1, sticky="e")
        tk.Label(
            self.stats_frame,
            textvariable=self.fragment_count_var,
            font=("Microsoft YaHei", 12, "bold"),
            fg="#ff6600",
        ).grid(row=1, column=2, sticky="w")

        tk.Label(self.stats_frame, text="运行时间：").grid(row=0, column=3, sticky="e")
        tk.Label(self.stats_frame, textvariable=self.time_str_var).grid(row=0, column=4, sticky="w")

        tk.Label(self.stats_frame, text="平均掉落：").grid(row=1, column=3, sticky="e")
        tk.Label(self.stats_frame, textvariable=self.rate_str_var).grid(row=1, column=4, sticky="w")

        tk.Label(self.stats_frame, text="效率：").grid(row=2, column=3, sticky="e")
        tk.Label(self.stats_frame, textvariable=self.eff_str_var).grid(row=2, column=4, sticky="w")

        if self.enable_letter_paging:
            letter_tip = (
                f"2. {self.letter_label}图片放入 {self.letters_dir_hint}/ 目录，界面支持分页浏览全部图片。\n"
            )
        else:
            letter_tip = (
                f"2. {self.letter_label}图片放入 {self.letters_dir_hint}/ 目录，最多显示前 {self.MAX_LETTERS} 张。\n"
            )

        tip_text = (
            "提示：\n"
            f"1. 本模式无需 mapA / mapB 宏，确认{self.letter_label}后默认 7 秒进入地图。\n"
            + letter_tip
            + "3. 若卡死会自动执行 Esc→G→Q→exit_step1 的防卡死流程，并重新开始当前波。\n"
        )
        self.tip_label = tk.Label(
            self.parent,
            text=tip_text,
            fg="#666666",
            anchor="w",
            justify="left",
        )
        self.tip_label.pack(fill="x", padx=10, pady=(0, 8))

    def _update_auto_skill_states(self):
        state_e = tk.NORMAL if self.auto_e_enabled_var.get() else tk.DISABLED
        state_q = tk.NORMAL if self.auto_q_enabled_var.get() else tk.DISABLED
        if self.auto_e_interval_entry is not None:
            self.auto_e_interval_entry.config(state=state_e)
        if self.auto_q_interval_entry is not None:
            self.auto_q_interval_entry.config(state=state_q)

    def _validate_auto_skill_settings(self) -> bool:
        try:
            e_interval = float(self.auto_e_interval_var.get().strip())
            if e_interval <= 0:
                raise ValueError
        except (ValueError, AttributeError):
            messagebox.showwarning("提示", "E 键间隔请输入大于 0 的数字秒数。")
            return False

        try:
            q_interval = float(self.auto_q_interval_var.get().strip())
            if q_interval <= 0:
                raise ValueError
        except (ValueError, AttributeError):
            messagebox.showwarning("提示", "Q 键间隔请输入大于 0 的数字秒数。")
            return False

        self.auto_e_interval_seconds = e_interval
        self.auto_q_interval_seconds = q_interval
        return True

    def log(self, msg: str):
        widget = getattr(self, "log_text", None)
        if widget is None:
            return

        def _append():
            append_formatted_log(widget, msg)

        post_to_main_thread(_append)

    def _load_letters(self):
        for b in self.letter_buttons:
            parent = b.master
            b.destroy()
            if parent not in (None, self.letters_grid):
                try:
                    parent.destroy()
                except Exception:
                    pass
        self.letter_buttons.clear()
        self.letter_images.clear()
        self.multi_order_labels = []

        files = []
        for name in os.listdir(self.letters_dir):
            low = name.lower()
            if low.endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp")):
                files.append(name)
        files.sort()
        self.all_letter_files = files

        if self.enable_letter_paging:
            total = len(files)
            if total == 0:
                self.total_letter_pages = 0
                self.letter_page = 0
                display_files = []
            else:
                page_size = self.letter_page_size
                self.total_letter_pages = math.ceil(total / page_size)
                if self.letter_page >= self.total_letter_pages:
                    self.letter_page = self.total_letter_pages - 1
                start = self.letter_page * page_size
                end = start + page_size
                display_files = files[start:end]
            self.visible_letter_files = display_files
        else:
            display_files = files[: self.MAX_LETTERS]
            self.visible_letter_files = display_files
            self.total_letter_pages = 1 if display_files else 0
            self.letter_page = 0

        if not display_files:
            if not files:
                self.selected_label_var.set(
                    f"当前未选择{self.letter_label}（{self.letters_dir_hint}/ 目录为空）"
                )
                self.selected_letter_path = None
            self._highlight_button(None)
            self._update_letter_paging_controls()
            return

        max_per_row = 5
        for col in range(max_per_row):
            self.letters_grid.grid_columnconfigure(col, weight=1, uniform="expel_letters")
        for idx, name in enumerate(display_files):
            full_path = os.path.join(self.letters_dir, name)
            img = load_uniform_letter_image(full_path)
            if img is None:
                continue
            self.letter_images.append(img)
            r = idx // max_per_row
            c = idx % max_per_row
            cell = tk.Frame(
                self.letters_grid,
                width=LETTER_IMAGE_SIZE + 8,
                height=LETTER_IMAGE_SIZE + 8,
                bd=0,
                highlightthickness=0,
            )
            cell.grid(row=r, column=c, padx=4, pady=4, sticky="nsew")
            cell.grid_propagate(False)
            btn = tk.Button(
                cell,
                image=img,
                relief="raised",
                borderwidth=2,
                command=lambda p=full_path, b_idx=idx: self._on_letter_clicked(p, b_idx),
            )
            btn.pack(expand=True, fill="both")
            self.letter_buttons.append(btn)

            order_label = tk.Label(
                cell,
                text="",
                fg="white",
                bg="#ff6600",
                font=("Microsoft YaHei", 9, "bold"),
                width=2,
            )
            order_label.place(x=2, y=2)
            self.multi_order_labels.append(order_label)

        highlight_idx = None
        if self.selected_letter_path:
            base = os.path.basename(self.selected_letter_path)
            if base in display_files:
                highlight_idx = display_files.index(base)
            elif not os.path.exists(self.selected_letter_path):
                self.selected_letter_path = None
                self.selected_label_var.set(f"当前未选择{self.letter_label}")

        self._highlight_button(highlight_idx)
        self._update_letter_order_labels()
        self._update_letter_paging_controls()

    def _on_letter_clicked(self, path: str, idx: int):
        if self._multi_mode_active():
            self._multi_handle_letter_click(path, idx)
        else:
            self.selected_letter_path = path
            base = os.path.basename(path)
            self.selected_label_var.set(f"当前选择{self.letter_label}：{base}")
            self._highlight_button(idx)
            self.stat_name_var.set(base)
            self.stat_image = self.letter_images[idx]
            self.stat_image_label.config(image=self.stat_image)

    def _highlight_button(self, idx: Optional[int]):
        if self._multi_mode_active():
            selected = set(self.multi_letter_names)
            for btn, name in zip(self.letter_buttons, self.visible_letter_files):
                if name in selected:
                    btn.config(relief="sunken", bg="#a0cfff")
                else:
                    btn.config(relief="raised", bg="#f0f0f0")
            return

        for i, btn in enumerate(self.letter_buttons):
            if idx is not None and i == idx:
                btn.config(relief="sunken", bg="#a0cfff")
            else:
                btn.config(relief="raised", bg="#f0f0f0")

    def _update_letter_paging_controls(self):
        if not self.enable_letter_paging or self.letter_page_info_var is None:
            return

        total = len(self.all_letter_files)
        if total == 0:
            self.total_letter_pages = 0
            self.letter_page = 0
            self.letter_page_info_var.set("暂无图片")
            if self.prev_letter_btn is not None:
                self.prev_letter_btn.config(state="disabled")
            if self.next_letter_btn is not None:
                self.next_letter_btn.config(state="disabled")
            return

        page_size = self.letter_page_size
        total_pages = max(1, math.ceil(total / page_size))
        if self.letter_page >= total_pages:
            self.letter_page = total_pages - 1
        self.total_letter_pages = total_pages
        self.letter_page_info_var.set(
            f"第 {self.letter_page + 1}/{total_pages} 页（共 {total} 张）"
        )
        if self.prev_letter_btn is not None:
            self.prev_letter_btn.config(state="normal" if self.letter_page > 0 else "disabled")
        if self.next_letter_btn is not None:
            self.next_letter_btn.config(
                state="normal" if self.letter_page < total_pages - 1 else "disabled"
            )

    def _prev_letter_page(self):
        if not self.enable_letter_paging:
            return
        if self.letter_page > 0:
            self.letter_page -= 1
            self._load_letters()

    def _next_letter_page(self):
        if not self.enable_letter_paging:
            return
        if self.total_letter_pages and self.letter_page < self.total_letter_pages - 1:
            self.letter_page += 1
            self._load_letters()

    # ---- 热键与设置 ----
    def _capture_hotkey(self):
        if keyboard is None:
            messagebox.showerror("错误", "当前环境未安装 keyboard，无法录制热键。")
            return

        def worker():
            try:
                hk = keyboard.read_hotkey(suppress=False)
            except Exception as e:
                log(f"{self.log_prefix} 录制热键失败：{e}")
                return
            post_to_main_thread(lambda: self._set_hotkey(hk))

        threading.Thread(target=worker, daemon=True).start()

    def _set_hotkey(self, hotkey: str):
        self.hotkey_var.set(hotkey or "")
        self._bind_hotkey(show_popup=False)

    def _release_hotkey(self):
        if self.hotkey_handle is None or keyboard is None:
            return
        try:
            keyboard.remove_hotkey(self.hotkey_handle)
        except Exception:
            pass
        self.hotkey_handle = None
        self._bound_hotkey_key = None

    def _bind_hotkey(self, show_popup: bool = True):
        if keyboard is None:
            return
        self._release_hotkey()
        key = self.hotkey_var.get().strip()
        if not key:
            return
        try:
            handle = keyboard.add_hotkey(
                key,
                self._on_hotkey_trigger,
            )
        except Exception as e:
            log(f"{self.log_prefix} 绑定热键失败：{e}")
            messagebox.showerror("错误", f"绑定热键失败：{e}")
            return

        self.hotkey_handle = handle
        self._bound_hotkey_key = key
        log(f"{self.log_prefix} 已绑定热键：{key}")

    def _on_hotkey_trigger(self):
        post_to_main_thread(self._handle_hotkey_if_active)

    def _handle_hotkey_if_active(self):
        active = get_active_fragment_gui()
        if active is not self and not self.is_farming:
            return
        self._toggle_by_hotkey()

    def _toggle_by_hotkey(self):
        if self.is_farming:
            log(f"{self.log_prefix} 热键触发：请求停止刷{self.product_short_label}。")
            self.stop_farming(from_hotkey=True)
        else:
            log(f"{self.log_prefix} 热键触发：开始刷{self.product_short_label}。")
            self.start_farming(from_hotkey=True)

    def _save_settings(self):
        try:
            waves = int(self.wave_var.get().strip())
            if waves <= 0:
                raise ValueError
        except ValueError:
            messagebox.showwarning("提示", "总波数请输入大于 0 的整数。")
            return
        try:
            timeout = float(self.timeout_var.get().strip())
            if timeout <= 0:
                raise ValueError
        except ValueError:
            messagebox.showwarning("提示", "局内超时请输入大于 0 的数字秒数。")
            return
        if not self._validate_auto_skill_settings():
            return
        section = self.cfg.setdefault(self.cfg_key, {})
        section["waves"] = waves
        section["timeout"] = timeout
        section["hotkey"] = self.hotkey_var.get().strip()
        section["auto_e_enabled"] = bool(self.auto_e_enabled_var.get())
        section["auto_e_interval"] = self.auto_e_interval_seconds
        section["auto_q_enabled"] = bool(self.auto_q_enabled_var.get())
        section["auto_q_interval"] = self.auto_q_interval_seconds
        if self._multi_feature_available():
            section[self.multi_config_enabled_key] = bool(
                self.multi_enabled_var.get() if self.multi_enabled_var else False
            )
            section[self.multi_config_files_key] = list(self.multi_letter_names)
        self._bind_hotkey()
        save_config(self.cfg)
        messagebox.showinfo("提示", "设置已保存。")

    def start_farming(self, from_hotkey: bool = False):
        if self._multi_mode_active():
            if not self.multi_letter_names:
                messagebox.showwarning("提示", "多刷模式下请至少选择一个密函。")
                return
        elif not self.selected_letter_path:
            messagebox.showwarning("提示", f"请先选择一个{self.letter_label}。")
            return

        try:
            total_waves = int(self.wave_var.get().strip())
            if total_waves <= 0:
                raise ValueError
        except ValueError:
            messagebox.showwarning("提示", "总波数请输入大于 0 的整数。")
            return

        try:
            self.timeout_seconds = float(self.timeout_var.get().strip())
            if self.timeout_seconds <= 0:
                raise ValueError
        except ValueError:
            messagebox.showwarning("提示", "局内超时请输入大于 0 的数字秒数。")
            return
        if not self._validate_auto_skill_settings():
            return

        if pyautogui is None or cv2 is None or np is None:
            messagebox.showerror("错误", "缺少 pyautogui 或 opencv/numpy，无法刷碎片。")
            return
        if keyboard is None and not hasattr(pyautogui, "keyDown"):
            messagebox.showerror("错误", "当前环境无法发送键盘输入。")
            return

        if not self._prepare_multi_runtime_cycle():
            return

        if not round_running_lock.acquire(blocking=False):
            messagebox.showwarning("提示", "当前已有其它任务在运行，请先停止后再试。")
            return

        self.fragment_count = 0
        self.fragment_count_var.set("0")
        self.finished_waves = 0
        self.run_start_time = time.time()
        self.is_farming = True
        self._update_stats_ui()
        self.parent.after(1000, self._stats_timer)

        worker_stop.clear()
        self.start_btn.config(state="disabled")

        t = threading.Thread(target=self._expel_worker, args=(total_waves,), daemon=True)
        t.start()

        if from_hotkey:
            log(f"{self.log_prefix} 热键启动刷{self.product_short_label}成功。")

    def stop_farming(self, from_hotkey: bool = False):
        worker_stop.set()
        if not from_hotkey:
            messagebox.showinfo(
                "提示",
                f"已请求停止刷{self.product_short_label}，本波结束后将自动退出。",
            )
        else:
            log(f"{self.log_prefix} 热键停止请求已发送，等待当前波结束。")

    def _add_fragments(self, delta: int):
        if delta <= 0:
            return
        self.fragment_count += delta
        val = self.fragment_count

        def _update():
            self.fragment_count_var.set(str(val))

        post_to_main_thread(_update)

    def _update_stats_ui(self):
        if self.run_start_time is None:
            elapsed = 0
        else:
            elapsed = time.time() - self.run_start_time
        self.time_str_var.set(format_hms(elapsed))
        if self.finished_waves > 0:
            rate = self.fragment_count / self.finished_waves
        else:
            rate = 0.0
        self.rate_str_var.set(f"{rate:.2f} {self.product_short_label}/波")
        if elapsed > 0:
            eff = self.fragment_count / (elapsed / 3600.0)
        else:
            eff = 0.0
        self.eff_str_var.set(f"{eff:.2f} {self.product_short_label}/小时")

    def _stats_timer(self):
        if not self.is_farming:
            return
        self._update_stats_ui()
        self.parent.after(1000, self._stats_timer)

    def _expel_worker(self, total_waves: int):
        try:
            log("===== 驱离刷取 开始 =====")
            if not init_game_region():
                messagebox.showerror(
                    "错误", f"未找到{get_window_name_hint()}窗口，无法开始驱离刷取。"
                )
                return

            if not self._prepare_first_wave():
                log(f"{self.log_prefix} 首次进入失败，结束刷取。")
                return

            current_wave = 1
            max_wave = total_waves

            while not worker_stop.is_set():
                log(f"{self.log_prefix} 开始第 {current_wave} 波战斗挂机…")
                result = self._run_wave_actions(current_wave)
                if worker_stop.is_set():
                    break

                if result == "timeout":
                    log(f"{self.log_prefix} 第 {current_wave} 波判定卡死，执行防卡死逻辑…")
                    if not self._anti_stuck_and_reset():
                        log(f"{self.log_prefix} 防卡死失败，结束刷取。")
                        break
                    continue

                if result != "ok":
                    break

                self.finished_waves += 1

                if max_wave > 0 and current_wave >= max_wave:
                    if self.auto_loop_var.get():
                        current_wave = 1
                    else:
                        log(f"{self.log_prefix} 到达设定波数（未启用自动循环），撤退并结束。")
                        self._retreat_only()
                        break
                else:
                    current_wave += 1

                if worker_stop.is_set():
                    break
                if not self.auto_loop_var.get() and max_wave > 0 and self.finished_waves >= max_wave:
                    # 已经完成指定波数且不循环，直接退出
                    self._retreat_only()
                    break

                if not self._prepare_next_wave():
                    log(f"{self.log_prefix} 进入下一波失败，结束刷取。")
                    break

            log("===== 驱离刷取 结束 =====")

        except Exception as e:
            log(f"{self.log_prefix} 后台线程异常：{e}")
            traceback.print_exc()
        finally:
            worker_stop.clear()
            round_running_lock.release()
            self.is_farming = False
            self._update_stats_ui()

            def restore():
                try:
                    self.start_btn.config(state="normal")
                except Exception:
                    pass

            post_to_main_thread(restore)

            if self.run_start_time is not None:
                elapsed = time.time() - self.run_start_time
                time_str = format_hms(elapsed)
                if self.finished_waves > 0:
                    rate = self.fragment_count / self.finished_waves
                else:
                    rate = 0.0
                if elapsed > 0:
                    eff = self.fragment_count / (elapsed / 3600.0)
                else:
                    eff = 0.0
                msg = (
                    f"驱离刷{self.product_short_label}已结束。\n\n"
                    f"总运行时间：{time_str}\n"
                    f"完成波数：{self.finished_waves}\n"
                    f"累计{self.product_label}：{self.fragment_count}\n"
                    f"平均掉落：{rate:.2f} {self.product_short_label}/波\n"
                    f"效率：{eff:.2f} {self.product_short_label}/小时\n"
                )
                post_to_main_thread(
                    lambda: messagebox.showinfo(
                        f"驱离刷{self.product_short_label}完成", msg
                    )
                )

    def _prepare_first_wave(self) -> bool:
        log(f"{self.log_prefix} 首次进图：{self.letter_label} → 确认选择")
        return self._select_letter_sequence(f"{self.log_prefix} 首次", need_open_button=True)

    def _prepare_next_wave(self) -> bool:
        log(f"{self.log_prefix} 下一波：再次进行 → {self.letter_label} → 确认")
        if not wait_and_click_template(BTN_EXPEL_NEXT_WAVE, f"{self.log_prefix} 下一波：再次进行按钮", 25.0, 0.8):
            log(f"{self.log_prefix} 下一波：未能点击 再次进行.png。")
            return False
        return self._select_letter_sequence(f"{self.log_prefix} 下一波", need_open_button=False)

    def _select_letter_sequence(self, prefix: str, need_open_button: bool) -> bool:
        return super()._select_letter_sequence(prefix, need_open_button)

    def _run_wave_actions(self, wave_index: int) -> str:
        if not self._wait_for_map_entry():
            return "stopped"
        if not self._hold_forward(12.0):
            return "stopped"
        return self._random_move_and_loot(self.timeout_seconds)

    def _wait_for_map_entry(self, wait_seconds: float = 7.0) -> bool:
        log(f"{self.log_prefix} 确认后等待 {wait_seconds:.1f} 秒让地图载入…")
        start = time.time()
        while time.time() - start < wait_seconds:
            if worker_stop.is_set():
                return False
            time.sleep(0.1)
        return True

    def _hold_forward(self, duration: float) -> bool:
        if keyboard is None and not hasattr(pyautogui, "keyDown"):
            log(f"{self.log_prefix} 无法发送按键，无法执行长按 W。")
            return False
        log(f"{self.log_prefix} 长按 W {duration:.1f} 秒…")
        self._press_key("w")
        try:
            start = time.time()
            while time.time() - start < duration:
                if worker_stop.is_set():
                    return False
                time.sleep(0.1)
        finally:
            self._release_key("w")
        return True

    def _random_move_and_loot(self, max_wait: float) -> str:
        if keyboard is None and not hasattr(pyautogui, "keyDown"):
            log(f"{self.log_prefix} 无法发送按键。")
            return "stopped"

        auto_e_enabled = bool(self.auto_e_enabled_var.get())
        auto_q_enabled = bool(self.auto_q_enabled_var.get())
        e_interval = getattr(self, "auto_e_interval_seconds", 5.0)
        q_interval = getattr(self, "auto_q_interval_seconds", 5.0)

        desc_parts = []
        if auto_e_enabled:
            desc_parts.append(f"E 每 {e_interval:g} 秒")
        if auto_q_enabled:
            desc_parts.append(f"Q 每 {q_interval:g} 秒")
        if not desc_parts:
            desc = "不自动释放技能"
        else:
            desc = "，".join(desc_parts)

        log(
            f"{self.log_prefix} 顺序执行 W/A/S/D（每个 2 秒），{desc}（超时 {max_wait:.1f} 秒）。"
        )
        start = time.time()
        last_e = start
        last_q = start
        drop_ui_visible = False
        last_ui_log = 0.0
        min_drop_check_time = 10.0

        sequence = ["w", "a", "s", "d"]
        idx = 0
        active_key = None
        key_end_time = start

        try:
            while not worker_stop.is_set():
                now = time.time()

                if active_key is None or now >= key_end_time:
                    if active_key:
                        self._release_key(active_key)
                    active_key = sequence[idx]
                    idx = (idx + 1) % len(sequence)
                    self._press_key(active_key)
                    key_end_time = now + 2.0

                if auto_e_enabled and now - last_e >= e_interval:
                    self._tap_key("e")
                    last_e = now
                if auto_q_enabled and now - last_q >= q_interval:
                    self._tap_key("q")
                    last_q = now

                if now - start >= min_drop_check_time:
                    if not drop_ui_visible:
                        if self._is_drop_ui_visible():
                            drop_ui_visible = True
                            log(f"{self.log_prefix} 检测到物品掉落界面，开始识别掉落物。")
                        else:
                            if now - last_ui_log > 3.0:
                                self._is_drop_ui_visible(log_detail=True)
                                last_ui_log = now
                    else:
                        if self._detect_and_pick_drop():
                            log(f"{self.log_prefix} 本波掉落已选择。")
                            return "ok"

                if now - start > max_wait:
                    log(f"{self.log_prefix} 超过 {max_wait:.1f} 秒未检测到掉落，判定卡死。")
                    return "timeout"

                time.sleep(0.1)

        finally:
            if active_key:
                self._release_key(active_key)

        return "stopped"

    def _press_key(self, key: str):
        try:
            if keyboard is not None:
                keyboard.press(key)
            else:
                pyautogui.keyDown(key)
        except Exception as e:
            log(f"{self.log_prefix} 按下 {key} 失败：{e}")

    def _release_key(self, key: str):
        try:
            if keyboard is not None:
                keyboard.release(key)
            else:
                pyautogui.keyUp(key)
        except Exception as e:
            log(f"{self.log_prefix} 松开 {key} 失败：{e}")

    def _tap_key(self, key: str):
        try:
            if keyboard is not None:
                keyboard.press_and_release(key)
            else:
                pyautogui.press(key)
        except Exception as e:
            log(f"{self.log_prefix} 发送 {key} 失败：{e}")

    def _is_drop_ui_visible(self, log_detail: bool = False, threshold: float = 0.7) -> bool:
        score, _, _ = match_template(BTN_CONFIRM_LETTER)
        if log_detail:
            log(f"{self.log_prefix} 掉落界面检查：确认选择 匹配度 {score:.3f}")
        return score >= threshold

    def _detect_and_pick_drop(self, threshold=0.8) -> bool:
        if click_template(
            BTN_CONFIRM_LETTER,
            f"{self.log_prefix} 掉落确认：确认选择",
            threshold=0.7,
        ):
            time.sleep(1.0)
            return True
        return False

    def _anti_stuck_and_reset(self) -> bool:
        try:
            if keyboard is not None:
                keyboard.press_and_release("esc")
            else:
                pyautogui.press("esc")
        except Exception as e:
            log(f"{self.log_prefix} 发送 ESC 失败：{e}")
        time.sleep(1.0)
        click_template("G.png", f"{self.log_prefix} 防卡死：点击 G.png", 0.6)
        time.sleep(1.0)
        click_template("Q.png", f"{self.log_prefix} 防卡死：点击 Q.png", 0.6)
        time.sleep(1.0)

        if not wait_and_click_template(
            BTN_EXPEL_NEXT_WAVE,
            f"{self.log_prefix} 防卡死：再次进行按钮",
            25.0,
            0.8,
        ):
            log(f"{self.log_prefix} 防卡死：未能点击 再次进行.png。")
            return False
        return self._select_letter_sequence(f"{self.log_prefix} 防卡死", need_open_button=False)

    def _retreat_only(self):
        wait_and_click_template(BTN_RETREAT_START, f"{self.log_prefix} 撤退按钮", 20.0, 0.8)


class ModFragmentGUI(FragmentFarmGUI):
    multi_toggle_text = "同时刷多个mod"
    multi_list_title = "mod刷取顺序"
    multi_item_prefix = "刷取"
    def __init__(self, parent, cfg):
        self.cfg_key = "mod_guard_settings"
        self.letter_label = "mod密函"
        self.product_label = "mod成品"
        self.product_short_label = "mod成品"
        self.entity_label = "mod"
        self.letters_dir = MOD_DIR
        self.letters_dir_hint = "mod"
        self.preview_dir_hint = "mod"
        self.log_prefix = "[MOD]"
        super().__init__(parent, cfg, enable_no_trick_decrypt=True)


class ModExpelGUI(ExpelFragmentGUI):
    multi_toggle_text = "同时刷多个mod"
    multi_list_title = "mod驱离顺序"
    multi_item_prefix = "刷取"
    def __init__(self, parent, cfg):
        self.cfg_key = "mod_expel_settings"
        self.letter_label = "mod密函"
        self.product_label = "mod成品"
        self.product_short_label = "mod成品"
        self.entity_label = "mod"
        self.letters_dir = MOD_DIR
        self.letters_dir_hint = "mod"
        self.preview_dir_hint = "mod"
        self.log_prefix = "[MOD-驱离]"
        super().__init__(parent, cfg)


class WeaponBlueprintFragmentGUI(FragmentFarmGUI):
    multi_toggle_text = "同时刷取多张武器图纸"
    multi_list_title = "武器图纸刷取顺序"

    def __init__(self, parent, cfg):
        self.cfg_key = "weapon_blueprint_guard_settings"
        self.letter_label = "武器图纸密函"
        self.product_label = "武器图纸成品"
        self.product_short_label = "武器图纸"
        self.entity_label = "武器图纸"
        self.letters_dir = WQ_DIR
        self.letters_dir_hint = "武器图纸"
        self.preview_dir_hint = "武器图纸"
        self.log_prefix = "[武器图纸]"
        self.enable_letter_paging = True
        self.letter_nav_position = "top"
        super().__init__(parent, cfg, enable_no_trick_decrypt=True)


class WeaponBlueprintExpelGUI(ExpelFragmentGUI):
    multi_toggle_text = "同时刷取多张武器图纸"
    multi_list_title = "武器图纸驱离顺序"

    def __init__(self, parent, cfg):
        self.cfg_key = "weapon_blueprint_expel_settings"
        self.letter_label = "武器图纸密函"
        self.product_label = "武器图纸成品"
        self.product_short_label = "武器图纸"
        self.entity_label = "武器图纸"
        self.letters_dir = WQ_DIR
        self.letters_dir_hint = "武器图纸"
        self.preview_dir_hint = "武器图纸"
        self.log_prefix = "[武器图纸-驱离]"
        self.enable_letter_paging = True
        self.letter_nav_position = "top"
        super().__init__(parent, cfg)


class ClueFarmGUI(FragmentFarmGUI):
    cfg_key = "clue_guard_settings"
    letter_label = "密函"
    product_label = "密函线索"
    product_short_label = "线索"
    entity_label = "密函线索"
    log_prefix = "[线索]"
    multi_toggle_text = ""
    multi_list_title = ""
    multi_item_prefix = ""
    enable_letter_paging = False

    def __init__(self, parent, cfg):
        clue_cfg = cfg.get(self.cfg_key, {})
        initial_level = str(clue_cfg.get("level", "10"))
        auto_switch_enabled = bool(clue_cfg.get("auto_switch", False))
        auto_switch_minutes = clue_cfg.get("auto_switch_minutes", 60)

        self._pending_decrypt_mode = "line"
        self._auto_switch_deadline: Optional[float] = None
        self._auto_switch_origin_level = initial_level if initial_level in ("10", "30") else "10"

        self.level10_macro_a_path = os.path.join(SCRIPTS_DIR, CLUE_MACRO_A)
        self.level10_macro_b_path = os.path.join(SCRIPTS_DIR, CLUE_MACRO_B)
        self.level30_macro_path = os.path.join(SCRIPTS_DIR, CLUE_MACRO_30)

        super().__init__(parent, cfg, enable_no_trick_decrypt=True)

        self.selected_letter_path = "__AUTO__"
        self.level_var = tk.StringVar(
            value=initial_level if initial_level in ("10", "30") else "10"
        )
        coerced_minutes = self._coerce_minutes(auto_switch_minutes)
        self.auto_switch_var = tk.BooleanVar(value=auto_switch_enabled)
        self.auto_switch_minutes_var = tk.StringVar(
            value=self._format_minutes(coerced_minutes)
        )
        self.decrypt_mode_var = tk.StringVar(value="划线无巧手解密")
        self.auto_switch_remaining_var = tk.StringVar(value="未开启")

        self.level_var.trace_add("write", self._on_level_var_changed)

        self._setup_clue_ui()
        self._on_level_var_changed()
        self._update_auto_switch_ui()
        self._reset_auto_switch_timer(reset_origin=True)

    # ------------------------------------------------------------------
    # UI 布局
    # ------------------------------------------------------------------
    def _setup_clue_ui(self):
        self.hotkey_label_widget.config(text="刷线索热键:")
        try:
            self.start_btn.config(text="开始刷线索")
        except Exception:
            pass

        if getattr(self, "frame_letters", None) is not None:
            try:
                self.frame_letters.destroy()
            except Exception:
                pass
            self.frame_letters = None

        self.selected_label_var.set("无需选择密函（自动流程）")
        self.stat_name_var.set("自动模式")
        try:
            self.stat_image_label.config(image="", text="自动模式", fg="#666666")
        except Exception:
            pass

        if getattr(self, "frame_macros", None) is not None:
            try:
                self.frame_macros.config(text="地图宏脚本（自动加载）")
            except Exception:
                pass
        for widget in (
            getattr(self, "macro_a_entry", None),
            getattr(self, "macro_b_entry", None),
        ):
            if widget is not None:
                try:
                    widget.config(state="readonly")
                except Exception:
                    pass
        for widget in (
            getattr(self, "macro_a_button", None),
            getattr(self, "macro_b_button", None),
        ):
            if widget is not None:
                try:
                    widget.config(state="disabled")
                except Exception:
                    pass

        level_frame = tk.LabelFrame(self.left_panel, text="等级设置")
        if getattr(self, "frame_macros", None) is not None and self.frame_macros.winfo_exists():
            level_frame.pack(fill="x", padx=10, pady=5, before=self.frame_macros)
        else:
            level_frame.pack(fill="x", padx=10, pady=5)

        radios = tk.Frame(level_frame)
        radios.pack(fill="x", padx=8, pady=(6, 2))
        tk.Radiobutton(
            radios,
            text="10级（10火）",
            variable=self.level_var,
            value="10",
            anchor="w",
        ).pack(side="left", padx=(0, 10))
        tk.Radiobutton(
            radios,
            text="30级（30级火本）",
            variable=self.level_var,
            value="30",
            anchor="w",
        ).pack(side="left")

        auto_row = tk.Frame(level_frame)
        auto_row.pack(fill="x", padx=8, pady=(4, 6))
        tk.Checkbutton(
            auto_row,
            text="自动切换等级",
            variable=self.auto_switch_var,
            command=self._on_auto_switch_toggle,
        ).pack(side="left")
        tk.Label(auto_row, text="间隔(分钟)：").pack(side="left", padx=(10, 2))
        self.auto_switch_entry = tk.Entry(
            auto_row, textvariable=self.auto_switch_minutes_var, width=6
        )
        self.auto_switch_entry.pack(side="left")

        tip_text = (
            "提示：\n"
            "1. 本界面无需选择密函，自动识别继续挑战与开始挑战。\n"
            "2. 10级（10火）使用 scripts/ 目录下的人物碎片刷取A/B.json，30级（火本）使用 scripts/30级.json。\n"
            "3. 地图模板放入 templates/ 目录（mapa.png、mapb.png、30map.png）。"
        )
        try:
            self.tip_label.config(text=tip_text)
        except Exception:
            pass

        # 重新布置统计面板，仅保留关键信息
        for child in list(self.stats_frame.winfo_children()):
            child.destroy()
        self.stats_frame.config(text="线索运行信息")

        self.stat_image_label = tk.Label(self.stats_frame, relief="sunken")
        self.stat_image_label.grid(row=0, column=0, rowspan=4, padx=5, pady=5, sticky="ns")

        self.current_entity_label = tk.Label(self.stats_frame, text="当前模式：")
        self.current_entity_label.grid(row=0, column=1, sticky="e", pady=(4, 0))
        tk.Label(self.stats_frame, textvariable=self.stat_name_var).grid(
            row=0, column=2, sticky="w", padx=(4, 0), pady=(4, 0)
        )

        self.total_product_label = tk.Label(self.stats_frame, text="解密方式：")
        self.total_product_label.grid(row=1, column=1, sticky="e")
        tk.Label(self.stats_frame, textvariable=self.decrypt_mode_var).grid(
            row=1, column=2, sticky="w", padx=(4, 0)
        )

        tk.Label(self.stats_frame, text="运行时间：").grid(
            row=2, column=1, sticky="e", pady=(4, 0)
        )
        tk.Label(self.stats_frame, textvariable=self.time_str_var).grid(
            row=2, column=2, sticky="w", padx=(4, 0), pady=(4, 0)
        )

        tk.Label(self.stats_frame, text="距离自动切换等级还剩：").grid(
            row=3, column=1, sticky="e", pady=(4, 4)
        )
        tk.Label(self.stats_frame, textvariable=self.auto_switch_remaining_var).grid(
            row=3, column=2, sticky="w", padx=(4, 0), pady=(4, 4)
        )

        self._refresh_auto_switch_remaining()

    # ------------------------------------------------------------------
    # 配置与状态
    # ------------------------------------------------------------------
    def _coerce_minutes(self, value) -> float:
        try:
            minutes = float(value)
        except (TypeError, ValueError):
            minutes = 60.0
        if minutes < 1.0:
            minutes = 1.0
        return minutes

    def _format_minutes(self, minutes: float) -> str:
        if abs(minutes - int(minutes)) < 1e-6:
            return str(int(minutes))
        return f"{minutes:g}"

    def _on_level_var_changed(self, *_):
        self._update_macro_bindings_for_level()
        self._update_level_display()
        self._update_pending_decrypt_mode()

    def _update_macro_bindings_for_level(self):
        level = self.level_var.get()
        if level == "30":
            self.macro_a_var.set(self.level30_macro_path)
            self.macro_b_var.set(self.level30_macro_path)
        else:
            self.macro_a_var.set(self.level10_macro_a_path)
            self.macro_b_var.set(self.level10_macro_b_path)

    def _update_level_display(self):
        level = self.level_var.get()
        if level == "30":
            desc = "当前模式：30级（转盘解密）"
            decrypt_desc = "转盘无巧手解密"
        else:
            desc = "当前模式：10级（划线解密）"
            decrypt_desc = "划线无巧手解密"
        self.selected_label_var.set(desc)
        self.stat_name_var.set(desc)
        self.decrypt_mode_var.set(decrypt_desc)
        try:
            self.stat_image_label.config(image="", text=desc, fg="#444444")
        except Exception:
            pass

    def _update_pending_decrypt_mode(self):
        level = self.level_var.get()
        if level == "30":
            self._pending_decrypt_mode = "firework"
        else:
            self._pending_decrypt_mode = "line"

    def _on_auto_switch_toggle(self):
        self._update_auto_switch_ui()
        if not self.auto_switch_var.get():
            self._auto_switch_deadline = None
        else:
            self._reset_auto_switch_timer()

    def _update_auto_switch_ui(self):
        state = tk.NORMAL if self.auto_switch_var.get() else tk.DISABLED
        try:
            self.auto_switch_entry.config(state=state)
        except Exception:
            pass
        self._refresh_auto_switch_remaining()

    def _auto_switch_enabled(self) -> bool:
        return bool(self.auto_switch_var.get())

    def _get_auto_switch_seconds(self) -> float:
        minutes = self._coerce_minutes(self.auto_switch_minutes_var.get())
        self.auto_switch_minutes_var.set(self._format_minutes(minutes))
        seconds = max(CLUE_AUTO_SWITCH_MIN_SECONDS, minutes * 60.0)
        return seconds

    def _refresh_auto_switch_remaining(self):
        if not self._auto_switch_enabled() or self._auto_switch_deadline is None:
            self.auto_switch_remaining_var.set("未开启")
            return
        remaining = max(0.0, self._auto_switch_deadline - time.time())
        self.auto_switch_remaining_var.set(f"{int(remaining)} 秒")

    def _save_settings(self):
        try:
            waves = int(self.wave_var.get().strip())
            if waves <= 0:
                raise ValueError
        except ValueError:
            messagebox.showwarning("提示", "总波数请输入大于 0 的整数。")
            return

        try:
            timeout = float(self.timeout_var.get().strip())
            if timeout <= 0:
                raise ValueError
        except ValueError:
            messagebox.showwarning("提示", "局内超时请输入大于 0 的数字秒数。")
            return

        if not self._validate_auto_skill_settings():
            return

        minutes = self._coerce_minutes(self.auto_switch_minutes_var.get())
        self.auto_switch_minutes_var.set(self._format_minutes(minutes))

        section = self.cfg.setdefault(self.cfg_key, {})
        section["waves"] = waves
        section["timeout"] = timeout
        section["hotkey"] = self.hotkey_var.get().strip()
        if self.enable_no_trick_decrypt and self.no_trick_var is not None:
            section["no_trick_decrypt"] = bool(self.no_trick_var.get())
        section["auto_e_enabled"] = bool(self.auto_e_enabled_var.get())
        section["auto_e_interval"] = self.auto_e_interval_seconds
        section["auto_q_enabled"] = bool(self.auto_q_enabled_var.get())
        section["auto_q_interval"] = self.auto_q_interval_seconds
        section["level"] = self.level_var.get()
        section["auto_switch"] = bool(self.auto_switch_var.get())
        section["auto_switch_minutes"] = minutes

        self._bind_hotkey()
        save_config(self.cfg)
        messagebox.showinfo("提示", "设置已保存。")

    def _reset_auto_switch_timer(self, reset_origin: bool = False):
        if reset_origin:
            current = self.level_var.get()
            if current in ("10", "30"):
                self._auto_switch_origin_level = current
        if self._auto_switch_enabled():
            self._auto_switch_deadline = time.time() + self._get_auto_switch_seconds()
        else:
            self._auto_switch_deadline = None
        self._refresh_auto_switch_remaining()

    def _check_auto_switch(self) -> bool:
        if not self._auto_switch_enabled() or self._auto_switch_deadline is None:
            return False
        if time.time() < self._auto_switch_deadline:
            return False
        log(f"{self.log_prefix} 自动切换等级计时到达，执行模式切换。")
        success = self._perform_mode_switch()
        if success:
            self._reset_auto_switch_timer(reset_origin=True)
            return True
        self._auto_switch_deadline = time.time() + 60.0
        return False

    def _update_stats_ui(self):
        super()._update_stats_ui()
        self._refresh_auto_switch_remaining()

    # ------------------------------------------------------------------
    # 运行流程
    # ------------------------------------------------------------------
    def start_farming(self, from_hotkey: bool = False):
        level = self.level_var.get()
        if level == "30":
            if not os.path.exists(self.level30_macro_path):
                messagebox.showwarning(
                    "提示",
                    "未找到 scripts/30级.json，请放置对应宏文件后重试。",
                )
                return
        else:
            missing = []
            if not os.path.exists(self.level10_macro_a_path):
                missing.append(CLUE_MACRO_A)
            if not os.path.exists(self.level10_macro_b_path):
                missing.append(CLUE_MACRO_B)
            if missing:
                messagebox.showwarning(
                    "提示",
                    "缺少以下宏文件：\n" + "\n".join(missing),
                )
                return

        self.selected_letter_path = "__AUTO__"
        self._update_macro_bindings_for_level()

        try:
            total_waves = int(self.wave_var.get().strip())
            if total_waves <= 0:
                raise ValueError
        except ValueError:
            messagebox.showwarning("提示", "总波数请输入大于 0 的整数。")
            return

        try:
            self.timeout_seconds = float(self.timeout_var.get().strip())
            if self.timeout_seconds <= 0:
                raise ValueError
        except ValueError:
            messagebox.showwarning("提示", "局内超时请输入大于 0 的数字秒数。")
            return

        if not self._validate_auto_skill_settings():
            return

        if pyautogui is None or cv2 is None or np is None:
            messagebox.showerror("错误", "缺少 pyautogui 或 opencv/numpy，无法刷线索。")
            return
        if keyboard is None:
            messagebox.showerror("错误", "未安装 keyboard 模块，无法发送按键。")
            return

        if not self._prepare_multi_runtime_cycle():
            return

        if not round_running_lock.acquire(blocking=False):
            messagebox.showwarning("提示", "当前已有其它任务在运行，请先停止后再试。")
            return

        self.fragment_count = 0
        self.fragment_count_var.set("0")
        self.finished_waves = 0
        self.run_start_time = time.time()
        self.is_farming = True
        self._update_stats_ui()
        self.parent.after(1000, self._stats_timer)
        self._reset_wave_progress(total_waves)

        worker_stop.clear()
        try:
            self.start_btn.config(state="disabled")
        except Exception:
            pass

        self._reset_auto_switch_timer(reset_origin=True)

        t = threading.Thread(target=self._farm_worker, args=(total_waves,), daemon=True)
        t.start()

        if from_hotkey:
            log(f"{self.log_prefix} 热键启动刷线索成功。")

    def _enter_first_wave_and_setup(self) -> bool:
        log(f"{self.log_prefix} 首次进图：双击开始挑战 → 地图识别")
        if not self._click_start_challenge_twice():
            return False
        self._increment_wave_progress()
        return self._map_detect_and_run_macros()

    def _restart_from_lobby_after_retreat(self) -> bool:
        log(f"{self.log_prefix} 循环重开：再次进行 → 开始挑战 → 地图识别")
        if not wait_and_click_template(
            BTN_EXPEL_NEXT_WAVE,
            f"{self.log_prefix} 循环重开：再次进行按钮",
            25.0,
            0.8,
        ):
            log(f"{self.log_prefix} 循环重开：未能点击 再次进行.png。")
            return False
        time.sleep(0.4)
        if not self._click_single_start_challenge("循环重开：开始挑战"):
            return False
        self._increment_wave_progress()
        return self._map_detect_and_run_macros()

    def _enter_next_wave_without_map(self) -> bool:
        log(f"{self.log_prefix} 进入下一波：继续挑战 → 开始挑战")
        if not wait_and_click_template(
            BTN_CONTINUE_CHALLENGE,
            f"{self.log_prefix} 下一波：继续挑战按钮",
            25.0,
            0.8,
        ):
            log(f"{self.log_prefix} 下一波：未能点击 继续挑战.png。")
            return False
        time.sleep(0.4)
        return self._click_single_start_challenge("下一波：开始挑战")

    def _anti_stuck_and_reset(self) -> bool:
        log(f"{self.log_prefix} 防卡死：ESC → G → Q → 再次进行 → 开始挑战")
        try:
            if keyboard is not None:
                keyboard.press_and_release("esc")
            else:
                pyautogui.press("esc")
        except Exception as e:
            log(f"{self.log_prefix} 发送 ESC 失败：{e}")
        time.sleep(1.0)
        click_template("G.png", f"{self.log_prefix} 防卡死：点击 G.png", 0.6)
        time.sleep(1.0)
        click_template("Q.png", f"{self.log_prefix} 防卡死：点击 Q.png", 0.6)
        time.sleep(1.0)

        if not wait_and_click_template(
            BTN_EXPEL_NEXT_WAVE,
            f"{self.log_prefix} 防卡死：再次进行按钮",
            25.0,
            0.8,
        ):
            log(f"{self.log_prefix} 防卡死：未能点击 再次进行.png。")
            return False
        time.sleep(0.4)
        if not self._click_single_start_challenge("防卡死：开始挑战"):
            return False
        return self._map_detect_and_run_macros()

    def _map_detect_and_run_macros(self) -> bool:
        if self.level_var.get() == "30":
            return self._run_level30_macro()
        self._update_pending_decrypt_mode()
        return super()._map_detect_and_run_macros()

    def _run_level30_macro(self) -> bool:
        template_path = os.path.join(TEMPLATE_DIR, CLUE_MAP_30_TEMPLATE)
        if not os.path.exists(template_path):
            log(f"{self.log_prefix} 缺少地图模板：{CLUE_MAP_30_TEMPLATE}")
            return False

        log(f"{self.log_prefix} 开始识别 30map（最长 12 秒）…")
        deadline = time.time() + 12.0
        best_score = 0.0
        score = 0.0
        while time.time() < deadline and not worker_stop.is_set():
            score, _, _ = match_template(CLUE_MAP_30_TEMPLATE)
            best_score = max(best_score, score)
            log(f"{self.log_prefix} 30map 匹配度 {score:.3f}")
            if score >= 0.7:
                break
            time.sleep(0.4)

        if score < 0.7:
            log(
                f"{self.log_prefix} 12 秒内 30map 匹配度低于 0.7（最高 {best_score:.3f}），本趟放弃。"
            )
            return False

        log(f"{self.log_prefix} 识别到 30map（匹配度 {score:.3f}），等待 2 秒进入宏。")
        t0 = time.time()
        while time.time() - t0 < 2.0 and not worker_stop.is_set():
            time.sleep(0.1)

        self._pending_decrypt_mode = "firework"
        self._execute_map_macro(self.level30_macro_path, "30级 地图宏")
        return True

    def _execute_map_macro(self, macro_path: str, label: str):
        try:
            super()._execute_map_macro(macro_path, label)
        finally:
            self._update_pending_decrypt_mode()

    def _battle_and_loot(self, max_wait: float = 160.0) -> str:
        if keyboard is None and pyautogui is None:
            log(f"{self.log_prefix} 无法发送按键。")
            return "stopped"

        auto_e_enabled = bool(self.auto_e_enabled_var.get())
        auto_q_enabled = bool(self.auto_q_enabled_var.get())
        e_interval = getattr(self, "auto_e_interval_seconds", 5.0)
        q_interval = getattr(self, "auto_q_interval_seconds", 5.0)

        desc_parts = []
        if auto_e_enabled:
            desc_parts.append(f"E 每 {e_interval:g} 秒")
        if auto_q_enabled:
            desc_parts.append(f"Q 每 {q_interval:g} 秒")
        desc = "，".join(desc_parts) if desc_parts else "不自动释放技能"

        log(
            f"{self.log_prefix} 开始战斗挂机（{desc}，等待继续挑战，超时 {max_wait:.1f} 秒）。"
        )

        start = time.time()
        last_e = start
        last_q = start
        last_revive_check = start
        last_wait_log = start

        while not worker_stop.is_set():
            now = time.time()

            if now - last_revive_check >= AUTO_REVIVE_CHECK_INTERVAL:
                last_revive_check = now
                self._auto_revive_if_needed()

            if auto_e_enabled and now - last_e >= e_interval:
                try:
                    if keyboard is not None:
                        keyboard.press_and_release("e")
                    else:
                        pyautogui.press("e")
                except Exception as e:
                    log(f"{self.log_prefix} 发送 E 失败：{e}")
                last_e = now

            if auto_q_enabled and now - last_q >= q_interval:
                try:
                    if keyboard is not None:
                        keyboard.press_and_release("q")
                    else:
                        pyautogui.press("q")
                except Exception as e:
                    log(f"{self.log_prefix} 发送 Q 失败：{e}")
                last_q = now

            score, _, _ = match_template(BTN_CONTINUE_CHALLENGE)
            if score >= 0.7:
                log(f"{self.log_prefix} 检测到继续挑战按钮，战斗完成。")
                return "ok"

            if now - last_wait_log >= 3.0:
                log(
                    f"{self.log_prefix} 正在等待继续挑战按钮… 当前匹配度 {score:.3f}"
                )
                last_wait_log = now

            if now - start >= max_wait:
                log(
                    f"{self.log_prefix} 超过 {max_wait:.1f} 秒未检测到继续挑战按钮，判定卡死。"
                )
                return "timeout"

            time.sleep(0.2)

        return "stopped"

    def _start_no_trick_monitor(self):
        if not self.enable_no_trick_decrypt or not self.no_trick_var.get():
            return None
        mode = getattr(self, "_pending_decrypt_mode", "line")
        if mode == "firework":
            controller = FireworkNoTrickController(self, GAME_SQ_DIR)
        else:
            controller = NoTrickDecryptController(self, GAME_DIR)
        if controller.start():
            self.no_trick_controller = controller
            return controller
        return None

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------
    def _click_start_challenge_twice(self) -> bool:
        if not self._click_single_start_challenge("开始挑战（第一次）"):
            return False
        time.sleep(0.4)
        return self._click_single_start_challenge("开始挑战（第二次）")

    def _click_single_start_challenge(self, stage: str) -> bool:
        return wait_and_click_template(
            HS_START_TEMPLATE,
            f"{self.log_prefix} {stage}",
            25.0,
            0.8,
        )

    def _perform_mode_switch(self) -> bool:
        if not init_game_region():
            log(f"{self.log_prefix} 模式切换：初始化窗口失败。")
            return False

        try:
            if keyboard is not None:
                keyboard.press_and_release("esc")
            else:
                pyautogui.press("esc")
        except Exception as e:
            log(f"{self.log_prefix} 模式切换：发送 ESC 失败：{e}")
        time.sleep(0.8)
        click_template("G.png", f"{self.log_prefix} 模式切换：点击 G.png", 0.6)
        time.sleep(0.8)
        click_template("Q.png", f"{self.log_prefix} 模式切换：点击 Q.png", 0.6)
        time.sleep(0.8)
        wait_and_click_template(
            CLUE_EXIT_ENTRUST_TEMPLATE,
            f"{self.log_prefix} 模式切换：退出委托",
            20.0,
            0.75,
        )
        time.sleep(6.0)

        try:
            if keyboard is not None:
                keyboard.press_and_release("esc")
            else:
                pyautogui.press("esc")
        except Exception as e:
            log(f"{self.log_prefix} 模式切换：再次发送 ESC 失败：{e}")
        time.sleep(0.6)
        if not wait_and_click_template(
            CLUE_SETTINGS_TEMPLATE,
            f"{self.log_prefix} 模式切换：打开设置",
            20.0,
            0.75,
        ):
            log(f"{self.log_prefix} 模式切换：未能点击 设置.png。")
            return False
        time.sleep(0.6)
        for tpl, desc in (
            (CLUE_TRAINING_TEMPLATE, "设置 → 历练"),
            (CLUE_ENTRUST_TEMPLATE, "历练 → 委托"),
            (CLUE_ADVENTURE_TEMPLATE, "委托 → 探险"),
        ):
            if not wait_and_click_template(
                tpl,
                f"{self.log_prefix} 模式切换：点击 {desc}",
                20.0,
                0.75,
            ):
                log(f"{self.log_prefix} 模式切换：未能点击 {tpl}。")
                return False
            time.sleep(0.6)

        current = self.level_var.get()
        target = "30" if current == "10" else "10"
        level_template = CLUE_LEVEL_30_TEMPLATE if target == "30" else CLUE_LEVEL_10_TEMPLATE
        if not wait_and_click_template(
            level_template,
            f"{self.log_prefix} 模式切换：选择 {target}级",
            20.0,
            0.75,
        ):
            log(f"{self.log_prefix} 模式切换：未能选择 {target}级。")
            return False
        time.sleep(0.4)
        if not wait_and_click_template(
            CLUE_FIRE_TEMPLATE,
            f"{self.log_prefix} 模式切换：选择火密函",
            20.0,
            0.75,
        ):
            log(f"{self.log_prefix} 模式切换：未能点击 火.png。")
            return False

        self.level_var.set(target)
        log(f"{self.log_prefix} 模式切换完成，当前等级：{target}级。")
        return True

# ======================================================================
#  全自动 50 人物经验副本
# ======================================================================
# ======================================================================
#  自动 70 红珠副本
# ======================================================================
class HS70AutoGUI:
    LOG_PREFIX = "[70HS]"
    MAP_STABILIZE_DELAY = 2.0
    BETWEEN_ROUNDS_DELAY = 2.0
    ENTRY_DELAY = 0.4
    INITIAL_MAP_THRESHOLD = 0.7
    BRANCH_A_THRESHOLD = 0.82
    BRANCH_A_TIMEOUT = 3.0
    VARIATION_THRESHOLD = 0.8
    SUBMAP_THRESHOLD = 0.65
    BRANCH_SCAN_DURATION = 25.0
    BRANCH_SCAN_INTERVAL = 0.25
    DECRYPT_EXTRA_DELAY = 0.5
    DECRYPT_APPEAR_TIMEOUT = 2.0
    DECRYPT_COMPLETE_TIMEOUT = 30.0
    TARGET_THRESHOLD = 0.8
    TARGET_SCAN_INTERVAL = 0.05
    TARGET_PAUSE = 5.0
    WARNING_THRESHOLD = 0.8
    WARNING_TIMEOUT = 20.0
    WARNING_SCAN_INTERVAL = 0.1
    WARNING_BRANCH_DELAY = 3.0
    BRANCH_DECRYPT_DELAY = 1.0
    CLICK_ATTEMPTS = 12
    RETRY_ATTEMPTS = 22
    PROGRESS_SEGMENTS = {
        "mapa.json": (20.0, 35.0),
        "mapa-开锁1.json": (35.0, 50.0),
        "mapa-开锁1-校准.json": (50.0, 55.0),
        "mapa-开锁2.json": (55.0, 70.0),
        "mapa-开锁2-校准.json": (70.0, 78.0),
        "branch": (70.0, 82.0),
        "final_reset": (82.0, 88.0),
        "final_macro": (88.0, 100.0),
    }

    def __init__(self, root, cfg):
        self.root = root
        self.cfg = cfg
        self.log_prefix = self.LOG_PREFIX

        settings = cfg.get("hs70_settings", {})
        self.hotkey_var = tk.StringVar(value=settings.get("hotkey", ""))
        self.loop_count_var = tk.StringVar(value=str(settings.get("loop_count", 0)))
        self.auto_loop_var = tk.BooleanVar(value=bool(settings.get("auto_loop", True)))
        self.no_trick_var = tk.BooleanVar(value=bool(settings.get("no_trick_decrypt", True)))

        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_message_var = tk.StringVar(value="等待开始")
        self.detail_message_var = tk.StringVar(value="")

        self.log_text = None
        self.progress = None
        self.no_trick_controller = None
        self.prepared_no_trick_controller = None
        self.no_trick_status_var = tk.StringVar(value="未启用")
        self.no_trick_progress_var = tk.DoubleVar(value=0.0)
        self.no_trick_image_ref = None
        self.no_trick_status_frame = None
        self.no_trick_image_label = None
        self.no_trick_progress = None
        self.tip_image_ref = None

        self.hotkey_handle = None
        self.running = False
        self.entry_prepared = False
        self.last_macro_name = None
        self.target_detection_disabled = False

        self._build_ui()
        self._update_no_trick_ui()

    # ---- UI ----
    def _build_ui(self):
        self.content_frame = tk.Frame(self.root)
        self.content_frame.pack(fill="both", expand=True)

        self.left_panel = tk.Frame(self.content_frame)
        self.left_panel.pack(side="left", fill="both", expand=True)

        notice = "不需要巧手 飞天刀 主控猪妹 剩下的两个一定要可以快速清怪 不然容易被卡 这一关怪太多了 没办法 变猪本身容易卡 不变容易被怪卡 我暂时没有好的办法了 文件夹里的宏 我一个人录制了4个多小时 尽力了 你们有更好的宏的话可以发给我。"
        notice_frame = tk.Frame(self.left_panel)
        notice_frame.pack(fill="x", padx=10, pady=(6, 0))
        tk.Label(
            notice_frame,
            text=notice,
            fg="#d40000",
            justify="left",
            anchor="w",
            wraplength=420,
        ).pack(side="left", fill="both", expand=True)

        tip_path = hs_find_asset(HS_TIP_IMAGE)
        if tip_path:
            photo = self._load_no_trick_preview(tip_path, max_size=140)
            if photo is not None:
                tip_frame = tk.Frame(notice_frame)
                tip_frame.pack(side="left", padx=(8, 0))
                tk.Label(tip_frame, image=photo).pack(side="left")
                tk.Label(
                    tip_frame,
                    text="如果经常卡怪 请让猪妹携带攀岩mod",
                    fg="#d40000",
                    anchor="w",
                    justify="left",
                    wraplength=140,
                ).pack(side="left", padx=(6, 0))
                self.tip_image_ref = photo
            else:
                log(f"{self.LOG_PREFIX} 无法加载 {HS_TIP_IMAGE} 预览。")
        else:
            log(f"{self.LOG_PREFIX} 缺少 {HS_TIP_IMAGE}，无法显示提示图。")

        self.right_panel = tk.Frame(self.content_frame)
        self.right_panel.pack(side="right", fill="y", padx=(5, 10), pady=5)

        controls = tk.Frame(self.left_panel)
        controls.pack(fill="x", padx=10, pady=5)
        controls.grid_columnconfigure(4, weight=1)

        tk.Label(controls, text="热键:").grid(row=0, column=0, sticky="e")
        tk.Entry(controls, textvariable=self.hotkey_var, width=15).grid(row=0, column=1, sticky="w")
        ttk.Button(controls, text="录制热键", command=self.capture_hotkey).grid(row=0, column=2, padx=3)
        ttk.Button(controls, text="保存配置", command=self.save_cfg).grid(row=0, column=3, padx=3)

        tk.Checkbutton(controls, text="自动循环", variable=self.auto_loop_var).grid(row=1, column=0, sticky="w")
        tk.Label(controls, text="循环次数(0=无限):").grid(row=1, column=1, sticky="e")
        tk.Entry(controls, textvariable=self.loop_count_var, width=8).grid(row=1, column=2, sticky="w")

        toggle = tk.Frame(self.left_panel)
        toggle.pack(fill="x", padx=10, pady=(0, 5))
        tk.Checkbutton(
            toggle,
            text="开启无巧手解密",
            variable=self.no_trick_var,
            command=self._on_no_trick_toggle,
        ).pack(anchor="w")

        btns = tk.Frame(self.left_panel)
        btns.pack(padx=10, pady=5)
        ttk.Button(btns, text="开始执行", command=self.start_via_button).grid(row=0, column=0, padx=3)
        ttk.Button(btns, text="开始监听热键", command=self.start_listen).grid(row=0, column=1, padx=3)
        ttk.Button(btns, text="停止", command=self.stop_listen).grid(row=0, column=2, padx=3)
        ttk.Button(btns, text="只执行一轮", command=self.run_once).grid(row=0, column=3, padx=3)

        status_frame = tk.LabelFrame(self.left_panel, text="执行状态")
        status_frame.pack(fill="x", padx=10, pady=(0, 5))

        ensure_goal_progress_style()
        self.progress = ttk.Progressbar(
            status_frame,
            variable=self.progress_var,
            maximum=100.0,
            style="Goal.Horizontal.TProgressbar",
        )
        self.progress.pack(fill="x", padx=10, pady=(8, 4))

        tk.Label(
            status_frame,
            textvariable=self.progress_message_var,
            anchor="w",
            justify="left",
        ).pack(fill="x", padx=10, pady=(0, 2))

        tk.Label(
            status_frame,
            textvariable=self.detail_message_var,
            anchor="w",
            justify="left",
        ).pack(fill="x", padx=10, pady=(0, 4))

        self.log_panel = CollapsibleLogPanel(self.left_panel, "日志")
        self.log_panel.pack(fill="both", padx=10, pady=(0, 6))
        self.log_text = self.log_panel.text

        self.no_trick_status_frame = tk.LabelFrame(self.right_panel, text="无巧手解密状态")
        self.no_trick_status_frame.pack(fill="both", expand=True, padx=5, pady=5)

        status_inner = tk.Frame(self.no_trick_status_frame)
        status_inner.pack(fill="x", padx=5, pady=5)

        tk.Label(
            status_inner,
            textvariable=self.no_trick_status_var,
            anchor="w",
            justify="left",
        ).pack(fill="x", anchor="w")

        self.no_trick_image_label = tk.Label(
            self.no_trick_status_frame,
            relief="sunken",
            bd=1,
            bg="#f8f8f8",
        )
        self.no_trick_image_label.pack(fill="both", expand=True, padx=10, pady=(0, 5))

        self.no_trick_progress = ttk.Progressbar(
            self.no_trick_status_frame,
            variable=self.no_trick_progress_var,
            maximum=100.0,
            mode="determinate",
        )
        self.no_trick_progress.pack(fill="x", padx=10, pady=(0, 8))

    # ---- 日志 & 状态 ----
    def log(self, msg: str):
        widget = getattr(self, "log_text", None)
        if widget is None:
            return

        def _append():
            append_formatted_log(widget, msg)

        post_to_main_thread(_append)

    def set_progress(self, percent: float):
        def _():
            self.progress_var.set(max(0.0, min(100.0, percent)))
        post_to_main_thread(_)

    def set_status(self, text: str):
        def _():
            self.progress_message_var.set(text)
        post_to_main_thread(_)

    def set_detail(self, text: str):
        def _():
            self.detail_message_var.set(text)
        post_to_main_thread(_)

    def reset_round_ui(self):
        self.set_progress(0.0)
        self.set_status("等待开始")
        self.set_detail("")

    # ---- 配置 / 热键 ----
    def capture_hotkey(self):
        if keyboard is None:
            messagebox.showerror("错误", "未安装 keyboard，无法录制热键。")
            return
        log(f"{self.LOG_PREFIX} 请按下想要设置的热键组合…")

        def worker():
            try:
                hk = keyboard.read_hotkey(suppress=False)
                self.hotkey_var.set(hk)
                log(f"{self.LOG_PREFIX} 捕获热键：{hk}")
            except Exception as exc:
                log(f"{self.LOG_PREFIX} 录制热键失败：{exc}")

        threading.Thread(target=worker, daemon=True).start()

    def save_cfg(self):
        section = self.cfg.setdefault("hs70_settings", {})
        section["hotkey"] = self.hotkey_var.get().strip()
        try:
            section["loop_count"] = int(self.loop_count_var.get().strip() or 0)
        except Exception:
            section["loop_count"] = 0
        section["auto_loop"] = bool(self.auto_loop_var.get())
        section["no_trick_decrypt"] = bool(self.no_trick_var.get())
        save_config(self.cfg)

    def _parse_loop_count(self):
        text = self.loop_count_var.get().strip()
        if not text:
            return 0
        try:
            count = int(text)
            if count < 0:
                raise ValueError
            return count
        except ValueError:
            messagebox.showwarning("提示", "循环次数请输入不小于 0 的整数。")
            return None

    # ---- 控制 ----
    def start_via_button(self):
        self.start_worker(auto_loop=self.auto_loop_var.get())

    def start_listen(self):
        if keyboard is None:
            messagebox.showerror("错误", "未安装 keyboard，无法使用热键监听。")
            return
        if not self.ensure_assets():
            return
        hk = self.hotkey_var.get().strip()
        if not hk:
            messagebox.showwarning("提示", "请先设置一个热键。")
            return

        worker_stop.clear()
        if self.hotkey_handle is not None:
            try:
                keyboard.remove_hotkey(self.hotkey_handle)
            except Exception:
                pass
            self.hotkey_handle = None

        def on_hotkey():
            log(f"{self.LOG_PREFIX} 检测到热键，开始执行一轮。")
            self.start_worker(auto_loop=self.auto_loop_var.get())

        try:
            self.hotkey_handle = keyboard.add_hotkey(hk, on_hotkey)
        except Exception as exc:
            messagebox.showerror("错误", f"注册热键失败：{exc}")
            return
        log(f"{self.LOG_PREFIX} 开始监听热键：{hk}")

    def stop_listen(self):
        worker_stop.set()
        if keyboard is not None and self.hotkey_handle is not None:
            try:
                keyboard.remove_hotkey(self.hotkey_handle)
            except Exception:
                pass
        self.hotkey_handle = None
        log(f"{self.LOG_PREFIX} 已停止监听，当前轮结束后退出。")

    def run_once(self):
        self.start_worker(auto_loop=False, loop_override=1)

    def start_worker(self, auto_loop: bool = None, loop_override: int = None):
        if not self.ensure_assets():
            return
        loop_count = self._parse_loop_count()
        if loop_count is None:
            return
        if loop_override is not None:
            loop_count = loop_override
        if auto_loop is None:
            auto_loop = self.auto_loop_var.get()
        if not auto_loop:
            loop_count = max(1, loop_override or 1)

        if not round_running_lock.acquire(blocking=False):
            messagebox.showwarning("提示", "当前已有其它任务在运行，请先停止后再试。")
            return

        worker_stop.clear()
        self.running = True
        self.reset_round_ui()
        self.set_status("准备开始…")
        self.entry_prepared = False

        def worker():
            try:
                self._worker_loop(auto_loop, loop_count)
            finally:
                self.running = False
                round_running_lock.release()

        threading.Thread(target=worker, daemon=True).start()

    def _worker_loop(self, auto_loop: bool, loop_limit: int):
        loops_done = 0
        first_round_pending = True
        try:
            while not worker_stop.is_set():
                loops_done += 1
                log(f"===== {self.LOG_PREFIX} 新一轮开始 =====")
                expect_more = auto_loop and (loop_limit == 0 or loops_done < loop_limit)
                was_first = first_round_pending
                success = self._run_round(first_round_pending, expect_more)
                if was_first:
                    first_round_pending = False
                if worker_stop.is_set():
                    break
                if not auto_loop:
                    break
                if loop_limit > 0 and loops_done >= loop_limit:
                    log(f"{self.LOG_PREFIX} 达到循环次数限制，结束执行。")
                    break
                if not success:
                    log(f"{self.LOG_PREFIX} 本轮未完成，重新开始下一轮。")
                else:
                    log(f"{self.LOG_PREFIX} 本轮完成，{self.BETWEEN_ROUNDS_DELAY:.0f} 秒后继续。")
                self.set_status("等待下一轮开始…")
                self.set_detail("")
                delay = self.BETWEEN_ROUNDS_DELAY
                while delay > 0 and not worker_stop.is_set():
                    time.sleep(min(0.1, delay))
                    delay -= 0.1
        except Exception as exc:
            log(f"{self.LOG_PREFIX} 后台线程异常：{exc}")
            traceback.print_exc()
        finally:
            self.on_worker_finished()

    def on_worker_finished(self):
        self._stop_no_trick_monitor()

        def _():
            self.progress_var.set(0.0)
            if not worker_stop.is_set():
                self.progress_message_var.set("就绪")
                self.detail_message_var.set("")
        post_to_main_thread(_)

    # ---- 资产检查 ----
    def ensure_assets(self) -> bool:
        missing = []

        def require(name: str, *, allow_templates: bool = False, desc: Optional[str] = None):
            path = hs_find_asset(name, allow_templates=allow_templates)
            if not path:
                label = desc or name
                location = "templates" if allow_templates else "HS"
                missing.append(f"未找到 {label}（请放在 {location} 目录或其子目录）")
            return path

        require(HS_INITIAL_MAP_TEMPLATE)
        require(HS_SETTINGS_TEMPLATE)
        require(HS_MORE_TEMPLATE)
        require(HS_RESET_TEMPLATE)
        require(HS_RESET_CONFIRM_TEMPLATE, allow_templates=True, desc="Q.png")
        require(HS_BRANCH_TEMPLATE)
        require(HS_TARGET_TEMPLATE)
        require(HS_WARNING_TEMPLATE)
        for macro in HS_FINAL_MACROS.values():
            require(macro)
        require(HS_COMPENSATE_MACRO)
        require(HS_FINE_TUNE_MACRO)

        for macro in HS_MAIN_MACROS:
            require(macro)
            calib = HS_CALIBRATION_MACROS.get(macro)
            if calib:
                require(calib)

        for info in HS_BRANCH_OPTIONS.values():
            template_names = info.get("templates") or ([info.get("template")] if info.get("template") else [])
            for template_name in template_names:
                if template_name:
                    require(template_name)
            require(info["macro"])
            require(info["calibrate"])

        for tpl in HS_SUBMAP_TEMPLATES.values():
            require(tpl)

        require(HS_START_TEMPLATE, allow_templates=True, desc="开始挑战.png")
        require(HS_RETRY_TEMPLATE, allow_templates=True, desc="再次进行.png")
        require(HS_TIP_IMAGE, desc="提示.png")

        if missing:
            messagebox.showerror("错误", "\n".join(missing))
            return False
        return True

    # ---- 核心逻辑 ----
    def _sleep_with_check(self, duration: float):
        end = time.time() + max(0.0, duration)
        while time.time() < end and not worker_stop.is_set():
            time.sleep(0.05)

    def _run_fine_tune_after_calibration(self):
        if worker_stop.is_set():
            return
        path = hs_find_asset(HS_FINE_TUNE_MACRO)
        if not path:
            log(f"{self.LOG_PREFIX} 缺少 {HS_FINE_TUNE_MACRO}，无法执行校准微调。")
            return
        log(f"{self.LOG_PREFIX} 校准完成，执行微调宏 {HS_FINE_TUNE_MACRO}。")
        self.set_detail("校准后执行微调宏…")
        previous_last = self.last_macro_name
        success = self._run_macro_with_decrypt(HS_FINE_TUNE_MACRO, require_decrypt=False)
        self.last_macro_name = previous_last
        if not success:
            log(f"{self.LOG_PREFIX} 微调宏 {HS_FINE_TUNE_MACRO} 执行失败。")

    def _run_round(self, first_round: bool, prepare_next_round: bool) -> bool:
        if worker_stop.is_set():
            return False

        self.last_macro_name = None
        self.target_detection_disabled = False
        if self.no_trick_var.get():
            self._prime_no_trick_controller()

        if not init_game_region():
            log(f"{self.LOG_PREFIX} 初始化游戏区域失败，本轮结束。")
            self.set_status("初始化失败")
            return False

        use_prepared_entry = False
        if self.entry_prepared:
            use_prepared_entry = True
            self.entry_prepared = False

        if not use_prepared_entry:
            if not self._enter_map(first_round):
                return False
        else:
            self.set_status("等待地图加载…")
            self.set_progress(15.0)
            self._sleep_with_check(self.ENTRY_DELAY)
            if worker_stop.is_set():
                return False

        if not self._confirm_initial_map():
            return False
        if worker_stop.is_set():
            return False

        if not self._perform_reset_sequence():
            return False
        if worker_stop.is_set():
            return False

        if not self._ensure_branch_a():
            return False
        if worker_stop.is_set():
            return False

        self.set_status("等待画面稳定…")
        self.set_detail("初始地图识别成功。")
        self._sleep_with_check(self.MAP_STABILIZE_DELAY)
        if worker_stop.is_set():
            return False

        if not self._run_macro_with_decrypt("mapa.json"):
            self.set_status("执行 mapa.json 失败")
            return False
        if worker_stop.is_set():
            return False
        self._after_decrypt_actions("mapa.json", detect_target=False, wait_seconds=0.0)
        if worker_stop.is_set():
            return False

        if not self._run_macro_with_decrypt("mapa-开锁1.json"):
            self.set_status("执行 mapa-开锁1.json 失败")
            return False
        if worker_stop.is_set():
            return False

        first_target = self._after_decrypt_actions("mapa-开锁1.json", detect_target=True)
        if worker_stop.is_set():
            return False

        resolved_after_first = False
        if first_target:
            log(f"{self.LOG_PREFIX} 第一段检测到 {HS_TARGET_TEMPLATE}，执行校准宏。")
            if not self._run_macro_with_decrypt("mapa-开锁1-校准.json"):
                self.set_status("执行 mapa-开锁1-校准.json 失败")
                return False
            if worker_stop.is_set():
                return False
            self._run_fine_tune_after_calibration()
            if worker_stop.is_set():
                return False
            resolved_after_first = True
            end_progress = self.PROGRESS_SEGMENTS.get("mapa-开锁2.json", (0.0, 0.0))[1]
            if end_progress:
                self.set_progress(end_progress)
            self.set_detail("首段校准完成，跳过第二段开锁。")
        else:
            log(f"{self.LOG_PREFIX} 第一段未识别到 {HS_TARGET_TEMPLATE}，继续执行第二段开锁。")

        if not resolved_after_first:
            if not self._run_macro_with_decrypt("mapa-开锁2.json"):
                self.set_status("执行 mapa-开锁2.json 失败")
                return False
            if worker_stop.is_set():
                return False

            second_target = self._after_decrypt_actions("mapa-开锁2.json", detect_target=True)
            if worker_stop.is_set():
                return False

            if second_target:
                log(f"{self.LOG_PREFIX} 第二段检测到 {HS_TARGET_TEMPLATE}，执行校准宏。")
                if not self._run_macro_with_decrypt("mapa-开锁2-校准.json"):
                    self.set_status("执行 mapa-开锁2-校准.json 失败")
                    return False
                if worker_stop.is_set():
                    return False
                self._run_fine_tune_after_calibration()
                if worker_stop.is_set():
                    return False
            else:
                log(
                    f"{self.LOG_PREFIX} 第二段未识别到 {HS_TARGET_TEMPLATE}，判定本局缺失目标图像，进入分支识别。"
                )
                self.target_detection_disabled = True
                if not self._wait_for_warning_signal():
                    self.set_status("未检测到警告提示，执行防卡死…")
                    self.set_detail("等待警告图标失败。")
                    self._perform_retry_recover()
                    return False
                if not self._scan_branch_variation():
                    return False
        if worker_stop.is_set():
            return False

        if not self._perform_reset_sequence(stage="完成校准后复位"):
            return False
        if worker_stop.is_set():
            return False

        submap_label = self._identify_submap()
        if not submap_label:
            return False
        if worker_stop.is_set():
            return False

        if not self._run_final_macro(submap_label):
            return False

        self.set_status("本轮完成，准备处理循环。")
        self.set_detail("撤离宏执行完成。")
        self.set_progress(100.0)

        if worker_stop.is_set():
            return True

        if prepare_next_round:
            if not self._prepare_next_round():
                return False

        return True

    def _enter_map(self, first_round: bool) -> bool:
        if worker_stop.is_set():
            return False

        if first_round:
            self.set_status("点击开始挑战（第一次）…")
            if not hs_wait_and_click_template(
                HS_START_TEMPLATE,
                f"{self.LOG_PREFIX} 进入：开始挑战（第一次）",
                timeout=25.0,
            ):
                self.set_status("未能点击开始挑战。")
                return False
            self.set_progress(5.0)
            self._sleep_with_check(self.ENTRY_DELAY)
            if worker_stop.is_set():
                return False

            self.set_status("点击开始挑战（第二次）…")
            if not hs_wait_and_click_template(
                HS_START_TEMPLATE,
                f"{self.LOG_PREFIX} 进入：开始挑战（第二次）",
                timeout=20.0,
            ):
                self.set_status("第二次点击开始挑战失败。")
                return False
            self.set_progress(10.0)
            self._sleep_with_check(self.ENTRY_DELAY)
            if worker_stop.is_set():
                return False
        else:
            retry_path = hs_find_asset(HS_RETRY_TEMPLATE, allow_templates=True)
            if not retry_path:
                log(f"{self.LOG_PREFIX} 缺少 {HS_RETRY_TEMPLATE}，无法点击再次进行。")
                self.set_status("缺少 再次进行.png")
                return False

            self.set_status("点击再次进行…")
            if not self._click_template_from_path(retry_path, "再次进行"):
                self.set_status("未能点击再次进行。")
                return False
            self._sleep_with_check(0.35)
            if worker_stop.is_set():
                return False

            self.set_status("点击开始挑战…")
            if not hs_wait_and_click_template(
                HS_START_TEMPLATE,
                f"{self.LOG_PREFIX} 再次进入：开始挑战",
                timeout=20.0,
            ):
                self.set_status("未能点击开始挑战。")
                return False
            self.set_progress(10.0)
            self._sleep_with_check(self.ENTRY_DELAY)
            if worker_stop.is_set():
                return False
        return True

    def _confirm_initial_map(self) -> bool:
        path = hs_find_asset(HS_INITIAL_MAP_TEMPLATE)
        if not path:
            log(f"{self.LOG_PREFIX} 缺少 {HS_INITIAL_MAP_TEMPLATE}")
            self.set_status("缺少初始地图模板")
            return False

        self.set_status("识别初始地图…")
        deadline = time.time() + 12.0
        best = 0.0
        while time.time() < deadline and not worker_stop.is_set():
            score, _, _ = match_template_from_path(path)
            best = max(best, score)
            log(f"{self.LOG_PREFIX} 初始地图匹配度 {score:.3f}")
            if score >= self.INITIAL_MAP_THRESHOLD:
                self.set_progress(15.0)
                self.set_detail("初始地图匹配成功。")
                return True
            time.sleep(0.3)

        log(
            f"{self.LOG_PREFIX} 初始地图识别失败，最高匹配度 {best:.3f} < {self.INITIAL_MAP_THRESHOLD:.2f}。"
        )
        self.set_status("初始地图识别失败")
        return False

    def _perform_reset_sequence(self, stage: str = "复位流程") -> bool:
        self.set_status(f"执行{stage}…")
        self.set_detail("ESC → 设置 → 更多 → 复位 → Q")

        try:
            if keyboard is not None:
                keyboard.press_and_release("esc")
            elif pyautogui is not None:
                pyautogui.press("esc")
        except Exception as exc:
            log(f"{self.LOG_PREFIX} 发送 ESC 失败：{exc}")
        time.sleep(0.3)

        settings_path = hs_find_asset(HS_SETTINGS_TEMPLATE)
        if not settings_path:
            log(f"{self.LOG_PREFIX} 缺少 {HS_SETTINGS_TEMPLATE}")
            self.set_status("缺少 设置.png")
            return False
        if not self._click_template_from_path(settings_path, "设置", threshold=0.65):
            self.set_status("点击 设置.png 失败")
            return False
        time.sleep(0.3)

        more_path = hs_find_asset(HS_MORE_TEMPLATE)
        if not more_path:
            log(f"{self.LOG_PREFIX} 缺少 {HS_MORE_TEMPLATE}")
            self.set_status("缺少 更多.png")
            return False
        if not self._click_template_from_path(more_path, "更多", threshold=0.65):
            self.set_status("点击 更多.png 失败")
            return False
        time.sleep(0.3)

        reset_path = hs_find_asset(HS_RESET_TEMPLATE)
        if not reset_path:
            log(f"{self.LOG_PREFIX} 缺少 {HS_RESET_TEMPLATE}")
            self.set_status("缺少 复位.png")
            return False
        if not self._click_template_from_path(reset_path, "复位", threshold=0.65):
            self.set_status("点击 复位.png 失败")
            return False
        time.sleep(0.3)

        confirm_path = hs_find_asset(
            HS_RESET_CONFIRM_TEMPLATE, allow_templates=True
        )
        if not confirm_path:
            log(f"{self.LOG_PREFIX} 缺少 {HS_RESET_CONFIRM_TEMPLATE}")
            self.set_status("缺少 Q.png")
            return False
        if not self._click_template_from_path(
            confirm_path, "Q", threshold=0.6
        ):
            self.set_status("点击 Q.png 失败")
            return False
        time.sleep(0.4)
        return True

    def _ensure_branch_a(self) -> bool:
        path = hs_find_asset(HS_BRANCH_TEMPLATE)
        if not path:
            log(f"{self.LOG_PREFIX} 缺少 {HS_BRANCH_TEMPLATE}")
            self.set_status("缺少 分支A.png")
            return False

        self.set_status("匹配分支A…")
        deadline = time.time() + self.BRANCH_A_TIMEOUT
        best = 0.0
        while time.time() < deadline and not worker_stop.is_set():
            score, _, _ = match_template_from_path(path)
            best = max(best, score)
            log(f"{self.LOG_PREFIX} 分支A 匹配度 {score:.3f}")
            if score >= self.BRANCH_A_THRESHOLD:
                self.set_detail("分支A匹配成功。")
                return True
            time.sleep(0.1)

        log(
            f"{self.LOG_PREFIX} 分支A 匹配失败，最高 {best:.3f} < {self.BRANCH_A_THRESHOLD:.2f}，执行防卡死。"
        )
        self.set_status("分支A匹配失败，执行防卡死…")
        self._perform_retry_recover()
        return False

    def _macro_requires_decrypt(self, macro_name: str) -> bool:
        """HS 模块下的宏均为键盘录制，不直接依赖解密回放。"""
        return False

    def _pump_decrypt_until(
        self,
        controller,
        predicate,
        timeout: Optional[float],
    ) -> bool:
        deadline = time.time() + timeout if timeout is not None else None
        while True:
            if predicate():
                return True
            if worker_stop.is_set():
                return False
            pause_time = controller.run_decrypt_if_needed()
            if predicate():
                return True
            if deadline is not None and time.time() >= deadline:
                return predicate()
            if not pause_time:
                time.sleep(0.05)

    def _run_macro_with_decrypt(
        self,
        macro_name: str,
        segment_key: Optional[str] = None,
        *,
        require_decrypt: Optional[bool] = None,
    ) -> bool:
        macro_path = hs_find_asset(macro_name)
        if not macro_path:
            log(f"{self.LOG_PREFIX} 缺少宏文件：{macro_name}")
            self.set_detail(f"缺少 {macro_name}")
            return False

        if require_decrypt is None:
            require_decrypt = self._macro_requires_decrypt(macro_name)

        key = segment_key or macro_name
        start, end = self.PROGRESS_SEGMENTS.get(key, (0.0, 0.0))
        self.set_status(f"执行 {macro_name}…")

        controller = self._start_no_trick_monitor()
        if require_decrypt and controller is None:
            log(
                f"{self.LOG_PREFIX} {macro_name} 需要无巧手解密，但当前未启用无巧手解密控制器。"
            )
            self.set_status("未启用无巧手解密")
            self.set_detail("未启用无巧手解密，无法监控解密图案。")
            return False
        executed_before = controller.executed_macros if controller is not None else 0

        def progress_cb(local):
            span = max(0.0, end - start)
            percent = start + span * max(0.0, min(1.0, local))
            self.set_progress(percent)

        decrypt_triggered = False
        decrypt_completed = False
        try:
            executed = play_macro(
                macro_path,
                f"{self.LOG_PREFIX} {macro_name}",
                0.0,
                0.0,
                interrupt_on_exit=False,
                interrupter=controller,
                progress_callback=progress_cb,
            )
        finally:
            if controller is not None:
                triggered = False
                completed = False
                if require_decrypt:
                    if controller.executed_macros > executed_before:
                        triggered = True
                    else:
                        log(
                            f"{self.LOG_PREFIX} 等待解密图案出现（最多 {self.DECRYPT_APPEAR_TIMEOUT:.1f} 秒）…"
                        )
                        triggered = self._pump_decrypt_until(
                            controller,
                            lambda: controller.executed_macros > executed_before,
                            self.DECRYPT_APPEAR_TIMEOUT,
                        )

                    if triggered:
                        log(f"{self.LOG_PREFIX} 解密已触发，等待解密宏执行完成…")
                        event = getattr(controller, "macro_done_event", None)
                        if event is not None:
                            predicate = event.is_set
                        else:
                            predicate = lambda: getattr(controller, "macro_executed", False)
                        completed = self._pump_decrypt_until(
                            controller,
                            predicate,
                            self.DECRYPT_COMPLETE_TIMEOUT,
                        )
                        if completed:
                            log(f"{self.LOG_PREFIX} 解密宏已完成。")
                            if self.DECRYPT_EXTRA_DELAY > 0:
                                log(
                                    f"{self.LOG_PREFIX} 解密完成后等待 {self.DECRYPT_EXTRA_DELAY:.1f} 秒稳定…"
                                )
                                wait_after_decrypt_delay(self.DECRYPT_EXTRA_DELAY)
                        else:
                            log(
                                f"{self.LOG_PREFIX} 解密宏未在 {self.DECRYPT_COMPLETE_TIMEOUT:.1f} 秒内完成。"
                            )
                    else:
                        log(
                            f"{self.LOG_PREFIX} 在 {self.DECRYPT_APPEAR_TIMEOUT:.1f} 秒内未检测到解密图案。"
                        )

                decrypt_triggered = triggered
                decrypt_completed = completed if triggered else False
                controller.stop()
                controller.finish_session()
                if self.no_trick_controller is controller:
                    self.no_trick_controller = None
                if not worker_stop.is_set():
                    self._prime_no_trick_controller()

        if require_decrypt:
            if not decrypt_triggered:
                log(
                    f"{self.LOG_PREFIX} {macro_name} 执行后未在 {self.DECRYPT_APPEAR_TIMEOUT:.1f} 秒内触发解密，执行防卡死。"
                )
                self.set_status("解密未触发，执行防卡死…")
                self.set_detail("2 秒内未检测到解密图案。")
                self._perform_retry_recover()
                return False
            if not decrypt_completed:
                log(
                    f"{self.LOG_PREFIX} {macro_name} 的解密宏未在 {self.DECRYPT_COMPLETE_TIMEOUT:.1f} 秒内完成，执行防卡死。"
                )
                self.set_status("解密超时，执行防卡死…")
                self.set_detail("解密宏执行超时。")
                self._perform_retry_recover()
                return False

        if executed:
            if end:
                self.set_progress(end)
            self.last_macro_name = macro_name
            return True
        return False

    def _after_decrypt_actions(
        self,
        macro_name: str,
        *,
        detect_target: bool = True,
        wait_seconds: Optional[float] = None,
    ) -> bool:
        wait_time = self.TARGET_PAUSE if wait_seconds is None else max(0.0, wait_seconds)
        start_time = time.time()
        target_found = False

        controller = None
        if self.no_trick_var.get() and not worker_stop.is_set():
            controller = self._start_no_trick_monitor()

        should_detect = detect_target and not self.target_detection_disabled
        target_path = None
        best_score = 0.0
        if should_detect:
            target_path = hs_find_asset(HS_TARGET_TEMPLATE)
            if not target_path:
                log(f"{self.LOG_PREFIX} 缺少 {HS_TARGET_TEMPLATE}，无法检测目标。")
                should_detect = False

        if wait_time > 0:
            if should_detect:
                log(
                    f"{self.LOG_PREFIX} {macro_name} 执行完成，暂停 {wait_time:.1f} 秒识别 {HS_TARGET_TEMPLATE}，阈值"
                    f" {self.TARGET_THRESHOLD:.2f}。"
                )
                self.set_detail("检测目标位置…")
            deadline = start_time + wait_time
            while time.time() < deadline and not worker_stop.is_set():
                if controller is not None:
                    pause_time = controller.run_decrypt_if_needed()
                    if pause_time:
                        continue
                if should_detect and target_path is not None:
                    score, _, _ = match_template_from_path(target_path)
                    best_score = max(best_score, score)
                    if score >= self.TARGET_THRESHOLD:
                        target_found = True
                        log(
                            f"{self.LOG_PREFIX} 检测到 {HS_TARGET_TEMPLATE}，得分 {score:.3f}。"
                        )
                        break
                time.sleep(self.TARGET_SCAN_INTERVAL)

        elapsed = time.time() - start_time
        remaining = max(0.0, wait_time - elapsed)
        if remaining > 0:
            self._sleep_with_check(remaining)

        if should_detect and not target_found:
            log(
                f"{self.LOG_PREFIX} 在 {wait_time:.1f} 秒内未识别到 {HS_TARGET_TEMPLATE}，"
                f"最高得分 {best_score:.3f}。"
            )

        return target_found

    def _wait_for_warning_signal(self) -> bool:
        path = hs_find_asset(HS_WARNING_TEMPLATE)
        if not path:
            log(f"{self.LOG_PREFIX} 缺少 {HS_WARNING_TEMPLATE}，无法等待警告提示。")
            return False

        self.set_status("等待警告提示…")
        self.set_detail("监控警告图标以进入分支判断。")

        controller = None
        if self.no_trick_var.get() and not worker_stop.is_set():
            controller = self._start_no_trick_monitor()

        deadline = time.time() + self.WARNING_TIMEOUT
        best_score = 0.0
        while time.time() < deadline and not worker_stop.is_set():
            if controller is not None:
                pause_time = controller.run_decrypt_if_needed()
                if pause_time:
                    continue
            score, _, _ = match_template_from_path(path)
            best_score = max(best_score, score)
            log(f"{self.LOG_PREFIX} 警告提示匹配度 {score:.3f}")
            if score >= self.WARNING_THRESHOLD:
                log(f"{self.LOG_PREFIX} 检测到警告提示，准备识别 2-3 / 2-4 分支。")
                wait_deadline = time.time() + self.WARNING_BRANCH_DELAY
                if self.WARNING_BRANCH_DELAY > 0:
                    log(
                        f"{self.LOG_PREFIX} 警告提示确认后延迟 {self.WARNING_BRANCH_DELAY:.1f} 秒再开始分支识别。"
                    )
                    self.set_detail("警告提示已确认，稍候开始分支识别…")
                    while time.time() < wait_deadline and not worker_stop.is_set():
                        if controller is not None:
                            pause = controller.run_decrypt_if_needed()
                            if pause:
                                continue
                        time.sleep(0.05)
                return not worker_stop.is_set()
            time.sleep(self.WARNING_SCAN_INTERVAL)

        log(
            f"{self.LOG_PREFIX} 未在 {self.WARNING_TIMEOUT:.1f} 秒内识别到 {HS_WARNING_TEMPLATE}，最高匹配度 {best_score:.3f}。"
        )
        return False

    def _scan_branch_variation(self) -> bool:
        self.set_status("识别二阶段分支…")
        self.set_detail("识别 2-3 / 2-4 路线。")

        controller = None
        if self.no_trick_var.get() and not worker_stop.is_set():
            controller = self._start_no_trick_monitor()

        end = time.time() + self.BRANCH_SCAN_DURATION
        template_sets: Dict[str, List[Tuple[str, str]]] = {}
        for label, info in HS_BRANCH_OPTIONS.items():
            names = info.get("templates") or ([info.get("template")] if info.get("template") else [])
            paths: List[Tuple[str, str]] = []
            for name in names:
                if not name:
                    continue
                tpl_path = hs_find_asset(name)
                if tpl_path:
                    paths.append((name, tpl_path))
                else:
                    log(f"{self.LOG_PREFIX} 缺少 {name}，无法用于分支 {label} 识别。")
            if paths:
                template_sets[label] = paths

        if not template_sets:
            log(f"{self.LOG_PREFIX} 未找到任何分支模板，无法识别分支。")
            self.set_status("缺少分支模板")
            return False

        best_label = None
        best_score = 0.0
        best_template = None
        while time.time() < end and not worker_stop.is_set():
            if controller is not None:
                pause_time = controller.run_decrypt_if_needed()
                if pause_time:
                    continue
            for label, entries in template_sets.items():
                label_best = 0.0
                label_template = None
                for template_name, tpl_path in entries:
                    score, _, _ = match_template_from_path(tpl_path)
                    log(
                        f"{self.LOG_PREFIX} 分支 {label} ({template_name}) 匹配度 {score:.3f}"
                    )
                    if score > label_best:
                        label_best = score
                        label_template = template_name
                if label_best > best_score:
                    best_score = label_best
                    best_label = label
                    best_template = label_template
            if best_score >= self.VARIATION_THRESHOLD:
                break
            time.sleep(self.BRANCH_SCAN_INTERVAL)

        if not best_label or best_score < self.VARIATION_THRESHOLD:
            log(
                f"{self.LOG_PREFIX} 分支识别失败，最高匹配度 {best_score:.3f} < {self.VARIATION_THRESHOLD:.2f}。"
            )
            self.set_status("分支识别失败")
            return False

        info = HS_BRANCH_OPTIONS[best_label]
        if best_template:
            detail_template = f"，模板 {best_template}"
        else:
            detail_template = ""
        self.set_detail(
            f"识别到 {best_label} 分支（匹配度 {best_score:.3f}{detail_template}），执行对应宏。"
        )
        if not self._run_macro_with_decrypt(info["macro"], segment_key="branch"):
            self.set_status(f"执行 {info['macro']} 失败")
            return False
        if worker_stop.is_set():
            return False
        self._after_decrypt_actions(
            info["macro"], detect_target=False, wait_seconds=self.BRANCH_DECRYPT_DELAY
        )
        if worker_stop.is_set():
            return False
        if not self._run_macro_with_decrypt(info["calibrate"], segment_key="branch"):
            self.set_status(f"执行 {info['calibrate']} 失败")
            return False
        if worker_stop.is_set():
            return False
        self._run_fine_tune_after_calibration()
        if worker_stop.is_set():
            return False
        return True

    def _identify_submap(self) -> Optional[str]:
        self.set_status("识别子地图…")
        template_paths = {
            label: hs_find_asset(tpl)
            for label, tpl in HS_SUBMAP_TEMPLATES.items()
        }
        deadline = time.time() + 12.0
        best_label = None
        best_score = 0.0
        while time.time() < deadline and not worker_stop.is_set():
            for label, path in template_paths.items():
                if not path:
                    continue
                score, _, _ = match_template_from_path(path)
                log(f"{self.LOG_PREFIX} 子地图 {label} 匹配度 {score:.3f}")
                if score > best_score:
                    best_score = score
                    best_label = label
            if best_label and best_score >= self.SUBMAP_THRESHOLD:
                break
            time.sleep(0.35)

        if best_label is None:
            log(f"{self.LOG_PREFIX} 子地图识别失败，最高匹配度 {best_score:.3f}。")
            self.set_status("子地图识别失败")
            return None

        self.set_detail(f"识别到 {best_label} 类地图，准备执行撤离宏。")
        self.set_progress(self.PROGRESS_SEGMENTS["final_reset"][0])
        return best_label

    def _run_final_macro(self, submap_label: str) -> bool:
        macro_name = HS_FINAL_MACROS.get(submap_label)
        if not macro_name:
            log(f"{self.LOG_PREFIX} 未配置 {submap_label} 类撤离宏。")
            self.set_status(f"缺少{submap_label}类撤离宏")
            return False

        if not self._run_macro_with_decrypt(macro_name, segment_key="final_macro"):
            self.set_status(f"执行 {macro_name} 失败")
            return False
        return True

    def _prepare_next_round(self) -> bool:
        retry_path = hs_find_asset(HS_RETRY_TEMPLATE, allow_templates=True)
        if not retry_path:
            log(f"{self.LOG_PREFIX} 未找到 {HS_RETRY_TEMPLATE}，无法准备下一轮。")
            return False

        self.set_status("识别再次进行，准备下一轮…")
        if self._attempt_retry_sequence(retry_path):
            return True

        compensate_path = hs_find_asset(HS_COMPENSATE_MACRO)
        if compensate_path:
            self.set_detail("未识别到再次进行，执行补偿宏…")
            executed = play_macro(
                compensate_path,
                f"{self.LOG_PREFIX} {HS_COMPENSATE_MACRO}",
                0.0,
                0.0,
                interrupt_on_exit=False,
            )
            if worker_stop.is_set():
                return False
            if executed:
                self.set_detail("补偿宏执行完成，重试识别再次进行…")
                self._sleep_with_check(0.3)
                if self._attempt_retry_sequence(retry_path):
                    return True
            else:
                self.set_detail("补偿宏执行失败，尝试防卡死…")
        else:
            log(f"{self.LOG_PREFIX} 缺少 {HS_COMPENSATE_MACRO}，无法执行补偿流程。")

        self.set_status("仍未识别到再次进行，执行防卡死…")
        self._perform_retry_recover()
        return False

    def _attempt_retry_sequence(self, retry_path: str) -> bool:
        if worker_stop.is_set():
            return False
        if not self._click_template_from_path(
            retry_path, "再次进行", attempts=self.RETRY_ATTEMPTS
        ):
            self.set_detail("未能点击 再次进行。")
            return False
        self._sleep_with_check(0.4)
        if worker_stop.is_set():
            return False
        if not hs_wait_and_click_template(
            HS_START_TEMPLATE,
            f"{self.LOG_PREFIX} 再次进入：开始挑战",
            timeout=20.0,
        ):
            self.set_detail("点击开始挑战失败。")
            return False
        self._sleep_with_check(self.ENTRY_DELAY)
        if worker_stop.is_set():
            return False
        self.entry_prepared = True
        return True

    def _perform_retry_recover(self):
        log(f"{self.LOG_PREFIX} 防卡死：ESC → G.png → Q.png")
        try:
            if keyboard is not None:
                keyboard.press_and_release("esc")
            elif pyautogui is not None:
                pyautogui.press("esc")
        except Exception as exc:
            log(f"{self.LOG_PREFIX} 发送 ESC 失败：{exc}")
        time.sleep(0.4)
        click_template("G.png", f"{self.LOG_PREFIX} 防卡死：点击 G.png", 0.6)
        time.sleep(0.4)
        click_template("Q.png", f"{self.LOG_PREFIX} 防卡死：点击 Q.png", 0.6)
        time.sleep(0.6)

    def _click_template_from_path(
        self,
        path: str,
        label: str,
        *,
        threshold: float = HS_CLICK_THRESHOLD,
        attempts: Optional[int] = None,
    ) -> bool:
        attempt_count = attempts if attempts is not None else self.CLICK_ATTEMPTS
        for attempt in range(1, attempt_count + 1):
            score, x, y = match_template_from_path(path)
            log(
                f"{self.LOG_PREFIX} {label} 匹配[{attempt}/{attempt_count}] {score:.3f}"
            )
            if score >= threshold and x is not None:
                if perform_click(x, y):
                    time.sleep(0.3)
                    return True
            time.sleep(0.2)
        return False

    # ---- 无巧手解密 ----
    def _on_no_trick_toggle(self):
        if not self.no_trick_var.get():
            self._stop_no_trick_monitor()
        self._update_no_trick_ui()

    def _update_no_trick_ui(self):
        if self.no_trick_var.get():
            self._ensure_no_trick_frame_visible()
            if self.no_trick_controller is None:
                self._set_no_trick_status_direct("等待识别解密图像…")
                self._set_no_trick_progress_value(0.0)
                self._set_no_trick_image(None)
        else:
            self._hide_no_trick_frame()
            self._set_no_trick_status_direct("未启用")
            self._set_no_trick_progress_value(0.0)
            self._set_no_trick_image(None)

    def _ensure_no_trick_frame_visible(self):
        if self.no_trick_status_frame is None:
            return
        if not self.no_trick_status_frame.winfo_ismapped():
            self.no_trick_status_frame.pack(fill="both", expand=True, padx=5, pady=5)

    def _hide_no_trick_frame(self):
        if self.no_trick_status_frame is None:
            return
        if self.no_trick_status_frame.winfo_manager():
            self.no_trick_status_frame.pack_forget()

    def _set_no_trick_status_direct(self, text: str):
        self.no_trick_status_var.set(text)

    def _set_no_trick_progress_value(self, percent: float):
        self.no_trick_progress_var.set(max(0.0, min(100.0, percent)))

    def _set_no_trick_image(self, photo):
        if self.no_trick_image_label is None:
            return
        if photo is None:
            self.no_trick_image_label.config(image="")
        else:
            self.no_trick_image_label.config(image=photo)
        self.no_trick_image_ref = photo

    def _load_no_trick_preview(self, path: str, max_size: int = 240):
        if not path or not os.path.exists(path):
            return None
        if Image is not None and ImageTk is not None:
            try:
                with Image.open(path) as pil_img:
                    pil_img = pil_img.convert("RGBA")
                    w, h = pil_img.size
                    scale = 1.0
                    if max(w, h) > max_size:
                        scale = max_size / max(w, h)
                        pil_img = pil_img.resize(
                            (
                                max(1, int(w * scale)),
                                max(1, int(h * scale)),
                            ),
                            Image.LANCZOS,
                        )
                    return ImageTk.PhotoImage(pil_img)
            except Exception:
                pass
        try:
            img = tk.PhotoImage(file=path)
        except Exception:
            return None
        w = max(img.width(), 1)
        h = max(img.height(), 1)
        factor = max(1, (max(w, h) + max_size - 1) // max_size)
        if factor > 1:
            img = img.subsample(factor, factor)
        return img

    def _clear_prepared_no_trick(self):
        if self.prepared_no_trick_controller is not None:
            try:
                self.prepared_no_trick_controller.stop()
                self.prepared_no_trick_controller.finish_session()
            except Exception:
                pass
            self.prepared_no_trick_controller = None

    def _prime_no_trick_controller(self):
        if not self.no_trick_var.get():
            self._clear_prepared_no_trick()
            return
        if self.no_trick_controller is not None or self.prepared_no_trick_controller is not None:
            return
        controller = NoTrickDecryptController(self, GAME_DIR)
        if controller.start():
            self.prepared_no_trick_controller = controller

    def _start_no_trick_monitor(self):
        if not self.no_trick_var.get():
            return None
        if self.no_trick_controller is not None:
            return self.no_trick_controller
        controller = None
        if self.prepared_no_trick_controller is not None:
            controller = self.prepared_no_trick_controller
            self.prepared_no_trick_controller = None
        else:
            controller = NoTrickDecryptController(self, GAME_DIR)
            if not controller.start():
                return None
        self.no_trick_controller = controller
        return controller

    def _stop_no_trick_monitor(self):
        controller = self.no_trick_controller
        if controller is not None:
            controller.stop()
            controller.finish_session()
            self.no_trick_controller = None
        self._clear_prepared_no_trick()

    def on_no_trick_unavailable(self, reason: str):
        def _():
            if not self.no_trick_var.get():
                return
            self._ensure_no_trick_frame_visible()
            self._set_no_trick_status_direct(f"无巧手解密不可用：{reason}。")
            self._set_no_trick_progress_value(0.0)
            self._set_no_trick_image(None)

        post_to_main_thread(_)

    def on_no_trick_no_templates(self, game_dir: str):
        def _():
            if not self.no_trick_var.get():
                return
            self._ensure_no_trick_frame_visible()
            self._set_no_trick_status_direct("Game 文件夹中未找到解密图像模板，请放置 1.png 等文件。")
            self._set_no_trick_progress_value(0.0)
            self._set_no_trick_image(None)

        post_to_main_thread(_)

    def on_no_trick_monitor_started(self, templates):
        def _():
            if not self.no_trick_var.get():
                return
            self._ensure_no_trick_frame_visible()
            self._set_no_trick_status_direct("等待识别解密图像…")
            self._set_no_trick_progress_value(0.0)
            self._set_no_trick_image(None)

        post_to_main_thread(_)

    def on_no_trick_detected(self, entry, score: float):
        """无巧手监控线程识别到图像时的回调。"""

        if not self.no_trick_var.get():
            return

        photo = self._load_no_trick_preview(entry.get("png_path"))

        def _():
            if not self.no_trick_var.get():
                return
            self._ensure_no_trick_frame_visible()
            base = os.path.splitext(entry.get("name", ""))[0]
            self._set_no_trick_status_direct(
                f"识别到 {base}.png，得分 {score:.3f}，等待回放解密宏…"
            )
            self._set_no_trick_image(photo)
            self._set_no_trick_progress_value(0.0)

        post_to_main_thread(_)

    def on_no_trick_macro_start(self, entry, score: float):
        def _():
            if not self.no_trick_var.get():
                return
            self._set_no_trick_progress_value(0.0)
            base = os.path.splitext(entry.get("name", ""))[0]
            self._set_no_trick_status_direct(f"识别到 {base}.png，开始回放…")

        post_to_main_thread(_)

    def on_no_trick_progress(self, progress: float):
        def _():
            if not self.no_trick_var.get():
                return
            self._set_no_trick_progress_value(progress * 100.0)

        post_to_main_thread(_)

    def on_no_trick_macro_complete(self, entry):
        def _():
            if not self.no_trick_var.get():
                return
            self._set_no_trick_status_direct("解密完成，恢复原宏执行。")
            self._set_no_trick_progress_value(100.0)

        post_to_main_thread(_)

    def on_no_trick_macro_missing(self, entry):
        def _():
            if not self.no_trick_var.get():
                return
            base = os.path.splitext(entry.get("name", ""))[0]
            self._set_no_trick_status_direct(
                f"未找到 {base}.json，跳过无巧手解密。"
            )
            self._set_no_trick_progress_value(0.0)

        post_to_main_thread(_)

    def on_no_trick_macro_preview(self, entry, path: str):
        photo = self._load_no_trick_preview(path)

        def _():
            if not self.no_trick_var.get():
                return
            self._set_no_trick_image(photo)

        post_to_main_thread(_)

    def on_no_trick_idle(self, remaining: float):
        def _():
            if not self.no_trick_var.get():
                return
            self._set_no_trick_status_direct(
                f"等待下一个解密图像（剩余 {remaining:.1f} 秒）"
            )

        post_to_main_thread(_)

    def on_no_trick_idle_complete(self):
        def _():
            if not self.no_trick_var.get():
                return
            self._set_no_trick_status_direct("本轮解密结束。")
            self._set_no_trick_progress_value(100.0)

        post_to_main_thread(_)

    def on_no_trick_session_finished(
        self, triggered: bool, macro_executed: bool, macro_missing: bool
    ):
        def _():
            if not self.no_trick_var.get():
                return
            if not triggered:
                self._set_no_trick_status_direct("本轮未识别到解密图像。")
                self._set_no_trick_progress_value(0.0)
                self._set_no_trick_image(None)
            elif macro_executed:
                self._set_no_trick_status_direct("解密完成，恢复原宏执行。")
                self._set_no_trick_progress_value(100.0)
            elif macro_missing:
                # 缺失宏时状态已在 on_no_trick_macro_missing 中提示
                pass

        post_to_main_thread(_)


class XP50AutoGUI:
    LOG_PREFIX = "[50XP]"
    MAP_STABILIZE_DELAY = 2.0
    BETWEEN_ROUNDS_DELAY = 3.0
    WAIT_POLL_INTERVAL = 0.3
    RETRY_MAX_ATTEMPTS = 20
    RETRY_CHECK_INTERVAL = 0.3
    PROGRESS_SEGMENTS = (
        (20.0, 45.0),
        (45.0, 65.0),
        (85.0, 100.0),
    )
    WAIT_PROGRESS_RANGE = (65.0, 85.0)

    def __init__(self, root, cfg):
        self.root = root
        self.cfg = cfg
        self.log_prefix = self.LOG_PREFIX

        settings = cfg.get("xp50_settings", {})
        self.hotkey_var = tk.StringVar(value=settings.get("hotkey", ""))
        self.wait_var = tk.StringVar(value=str(settings.get("wait_seconds", 120.0)))
        self.loop_count_var = tk.StringVar(value=str(settings.get("loop_count", 0)))
        self.auto_loop_var = tk.BooleanVar(value=bool(settings.get("auto_loop", True)))
        self.no_trick_var = tk.BooleanVar(value=bool(settings.get("no_trick_decrypt", True)))

        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_message_var = tk.StringVar(value="等待开始")
        self.wait_message_var = tk.StringVar(value="")
        self.serum_status_var = tk.StringVar(value="尚未识别血清完成")

        self.serum_image_ref = None
        self.no_trick_controller = None
        self.no_trick_status_var = tk.StringVar(value="未启用")
        self.no_trick_progress_var = tk.DoubleVar(value=0.0)
        self.no_trick_image_ref = None

        self.log_text = None
        self.progress = None
        self.serum_preview_label = None
        self.no_trick_status_frame = None
        self.no_trick_image_label = None
        self.no_trick_progress = None
        self.hotkey_handle = None
        self.running = False
        self.entry_prepared = False

        self._build_ui()
        self._update_no_trick_ui()

    # ---- UI 构建 ----
    def _build_ui(self):
        self.content_frame = tk.Frame(self.root)
        self.content_frame.pack(fill="both", expand=True)

        self.left_panel = tk.Frame(self.content_frame)
        self.left_panel.pack(side="left", fill="both", expand=True)

        notice_text = (
            "主控必须要用🐷猪 必须！ 划线无巧手解密的速度和精度已经调到最佳了 速度基本上和巧手"
            "一样快 。撤离的时候因为回放精度问题和爆炸怪 有时候会卡住 没办法尽力了理解一下 会自动执行退图重开"
            "的 大一学生摸鱼写的 有问题 群里 at 我 看到就修！"
        )
        tk.Label(
            self.left_panel,
            text=notice_text,
            fg="#d40000",
            justify="left",
            anchor="w",
            wraplength=520,
        ).pack(fill="x", padx=10, pady=(6, 0))

        self.right_panel = tk.Frame(self.content_frame)
        self.right_panel.pack(side="right", fill="y", padx=(5, 10), pady=5)

        top = tk.Frame(self.left_panel)
        top.pack(fill="x", padx=10, pady=5)
        top.grid_columnconfigure(4, weight=1)

        tk.Label(top, text="热键:").grid(row=0, column=0, sticky="e")
        tk.Entry(top, textvariable=self.hotkey_var, width=15).grid(row=0, column=1, sticky="w")
        ttk.Button(top, text="录制热键", command=self.capture_hotkey).grid(row=0, column=2, padx=3)
        ttk.Button(top, text="保存配置", command=self.save_cfg).grid(row=0, column=3, padx=3)

        tk.Label(top, text="局内等待(秒):").grid(row=1, column=0, sticky="e")
        tk.Entry(top, textvariable=self.wait_var, width=10).grid(row=1, column=1, sticky="w")
        tk.Checkbutton(top, text="自动循环", variable=self.auto_loop_var).grid(row=1, column=2, sticky="w")
        tk.Label(top, text="循环次数(0=无限):").grid(row=1, column=3, sticky="e")
        tk.Entry(top, textvariable=self.loop_count_var, width=8).grid(row=1, column=4, sticky="w")

        toggle = tk.Frame(self.left_panel)
        toggle.pack(fill="x", padx=10, pady=(0, 5))
        tk.Checkbutton(
            toggle,
            text="开启无巧手解密",
            variable=self.no_trick_var,
            command=self._on_no_trick_toggle,
        ).pack(anchor="w")

        status_frame = tk.LabelFrame(self.left_panel, text="执行状态")
        status_frame.pack(fill="x", padx=10, pady=(0, 5))

        ensure_goal_progress_style()
        self.progress = ttk.Progressbar(
            status_frame,
            variable=self.progress_var,
            maximum=100.0,
            style="Goal.Horizontal.TProgressbar",
        )
        self.progress.pack(fill="x", padx=10, pady=(8, 4))

        tk.Label(
            status_frame,
            textvariable=self.progress_message_var,
            anchor="w",
            justify="left",
        ).pack(fill="x", padx=10, pady=(0, 2))

        tk.Label(
            status_frame,
            textvariable=self.wait_message_var,
            anchor="w",
            justify="left",
        ).pack(fill="x", padx=10, pady=(0, 2))

        self.serum_preview_label = tk.Label(
            status_frame,
            relief="sunken",
            bd=1,
            bg="#f3f3f3",
            height=6,
            anchor="center",
            text="等待识别 血清完成.png",
        )
        self.serum_preview_label.pack(fill="x", padx=10, pady=(6, 4))

        tk.Label(
            status_frame,
            textvariable=self.serum_status_var,
            anchor="w",
            justify="left",
        ).pack(fill="x", padx=10, pady=(0, 6))

        self.log_panel = CollapsibleLogPanel(self.left_panel, "日志")
        self.log_panel.pack(fill="both", padx=10, pady=(0, 5))
        self.log_text = self.log_panel.text

        btns = tk.Frame(self.left_panel)
        btns.pack(padx=10, pady=5)
        ttk.Button(btns, text="开始执行", command=self.start_via_button).grid(row=0, column=0, padx=3)
        ttk.Button(btns, text="开始监听热键", command=self.start_listen).grid(row=0, column=1, padx=3)
        ttk.Button(btns, text="停止", command=self.stop_listen).grid(row=0, column=2, padx=3)
        ttk.Button(btns, text="只执行一轮", command=self.run_once).grid(row=0, column=3, padx=3)

        self.no_trick_status_frame = tk.LabelFrame(self.right_panel, text="无巧手解密状态")
        self.no_trick_status_frame.pack(fill="both", expand=True, padx=5, pady=5)

        status_inner = tk.Frame(self.no_trick_status_frame)
        status_inner.pack(fill="x", padx=5, pady=5)

        tk.Label(
            status_inner,
            textvariable=self.no_trick_status_var,
            anchor="w",
            justify="left",
        ).pack(fill="x", anchor="w")

        self.no_trick_image_label = tk.Label(
            self.no_trick_status_frame,
            relief="sunken",
            bd=1,
            bg="#f8f8f8",
        )
        self.no_trick_image_label.pack(fill="both", expand=True, padx=10, pady=(0, 5))

        self.no_trick_progress = ttk.Progressbar(
            self.no_trick_status_frame,
            variable=self.no_trick_progress_var,
            maximum=100.0,
            mode="determinate",
        )
        self.no_trick_progress.pack(fill="x", padx=10, pady=(0, 8))

    # ---- 日志 & 状态 ----
    def log(self, msg: str):
        widget = getattr(self, "log_text", None)
        if widget is None:
            return

        def _append():
            append_formatted_log(widget, msg)

        post_to_main_thread(_append)

    def on_global_progress(self, p: float):
        # 全局进度只用于主界面，这里忽略。
        return

    def set_progress(self, percent: float):
        def _():
            self.progress_var.set(max(0.0, min(100.0, percent)))
        post_to_main_thread(_)

    def set_status(self, text: str):
        def _():
            self.progress_message_var.set(text)
        post_to_main_thread(_)

    def set_wait_message(self, text: str):
        def _():
            self.wait_message_var.set(text)
        post_to_main_thread(_)

    def set_serum_status(self, text: str):
        def _():
            self.serum_status_var.set(text)
        post_to_main_thread(_)

    def show_serum_preview(self, photo, placeholder: str = "等待识别 血清完成.png"):
        def _():
            if self.serum_preview_label is None:
                return
            if photo is None:
                self.serum_preview_label.config(image="", text=placeholder)
            else:
                self.serum_preview_label.config(image=photo, text="")
            self.serum_image_ref = photo
        post_to_main_thread(_)

    def reset_round_ui(self):
        self.set_progress(0.0)
        self.set_status("等待开始")
        self.set_wait_message("")
        self.set_serum_status("尚未识别血清完成")
        self.show_serum_preview(None)

    # ---- 配置 / 热键 ----
    def capture_hotkey(self):
        if keyboard is None:
            messagebox.showerror("错误", "未安装 keyboard，无法录制热键。")
            return
        log(f"{self.LOG_PREFIX} 请按下想要设置的热键组合…")

        def worker():
            try:
                hk = keyboard.read_hotkey(suppress=False)
                self.hotkey_var.set(hk)
                log(f"{self.LOG_PREFIX} 捕获热键：{hk}")
            except Exception as exc:
                log(f"{self.LOG_PREFIX} 录制热键失败：{exc}")

        threading.Thread(target=worker, daemon=True).start()

    def _parse_wait_seconds(self):
        try:
            value = float(self.wait_var.get().strip())
            if value < 0:
                raise ValueError
            return value
        except ValueError:
            messagebox.showwarning("提示", "局内等待时间请输入不小于 0 的数字。")
            return None

    def _parse_loop_count(self):
        text = self.loop_count_var.get().strip()
        if not text:
            return 0
        try:
            count = int(text)
            if count < 0:
                raise ValueError
            return count
        except ValueError:
            messagebox.showwarning("提示", "循环次数请输入不小于 0 的整数。")
            return None

    def save_cfg(self):
        wait_seconds = self._parse_wait_seconds()
        if wait_seconds is None:
            return
        loop_count = self._parse_loop_count()
        if loop_count is None:
            return
        section = self.cfg.setdefault("xp50_settings", {})
        section["hotkey"] = self.hotkey_var.get().strip()
        section["wait_seconds"] = wait_seconds
        section["loop_count"] = loop_count
        section["auto_loop"] = bool(self.auto_loop_var.get())
        section["no_trick_decrypt"] = bool(self.no_trick_var.get())
        save_config(self.cfg)
        messagebox.showinfo("提示", "设置已保存。")

    def ensure_assets(self) -> bool:
        if pyautogui is None or cv2 is None or np is None:
            messagebox.showerror("错误", "缺少 pyautogui 或 opencv/numpy，无法执行副本。")
            return False
        if keyboard is None:
            messagebox.showerror("错误", "未安装 keyboard 模块，无法执行宏。")
            return False
        xp50_reset_asset_cache()
        missing = []

        start_path = xp50_find_asset(XP50_START_TEMPLATE, allow_templates=True)
        if not start_path:
            missing.append(
                f"未找到 {XP50_START_TEMPLATE}，请放置于 {XP50_DIR} 或 templates 的任意子目录内"
            )

        retry_path = xp50_find_asset(XP50_RETRY_TEMPLATE, allow_templates=True)
        if not retry_path:
            missing.append(
                f"未找到 {XP50_RETRY_TEMPLATE}，请放置于 {XP50_DIR} 或 templates 的任意子目录内"
            )

        serum_path = xp50_find_asset(XP50_SERUM_TEMPLATE)
        if not serum_path:
            missing.append(f"未找到 {XP50_SERUM_TEMPLATE}（期望位于 {XP50_DIR} 内）")

        for name in XP50_MAP_TEMPLATES.values():
            path = xp50_find_asset(name)
            if not path:
                missing.append(f"未找到 {name}（请放置在 {XP50_DIR} 目录或其子目录）")
        for files in XP50_MACRO_SEQUENCE.values():
            for fname in files:
                path = xp50_find_asset(fname)
                if not path:
                    missing.append(f"未找到 {fname}（请放置在 {XP50_DIR} 目录或其子目录）")
        if missing:
            msg = "\n".join(missing)
            messagebox.showerror("错误", f"以下文件缺失：\n{msg}")
            return False
        return True

    # ---- 控制 ----
    def start_via_button(self):
        """手动点击开始执行时进入主循环。"""

        self.start_worker(auto_loop=self.auto_loop_var.get())

    def start_listen(self):
        if keyboard is None:
            messagebox.showerror("错误", "未安装 keyboard，无法使用热键监听。")
            return
        if not self.ensure_assets():
            return
        hk = self.hotkey_var.get().strip()
        if not hk:
            messagebox.showwarning("提示", "请先设置一个热键。")
            return

        worker_stop.clear()
        if self.hotkey_handle is not None:
            try:
                keyboard.remove_hotkey(self.hotkey_handle)
            except Exception:
                pass
            self.hotkey_handle = None

        def on_hotkey():
            log(f"{self.LOG_PREFIX} 检测到热键，开始执行一轮。")
            self.start_worker(auto_loop=self.auto_loop_var.get())

        try:
            self.hotkey_handle = keyboard.add_hotkey(hk, on_hotkey)
        except Exception as exc:
            messagebox.showerror("错误", f"注册热键失败：{exc}")
            return
        log(f"{self.LOG_PREFIX} 开始监听热键：{hk}")

    def stop_listen(self):
        worker_stop.set()
        if keyboard is not None and self.hotkey_handle is not None:
            try:
                keyboard.remove_hotkey(self.hotkey_handle)
            except Exception:
                pass
        self.hotkey_handle = None
        log(f"{self.LOG_PREFIX} 已停止监听，当前轮结束后退出。")

    def start_worker(self, auto_loop: bool = None, loop_override: int = None):
        if not self.ensure_assets():
            return
        wait_seconds = self._parse_wait_seconds()
        if wait_seconds is None:
            return
        loop_count = self._parse_loop_count()
        if loop_count is None:
            return
        if loop_override is not None:
            loop_count = loop_override
        if auto_loop is None:
            auto_loop = self.auto_loop_var.get()
        if not auto_loop:
            loop_count = max(1, loop_override or 1)

        if not round_running_lock.acquire(blocking=False):
            messagebox.showwarning("提示", "当前已有其它任务在运行，请先停止后再试。")
            return

        worker_stop.clear()
        self.running = True
        self.reset_round_ui()
        self.set_status("准备开始…")
        self.entry_prepared = False

        def worker():
            try:
                self._worker_loop(wait_seconds, auto_loop, loop_count)
            finally:
                self.running = False
                round_running_lock.release()

        threading.Thread(target=worker, daemon=True).start()

    def run_once(self):
        self.start_worker(auto_loop=False, loop_override=1)

    def _worker_loop(self, wait_seconds: float, auto_loop: bool, loop_limit: int):
        loops_done = 0
        first_round_pending = True
        try:
            while not worker_stop.is_set():
                loops_done += 1
                log(f"===== {self.LOG_PREFIX} 新一轮开始 =====")
                expect_more = auto_loop and (loop_limit == 0 or loops_done < loop_limit)
                was_first_round = first_round_pending
                success = self._run_round(wait_seconds, first_round_pending, expect_more)
                if was_first_round:
                    first_round_pending = False
                if worker_stop.is_set():
                    break
                if not auto_loop:
                    break
                if loop_limit > 0 and loops_done >= loop_limit:
                    log(f"{self.LOG_PREFIX} 达到循环次数限制，结束执行。")
                    break
                if not success:
                    log(f"{self.LOG_PREFIX} 本轮未完成，重新开始下一轮。")
                else:
                    log(f"{self.LOG_PREFIX} 本轮完成，{self.BETWEEN_ROUNDS_DELAY:.0f} 秒后继续。")
                self.set_status("等待下一轮开始…")
                self.set_wait_message("")
                delay = self.BETWEEN_ROUNDS_DELAY
                step = 0.1
                while delay > 0 and not worker_stop.is_set():
                    time.sleep(min(step, delay))
                    delay -= step
        except Exception as exc:
            log(f"{self.LOG_PREFIX} 后台线程异常：{exc}")
            traceback.print_exc()
        finally:
            self.on_worker_finished()

    def on_worker_finished(self):
        self._stop_no_trick_monitor()

        def _():
            self.progress_var.set(0.0)
            if not worker_stop.is_set():
                self.progress_message_var.set("就绪")
            if self.hotkey_handle is None:
                self.set_wait_message("")
        post_to_main_thread(_)

    # ---- 核心逻辑 ----
    def _run_round(self, wait_seconds: float, first_round: bool, prepare_next_round: bool) -> bool:
        if worker_stop.is_set():
            return False

        if not init_game_region():
            log(f"{self.LOG_PREFIX} 初始化游戏区域失败，本轮结束。")
            self.set_status("初始化失败")
            return False

        use_prepared_entry = False
        if self.entry_prepared:
            use_prepared_entry = True
            self.entry_prepared = False

        if not use_prepared_entry:
            if first_round:
                self.set_status("点击开始挑战（第一次）…")
                if not xp50_wait_and_click(
                    XP50_START_TEMPLATE,
                    f"{self.LOG_PREFIX} 进入：开始挑战（第一次）",
                    25.0,
                    XP50_CLICK_THRESHOLD,
                ):
                    self.set_status("未能点击开始挑战。")
                    return False
                self.set_progress(5.0)
                time.sleep(0.4)
                if worker_stop.is_set():
                    return False

                self.set_status("点击开始挑战（第二次）…")
                if not xp50_wait_and_click(
                    XP50_START_TEMPLATE,
                    f"{self.LOG_PREFIX} 进入：开始挑战（第二次）",
                    20.0,
                    XP50_CLICK_THRESHOLD,
                ):
                    self.set_status("第二次点击开始挑战失败。")
                    return False
                self.set_progress(10.0)
                time.sleep(0.4)
                if worker_stop.is_set():
                    return False
            else:
                self.set_status("点击再次开始挑战…")
                if not xp50_wait_and_click(
                    XP50_START_TEMPLATE,
                    f"{self.LOG_PREFIX} 再次进入：开始挑战",
                    20.0,
                    XP50_CLICK_THRESHOLD,
                ):
                    self.set_status("未能点击再次开始挑战。")
                    return False
                self.set_progress(10.0)
                time.sleep(0.4)
                if worker_stop.is_set():
                    return False
        else:
            self.set_status("等待地图识别…")
            self.set_progress(15.0)
            time.sleep(0.4)
            if worker_stop.is_set():
                return False

        chosen = None
        scores = {label: 0.0 for label in XP50_MAP_TEMPLATES}
        map_paths = {}
        for label, tpl_name in XP50_MAP_TEMPLATES.items():
            path = xp50_find_asset(tpl_name)
            if not path:
                log(f"{self.LOG_PREFIX} 缺少地图模板：{tpl_name}")
                self.set_status("地图模板缺失")
                return False
            map_paths[label] = path
        self.set_status("识别地图模板…")
        deadline = time.time() + 12.0
        while time.time() < deadline and not worker_stop.is_set():
            for label, tpl_name in XP50_MAP_TEMPLATES.items():
                path = map_paths[label]
                score, _, _ = match_template_from_path(path)
                scores[label] = score
            log(
                f"{self.LOG_PREFIX} 地图匹配："
                f"mapa={scores['A']:.3f}，mapb={scores['B']:.3f}"
            )
            best_label = max(scores, key=scores.get)
            best_score = scores[best_label]
            if best_score >= XP50_MAP_THRESHOLD:
                chosen = best_label
                break
            time.sleep(0.4)

        if worker_stop.is_set():
            return False

        if chosen is None:
            log(f"{self.LOG_PREFIX} 地图识别失败，匹配度始终低于 {XP50_MAP_THRESHOLD:.2f}。")
            self.set_status("地图识别失败")
            return False

        map_label = f"map{chosen.lower()}"
        self.set_status(f"识别为 {map_label}，等待画面稳定…")
        self.set_progress(20.0)

        t0 = time.time()
        while time.time() - t0 < self.MAP_STABILIZE_DELAY and not worker_stop.is_set():
            time.sleep(0.1)

        if worker_stop.is_set():
            return False

        macros = XP50_MACRO_SEQUENCE.get(chosen, [])
        if len(macros) < 3:
            log(f"{self.LOG_PREFIX} {map_label} 的宏文件数量不足。")
            self.set_status("宏文件缺失")
            return False

        resolved_macros = []
        for macro_name in macros:
            macro_path = xp50_find_asset(macro_name)
            if not macro_path:
                log(f"{self.LOG_PREFIX} 缺少宏文件：{macro_name}")
                self.set_status("宏文件缺失")
                return False
            resolved_macros.append((macro_name, macro_path))

        for idx, (macro_name, macro_path) in enumerate(resolved_macros):
            segment = self.PROGRESS_SEGMENTS[min(idx, len(self.PROGRESS_SEGMENTS) - 1)]
            self.set_status(f"执行 {macro_name}…")
            executed = self._run_map_macro(macro_path, macro_name, *segment)
            if worker_stop.is_set():
                return False
            if not executed:
                self.set_status(f"执行 {macro_name} 失败")
                return False

            if idx == 1:
                self.set_progress(segment[1])
                success = self._wait_for_serum(wait_seconds)
                if worker_stop.is_set():
                    return False
                if not success:
                    log(f"{self.LOG_PREFIX} 等待血清完成超时，执行防卡死。")
                    self._on_serum_timeout()
                    emergency_recover()
                    if prepare_next_round and not worker_stop.is_set():
                        self._reenter_after_emergency()
                    return False

        self.set_status("执行撤离宏完成。")
        self.set_wait_message("")
        self.set_serum_status("撤离完成，等待下一轮。")
        self.set_progress(100.0)
        if worker_stop.is_set():
            return True

        if prepare_next_round:
            ready = self._prepare_next_round_after_retreat()
            if not ready:
                self.set_status("未能准备下一轮，已执行防卡死流程。")
                return False

        return True

    def _run_map_macro(self, macro_path: str, macro_name: str, start: float, end: float) -> bool:
        if not os.path.exists(macro_path):
            log(f"{self.LOG_PREFIX} 缺少宏文件：{macro_path}")
            return False

        controller = self._start_no_trick_monitor()

        def progress_cb(local):
            span = max(0.0, end - start)
            percent = start + span * max(0.0, min(1.0, local))
            self.set_progress(percent)

        try:
            executed = play_macro(
                macro_path,
                f"{self.LOG_PREFIX} {macro_name}",
                0.0,
                0.0,
                interrupt_on_exit=False,
                interrupter=controller,
                progress_callback=progress_cb,
            )
        finally:
            if controller is not None:
                controller.stop()
                controller.finish_session()
                if self.no_trick_controller is controller:
                    self.no_trick_controller = None

        if executed:
            self.set_progress(end)
        return bool(executed)

    def _prepare_next_round_after_retreat(self) -> bool:
        template_path = xp50_find_asset(XP50_RETRY_TEMPLATE, allow_templates=True)
        if not template_path:
            log(
                f"{self.LOG_PREFIX} 未找到 {XP50_RETRY_TEMPLATE}，跳过再次进行检测。"
            )
            return True

        self.set_status("识别再次进行，准备下一轮…")
        if self._ensure_retry_and_start(template_path, allow_recover=True):
            self.set_status("已准备下一轮，等待地图加载…")
            return True

        log(f"{self.LOG_PREFIX} 未能在防卡死后重新进入副本。")
        return False

    def _try_click_retry_button(self, template_path: str) -> bool:
        max_attempts = max(1, int(self.RETRY_MAX_ATTEMPTS))
        for attempt in range(1, max_attempts + 1):
            if worker_stop.is_set():
                return False
            score, x, y = match_template_from_path(template_path)
            log(
                f"{self.LOG_PREFIX} 再次进行检测[{attempt}/{max_attempts}] 匹配度 {score:.3f}"
            )
            if score >= XP50_CLICK_THRESHOLD and x is not None:
                if not perform_click(x, y):
                    log(f"{self.LOG_PREFIX} 点击 再次进行 ({x},{y}) 失败，重试。")
                    time.sleep(max(0.05, self.RETRY_CHECK_INTERVAL))
                    continue
                log(f"{self.LOG_PREFIX} 已点击 再次进行 ({x},{y})")
                time.sleep(0.4)
                return True
            time.sleep(max(0.05, self.RETRY_CHECK_INTERVAL))
        return False

    def _click_start_button(self, step_label: str, timeout: float = 20.0) -> bool:
        return xp50_wait_and_click(
            XP50_START_TEMPLATE,
            f"{self.LOG_PREFIX} {step_label}",
            timeout,
            XP50_CLICK_THRESHOLD,
        )

    def _click_retry_and_start(
        self,
        template_path: str,
        start_label: str = "再次进入：开始挑战",
        timeout: float = 20.0,
    ) -> bool:
        if not self._try_click_retry_button(template_path):
            return False
        if worker_stop.is_set():
            return False
        self.set_status("点击开始挑战，准备进入地图…")
        if not self._click_start_button(start_label, timeout=timeout):
            return False
        time.sleep(0.4)
        if worker_stop.is_set():
            return False
        self.entry_prepared = True
        return True

    def _ensure_retry_and_start(
        self,
        template_path: str,
        allow_recover: bool = True,
        start_label: str = "再次进入：开始挑战",
        timeout: float = 20.0,
    ) -> bool:
        if self._click_retry_and_start(template_path, start_label=start_label, timeout=timeout):
            return True
        if not allow_recover:
            return False
        self.set_status("多次未识别到再次进行，执行防卡死…")
        self._perform_retry_recover()
        if worker_stop.is_set():
            return False
        self.set_status("防卡死完成，重新识别再次进行…")
        return self._click_retry_and_start(
            template_path, start_label=start_label, timeout=timeout
        )

    def _reenter_after_emergency(self) -> bool:
        template_path = xp50_find_asset(XP50_RETRY_TEMPLATE, allow_templates=True)
        if not template_path:
            log(
                f"{self.LOG_PREFIX} 防卡死后未找到 {XP50_RETRY_TEMPLATE}，无法自动重新进入。"
            )
            return False

        self.set_status("防卡死完成，尝试重新进入…")
        success = self._ensure_retry_and_start(
            template_path, allow_recover=False, start_label="再次进入：开始挑战"
        )
        if success:
            self.set_status("重新进入成功，等待地图加载…")
        return success

    def _perform_retry_recover(self):
        log(f"{self.LOG_PREFIX} 防卡死：ESC → G.png → Q.png")
        try:
            if keyboard is not None:
                keyboard.press_and_release("esc")
            elif pyautogui is not None:
                pyautogui.press("esc")
        except Exception as exc:
            log(f"{self.LOG_PREFIX} 发送 ESC 失败：{exc}")
        time.sleep(0.4)
        click_template("G.png", f"{self.LOG_PREFIX} 防卡死：点击 G.png", 0.6)
        time.sleep(0.4)
        click_template("Q.png", f"{self.LOG_PREFIX} 防卡死：点击 Q.png", 0.6)
        time.sleep(0.6)

    def _wait_for_serum(self, wait_seconds: float) -> bool:
        template_path = xp50_find_asset(XP50_SERUM_TEMPLATE)
        if not template_path:
            log(f"{self.LOG_PREFIX} 缺少血清完成模板：{XP50_SERUM_TEMPLATE}")
            return False
        total = max(0.0, float(wait_seconds or 0.0))
        start_time = time.time()

        self.set_wait_message(
            "等待血清完成…" if total <= 0 else f"等待血清完成（剩余 {total:.1f} 秒）"
        )
        self.set_serum_status("尚未识别血清完成")
        self.show_serum_preview(None)

        while not worker_stop.is_set():
            elapsed = time.time() - start_time
            remaining = max(total - elapsed, 0.0)
            if total > 0:
                fraction = min(1.0, elapsed / total)
            else:
                fraction = 0.0
            self._update_wait_progress(fraction, remaining if total > 0 else None)

            score, _, _ = match_template_from_path(template_path)
            if score >= XP50_SERUM_THRESHOLD:
                self._on_serum_detected(template_path)
                return True

            if total > 0 and elapsed >= total:
                break
            time.sleep(self.WAIT_POLL_INTERVAL)

        return False

    def _update_wait_progress(self, fraction: float, remaining):
        start, end = self.WAIT_PROGRESS_RANGE
        percent = start + (end - start) * max(0.0, min(1.0, fraction))

        def _():
            self.progress_var.set(max(0.0, min(100.0, percent)))
            if remaining is None:
                self.wait_message_var.set("等待血清完成…")
            else:
                self.wait_message_var.set(f"等待血清完成（剩余 {remaining:.1f} 秒）")

        post_to_main_thread(_)

    def _on_serum_detected(self, template_path: str):
        self.set_progress(self.WAIT_PROGRESS_RANGE[1])
        self.set_wait_message("识别到血清完成，开始撤退。")
        self.set_serum_status("识别到血清完成，准备执行撤离宏。")
        photo = self._load_serum_preview(template_path)
        self.show_serum_preview(photo, placeholder="识别到血清完成")

    def _on_serum_timeout(self):
        self.set_wait_message("等待血清完成超时。")
        self.set_serum_status("超时未识别血清完成，已执行防卡死。")

    def _load_serum_preview(self, path: str, max_size: int = 280):
        if not path or not os.path.exists(path):
            return None
        if Image is not None and ImageTk is not None:
            try:
                with Image.open(path) as pil_img:
                    pil_img = pil_img.convert("RGBA")
                    w, h = pil_img.size
                    scale = 1.0
                    if max(w, h) > max_size:
                        scale = max_size / max(w, h)
                        pil_img = pil_img.resize(
                            (
                                max(1, int(w * scale)),
                                max(1, int(h * scale)),
                            ),
                            Image.LANCZOS,
                        )
                    return ImageTk.PhotoImage(pil_img)
            except Exception:
                pass
        try:
            img = tk.PhotoImage(file=path)
        except Exception:
            return None
        w = max(img.width(), 1)
        h = max(img.height(), 1)
        factor = max(1, (max(w, h) + max_size - 1) // max_size)
        if factor > 1:
            img = img.subsample(factor, factor)
        return img

    # ---- 无巧手解密 ----
    def _on_no_trick_toggle(self):
        if not self.no_trick_var.get():
            self._stop_no_trick_monitor()
        self._update_no_trick_ui()

    def _update_no_trick_ui(self):
        if self.no_trick_var.get():
            self._ensure_no_trick_frame_visible()
            if self.no_trick_controller is None:
                self._set_no_trick_status_direct("等待识别解密图像…")
                self._set_no_trick_progress_value(0.0)
                self._set_no_trick_image(None)
        else:
            self._hide_no_trick_frame()
            self._set_no_trick_status_direct("未启用")
            self._set_no_trick_progress_value(0.0)
            self._set_no_trick_image(None)

    def _ensure_no_trick_frame_visible(self):
        if self.no_trick_status_frame is None:
            return
        if not self.no_trick_status_frame.winfo_ismapped():
            self.no_trick_status_frame.pack(fill="both", expand=True, padx=5, pady=5)

    def _hide_no_trick_frame(self):
        if self.no_trick_status_frame is None:
            return
        if self.no_trick_status_frame.winfo_manager():
            self.no_trick_status_frame.pack_forget()

    def _set_no_trick_status_direct(self, text: str):
        self.no_trick_status_var.set(text)

    def _set_no_trick_progress_value(self, percent: float):
        self.no_trick_progress_var.set(max(0.0, min(100.0, percent)))

    def _set_no_trick_image(self, photo):
        if self.no_trick_image_label is None:
            return
        if photo is None:
            self.no_trick_image_label.config(image="")
        else:
            self.no_trick_image_label.config(image=photo)
        self.no_trick_image_ref = photo

    def _load_no_trick_preview(self, path: str, max_size: int = 240):
        if not path or not os.path.exists(path):
            return None
        if Image is not None and ImageTk is not None:
            try:
                with Image.open(path) as pil_img:
                    pil_img = pil_img.convert("RGBA")
                    w, h = pil_img.size
                    scale = 1.0
                    if max(w, h) > max_size:
                        scale = max_size / max(w, h)
                        pil_img = pil_img.resize(
                            (
                                max(1, int(w * scale)),
                                max(1, int(h * scale)),
                            ),
                            Image.LANCZOS,
                        )
                    return ImageTk.PhotoImage(pil_img)
            except Exception:
                pass
        try:
            img = tk.PhotoImage(file=path)
        except Exception:
            return None
        w = max(img.width(), 1)
        h = max(img.height(), 1)
        factor = max(1, (max(w, h) + max_size - 1) // max_size)
        if factor > 1:
            img = img.subsample(factor, factor)
        return img

    def _start_no_trick_monitor(self):
        if not self.no_trick_var.get():
            return None
        controller = NoTrickDecryptController(self, GAME_DIR)
        if controller.start():
            self.no_trick_controller = controller
            return controller
        return None

    def _stop_no_trick_monitor(self):
        controller = self.no_trick_controller
        if controller is not None:
            controller.stop()
            controller.finish_session()
            self.no_trick_controller = None

    def on_no_trick_unavailable(self, reason: str):
        def _():
            if not self.no_trick_var.get():
                return
            self._ensure_no_trick_frame_visible()
            self._set_no_trick_status_direct(f"无巧手解密不可用：{reason}。")
            self._set_no_trick_progress_value(0.0)
            self._set_no_trick_image(None)

        post_to_main_thread(_)

    def on_no_trick_no_templates(self, game_dir: str):
        def _():
            if not self.no_trick_var.get():
                return
            self._ensure_no_trick_frame_visible()
            self._set_no_trick_status_direct("Game 文件夹中未找到解密图像模板，请放置 1.png 等文件。")
            self._set_no_trick_progress_value(0.0)
            self._set_no_trick_image(None)

        post_to_main_thread(_)

    def on_no_trick_monitor_started(self, templates):
        total = len(templates)
        valid = sum(1 for t in templates if t.get("template") is not None)

        def _():
            if not self.no_trick_var.get():
                return
            self._ensure_no_trick_frame_visible()
            if valid <= 0:
                self._set_no_trick_status_direct("Game 模板加载失败，无法识别解密图像。")
            else:
                self._set_no_trick_status_direct(f"等待识别解密图像（共 {total} 张模板）…")
            self._set_no_trick_progress_value(0.0)
            self._set_no_trick_image(None)

        post_to_main_thread(_)

    def on_no_trick_detected(self, entry, score: float):
        def _():
            if not self.no_trick_var.get():
                return
            self._ensure_no_trick_frame_visible()
            name = entry.get("name", "")
            self._set_no_trick_status_direct(f"识别到解密图像 - {name}，正在解密…")
            photo = self._load_no_trick_preview(entry.get("png_path"))
            self._set_no_trick_image(photo)
            self._set_no_trick_progress_value(0.0)

        post_to_main_thread(_)

    def on_no_trick_macro_start(self, entry, score: float):
        def _():
            if not self.no_trick_var.get():
                return
            self._set_no_trick_progress_value(0.0)

        post_to_main_thread(_)

    def on_no_trick_progress(self, progress: float):
        def _():
            if not self.no_trick_var.get():
                return
            self._set_no_trick_progress_value(progress * 100.0)

        post_to_main_thread(_)

    def on_no_trick_macro_complete(self, entry):
        def _():
            if not self.no_trick_var.get():
                return
            self._set_no_trick_status_direct("解密完成，恢复原宏执行。")
            self._set_no_trick_progress_value(100.0)

        post_to_main_thread(_)

    def on_no_trick_macro_missing(self, entry):
        def _():
            if not self.no_trick_var.get():
                return
            base = os.path.splitext(entry.get("name", ""))[0]
            self._set_no_trick_status_direct(f"未找到 {base}.json，跳过无巧手解密。")
            self._set_no_trick_progress_value(0.0)
            self._set_no_trick_image(None)

        post_to_main_thread(_)

    def on_no_trick_session_finished(self, triggered: bool, macro_executed: bool, macro_missing: bool):
        def _():
            if not self.no_trick_var.get():
                return
            if not triggered:
                self._set_no_trick_status_direct("本轮未识别到解密图像。")
                self._set_no_trick_progress_value(0.0)
                self._set_no_trick_image(None)
            elif macro_executed:
                self._set_no_trick_status_direct("解密流程完成，继续执行原宏。")
                self._set_no_trick_progress_value(100.0)

        post_to_main_thread(_)

# ======================================================================
#  main
# ======================================================================
def ensure_manual_firework_service():
    global manual_firework_service
    if manual_firework_service is None:
        manual_firework_service = StandaloneDecryptService(
            FireworkNoTrickController,
            GAME_SQ_DIR,
            "赛琪无巧手解密",
            cooldown_seconds=60.0,
        )
    return manual_firework_service


def ensure_manual_line_service():
    global manual_line_service
    if manual_line_service is None:
        manual_line_service = StandaloneDecryptService(
            NoTrickDecryptController,
            GAME_DIR,
            "无巧手解密",
        )
    return manual_line_service


def set_firework_no_trick_enabled(enabled: bool):
    if app is None:
        return
    var = getattr(app, "no_trick_var", None)
    if var is None:
        return
    try:
        current = bool(var.get())
    except Exception:
        current = None
    if current == enabled:
        return
    try:
        var.set(enabled)
    except Exception:
        return
    handler = getattr(app, "_on_no_trick_toggle", None)
    if callable(handler):
        try:
            handler()
        except Exception:
            pass


def set_line_no_trick_enabled(enabled: bool):
    targets = []
    if xp50_app is not None:
        targets.append(xp50_app)
    if hs70_app is not None:
        targets.append(hs70_app)
    targets.extend(fragment_apps)
    for gui in targets:
        var = getattr(gui, "no_trick_var", None)
        if var is None:
            continue
        try:
            current = bool(var.get())
        except Exception:
            current = None
        if current == enabled:
            continue
        try:
            var.set(enabled)
        except Exception:
            continue
        handler = getattr(gui, "_on_no_trick_toggle", None)
        if callable(handler):
            try:
                handler()
            except Exception:
                pass


def _manual_any_active():
    if manual_firework_var is not None and manual_firework_var.get():
        return True
    if manual_line_var is not None and manual_line_var.get():
        return True
    return False


def on_international_support_toggle():
    global config_data
    if international_support_var is None:
        return
    enabled = bool(international_support_var.get())
    set_international_support_enabled(enabled)
    if config_data is not None:
        config_data["support_international"] = enabled
        save_config(config_data)
    hint = get_window_name_hint()
    log(f"国际服识别支持已{'开启' if enabled else '关闭'}（当前匹配 {hint}）。")


def collapse_for_manual_mode():
    global manual_collapse_active, manual_original_geometry
    if root_window is None or toolbar_frame is None:
        return
    if manual_collapse_active:
        return
    root_window.update_idletasks()
    manual_original_geometry = root_window.geometry()
    if manual_previous_minsize is not None:
        root_window.minsize(200, 120)
    width = max(toolbar_frame.winfo_reqwidth() + 40, 360)
    height = max(toolbar_frame.winfo_reqheight() + 20, 140)
    try:
        win = find_game_window()
    except Exception:
        win = None
    if win is not None:
        x = int(win.left)
        y = int(max(0, win.top - height))
    else:
        x = max(root_window.winfo_rootx(), 0)
        y = max(root_window.winfo_rooty() - height, 0)
    root_window.geometry(f"{width}x{height}+{x}+{y}")
    manual_collapse_active = True
    if manual_expand_button is not None:
        try:
            manual_expand_button.state(["!disabled"])
        except Exception:
            pass


def restore_main_window():
    global manual_collapse_active
    if root_window is None:
        return
    if manual_original_geometry:
        root_window.geometry(manual_original_geometry)
    if manual_previous_minsize is not None:
        root_window.minsize(*manual_previous_minsize)
    manual_collapse_active = False
    if manual_expand_button is not None:
        try:
            manual_expand_button.state(["disabled"])
        except Exception:
            pass


def on_manual_firework_toggle():
    if manual_firework_var is None:
        return
    enabled = bool(manual_firework_var.get())
    if enabled:
        ensure_manual_firework_service().start()
        set_firework_no_trick_enabled(True)
        collapse_for_manual_mode()
    else:
        if manual_firework_service is not None:
            manual_firework_service.stop()
        if not _manual_any_active() and manual_collapse_active:
            restore_main_window()


def on_manual_line_toggle():
    if manual_line_var is None:
        return
    enabled = bool(manual_line_var.get())
    if enabled:
        ensure_manual_line_service().start()
        set_line_no_trick_enabled(True)
        collapse_for_manual_mode()
    else:
        if manual_line_service is not None:
            manual_line_service.stop()
        if not _manual_any_active() and manual_collapse_active:
            restore_main_window()




class CustomScriptGUI:
    """可视化自定义脚本编辑器，允许用户组合模块并执行。"""

    CANVAS_HEIGHT = 220
    BOX_WIDTH = 220
    BOX_HEIGHT = 60
    BOX_GAP = 40

    def __init__(self, root):
        self.root = root
        self.modules: List[ScriptNode] = []
        self.worker_thread = None
        self.library_modules = sorted(
            CUSTOM_MODULE_REGISTRY.values(), key=lambda m: m.display_name
        )

        self.progress_var = tk.DoubleVar(value=0.0)
        self.status_var = tk.StringVar(value='未运行')
        self.library_desc_var = tk.StringVar(value='请选择一个模块查看说明。')

        self._build_ui()

    def _build_ui(self):
        self.content = tk.Frame(self.root)
        self.content.pack(fill='both', expand=True)

        controls = tk.Frame(self.content)
        controls.pack(fill='x', padx=10, pady=(10, 5))

        self.start_btn = ttk.Button(controls, text='开始执行', command=self.start_script)
        self.start_btn.pack(side='left', padx=(0, 6))
        self.stop_btn = ttk.Button(
            controls, text='停止', command=self.stop_script, state='disabled'
        )
        self.stop_btn.pack(side='left', padx=6)
        ttk.Label(controls, textvariable=self.status_var).pack(side='left', padx=6)

        progress_frame = tk.Frame(self.content)
        progress_frame.pack(fill='x', padx=10, pady=(0, 5))
        ensure_goal_progress_style()
        self.progress_bar = ttk.Progressbar(
            progress_frame,
            variable=self.progress_var,
            maximum=100.0,
            style='Goal.Horizontal.TProgressbar',
        )
        self.progress_bar.pack(fill='x')

        body = tk.Frame(self.content)
        body.pack(fill='both', expand=True, padx=10, pady=5)

        left = tk.Frame(body)
        left.pack(side='left', fill='both', expand=True)

        self.canvas = tk.Canvas(left, height=self.CANVAS_HEIGHT, bg='#fafafa')
        self.canvas.pack(fill='x', pady=(0, 6))

        columns = ('index', 'module', 'summary')
        self.tree = ttk.Treeview(
            left,
            columns=columns,
            show='headings',
            selectmode='browse',
            height=8,
        )
        self.tree.heading('index', text='#')
        self.tree.heading('module', text='模块')
        self.tree.heading('summary', text='概述')
        self.tree.column('index', width=40, anchor='center')
        self.tree.column('module', width=160, anchor='w')
        self.tree.column('summary', width=320, anchor='w')
        self.tree.pack(fill='both', expand=True)

        btns = tk.Frame(left)
        btns.pack(fill='x', pady=5)
        ttk.Button(btns, text='添加模块', command=self.add_selected_module).pack(
            side='left', padx=3
        )
        ttk.Button(btns, text='移除', command=self.remove_selected_module).pack(
            side='left', padx=3
        )
        ttk.Button(btns, text='上移', command=lambda: self.move_selected_module(-1)).pack(
            side='left', padx=3
        )
        ttk.Button(btns, text='下移', command=lambda: self.move_selected_module(1)).pack(
            side='left', padx=3
        )
        ttk.Button(btns, text='编辑配置', command=self.edit_selected_module).pack(
            side='left', padx=3
        )

        right = tk.Frame(body, width=240)
        right.pack(side='right', fill='y', padx=(10, 0))
        tk.Label(right, text='模块库').pack(anchor='w')
        self.library_list = tk.Listbox(right, height=12)
        self.library_list.pack(fill='both', expand=True)
        for mod in self.library_modules:
            self.library_list.insert('end', mod.display_name)
        self.library_list.bind('<<ListboxSelect>>', self._on_library_select)
        ttk.Label(right, textvariable=self.library_desc_var, wraplength=220).pack(
            fill='x', pady=(6, 0)
        )

        self.log_panel = CollapsibleLogPanel(self.content, '日志')
        self.log_panel.pack(fill='both', expand=True, padx=10, pady=(0, 10))
        self.log_text = self.log_panel.text

        self._render_diagram()

    def _on_library_select(self, event=None):
        module = self._get_library_module()
        if module is None:
            self.library_desc_var.set('请选择一个模块查看说明。')
        else:
            self.library_desc_var.set(module.description or '该模块暂无说明。')

    def _get_library_module(self):
        selection = self.library_list.curselection()
        if not selection:
            return None
        idx = selection[0]
        if 0 <= idx < len(self.library_modules):
            return self.library_modules[idx]
        return None

    def add_selected_module(self):
        module = self._get_library_module()
        if module is None:
            messagebox.showinfo('提示', '请在右侧选择一个模块。')
            return
        node = create_script_node(module.module_type)
        self.modules.append(node)
        self._refresh_module_list(select_id=node.node_id)
        self.queue_log(f'添加模块：{module.display_name}')

    def remove_selected_module(self):
        idx = self._get_selected_index()
        if idx is None:
            messagebox.showinfo('提示', '请先选择要移除的模块。')
            return
        node = self.modules.pop(idx)
        self._refresh_module_list()
        self.queue_log(f'已移除模块：{node.module.display_name}')

    def move_selected_module(self, delta: int):
        idx = self._get_selected_index()
        if idx is None:
            return
        target = idx + delta
        if target < 0 or target >= len(self.modules):
            return
        self.modules[idx], self.modules[target] = self.modules[target], self.modules[idx]
        self._refresh_module_list(select_id=self.modules[target].node_id)

    def edit_selected_module(self):
        idx = self._get_selected_index()
        if idx is None:
            messagebox.showinfo('提示', '请先选择要编辑的模块。')
            return
        node = self.modules[idx]
        top = tk.Toplevel(self.root)
        top.title(f'编辑模块：{node.module.display_name}')
        top.geometry('620x520')
        tk.Label(
            top,
            text='在下方编辑模块配置（JSON 格式），可自定义模板、宏与步骤。',
        ).pack(anchor='w', padx=10, pady=(10, 4))
        text_widget = tk.Text(top, wrap='none')
        text_widget.pack(fill='both', expand=True, padx=10, pady=(0, 10))
        text_widget.insert('1.0', json.dumps(node.config, indent=2, ensure_ascii=False))

        def browse_file():
            path = filedialog.askopenfilename(title='选择文件')
            if path:
                text_widget.insert('insert', path)

        btns = tk.Frame(top)
        btns.pack(pady=(0, 10))
        ttk.Button(btns, text='插入文件路径', command=browse_file).pack(side='left', padx=5)

        def save_and_close():
            raw = text_widget.get('1.0', 'end').strip()
            try:
                data = json.loads(raw)
            except json.JSONDecodeError as exc:
                messagebox.showerror('错误', f'JSON 解析失败：{exc}')
                return
            node.config = data
            self._refresh_module_list(select_id=node.node_id)
            top.destroy()

        ttk.Button(btns, text='保存', command=save_and_close).pack(side='left', padx=5)
        ttk.Button(btns, text='取消', command=top.destroy).pack(side='left', padx=5)

    def _get_selected_index(self):
        selection = self.tree.selection()
        if not selection:
            return None
        item_id = selection[0]
        for idx, node in enumerate(self.modules):
            if str(node.node_id) == item_id:
                return idx
        return None

    def _refresh_module_list(self, select_id=None):
        for item in self.tree.get_children():
            self.tree.delete(item)
        for idx, node in enumerate(self.modules, start=1):
            self.tree.insert(
                '',
                'end',
                iid=str(node.node_id),
                values=(idx, node.module.display_name, node.module.summary(node.config)),
            )
        if select_id is not None:
            self.tree.selection_set(str(select_id))
        self._render_diagram()

    def _render_diagram(self):
        self.canvas.delete('all')
        x = 20
        for idx, node in enumerate(self.modules):
            top = 20 + idx * (self.BOX_HEIGHT + self.BOX_GAP)
            self.canvas.create_rectangle(
                x,
                top,
                x + self.BOX_WIDTH,
                top + self.BOX_HEIGHT,
                fill='#fdfdfd',
                outline='#b0b0b0',
            )
            self.canvas.create_text(
                x + self.BOX_WIDTH / 2,
                top + 18,
                text=f"{idx + 1}. {node.module.display_name}",
                font=('Microsoft YaHei', 10, 'bold'),
            )
            summary = node.module.summary(node.config)
            self.canvas.create_text(
                x + self.BOX_WIDTH / 2,
                top + 40,
                text=summary[:32],
                font=('Microsoft YaHei', 9),
            )
            if idx < len(self.modules) - 1:
                self.canvas.create_line(
                    x + self.BOX_WIDTH,
                    top + self.BOX_HEIGHT / 2,
                    x + self.BOX_WIDTH + 40,
                    top + self.BOX_HEIGHT / 2,
                    arrow=tk.LAST,
                    width=2,
                )

        total_height = len(self.modules) * (self.BOX_HEIGHT + self.BOX_GAP) + 40
        self.canvas.config(
            scrollregion=(0, 0, self.BOX_WIDTH + 80, max(total_height, self.CANVAS_HEIGHT))
        )

    def start_script(self):
        if self.worker_thread and self.worker_thread.is_alive():
            return
        if not self.modules:
            messagebox.showwarning('提示', '请先添加至少一个模块。')
            return
        if not round_running_lock.acquire(blocking=False):
            messagebox.showwarning('提示', '当前已有其它任务在运行，请稍后再试。')
            return

        worker_stop.clear()
        self.set_status('运行中…')
        self.update_progress(0.0)
        self.start_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        self.queue_log('开始执行自定义脚本…')

        module_snapshot = [copy.deepcopy(node) for node in self.modules]

        def worker():
            success = True
            context = CustomScriptContext(self)
            loops = 0
            try:
                while not context.should_stop():
                    for node in module_snapshot:
                        if context.should_stop():
                            break
                        config = copy.deepcopy(node.config)
                        try:
                            result = node.module.execute(context, config)
                        except Exception as exc:
                            traceback.print_exc()
                            context.fail(f"{node.module.display_name} 执行异常：{exc}")
                            success = False
                            break
                        if not result and context.last_result is False:
                            success = False
                            break
                    if context.should_stop() or not success:
                        break
                    if context.loop_enabled:
                        loops += 1
                        if context.loop_limit and loops >= context.loop_limit:
                            break
                        context.log(f'循环完成 {loops} 次，准备下一轮…')
                        continue
                    break
            except Exception as exc:
                traceback.print_exc()
                context.fail(f'脚本运行异常：{exc}')
                success = False
            finally:
                worker_stop.clear()
                round_running_lock.release()
                post_to_main_thread(lambda: self._on_script_finished(success))

        self.worker_thread = threading.Thread(target=worker, daemon=True)
        self.worker_thread.start()

    def stop_script(self):
        if self.worker_thread and self.worker_thread.is_alive():
            worker_stop.set()
            self.queue_log('已请求停止脚本，等待当前模块结束…')

    def _on_script_finished(self, success: bool):
        self.worker_thread = None
        self.start_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        self.update_progress(0.0)
        self.set_status('已完成' if success else '已停止')
        self.queue_log(f"自定义脚本{'已完成' if success else '已停止'}。")

    def queue_log(self, message: str):
        def _append():
            widget = getattr(self, 'log_text', None)
            if widget is None:
                return
            append_formatted_log(widget, message)

        post_to_main_thread(_append)

    def update_progress(self, value: float):
        value = max(0.0, min(100.0, float(value)))
        post_to_main_thread(lambda: self.progress_var.set(value))

    def set_status(self, text: str):
        post_to_main_thread(lambda: self.status_var.set(text))


def main():
    global app, uid_mask_manager, xp50_app, hs70_app, root_window, toolbar_frame
    global manual_firework_var, manual_line_var, manual_expand_button
    global manual_previous_minsize, international_support_var, config_data
    cfg = load_config()
    config_data = cfg
    set_international_support_enabled(cfg.get("support_international", True))

    root = tk.Tk()
    root.title("苏苏多功能自动化工具")
    start_ui_dispatch_loop(root)
    uid_mask_manager = UIDMaskManager(root)
    root_window = root

    # 简单自适应分辨率 + DPI 缩放
    sw = root.winfo_screenwidth()
    sh = root.winfo_screenheight()

    try:
        base_h = 1080
        dpi_scale = max(0.85, min(2.0, sh / base_h))
        root.tk.call("tk", "scaling", dpi_scale)
    except Exception:
        pass

    base_w, base_h = 1400, 950
    margin_ratio = 0.95
    avail_w = max(int(sw * margin_ratio), int(sw * 0.75))
    avail_h = max(int(sh * margin_ratio), int(sh * 0.75))
    if avail_w <= 0 or avail_h <= 0:
        avail_w, avail_h = sw, sh

    scale_ratio = min(avail_w / base_w, avail_h / base_h)
    if scale_ratio <= 0:
        scale_ratio = 1.0
    scale_ratio = max(0.85, min(scale_ratio, 1.9))

    win_w = int(base_w * scale_ratio)
    win_h = int(base_h * scale_ratio)
    win_w = min(win_w, avail_w, sw)
    win_h = min(win_h, avail_h, sh)

    pos_x = max((sw - win_w) // 2, 0)
    pos_y = max((sh - win_h) // 2, 0)
    root.geometry(f"{win_w}x{win_h}+{pos_x}+{pos_y}")
    min_w = min(win_w, max(1000, int(base_w * 0.85)))
    min_h = min(win_h, max(650, int(base_h * 0.85)))
    root.minsize(min_w, min_h)
    manual_previous_minsize = (min_w, min_h)

    toolbar = ttk.Frame(root)
    toolbar.pack(fill="x", padx=10, pady=5)
    toolbar_frame = toolbar
    ttk.Button(toolbar, text="打开UID遮挡", command=lambda: uid_mask_manager.start()).pack(
        side="left", padx=4
    )
    ttk.Button(toolbar, text="关闭UID遮挡", command=lambda: uid_mask_manager.stop()).pack(
        side="left", padx=4
    )

    international_support_var = tk.BooleanVar(
        value=cfg.get("support_international", True)
    )
    tk.Checkbutton(
        toolbar,
        text="打开对国际服的支持",
        variable=international_support_var,
        command=on_international_support_toggle,
    ).pack(side="left", padx=4)

    manual_firework_var = tk.BooleanVar(value=False)
    manual_line_var = tk.BooleanVar(value=False)

    tk.Checkbutton(
        toolbar,
        text="单独开启转盘无巧手解密",
        variable=manual_firework_var,
        command=on_manual_firework_toggle,
    ).pack(side="left", padx=4)
    tk.Checkbutton(
        toolbar,
        text="单独开启划线无巧手解密",
        variable=manual_line_var,
        command=on_manual_line_toggle,
    ).pack(side="left", padx=4)

    manual_notice = tk.Label(
        toolbar,
        text=(
            "因群有要求 现在 两种无巧手解密都可以单独打开 常驻后台 配合其他作者的脚本使用 打开后你就相当于巧手了：注意 划线无巧手解密速度很快没什么影响 但是转盘无巧手解密需要一点时间 你要自己设置好脚本的延迟 配合解密完成！"
        ),
        fg="red",
        justify="left",
        wraplength=420,
    )
    manual_notice.pack(side="left", padx=8)

    manual_expand_button = ttk.Button(toolbar, text="展开界面", command=restore_main_window)
    manual_expand_button.pack(side="left", padx=4)
    try:
        manual_expand_button.state(["disabled"])
    except Exception:
        pass

    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True)

    frame_firework = ttk.Frame(notebook)
    notebook.add(frame_firework, text="赛琪大烟花")
    app = MainGUI(frame_firework, cfg)
    if manual_firework_var is not None and manual_firework_var.get():
        set_firework_no_trick_enabled(True)

    frame_xp50 = ttk.Frame(notebook)
    notebook.add(frame_xp50, text="全自动50人物经验副本")
    xp50_gui = XP50AutoGUI(frame_xp50, cfg)
    xp50_app = xp50_gui

    frame_hs70 = ttk.Frame(notebook)
    notebook.add(frame_hs70, text="自动70红珠")
    hs70_gui = HS70AutoGUI(frame_hs70, cfg)
    hs70_app = hs70_gui

    frame_fragment = ttk.Frame(notebook)
    notebook.add(frame_fragment, text="人物碎片刷取")

    fragment_notebook = ttk.Notebook(frame_fragment)
    fragment_notebook.pack(fill="both", expand=True)

    frame_guard = ttk.Frame(fragment_notebook)
    fragment_notebook.add(frame_guard, text="探险无尽血清")
    guard_gui = FragmentFarmGUI(frame_guard, cfg, enable_no_trick_decrypt=True)
    register_fragment_app(guard_gui)

    frame_expel = ttk.Frame(fragment_notebook)
    fragment_notebook.add(frame_expel, text="驱离")
    expel_gui = ExpelFragmentGUI(frame_expel, cfg)
    register_fragment_app(expel_gui)

    frame_mod = ttk.Frame(notebook)
    notebook.add(frame_mod, text="mod刷取")

    mod_notebook = ttk.Notebook(frame_mod)
    mod_notebook.pack(fill="both", expand=True)

    mod_guard_frame = ttk.Frame(mod_notebook)
    mod_notebook.add(mod_guard_frame, text="探险无尽血清")
    mod_guard_gui = ModFragmentGUI(mod_guard_frame, cfg)
    register_fragment_app(mod_guard_gui)

    mod_expel_frame = ttk.Frame(mod_notebook)
    mod_notebook.add(mod_expel_frame, text="驱离")
    mod_expel_gui = ModExpelGUI(mod_expel_frame, cfg)
    register_fragment_app(mod_expel_gui)

    frame_weapon = ttk.Frame(notebook)
    notebook.add(frame_weapon, text="刷武器图纸")

    weapon_notebook = ttk.Notebook(frame_weapon)
    weapon_notebook.pack(fill="both", expand=True)

    weapon_guard_frame = ttk.Frame(weapon_notebook)
    weapon_notebook.add(weapon_guard_frame, text="探险无尽血清")
    weapon_guard_gui = WeaponBlueprintFragmentGUI(weapon_guard_frame, cfg)
    register_fragment_app(weapon_guard_gui)

    weapon_expel_frame = ttk.Frame(weapon_notebook)
    weapon_notebook.add(weapon_expel_frame, text="驱离")
    weapon_expel_gui = WeaponBlueprintExpelGUI(weapon_expel_frame, cfg)
    register_fragment_app(weapon_expel_gui)

    frame_clue = ttk.Frame(notebook)
    notebook.add(frame_clue, text="密函线索刷取")
    clue_gui = ClueFarmGUI(frame_clue, cfg)
    register_fragment_app(clue_gui)

    frame_custom = ttk.Frame(notebook)
    custom_gui = CustomScriptGUI(frame_custom)

    fragment_gui_map = {
        frame_guard: guard_gui,
        frame_expel: expel_gui,
        mod_guard_frame: mod_guard_gui,
        mod_expel_frame: mod_expel_gui,
        weapon_guard_frame: weapon_guard_gui,
        weapon_expel_frame: weapon_expel_gui,
        frame_clue: clue_gui,
    }
    if manual_line_var is not None and manual_line_var.get():
        set_line_no_trick_enabled(True)

    fragment_notebooks = [
        (frame_fragment, fragment_notebook),
        (frame_mod, mod_notebook),
        (frame_weapon, weapon_notebook),
    ]

    def update_active_fragment_gui(event=None):
        current_main = notebook.select()
        if not current_main:
            set_active_fragment_gui(None)
            return
        main_widget = notebook.nametowidget(current_main)
        direct_gui = fragment_gui_map.get(main_widget)
        if direct_gui is not None:
            set_active_fragment_gui(direct_gui)
            return
        for container, sub_nb in fragment_notebooks:
            if main_widget is container:
                current_sub = sub_nb.select()
                if not current_sub:
                    set_active_fragment_gui(None)
                    return
                frame = sub_nb.nametowidget(current_sub)
                gui = fragment_gui_map.get(frame)
                set_active_fragment_gui(gui)
                return
        set_active_fragment_gui(None)

    fragment_notebook.bind("<<NotebookTabChanged>>", update_active_fragment_gui)
    mod_notebook.bind("<<NotebookTabChanged>>", update_active_fragment_gui)
    notebook.bind("<<NotebookTabChanged>>", update_active_fragment_gui)
    update_active_fragment_gui()

    def on_close():
        if uid_mask_manager is not None:
            uid_mask_manager.stop(manual=False, silent=True)
        if manual_firework_service is not None:
            manual_firework_service.stop()
        if manual_line_service is not None:
            manual_line_service.stop()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)

    log("苏苏多功能自动化工具 已启动。")
    root.mainloop()


if __name__ == "__main__":
    main()
