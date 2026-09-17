"""Tk controls for screen/video-file narration. All widget access stays on Tk's thread."""
from __future__ import annotations

import io
from pathlib import Path
from queue import Empty
import tkinter as tk
from tkinter import filedialog, ttk
from tkinter.scrolledtext import ScrolledText
from typing import Callable

from chat_backend import ApiError, Settings
from live_narration import LiveNarrator, LiveOptions, narration_threshold
from speech_panel import SpeechPanel


class LivePanel(ttk.Frame):
    def __init__(self, parent: ttk.Notebook, root: tk.Tk,
                 settings: Callable[[], Settings], set_busy: Callable[[bool], None],
                 is_busy: Callable[[], bool]) -> None:
        super().__init__(parent, padding=8)
        self.root, self.get_settings = root, settings
        self.set_chat_busy, self.is_busy = set_busy, is_busy
        self.narrator: LiveNarrator | None = None
        self.speech_panel = SpeechPanel(parent)
        parent.add(self.speech_panel, text="Speech")
        self.source = tk.StringVar(value="Screen")
        self.file_path = tk.StringVar()
        self.interval = tk.IntVar(value=3)
        self.rate = tk.IntVar(value=0)
        self.sensitivity = tk.IntVar(value=50)
        self.sensitivity_label = tk.StringVar(value="Live updates: 50/100\nNew-information threshold: 60%")
        self.muted = tk.BooleanVar(value=False)
        self.status = tk.StringVar(value="Stopped. Select a vision model in the Vision tab, then choose a source here.")
        self.region: tuple[int, int, int, int] | None = None
        self.region_text = tk.StringVar(value="Whole primary screen")
        self.last_preview = -1
        self.stopping = False
        self.preview_image = None
        self.columnconfigure(1, weight=1)
        self.rowconfigure(7, weight=1)
        ttk.Label(self, text="Spoken video descriptions · F8 Start · F9 Stop · F10 Mute (while this app has focus)\n"
                  "Descriptions can lag or miss events. Not for hazard detection or navigation.",
                  wraplength=740).grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 6))
        ttk.Label(self, text="Source").grid(row=1, column=0, sticky="w")
        self.source_box = ttk.Combobox(self, textvariable=self.source, values=("Screen", "Video file"),
                                       state="readonly", width=15)
        self.source_box.grid(row=1, column=1, sticky="w")
        self.file_button = ttk.Button(self, text="Choose video…", command=self.choose_file)
        self.file_button.grid(row=1, column=2, padx=5)
        self.region_button = ttk.Button(self, text="Select screen area…", command=self.select_region)
        self.region_button.grid(row=1, column=3)
        self.file_label = ttk.Label(self, text="Video files play silently here; screen mode keeps your player's audio.",
                                    wraplength=700)
        self.file_label.grid(row=2, column=0, columnspan=4, sticky="w", pady=3)
        ttk.Label(self, textvariable=self.region_text).grid(row=3, column=0, columnspan=3, sticky="w")
        self.full_button = ttk.Button(self, text="Whole screen", command=self.whole_screen)
        self.full_button.grid(row=3, column=3)
        options = ttk.Frame(self)
        options.grid(row=4, column=0, columnspan=4, sticky="ew", pady=5)
        ttk.Label(options, text="Check interval (seconds)").pack(side="left")
        self.interval_box = ttk.Spinbox(options, from_=1, to=30, textvariable=self.interval, width=4)
        self.interval_box.pack(side="left", padx=(5, 12))
        ttk.Label(options, text="Speech speed (-5 to 5)").pack(side="left")
        self.rate_box = ttk.Spinbox(options, from_=-5, to=5, textvariable=self.rate, width=4)
        self.rate_box.pack(side="left", padx=5)
        ttk.Checkbutton(options, text="Mute", variable=self.muted, command=self.change_mute).pack(side="right")
        sensitivity_row = ttk.Frame(self)
        sensitivity_row.grid(row=5, column=0, columnspan=4, sticky="ew", pady=(0, 5))
        ttk.Label(sensitivity_row, textvariable=self.sensitivity_label).pack(side="left", padx=(0, 10))
        ttk.Label(sensitivity_row, text="Fewer updates").pack(side="left")
        self.sensitivity_slider = tk.Scale(sensitivity_row, from_=0, to=100, resolution=5,
                                            orient="horizontal", variable=self.sensitivity, showvalue=False,
                                            command=self.change_sensitivity, highlightthickness=0)
        self.sensitivity_slider.pack(side="left", fill="x", expand=True, padx=5)
        ttk.Label(sensitivity_row, text="More updates").pack(side="left")
        actions = ttk.Frame(self)
        actions.grid(row=6, column=0, columnspan=4, sticky="ew", pady=(0, 5))
        self.start_button = ttk.Button(actions, text="Start narration (F8)", command=self.start)
        self.start_button.pack(side="left")
        self.stop_button = ttk.Button(actions, text="Stop (F9)", command=self.stop, state="disabled")
        self.stop_button.pack(side="left", padx=8)
        ttk.Label(actions, text="Jev receives frame descriptions, not video pixels.").pack(side="left")
        panes = ttk.Panedwindow(self, orient="horizontal")
        panes.grid(row=7, column=0, columnspan=4, sticky="nsew")
        preview_frame = ttk.Frame(panes, width=220)
        self.preview = ttk.Label(preview_frame, text="Video preview", anchor="center")
        self.preview.pack(fill="both", expand=True)
        self.log = ScrolledText(panes, wrap="word", height=6, width=42, state="disabled", font=("Segoe UI", 11))
        panes.add(preview_frame, weight=1)
        panes.add(self.log, weight=2)
        ttk.Label(self, textvariable=self.status, wraplength=740).grid(
            row=8, column=0, columnspan=4, sticky="w", pady=(5, 0))
        self.setting_widgets = [self.source_box, self.file_button, self.region_button, self.full_button,
                                self.interval_box, self.rate_box, self.start_button]
        root.bind("<F8>", lambda event: self.start())
        root.bind("<F9>", lambda event: self.stop())
        root.bind("<F10>", lambda event: self.toggle_mute())
        self.poll_id = root.after(100, self.poll)

    def choose_file(self) -> None:
        path = filedialog.askopenfilename(parent=self.root, title="Choose video",
                                         filetypes=[("Video", "*.mp4 *.mkv *.avi *.mov *.webm"), ("All files", "*.*")])
        if path:
            self.file_path.set(path)
            self.source.set("Video file")
            self.file_label.configure(text=f"{Path(path).name} · silent preview with spoken descriptions")

    def change_sensitivity(self, _value: str) -> None:
        value = self.sensitivity.get()
        self.sensitivity_label.set(f"Live updates: {value}/100\nNew-information threshold: {narration_threshold(value):.0%}")
        if self.narrator:
            self.narrator.set_sensitivity(value)

    def whole_screen(self) -> None:
        self.region = None
        self.region_text.set("Whole primary screen — select the video area to avoid capturing this app.")

    def select_region(self) -> None:
        if self.narrator or self.is_busy():
            return
        self.root.withdraw()
        overlay = tk.Toplevel(self.root)
        overlay.overrideredirect(True)
        overlay.geometry(f"{overlay.winfo_screenwidth()}x{overlay.winfo_screenheight()}+0+0")
        overlay.attributes("-topmost", True)
        overlay.attributes("-alpha", 0.35)
        canvas = tk.Canvas(overlay, background="black", cursor="crosshair", highlightthickness=0)
        canvas.pack(fill="both", expand=True)
        canvas.create_text(20, 20, anchor="nw", fill="white", text="Drag around the video. Escape cancels.",
                           font=("Segoe UI", 20))
        start: list[int] = []
        rectangle: list[int] = []

        def finish() -> None:
            overlay.destroy()
            self.root.deiconify()

        def press(event: tk.Event) -> None:
            start[:] = [event.x_root, event.y_root]
            if rectangle:
                canvas.delete(rectangle.pop())
            rectangle.append(canvas.create_rectangle(event.x, event.y, event.x, event.y, outline="yellow", width=3))

        def move(event: tk.Event) -> None:
            if start and rectangle:
                canvas.coords(rectangle[0], start[0], start[1], event.x_root, event.y_root)

        def release(event: tk.Event) -> None:
            if start:
                left, right = sorted((start[0], event.x_root))
                top, bottom = sorted((start[1], event.y_root))
                if right - left >= 20 and bottom - top >= 20:
                    self.region = (left, top, right, bottom)
                    self.region_text.set(f"Screen area: {left}, {top} to {right}, {bottom}")
                    self.source.set("Screen")
            finish()

        canvas.bind("<ButtonPress-1>", press)
        canvas.bind("<B1-Motion>", move)
        canvas.bind("<ButtonRelease-1>", release)
        overlay.bind("<Escape>", lambda event: finish())
        overlay.focus_force()

    def start(self) -> None:
        if self.narrator or self.is_busy():
            return
        try:
            interval, rate = self.interval.get(), self.rate.get()
            if not -5 <= rate <= 5:
                raise ValueError
            options = LiveOptions("screen" if self.source.get() == "Screen" else "file",
                                  self.file_path.get(), self.region, interval, rate, self.sensitivity.get(),
                                  self.speech_panel.provider.get(), self.speech_panel.nc_config(),
                                  self.speech_panel.server_config())
            narrator = LiveNarrator(self.get_settings(), options)
            narrator.set_muted(self.muted.get())
            narrator.start()
        except (ApiError, ValueError, tk.TclError) as exc:
            self.status.set(str(exc) if isinstance(exc, ApiError) else "Enter a valid interval and speech speed.")
            return
        self.narrator = narrator
        self.stopping = False
        self.last_preview = -1
        self.set_chat_busy(True)
        for widget in self.setting_widgets:
            widget.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.status.set("Narration starting…")

    def stop(self) -> None:
        if self.narrator:
            self.stopping = True
            self.narrator.stop()
            self.status.set("Stopping. Speech and capture stop now; any pending model response will be discarded.")

    def toggle_mute(self) -> None:
        self.muted.set(not self.muted.get())
        self.change_mute()

    def change_mute(self) -> None:
        if self.narrator:
            self.narrator.set_muted(self.muted.get())

    def poll(self) -> None:
        narrator = self.narrator
        if narrator:
            for _ in range(30):
                try:
                    kind, text = narrator.events.get_nowait()
                except Empty:
                    break
                if kind in ("description", "error") and not self.stopping:
                    self.log.configure(state="normal")
                    self.log.insert("end", text + "\n\n")
                    if int(self.log.index("end-1c").split(".")[0]) > 500:
                        self.log.delete("1.0", "100.0")
                    self.log.see("end")
                    self.log.configure(state="disabled")
                elif kind != "description" and not self.stopping:
                    self.status.set(text)
            frame = narrator.frames.get()
            if frame and frame.sequence != self.last_preview and not narrator.stop_event.is_set():
                try:
                    from PIL import Image, ImageTk
                    picture = Image.open(io.BytesIO(frame.jpeg))
                    picture.thumbnail((240, 150))
                    self.preview_image = ImageTk.PhotoImage(picture, master=self.root)
                    self.preview.configure(image=self.preview_image, text="")
                    self.last_preview = frame.sequence
                except Exception:
                    self.preview.configure(text="Preview unavailable")
            if not narrator.running():
                self.narrator = None
                self.set_chat_busy(False)
                for widget in self.setting_widgets:
                    widget.configure(state="normal")
                self.source_box.configure(state="readonly")
                self.stop_button.configure(state="disabled")
                self.status.set("Stopped." if self.stopping else "Stopped. " + self.status.get())
        self.poll_id = self.root.after(100, self.poll)

    def close(self) -> None:
        if self.narrator:
            self.narrator.stop()
        self.root.after_cancel(self.poll_id)
