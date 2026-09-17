"""Run with: python chatbot.py (Python 3.10+ with Tkinter)."""
from __future__ import annotations

import os
from pathlib import Path
from queue import Empty, Queue
from threading import Thread
import tkinter as tk
from tkinter import ttk, filedialog
from tkinter.scrolledtext import ScrolledText
from typing import Any, Callable

from chat_backend import ApiError, ChatResult, Settings, chat, list_models, mood_profile, review_threshold, vision_settings
from live_panel import LivePanel


class ChatApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        root.title("LM Studio + TypeSafe Chat")
        root.geometry("880x870")
        root.minsize(680, 710)
        self.history: list[dict[str, str]] = []
        self.events: Queue[tuple[str, Any]] = Queue()
        self.busy = False
        self.pending = ""
        self.image_path = ""
        self.attachment_label = tk.StringVar(value="No image attached")
        self.base_url = tk.StringVar(value="http://localhost:1234/v1")
        self.model = tk.StringVar()
        self.typesafe_key = tk.StringVar(value=os.environ.get("TYPESAFE_API_KEY", ""))
        self.lm_key = tk.StringVar(value=os.environ.get("LM_STUDIO_API_KEY", ""))
        self.refine = tk.BooleanVar(value=True)
        self.code_mode = tk.BooleanVar(value=False)
        self.vision_model = tk.StringVar()
        self.vision_url = tk.StringVar()
        self.vision_key = tk.StringVar()
        self.mood = tk.IntVar(value=0)
        self.mood_label = tk.StringVar(value="Assistant mood: Neutral (+0)")
        self.strictness = tk.IntVar(value=25)
        self.strictness_label = tk.StringVar(value="Chat rewrite strength: 25/100 — issue cutoff 75% (chat only)")
        self.status = tk.StringVar(value="Start LM Studio's server, then click Load models.")

        frame = ttk.Frame(root, padding=16)
        frame.pack(fill="both", expand=True)
        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(6, weight=1)
        ttk.Label(frame, text="LM Studio + TypeSafe", font=("Segoe UI", 17, "bold")).grid(
            row=0, column=0, columnspan=3, sticky="w", pady=(0, 12))
        ttk.Label(frame, text="LM Studio URL").grid(row=1, column=0, sticky="w", padx=(0, 10))
        self.url_entry = ttk.Entry(frame, textvariable=self.base_url)
        self.url_entry.grid(row=1, column=1, sticky="ew", pady=3)
        self.load_button = ttk.Button(frame, text="Load models", command=self.load_models)
        self.load_button.grid(row=1, column=2, padx=(8, 0))
        ttk.Label(frame, text="Chat model").grid(row=2, column=0, sticky="w")
        self.model_box = ttk.Combobox(frame, textvariable=self.model)
        self.model_box.grid(row=2, column=1, columnspan=2, sticky="ew", pady=3)
        ttk.Label(frame, text="TypeSafe API key").grid(row=3, column=0, sticky="w")
        self.key_entry = ttk.Entry(frame, textvariable=self.typesafe_key, show="*")
        self.key_entry.grid(row=3, column=1, columnspan=2, sticky="ew", pady=3)
        ttk.Label(frame, text="LM API key (optional)").grid(row=4, column=0, sticky="w", padx=(0, 10))
        self.lm_entry = ttk.Entry(frame, textvariable=self.lm_key, show="*")
        self.lm_entry.grid(row=4, column=1, columnspan=2, sticky="ew", pady=3)
        ttk.Label(frame, text="Jev receives your prompt, draft, recent context, and vision description when refining.\n"
                  "Images go only to the vision server. Clear the TypeSafe key for local-only chat.",
                  foreground="#555555").grid(row=5, column=0, columnspan=3, sticky="w", pady=10)
        self.tabs = ttk.Notebook(frame)
        self.tabs.grid(row=6, column=0, columnspan=3, sticky="nsew")
        chat_tab = ttk.Frame(self.tabs)
        prompt_tab = ttk.Frame(self.tabs, padding=8)
        self.tabs.add(chat_tab, text="Chat")
        self.tabs.add(prompt_tab, text="Master prompt")
        vision_tab = ttk.Frame(self.tabs, padding=8)
        self.tabs.add(vision_tab, text="Vision")
        self.live_panel = LivePanel(self.tabs, root, self.settings, self.set_busy, lambda: self.busy)
        self.tabs.add(self.live_panel, text="Live narration")
        vision_tab.columnconfigure(1, weight=1)
        vision_tab.rowconfigure(5, weight=1)
        ttk.Label(vision_tab, text="Choose an image-capable model such as LFM2-VL. Load it in LM Studio first.\n"
                  "Blank vision URL uses the main server. A separate server can use its own API key.\n"
                  "Attach an image below, then Send. Jev checks the description, not the image itself.",
                  wraplength=620).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 8))
        ttk.Label(vision_tab, text="Vision URL (optional)").grid(row=1, column=0, sticky="w", padx=(0, 8))
        self.vision_url_entry = ttk.Entry(vision_tab, textvariable=self.vision_url)
        self.vision_url_entry.grid(row=1, column=1, sticky="ew", pady=3)
        self.vision_load = ttk.Button(vision_tab, text="Refresh models", command=self.load_vision_models)
        self.vision_load.grid(row=1, column=2, padx=(8, 0))
        ttk.Label(vision_tab, text="Vision model").grid(row=2, column=0, sticky="w")
        self.vision_box = ttk.Combobox(vision_tab, textvariable=self.vision_model)
        self.vision_box.grid(row=2, column=1, columnspan=2, sticky="ew", pady=3)
        ttk.Label(vision_tab, text="Vision key (optional)").grid(row=3, column=0, sticky="w")
        self.vision_key_entry = ttk.Entry(vision_tab, textvariable=self.vision_key, show="*")
        self.vision_key_entry.grid(row=3, column=1, columnspan=2, sticky="ew", pady=3)
        ttk.Label(vision_tab, text="Last extracted description (also shown in chat)").grid(
            row=4, column=0, columnspan=3, sticky="w", pady=(8, 3))
        self.vision_output = ScrolledText(vision_tab, wrap="word", height=5, state="disabled")
        self.vision_output.grid(row=5, column=0, columnspan=3, sticky="nsew")
        ttk.Label(prompt_tab, text="Define the personality, role, and response style. Changes apply to the next reply.\n"
                  "Blank uses the default assistant. Clear chat for a fresh personality test.\n"
                  "The prompt stays in memory for this session; Jev receives it when refinement is enabled.",
                  wraplength=600).pack(anchor="w", pady=(0, 8))
        self.master_prompt = ScrolledText(prompt_tab, wrap="word", font=("Segoe UI", 11),
                                          height=8, padx=8, pady=8)
        self.master_prompt.pack(fill="both", expand=True)
        self.transcript = ScrolledText(chat_tab, wrap="word", state="disabled", font=("Segoe UI", 11),
                                       padx=12, pady=12, height=15)
        self.transcript.pack(fill="both", expand=True)
        self.transcript.tag_configure("You", foreground="#185a9d")
        self.transcript.tag_configure("Assistant", foreground="#236a3c")
        self.transcript.tag_configure("TypeSafe", foreground="#775496", font=("Segoe UI", 9))
        self.transcript.tag_configure("Error", foreground="#a4262c")
        self.transcript.tag_configure("Vision", foreground="#6a4d22", font=("Segoe UI", 10))
        ttk.Label(frame, textvariable=self.status, wraplength=800).grid(
            row=7, column=0, columnspan=3, sticky="w", pady=8)
        composer = ttk.Frame(frame)
        composer.grid(row=8, column=0, columnspan=3, sticky="ew")
        attachments = ttk.Frame(composer)
        attachments.pack(fill="x", pady=(0, 4))
        self.attach_button = ttk.Button(attachments, text="Attach image", command=self.attach_image)
        self.attach_button.pack(side="left")
        self.remove_image_button = ttk.Button(attachments, text="Remove image", command=self.remove_image)
        self.remove_image_button.pack(side="left", padx=6)
        ttk.Label(attachments, textvariable=self.attachment_label).pack(side="left")
        self.input = ScrolledText(composer, height=3, wrap="word", font=("Segoe UI", 11))
        self.input.pack(fill="x")
        self.input.bind("<Return>", self.on_enter)
        controls = ttk.Frame(frame)
        controls.grid(row=9, column=0, columnspan=3, sticky="ew", pady=(8, 0))
        ttk.Label(controls, text="Enter to send · Shift+Enter for a new line").pack(side="left")
        self.send_button = ttk.Button(controls, text="Send", command=self.send)
        self.send_button.pack(side="right")
        self.clear_button = ttk.Button(controls, text="Clear chat", command=self.clear)
        self.clear_button.pack(side="right", padx=8)
        review_controls = ttk.Frame(frame)
        review_controls.grid(row=10, column=0, columnspan=3, sticky="ew", pady=(8, 0))
        self.refine_check = ttk.Checkbutton(review_controls, text="Refine replies with Jev (may take longer)",
                                            variable=self.refine)
        self.refine_check.pack(side="left")
        self.code_check = ttk.Checkbutton(review_controls, text="Python code review (no execution)",
                                         variable=self.code_mode)
        self.code_check.pack(side="left", padx=(16, 0))
        mood_frame = ttk.Frame(frame)
        mood_frame.grid(row=11, column=0, columnspan=3, sticky="ew", pady=(8, 0))
        mood_frame.columnconfigure(1, weight=1)
        ttk.Label(mood_frame, textvariable=self.mood_label).grid(row=0, column=0, columnspan=3, sticky="w")
        ttk.Label(mood_frame, text="Grumpy").grid(row=1, column=0, padx=(0, 8))
        self.mood_slider = tk.Scale(mood_frame, from_=-100, to=100, resolution=10,
                                    orient="horizontal", variable=self.mood, showvalue=False,
                                    command=self.update_mood_label, highlightthickness=0)
        self.mood_slider.grid(row=1, column=1, sticky="ew")
        ttk.Label(mood_frame, text="Cheerful").grid(row=1, column=2, padx=(8, 0))
        strictness_frame = ttk.Frame(frame)
        strictness_frame.grid(row=12, column=0, columnspan=3, sticky="ew", pady=(4, 0))
        strictness_frame.columnconfigure(1, weight=1)
        ttk.Label(strictness_frame, textvariable=self.strictness_label).grid(
            row=0, column=0, columnspan=3, sticky="w")
        ttk.Label(strictness_frame, text="Fewer rewrites").grid(row=1, column=0, padx=(0, 8))
        self.strictness_slider = tk.Scale(strictness_frame, from_=0, to=100, resolution=5,
                                          orient="horizontal", variable=self.strictness, showvalue=False,
                                          command=self.update_strictness_label, highlightthickness=0)
        self.strictness_slider.grid(row=1, column=1, sticky="ew")
        ttk.Label(strictness_frame, text="More rewrites").grid(row=1, column=2, padx=(8, 0))
        self.controls = [self.url_entry, self.model_box, self.key_entry, self.lm_entry,
                         self.load_button, self.send_button, self.clear_button, self.refine_check,
                         self.mood_slider, self.strictness_slider, self.master_prompt, self.code_check,
                         self.vision_url_entry, self.vision_box, self.vision_key_entry, self.vision_load,
                         self.attach_button, self.remove_image_button]
        self.poll_id = root.after(100, self.poll)
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.input.focus_set()

    def settings(self) -> Settings:
        return Settings(self.base_url.get().strip(), self.model.get().strip(),
                        self.typesafe_key.get().strip(), self.lm_key.get().strip(), self.refine.get(),
                        self.mood.get(), self.strictness.get(), self.master_prompt.get("1.0", "end").strip(),
                        self.code_mode.get(), self.vision_model.get().strip(), self.vision_url.get().strip(),
                        self.vision_key.get().strip())

    def attach_image(self) -> None:
        if self.busy:
            return
        path = filedialog.askopenfilename(parent=self.root, title="Attach image",
                                         filetypes=[("Images", "*.png *.jpg *.jpeg *.webp")])
        if path:
            self.image_path = path
            self.attachment_label.set(Path(path).name[:55])

    def remove_image(self) -> None:
        if not self.busy:
            self.image_path = ""
            self.attachment_label.set("No image attached")

    def load_vision_models(self) -> None:
        if self.busy:
            return
        settings = self.settings()
        self.status.set("Checking the vision server's model list…")
        self.start_worker("vision_models", lambda: list_models(vision_settings(settings)))

    def update_strictness_label(self, _value: str) -> None:
        value = self.strictness.get()
        self.strictness_label.set(
            f"Chat rewrite strength: {value}/100 — issue cutoff {review_threshold(value):.0%} (chat only)")

    def update_mood_label(self, _value: str) -> None:
        mood = mood_profile(self.mood.get())
        self.mood_label.set(f"Assistant mood: {mood['label']} ({mood['value']:+d})")

    def append(self, role: str, text: str) -> None:
        self.transcript.configure(state="normal")
        self.transcript.insert("end", f"{role}\n{text}\n\n", role)
        self.transcript.configure(state="disabled")
        self.transcript.see("end")

    def set_busy(self, busy: bool) -> None:
        self.busy = busy
        for control in self.controls:
            control.configure(state="disabled" if busy else "normal")
        self.input.configure(state="disabled" if busy else "normal")

    def start_worker(self, kind: str, operation: Callable[[], Any]) -> None:
        self.set_busy(True)

        def work() -> None:
            # Never access Tk widgets or variables from this thread.
            try:
                self.events.put((kind, operation()))
            except ApiError as exc:
                self.events.put(("error", str(exc)))
            except Exception:
                self.events.put(("error", "Unexpected error. Check your settings and retry."))

        Thread(target=work, daemon=True).start()

    def load_models(self) -> None:
        if self.busy:
            return
        settings = self.settings()
        self.status.set("Connecting to LM Studio…")
        self.start_worker("models", lambda: list_models(settings))

    def on_enter(self, event: tk.Event) -> str | None:
        if event.state & 0x1:
            return None
        self.send()
        return "break"

    def send(self) -> None:
        if self.busy:
            return
        message = self.input.get("1.0", "end").strip()
        image_path = self.image_path
        if not message and image_path:
            message = "Describe this image."
        if not message:
            return
        settings = self.settings()
        if not settings.model:
            self.status.set("Click Load models and select a chat model first.")
            return
        if image_path and not settings.vision_model:
            self.tabs.select(2)
            self.status.set("Choose a vision-capable model in the Vision tab first.")
            return
        self.pending = message
        self.tabs.select(0)
        self.append("You", message + (f"\n[Image: {Path(image_path).name}]" if image_path else ""))
        self.input.delete("1.0", "end")
        history = [dict(item) for item in self.history]
        self.status.set("Generating reply and reviewing with Jev…" if settings.refine and settings.typesafe_key
                        else "Generating reply…")
        if settings.code_mode:
            self.status.set("Generating Python and checking code… (code will not be executed)")
        if image_path:
            self.status.set("Extracting the image description, then generating and reviewing the reply…")
        self.start_worker("reply", lambda: chat(settings, history, message, image_path=image_path))

    def poll(self) -> None:
        try:
            kind, result = self.events.get_nowait()
        except Empty:
            pass
        else:
            self.set_busy(False)
            if kind == "vision_models":
                self.vision_box.configure(values=result)
                self.status.set("Choose a vision-capable model from the list (text-only models may also appear)."
                                if result else "No vision-server models listed. Load a vision model in LM Studio first.")
            elif kind == "models":
                self.model_box.configure(values=result)
                if result:
                    if self.model.get() not in result:
                        self.model.set(result[0])
                    self.status.set("Models loaded. Select a chat model and send a message.")
                else:
                    self.model.set("")
                    self.status.set("No models found. Load a chat model in LM Studio and try again.")
            elif kind == "reply":
                assert isinstance(result, ChatResult)
                self.history.extend([{"role": "user", "content": result.user_context or self.pending},
                                     {"role": "assistant", "content": result.reply}])
                self.history = self.history[-20:]
                if result.vision_description:
                    self.append("Vision", result.vision_description)
                    self.vision_output.configure(state="normal")
                    self.vision_output.delete("1.0", "end")
                    self.vision_output.insert("1.0", result.vision_description)
                    self.vision_output.configure(state="disabled")
                self.remove_image()
                self.append("TypeSafe", result.analysis)
                self.append("Assistant", result.reply)
                self.pending = ""
                self.status.set("Ready · Last 10 exchanges are sent as conversation context.")
            else:
                self.append("Error", result)
                self.status.set("Request failed. Check the message above and try again.")
                if self.pending:
                    self.input.insert("1.0", self.pending)
                    self.pending = ""
            self.input.focus_set()
        self.poll_id = self.root.after(100, self.poll)

    def clear(self) -> None:
        if self.busy:
            return
        self.history.clear()
        self.remove_image()
        self.vision_output.configure(state="normal")
        self.vision_output.delete("1.0", "end")
        self.vision_output.configure(state="disabled")
        self.transcript.configure(state="normal")
        self.transcript.delete("1.0", "end")
        self.transcript.configure(state="disabled")
        self.status.set("Chat cleared. Ready for a new conversation.")

    def close(self) -> None:
        self.live_panel.close()
        self.root.after_cancel(self.poll_id)
        self.root.destroy()


if __name__ == "__main__":
    window = tk.Tk()
    ChatApp(window)
    window.mainloop()
