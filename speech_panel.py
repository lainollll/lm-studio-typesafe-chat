"""Speech configuration only; synthesis runs in LiveNarrator's worker thread."""
import tkinter as tk
from tkinter import filedialog, ttk

from chatterbox_client import ChatterboxConfig
from nc_chatterbox import NCConfig, find_nc_python


class SpeechPanel(ttk.Frame):
    def __init__(self, parent: ttk.Notebook) -> None:
        super().__init__(parent, padding=8)
        self.columnconfigure(1, weight=1)
        self.provider = tk.StringVar(value="Project Chatterbox")
        self.folder = tk.StringVar(value=r"Q:\NEURALCOMPANION DEV MAIN\NeuralCompanion-dev-dirty")
        self.python = tk.StringVar(value=find_nc_python(self.folder.get()))
        self.device = tk.StringVar(value="auto")
        self.reference = tk.StringVar()
        self.endpoint = tk.StringVar(value="http://localhost:8004/tts")
        self.voice = tk.StringVar()
        self.model = tk.StringVar(value="tts-1")
        self.key = tk.StringVar()
        ttk.Label(self, text="Speech for live narration. Settings apply when you next press Start.\n"
                  "Project Chatterbox uses this project's installation. NC Chatterbox uses NC instead.\n"
                  "Speech plays in punctuation-aware chunks. Initial model loading is still required.\n"
                  "It uses additional memory. Windows speech speed applies only to Windows speech.",
                  wraplength=700).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 10))
        ttk.Label(self, text="Speech provider").grid(row=1, column=0, sticky="w")
        ttk.Combobox(self, textvariable=self.provider, state="readonly",
                     values=("Project Chatterbox", "Windows", "NC Chatterbox", "Chatterbox server")).grid(row=1, column=1, sticky="ew")
        self.entry(2, "NC folder", self.folder, self.choose_folder)
        self.entry(3, "NC Python executable", self.python, lambda: self.choose_file(self.python, "Python", "*.exe"))
        ttk.Label(self, text="Chatterbox device").grid(row=4, column=0, sticky="w")
        ttk.Combobox(self, textvariable=self.device, state="readonly", values=("auto", "cuda", "cpu")).grid(row=4, column=1, sticky="ew")
        self.entry(5, "Reference voice (optional)", self.reference, lambda: self.choose_file(self.reference, "Audio", "*.wav *.mp3 *.flac"))
        ttk.Separator(self).grid(row=6, column=0, columnspan=3, sticky="ew", pady=12)
        self.entry(7, "Server speech endpoint", self.endpoint)
        self.entry(8, "Server voice (optional)", self.voice)
        self.entry(9, "Server model", self.model)
        self.entry(10, "Server API key (optional)", self.key, secret=True)
        ttk.Label(self, text="Server mode supports /tts or /v1/audio/speech returning WAV audio.\n"
                  "Project/NC modes need no server; narration uses cached weights offline.\n"
                  "Settings and keys stay in memory for this session.", wraplength=700).grid(
                      row=11, column=0, columnspan=3, sticky="w", pady=10)

    def entry(self, row: int, label: str, variable: tk.StringVar, browse=None, secret: bool = False) -> None:
        ttk.Label(self, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8))
        ttk.Entry(self, textvariable=variable, show="*" if secret else "").grid(row=row, column=1, sticky="ew", pady=3)
        if browse:
            ttk.Button(self, text="Browse...", command=browse).grid(row=row, column=2, padx=5)

    def choose_folder(self) -> None:
        selected = filedialog.askdirectory(parent=self, title="NeuralCompanion folder")
        if selected:
            self.folder.set(selected)
            self.python.set(find_nc_python(selected))

    def choose_file(self, variable: tk.StringVar, label: str, pattern: str) -> None:
        selected = filedialog.askopenfilename(parent=self, title=label, filetypes=[(label, pattern)])
        if selected:
            variable.set(selected)

    def nc_config(self) -> NCConfig:
        return NCConfig(self.folder.get().strip(), self.python.get().strip(), self.device.get(), self.reference.get().strip(),
                        local=self.provider.get() == "Project Chatterbox")

    def server_config(self) -> ChatterboxConfig:
        return ChatterboxConfig(self.endpoint.get().strip(), self.voice.get().strip(), self.model.get().strip(), self.key.get())
