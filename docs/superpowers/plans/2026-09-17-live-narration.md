# Live video description

User scope: screen video and local video files, with spoken descriptions for a
blind person. Reuse the configured vision model and TypeSafe API. No camera source.

Implementation:
- Add a latest-frame buffer, bounded capture and inference workers, and Windows
  speech output. Stop/mute must be responsive and late responses discarded.
- Screen source: explicit Start and optional user-selected rectangle. File source:
  OpenCV playback at the file's frame rate, with silent preview and narration.
- Sample at a user-selected interval, resize frames, keep no frame backlog.
- Ask the vision model for brief observable scene/action descriptions. Jev judges
  meaningful changes against the previous report. No personality or code modes.
- Expose Start/Stop, mute, source, file picker, crop, interval, preview, transcript.
  Keep all Tk updates on the main thread. No recording or saved narration.
- Verify buffer replacement, filtering/failure behavior, stale results, capture
  and UI lifecycle. Do not claim realtime accuracy or hazard/navigation reliability.

Runtime libraries already present: OpenCV, Pillow, pywin32. Imports stay lazy so
the basic chatbot still starts without optional live dependencies.
