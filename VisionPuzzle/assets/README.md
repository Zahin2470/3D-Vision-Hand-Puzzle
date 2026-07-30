# Audio assets

Drop your own sound files here — nothing is bundled, and the app runs
perfectly fine (just silently) until you do.

```
assets/audio/
├── music/
│   ├── select.ogg     looping ambient bed for Selection mode
│   └── play.ogg       looping ambient bed for Play mode
└── sfx/
    ├── pinch.wav       short tick — a piece/frame is picked up
    ├── lock.wav        selection frame locked in
    ├── snap.wav        piece snapped into its slot
    ├── shuffle.wav      board reshuffled (R key)
    └── win.wav          puzzle completed
```

Notes
-----
- Formats: `.ogg` is recommended for the looping music tracks (small,
  loops cleanly with pygame). `.wav` is recommended for SFX (lowest
  latency, no decode delay on trigger).
- Any file you don't add is just skipped — `AudioManager` checks
  `Path.is_file()` before touching it, so a partial set works fine.
- Good free sources if you don't want to record your own:
  freesound.org, opengameart.org, kenney.nl/assets (all have
  CC0 / permissively-licensed packs — check each pack's license).
- Keep SFX short (under ~1s) so they don't overlap awkwardly when
  triggered rapidly (e.g. picking up two pieces with both hands).
- Volumes are controlled in code via `AudioManager.set_music_volume()`
  / `set_sfx_volume()` (0.0–1.0), independent of your source files'
  loudness — normalize your files to a consistent level for the best
  balance between music and SFX.
