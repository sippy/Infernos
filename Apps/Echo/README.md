# Echo

Echo auto-answers incoming SIP calls and returns each recognized VAD speech
segment to the same caller through TTS. It uses no LLM, prompt file, outbound
call, or NLTK tokenizer. This is a speech echo, not a raw audio loopback.

See [examples/echo.yaml](../../examples/echo.yaml). Set the SIP connector sink
to `apps/echo/<profile-name>`, then run `python Infernos.py -f examples/echo.yaml`
in the normal configured runtime. Restrict the SIP connector to your caller/PBX.

Profiles accept `stt_lang` (input language, default `en`), `tts_lang`
(output language, default `en`), and `greeting` (optional greeting text, default
unset). The `greeting` parameter is analogous to AIAttendant's greeting options.
When present and non-empty, the greeting text is enqueued to the TTS queue once
per session on connection, using the configured `tts_lang`. The greeting is spoken
before any recognized speech and respects the existing TTS completion queue.
Absent or empty `greeting` preserves existing behavior (no greeting). Unknown
options, including `llm_prompt`, are rejected. STT uses `transcribe`, not
Whisper English translation. Only differing configured languages instantiate
the existing translator; equal languages pass recognized text directly to TTS.
Blank results and results over the existing STT no-speech threshold are ignored.
Speech requests queue in recognition-result order and advance on RTP playback
completion; there is no barge-in or acoustic echo cancellation. Use headphones
or an endpoint with echo cancellation to avoid feeding playback back into STT.

Each active profile needs Ray resources `echo: 1`, `stt: 1`, `tts: 1`, plus
the existing SIP/RTP resources and STT/TTS GPU reservations. The launcher
registers `echo: 1` locally; an existing cluster must advertise it separately.
STT/TTS models must be available for the chosen languages. The shared translator
currently updates the Argos package index and may download language packages
at startup; this app does not change that existing behavior. Same-language
Echo never constructs that translator.

Offline fake-service tests cover wiring and cleanup, not live SIP/model quality:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES= RAY_ENABLE_AUTO_CONNECT=0 \
  .venv/bin/python -m unittest discover -s tests -p test_echo.py -v
```
