# Infernos RTP integration

The active actor path uses RTP.InfernRTPEPoint.InfernRTPEPoint and the existing
four-argument constructor (rc, rtp_params, ring, get_direct_soundout).
The get_direct_soundout function resolves a session UUID to a callback; it is not
an audio sink. Use these active modules rather than earlier experimental SippyRtp examples.

## Responsibilities

- RTPParams extends sippy media parameters and stores real sippy G711/G722 codec classes.
- InfernRTPEPoint extends sippy's endpoint, installs factories before construction,
  and exposes its writer, transport channel, connect/update/shutdown methods.
- RTPOutputWorker adapts Infernos tracks and markers to the existing output muxer,
  then delegates encoding, RTP construction, pacing, counters and sender registration
  to sippy. An idle writer does not synthesize audio. Completion markers run after
  prior encoded audio drains; a final incomplete packet may be silence-padded.
- AudioBridge converts float tensors to clamped PCM16 and back using sippy's actual
  audio and samplerate fields, preserving available metadata.
- RTPInStream delegates jitter/codec processing to sippy and queues decoded PCM
  for the ingest worker. Existing VAD receives fixed-size windows and dispatches
  connected audio/VAD callbacks. No placeholder codec or audio producer is used.
- The actor returns the transport channel's local_addr, not Udp_server internals.

## Environment and checks

Use the existing project .venv; do not install into a different system interpreter.
Run offline checks from the repository root:

    CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m unittest discover -s tests -p 'test_rtp*.py' -v

The suite exercises actual sippy codecs, RTP packet construction and jitter decoding,
PCM conversion, track mixing, marker ordering, session interfaces, updates and cleanup.
Transport channels and the scheduler are isolated fakes; VAD inference is mocked.
Startup and Ray actor serialization checks run without starting a cluster.
These tests do not establish full live SIP/Ray/Silero/TTS operation or GPU performance.
No package versions, services, or model downloads are changed by this test procedure.

## G.722 packet-loss compatibility

The installed sippy G.722 silence helper passes a raw array to an encoder that
requires an AudioChunk. The local ingress-only compatibility subclass wraps native-rate
PCM in the codec's own chunk type; encoding and decoding remain inherited from sippy.
SIP negotiation and the writer still use the actual sippy G722Codec class.
No installed package is edited. Dropped-packet tests cover both 8 kHz and 16 kHz
resampled receive outputs and check concealment payload length in the RTP clock domain.
The SDP tests use the actual UAS/UAC codec policies, not manually substituted codecs.

## G.711 A-law (PCMA)

Requires sippy >= 2.5.1 and rtpsynth >= 1.4.0. Infernos uses upstream
`G711ALawCodec` (static RTP payload type 8, PCMA/8000), including native
A-law conversion and packet-loss silence. It is accepted on incoming calls
and included in outgoing SDP offers. Existing codec preferences are retained:
incoming G722, PCMU, PCMA; outgoing PCMU, G722, PCMA. PCMU remains the RTP
parameter default. Codec changes on SIP updates use the same endpoint adapters.
