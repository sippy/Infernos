# I.N.F.E.R.N.O.S.

### Interactive Neural Framework for Efficient Realtime Network Operations on Streams

🔥 Welcome to Infernos, where data comes to life in real-time! 🔥

## Overview

Harness the power of **I.N.F.E.R.N.O.S.** to transform audio, video, and
text streams with state-of-the-art inference in an instant. Embrace a
blazing-fast future, free from lag.

## News

Initial integration of the LLM (Qwen 2.5) and addition of the A.I.
Attendant application.

Upcoming presentation at the OpenSIPS Summit 2025.

## Features

-   **Interactive:** Infernos isn't just another tool; it's an
    experience. Speak in one voice and marvel as it's automatically
    translated into a completely different tone or even language, and
    then seamlessly transmitted in real-time during phone or web
    meetings.

-   **Neural Power:** With deep learning at its core, Infernos is
    optimized for top-notch performance.

-   **Multimodal Support:** Whether it's audio, video, or text, Infernos
    handles them with elegance.

-   **Efficiency:** Designed for low-latency, high-throughput
    operations.

-   **Realtime:** Don't wait. Experience the magic as it unfolds.

## Quick Start

1.  Clone the repository:

    ```bash
    git clone https://github.com/sippy/Infernos.git
    ```

2.  Navigate to the project directory and install dependencies:

    ```bash
    cd Infernos && pip install -r requirements.txt
    ```

3.  Create a configuration file. In the following example we would
    listen and accept SIP calls from `MY_IP` and pass them into Live
    Translator application. Then use SIP account to send
    outbound call legs to `DEST_NUM`@`MY_SIP_SRV`:

    ```bash
    MY_IP="A.B.C.D"
    MY_SIP_SRV="E.F.G.H"
    DEST_NUM="12345"
    DEST_USER="foo"
    DEST_PWD="bar"
    cat > config.yaml <<EOF
    sip:
      settings:
        bind: ${MY_IP}:5060
      profiles:
        me:
          sip_server: ${MY_IP}:*
          sink: apps/live_translator/configuration1
        bar:
          sip_server: ${MY_SIP_SRV}:5060
          username: '${DEST_NUM}'
          password: '${DEST_PWD}'
          register: True
    apps:
      live_translator:
        profiles:
          configuration1:
            stt_langs: ['en', 'pt']
            tts_langs: ['pt', 'en']
            outbound: sip/bar;cld=${DEST_NUM}
    EOF
    ```

4.  Light the fire:

    ```bash
    python Infernos.py -f -L ~/Infernos.log
    ```

5.  Use SIP device or software such as Linphone to place a SIP
    call to `sip:anything@localhost:5060`. Replace `localhost`
    with a local IP of machine running Infernos if testing over
    LAN.

Ready to experience real-time inferencing?

## Contribute

Feeling the warmth? 🔥 Eager to stoke the flames of Infernos? Delve into
our contribution guidelines and join the firestorm!

## License & Acknowledgements

Powered by the 2-clause BSD license. A heartfelt shoutout to the
community for their priceless insights and tireless contributions.

## Media

- [Setting up live translation service with Infernos:](https://www.youtube.com/live/-mTH1BpIMqY?t=26160s)
  Live presentation during OpenSIPS Summit 2024 setting up realtime in-call
  translation inference service for Portugese->English / English->Portugese
  on a AWS instance "from zero to hero" in less than 60 minutes.
- [Infernos: cost efficient AI inference for real-time applications:](https://www.youtube.com/watch?v=eawO0hXeO5Y)
  Overview of the Infernos architecture and progress over the past few months.

## Join US

- [Discord](https://discord.gg/bb95ZWhrhQ)

------------------------------------------------------------------------

Stay on the lookout for more sizzling updates, and always remember:
**Infernos** makes the future sizzle!
