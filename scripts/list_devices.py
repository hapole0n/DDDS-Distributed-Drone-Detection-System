"""Quick utility: print every sounddevice-visible input."""

import sounddevice as sd


def main() -> None:
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            print(
                f"[{i}] {d['name']}  "
                f"in_ch={d['max_input_channels']}  "
                f"sr={int(d['default_samplerate'])}"
            )


if __name__ == "__main__":
    main()
