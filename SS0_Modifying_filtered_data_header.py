from pathlib import Path

folder = Path(r"F:\Lab\Raw_data\TG915\Fixed_header_2026-05-27_19-14-21")

NLX_HEADER_SIZE = 16 * 1024

BOOL_FIELDS = [
    "-InputInverted",
    "-DSPLowCutFilterEnabled",
    "-DSPHighCutFilterEnabled",
]

for p in sorted(folder.glob("*.ncs")):
    data = p.read_bytes()
    header = data[:NLX_HEADER_SIZE]
    body = data[NLX_HEADER_SIZE:]

    text = header.decode("latin-1", errors="replace").rstrip("\x00")

    # 1) 修 bool-like fields: 1/0 -> True/False
    for field in BOOL_FIELDS:
        text = text.replace(f"{field} 1", f"{field} True")
        text = text.replace(f"{field} 0", f"{field} False")

    # 2) 修 ApplicationName 格式
    # filtered: -ApplicationName Cheetah 6.4.2 Development
    # raw:      -ApplicationName Cheetah "6.4.2 Development"
    text = text.replace(
        "-ApplicationName Cheetah 6.4.2 Development",
        '-ApplicationName Cheetah "6.4.2 Development"',
    )

    # 3) 可选：修奇怪的 delay 字段名
    text = text.replace("-DspFilterDelay__s", "-DspFilterDelay_µs")

    new_header = text.encode("latin-1", errors="replace")

    if len(new_header) > NLX_HEADER_SIZE:
        print("Header too long:", p.name)
        print("Length:", len(new_header))
        print("Limit:", NLX_HEADER_SIZE)
        raise ValueError(f"Header became too long for {p.name}")

    new_header = new_header.ljust(NLX_HEADER_SIZE, b"\x00")
    p.write_bytes(new_header + body)

    print("Fixed:", p.name)

print("\nDone. Modified folder in place:")
print(folder)