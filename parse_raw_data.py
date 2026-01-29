#!/usr/bin/env python3
"""Parse raw Yahoo Finance copy-paste data into standard CSV format.

Yahoo Finance format (tab-separated, newest first):
  Price:    Jan 28, 2026  14.72  14.78  14.64  14.72  14.72  3,156,700
  Dividend: Jan 22, 2026  0.085 Dividend
  Split:    Dec 1, 2025   1:5 Stock Splits

Output CSV format (oldest first):
  Date,Close,Dividend,Split
  2025-01-30,19.36,0.829,1
"""

from datetime import datetime
from pathlib import Path


def parse_yahoo_raw(raw_text: str) -> list[dict]:
    """Parse Yahoo Finance raw data into structured records."""
    records = {}

    for line in raw_text.strip().split("\n"):
        line = line.strip()
        if not line or line.startswith("Date"):
            continue

        parts = line.split("\t")
        if len(parts) < 2:
            continue

        date_str = parts[0].strip()
        try:
            date = datetime.strptime(date_str, "%b %d, %Y")
        except ValueError:
            continue

        date_key = date.strftime("%Y-%m-%d")
        if date_key not in records:
            records[date_key] = {"close": None, "dividend": 0, "split": 1}

        second_field = parts[1].strip()

        if "Dividend" in second_field:
            amount = float(second_field.replace("Dividend", "").strip())
            records[date_key]["dividend"] = amount
        elif "Stock Splits" in second_field or "Stock Split" in second_field:
            ratio = second_field.replace("Stock Splits", "").replace("Stock Split", "").strip()
            num, denom = ratio.split(":")
            records[date_key]["split"] = int(int(denom) / int(num))
        elif len(parts) >= 5:
            try:
                close = float(parts[4].strip().replace(",", ""))
                records[date_key]["close"] = close
            except (ValueError, IndexError):
                pass

    result = []
    for dk in sorted(records.keys()):
        r = records[dk]
        if r["close"] is not None:
            result.append({
                "date": dk, "close": r["close"],
                "dividend": r["dividend"], "split": r["split"]
            })
    return result


def write_csv(records: list[dict], path: str):
    with open(path, "w") as f:
        f.write("Date,Close,Dividend,Split\n")
        for r in records:
            f.write(f"{r['date']},{r['close']},{r['dividend']},{r['split']}\n")
    divs = sum(1 for r in records if r["dividend"] > 0)
    splits = sum(1 for r in records if r["split"] != 1)
    print(f"  {path}: {len(records)} trading days, {divs} dividend dates, {splits} split dates")


def main():
    targets = {
        "raw_data_1.txt": "nvdy_data_with_dividends.csv",
        "raw_tsly.txt": "tsly_data_with_dividends.csv",
        "raw_crsh.txt": "crsh_data_with_dividends.csv",
        "raw_cony.txt": "cony_data_with_dividends.csv",
        "raw_fiat.txt": "fiat_data_with_dividends.csv",
        "raw_dips.txt": "dips_data_with_dividends.csv",
    }

    print("Parsing Yahoo Finance raw data → CSV...")
    for raw, csv in targets.items():
        p = Path(raw)
        if p.exists():
            data = parse_yahoo_raw(p.read_text())
            write_csv(data, csv)
        else:
            print(f"  SKIP: {raw} not found")


if __name__ == "__main__":
    main()
