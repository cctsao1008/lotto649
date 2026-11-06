
import csv
import io
import re
import time
import zipfile
from pathlib import Path
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

VERSION = "0.0.2"
INDEX_URL = (
    "https://gaze.nta.gov.tw/"
    "dntmb/OpenData/csvDw?ntaCode=D423F"
)

OUTPUT = Path("data/raw")
OUTPUT.mkdir(parents=True, exist_ok=True)

session = requests.Session()
retry = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
)
session.mount("https://", HTTPAdapter(max_retries=retry))


def decode_csv(data):
    for encoding in ("utf-8-sig", "cp950", "big5"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError("Unsupported CSV encoding")


def get_links():
    print("[INFO] Fetching official D423F index")

    response = session.get(INDEX_URL, timeout=60)
    response.raise_for_status()

    text = decode_csv(response.content)
    reader = csv.reader(io.StringIO(text))

    links = []
    seen = set()

    for row in reader:
        for cell in row:
            matches = re.findall(
                r"https?://[^\s,\"'<>]+",
                cell,
                flags=re.IGNORECASE,
            )

            for url in matches:
                url = url.strip().rstrip(");")
                if url not in seen:
                    links.append(url)
                    seen.add(url)

    print(f"[INFO] Found {len(links)} URLs")
    return links


def download(index, url):
    response = session.get(
        url, timeout=90, stream=True
    )
    response.raise_for_status()

    parsed = urlparse(response.url)
    filename = Path(parsed.path).name

    if not filename or not filename.lower().endswith(".zip"):
        filename = f"archive_{index:03d}.zip"

    filename = f"{index:03d}_{filename}"
    path = OUTPUT / filename

    if path.exists() and zipfile.is_zipfile(path):
        print(f"[SKIP] {filename}")
        return

    temp = path.with_suffix(path.suffix + ".part")

    try:
        with temp.open("wb") as file:
            for chunk in response.iter_content(65536):
                if chunk:
                    file.write(chunk)

        if not zipfile.is_zipfile(temp):
            raise ValueError("Response is not a ZIP archive")

        temp.replace(path)
        print(f"[PASS] {filename} ({path.stat().st_size:,} bytes)")
    finally:
        temp.unlink(missing_ok=True)


def main():
    print(f"Taiwan Lottery Downloader v{VERSION}")
    print("Mode: requests only")

    try:
        links = get_links()
    except (requests.RequestException, ValueError) as exc:
        print(f"[FAIL] Index: {exc}")
        return

    if not links:
        print("[FAIL] No download URLs found")
        return

    passed = 0

    for index, url in enumerate(links, 1):
        try:
            download(index, url)
            passed += 1
        except (requests.RequestException, OSError, ValueError) as exc:
            print(f"[FAIL] {index}: {exc}")
        time.sleep(0.3)

    print(f"[DONE] {passed}/{len(links)} processed")


if __name__ == "__main__":
    main()
