"""WikiText-2 (raw) as two plain text files: the CALIBRATION corpus and the
HELD-OUT corpus, from the dataset's own train / test split.

    python wikitext.py <out_dir>

writes `<out_dir>/wikitext2_train.txt` and `<out_dir>/wikitext2_test.txt`.

Why this corpus and not the Gutenberg books the first calibrations used: a
held-out split of the SAME corpus is the standard protocol, so the failure
rate it gives is a statement about unseen prompts rather than about a change
of domain -- and every shape can then be compared against every other.

The canonical copy is `Salesforce/wikitext`, config `wikitext-2-raw-v1`, in
parquet; the raw zip is tried if the hub is unreachable. The `= Heading =`
lines and the blank lines are kept: they are part of the corpus, and a
window that lands on one is a real prompt a service would see.
"""

import io
import os
import sys
import zipfile

SPLITS = {"train": "wikitext2_train.txt", "test": "wikitext2_test.txt"}


def from_hub(out):
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    for split, name in SPLITS.items():
        path = hf_hub_download(
            "Salesforce/wikitext", "wikitext-2-raw-v1/%s-00000-of-00001.parquet" % split,
            repo_type="dataset")
        text = "".join(pq.read_table(path).column("text").to_pylist())
        open(os.path.join(out, name), "w", encoding="utf-8").write(text)
        print("%-6s %9d chars -> %s" % (split, len(text), name))


def from_zip(out):
    import urllib.request

    url = "https://wikitext.smerity.com/wikitext-2-raw-v1.zip"
    raw = urllib.request.urlopen(url, timeout=120).read()
    z = zipfile.ZipFile(io.BytesIO(raw))
    for split, name in SPLITS.items():
        inner = "wikitext-2-raw/wiki.%s.raw" % split
        text = z.read(inner).decode("utf-8")
        open(os.path.join(out, name), "w", encoding="utf-8").write(text)
        print("%-6s %9d chars -> %s (zip)" % (split, len(text), name))


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(out, exist_ok=True)
    try:
        from_hub(out)
    except Exception as e:                                    # noqa: BLE001
        print("hub route failed (%s); falling back to the zip" % e)
        from_zip(out)
    for name in SPLITS.values():
        p = os.path.join(out, name)
        assert os.path.getsize(p) > 100000, p
    print("done ->", out)


if __name__ == "__main__":
    main()
