"""Download the DALI song audio referenced by the segment metadata.

Replaces the notebook-bound pytube downloader (and the old monitor_git.py that
auto-committed the media). Reads the unique (DALI id, YouTube id) pairs from a
segments JSON and downloads audio-only to ``Dataset/data/mp4/<dali_id>.mp4`` with
yt-dlp. Resumes automatically: already-downloaded ids and ids in
``broken_links.txt`` are skipped, and new failures are appended there.

Requires yt-dlp (`pip install yt-dlp`) and ffmpeg on PATH.

    python scripts/download_data.py                 # all songs
    python scripts/download_data.py --limit 50       # a quick subset
"""
import argparse
import json
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "Dataset", "data")


def load_song_ids(segments_file: str):
    with open(segments_file) as f:
        items = json.load(f)
    # dict preserves first-seen order and dedups by DALI id
    return {it["id"]: it["youtube"] for it in items if it.get("youtube")}


def read_broken(path: str):
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        return {line.strip() for line in f if line.strip()}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--segments", default=os.path.join(DATA_DIR, "segments_with_descriptions.json"),
                    help="Segments JSON providing (id, youtube) pairs.")
    ap.add_argument("--out-dir", default=os.path.join(DATA_DIR, "mp4"))
    ap.add_argument("--broken", default=os.path.join(DATA_DIR, "broken_links.txt"))
    ap.add_argument("--limit", type=int, default=None, help="Only download the first N songs.")
    args = ap.parse_args()

    try:
        import yt_dlp
    except ImportError:
        raise SystemExit("yt-dlp is required: pip install yt-dlp")

    os.makedirs(args.out_dir, exist_ok=True)
    songs = load_song_ids(args.segments)
    broken = read_broken(args.broken)

    todo = [(i, y) for i, y in songs.items()
            if i not in broken
            and not os.path.exists(os.path.join(args.out_dir, f"{i}.mp4"))]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(songs)} songs total; {len(todo)} to download "
          f"({len(broken)} known-broken, rest already present).")

    n_ok = 0
    for dali_id, yt_id in todo:
        out_path = os.path.join(args.out_dir, f"{dali_id}.mp4")
        opts = {"format": "bestaudio/best", "outtmpl": out_path,
                "quiet": True, "no_warnings": True, "noprogress": True}
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.download([f"https://www.youtube.com/watch?v={yt_id}"])
            n_ok += 1
            print(f"  ok  {dali_id}")
        except Exception as e:  # noqa: BLE001 - record any download failure
            print(f"  FAIL {dali_id} ({yt_id}): {e}")
            with open(args.broken, "a") as f:
                f.write(dali_id + "\n")

    print(f"Done: {n_ok}/{len(todo)} downloaded.")


if __name__ == "__main__":
    main()
