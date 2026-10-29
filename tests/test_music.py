"""Free music search: reading Openverse results, downloading a song and remembering its credit."""

import http.server
import threading
from functools import partial

from trip_memory import music

SAMPLE = {
    "results": [
        {"id": "a1", "title": "Sunny Days", "creator": "Ana", "duration": 185000, "url": "{url}",
         "foreign_landing_url": "https://www.jamendo.com/track/1", "license": "by", "license_version": "3.0",
         "license_url": "https://creativecommons.org/licenses/by/3.0/", "source": "jamendo"},
        {"id": "b2", "title": "Beep", "creator": "Bo", "duration": 2000, "url": "https://x/beep.mp3",
         "license": "cc0", "source": "freesound"},
    ]
}


def test_parse_keeps_songs_and_drops_jingles():
    songs = music.parse(SAMPLE)
    assert [s.title for s in songs] == ["Sunny Days"]
    assert songs[0].licence == "CC BY 3.0"
    assert "Sunny Days" in songs[0].credit and "Ana" in songs[0].credit


def test_download_and_credit(tmp_path, monkeypatch):
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.delenv(key, raising=False)
    (tmp_path / "web").mkdir()
    (tmp_path / "web" / "song.mp3").write_bytes(b"ID3fake-song")
    handler = partial(http.server.SimpleHTTPRequestHandler, directory=str(tmp_path / "web"))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{server.server_port}/song.mp3"
        song = music.parse({"results": [dict(SAMPLE["results"][0], url=url)]})[0]
        path = music.download(song, tmp_path / "music")
        assert path.read_bytes() == b"ID3fake-song" and path.suffix == ".mp3"
        assert music.credit_for(path) == song.credit
        assert music.credit_for(tmp_path / "other.mp3") is None
    finally:
        server.shutdown()


CCMIXTER = [
    {"upload_id": 7, "upload_name": "Road Song", "user_name": "bob", "user_real_name": "Bob B",
     "file_page_url": "http://ccmixter.org/files/bob/7", "license_name": "Attribution (3.0)",
     "license_url": "http://creativecommons.org/licenses/by/3.0/",
     "files": [{"download_url": "http://ccmixter.org/content/bob/road.mp3", "file_format_info": {"ps": "3:05"}}]},
    {"upload_id": 8, "upload_name": "No Edits", "user_name": "x", "license_url": "http://creativecommons.org/licenses/by-nd/3.0/",
     "files": [{"download_url": "http://ccmixter.org/content/x/a.mp3", "file_format_info": {"ps": "2:00"}}]},
]


def test_parse_ccmixter():
    songs = music.parse_ccmixter(CCMIXTER)
    assert [(s.title, s.artist, s.seconds) for s in songs] == [("Road Song", "Bob B", 185.0)]


def test_one_slow_library_is_fine(monkeypatch):
    def slow(*_args):
        raise TimeoutError("The read operation timed out")

    monkeypatch.setattr(music, "search_openverse", slow)
    monkeypatch.setattr(music, "search_ccmixter", lambda *_: music.parse_ccmixter(CCMIXTER))
    assert [s.title for s in music.search("road")] == ["Road Song"]


def test_credit_remembered_for_songs_you_download_yourself(tmp_path):
    song = music.parse_ccmixter(CCMIXTER)[0]
    name = music.remember_credit(song, tmp_path / "music")
    assert name == "road.mp3"
    assert music.credit_for(tmp_path / "Downloads" / "road.mp3", tmp_path / "music") == song.credit


def test_song_downloaded_yourself_is_found_and_adopted(tmp_path):
    import os
    import time

    song = music.parse_ccmixter(CCMIXTER)[0]
    downloads = tmp_path / "Downloads"
    downloads.mkdir()
    old = downloads / "old.mp3"
    old.write_bytes(b"old")
    os.utime(old, (time.time() - 3600, time.time() - 3600))
    since = time.time()
    assert music.find_download("road.mp3", since, downloads) is None
    (downloads / "road (1).mp3").write_bytes(b"new song")
    found = music.find_download("road.mp3", since, downloads)
    assert found.name == "road (1).mp3"
    path = music.adopt(found, song, tmp_path / "music")
    assert path.read_bytes() == b"new song"
    assert music.credit_for(path) == song.credit
