import urllib.error
import urllib.request

import pytest

from trip_memory import sharephone


def test_a_video_downloads_from_its_secret_link_and_nothing_else(tmp_path):
    video = tmp_path / "Croatia 2026.mp4"
    video.write_bytes(bytes(range(256)) * 100)
    link = sharephone.share(video, ip="127.0.0.1")
    with urllib.request.urlopen(link) as r:
        assert r.read() == video.read_bytes()
        assert "attachment" in r.headers["Content-Disposition"]
    part = urllib.request.Request(link, headers={"Range": "bytes=10-19"})
    with urllib.request.urlopen(part) as r:
        assert r.status == 206 and r.read() == video.read_bytes()[10:20]
    wrong = link.rsplit("/", 2)[0] + "/guess/x.mp4"
    with pytest.raises(urllib.error.HTTPError):
        urllib.request.urlopen(wrong)
    assert sharephone.qr_svg(link).lstrip().startswith("<svg")
