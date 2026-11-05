"""CLI / main.py: flag --render & --report, alias lama, dan subcommand auto-area."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import hauling_offline.cli as cli  # noqa: E402
from hauling_offline.pipeline import RunSummary  # noqa: E402

REQ = ["--video", "v.mp4", "--vehicle-model", "m.pt", "--config", "c.conf", "--output-dir", "out"]


def test_default_only_core_outputs():
    a = cli.build_parser().parse_args(REQ)
    assert a.render is False and a.report is False and a.track_cards is True


def test_render_and_report_flags():
    a = cli.build_parser().parse_args(REQ + ["--render", "--report"])
    assert a.render is True and a.report is True


def test_old_save_video_alias_still_works():
    assert cli.build_parser().parse_args(REQ + ["--save-video"]).render is True
    assert cli.build_parser().parse_args(REQ + ["--no-save-video"]).render is False


def test_flags_reach_run_options(monkeypatch):
    seen = {}

    def fake_run(opt):
        seen["opt"] = opt
        return RunSummary()

    monkeypatch.setattr(cli, "run", fake_run)
    assert cli.main(REQ + ["--render", "--report", "--no-track-cards"]) == 0
    opt = seen["opt"]
    assert opt.save_video is True and opt.save_report is True and opt.save_track_cards is False
    assert cli.main(REQ) == 0
    assert seen["opt"].save_video is False and seen["opt"].save_report is False


def test_auto_area_subcommand(monkeypatch):
    import hauling_offline.auto_area as aa
    got = {}
    monkeypatch.setattr(aa, "main", lambda argv: got.setdefault("argv", argv) and 0)
    cli.main(["auto-area", "--video", "v.mp4", "--vehicle-model", "m.pt", "--out", "a.json"])
    assert got["argv"] == ["--video", "v.mp4", "--vehicle-model", "m.pt", "--out", "a.json"]


def test_main_py_is_entry_point():
    r = subprocess.run([sys.executable, str(ROOT / "main.py"), "--help"], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0 and "--render" in r.stdout and "--report" in r.stdout
