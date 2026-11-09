"""run_verification_a.py の CLI エラーメッセージのテスト。評価自体は行わない。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_verification_a


def test_missing_records_csv_reports_path_and_exits_1(tmp_path, capsys):
    """存在しない登録CSVはトレースバックではなく原因明示で終了1になる。"""
    missing = tmp_path / "no-such-records.csv"
    assert run_verification_a.main([str(missing), str(tmp_path)]) == 1
    err = capsys.readouterr().err
    assert str(missing) in err
    assert "Traceback" not in err
