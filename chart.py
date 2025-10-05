#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
폴더 안의 CSV들을 읽어 gp_mean vs JD 그래프를 겹쳐(오버레이) 그립니다.
- CSV 파일은 JD, flux, error, gp_mean 필드를 포함한다고 가정합니다.
- 파일 10개 단위로 한 이미지(단일 축, 10개 라인 겹침)를 생성합니다.
- 결과 이미지는 chart/ 폴더에 PNG로 저장됩니다.

사용 예:
    python make_charts_overlay.py --input ./data --output ./chart --per 10
"""

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def chunked(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i+size]

def load_csv_required_cols(path: Path, required=("JD", "gp_mean")):
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"[SKIP] {path.name}: CSV 읽기 실패 -> {e}")
        return None

    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"[SKIP] {path.name}: 누락된 컬럼 {missing}")
        return None

    df = df[["JD", "gp_mean"]].dropna()
    if df.empty:
        print(f"[SKIP] {path.name}: 유효 데이터 없음")
        return None

    try:
        df = df.sort_values("JD")
    except Exception:
        pass
    return df

def make_overlay_figure_for_batch(files_batch, out_path: Path, title_prefix="", dpi=200):
    fig, ax = plt.subplots(figsize=(14, 8))

    used = 0
    jd_min, jd_max = None, None
    y_min, y_max = None, None

    for csv_path in files_batch:
        df = load_csv_required_cols(csv_path)
        if df is None:
            continue

        df["gp_mean"] -= df["gp_mean"].mean()  # 중앙값 보정
        ax.plot(df["JD"], df["gp_mean"], linewidth=1.0, alpha=0.9, label=csv_path.stem)

        # 범위 집계(서로 다른 스케일 파일 섞여 있을 때도 보기 좋게)
        cmin_x, cmax_x = df["JD"].min(), df["JD"].max()
        cmin_y, cmax_y = df["gp_mean"].min(), df["gp_mean"].max()
        jd_min = cmin_x if jd_min is None else min(jd_min, cmin_x)
        jd_max = cmax_x if jd_max is None else max(jd_max, cmax_x)
        y_min = cmin_y if y_min is None else min(y_min, cmin_y)
        y_max = cmax_y if y_max is None else max(y_max, cmax_y)

        used += 1

    if used == 0:
        plt.close(fig)
        print(f"[SKIP] 저장 건너뜀(유효 CSV 없음): {out_path.name}")
        return

    ax.set_xlabel("JD")
    ax.set_ylabel("gp_mean")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

    # 여백 약간, 레전드는 오른쪽 바깥에 배치
    # (항목이 많아도 그래프가 가려지지 않도록)
    ax.legend(title="Files", fontsize=9, title_fontsize=10, loc="center left",
              bbox_to_anchor=(1.01, 0.5))

    # 보기 좋은 여유를 두고 축 설정
    if jd_min is not None and jd_max is not None and jd_min != jd_max:
        pad_x = (jd_max - jd_min) * 0.02
        ax.set_xlim(jd_min - pad_x, jd_max + pad_x)
    if y_min is not None and y_max is not None and y_min != y_max:
        pad_y = (y_max - y_min) * 0.05
        ax.set_ylim(y_min - pad_y, y_max + pad_y)

    ax.set_title(f"{title_prefix}gp_mean vs JD  (lines: {used})")

    fig.tight_layout(rect=(0, 0, 0.82, 1))  # 우측 레전드 공간 확보
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] 저장: {out_path} (겹친 라인 {used}/{len(files_batch)})")

def main():
    parser = argparse.ArgumentParser(description="CSV들을 10개 단위로 겹쳐 그린 이미지 생성")
    parser.add_argument("--input", "-i", type=str, default=".", help="CSV 폴더 경로 (기본값: 현재 폴더)")
    parser.add_argument("--output", "-o", type=str, default="chart", help="이미지 저장 폴더 (기본값: chart)")
    parser.add_argument("--per", "-p", type=int, default=10, help="한 이미지당 CSV 개수 (기본값: 10)")
    parser.add_argument("--pattern", type=str, default="*.csv", help="CSV 파일 패턴 (기본값: *.csv)")
    args = parser.parse_args()

    in_dir = Path(args.input).expanduser().resolve()
    out_dir = Path(args.output).expanduser().resolve()

    if not in_dir.exists() or not in_dir.is_dir():
        print(f"[ERROR] 입력 폴더가 존재하지 않음: {in_dir}")
        return

    files = sorted(in_dir.glob(args.pattern))
    if not files:
        print(f"[WARN] CSV 파일이 없습니다: {in_dir}/{args.pattern}")
        return

    print(f"[INFO] 입력 폴더: {in_dir}")
    print(f"[INFO] 저장 폴더: {out_dir}")
    print(f"[INFO] 발견 CSV: {len(files)}개, {args.per}개씩 묶어서 오버레이 이미지 생성")

    batch_index = 1
    for batch in chunked(files, args.per):
        out_path = out_dir / f"charts_overlay_{batch_index:03d}.png"
        make_overlay_figure_for_batch(
            batch,
            out_path,
            title_prefix=f"Batch {batch_index:03d} — "
        )
        batch_index += 1

    print("[DONE] 모든 작업 완료")

if __name__ == "__main__":
    main()
