# モデル取得・検証（公開前準備）

このコピーはコード確認用です。重みを除外しただけで、再現可能な提出物が完成したとは扱いません。v85 元コード・人物停止機能は変更していません。

| 必要ファイル | 由来・取得状況 | 現行実装での用途 |
| --- | --- | --- |
| `models/yolo26n.pt` | Ultralytics 公式 YOLO26n、v8.4.0 assets。公式URLとSHA固定の既存取得スクリプトあり | 固定・車載RGBから球を検出 |
| `models/yolo26n-person-synthetic-v1.pt` | 既存の合成人物向けチェックポイント。元ZIPに公開配布URL・学習スクリプト・学習データ一覧なし | 固定RGBから人物を検出する別モデル |

## 1. 球用モデル

利用条件を確認後、利用者が以下を明示的に実行します。今回の準備ではダウンロードしていません。

```bash
python fetch_yolo_model.py
```

取得先: https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt

公式リリース: https://github.com/ultralytics/assets/releases/tag/v8.4.0

SHA-256: `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`

既存スクリプトはハッシュ不一致なら一時ファイルを削除して停止します。正しい重みが存在すれば再取得しません。モデル形式はPyTorchチェックポイントなので、信頼できる配布元と固定ハッシュを確認し、任意のファイルを読み込まないでください。

## 2. 人物用モデル（公開経路未整備）

提出用メタデータ除去版の必要SHA-256:
`2591906d7ca45eeb9cd199380cf42a73e42b82af0bce30f138e5ce8aa3abc1d5`

元の録画時モデルとの対応は `evidence/model-equivalence.json`、由来の記録は `models/person-model-provenance.json` にあります。元の録画時ハッシュと提出用ハッシュは異なります。モデル名が同じでも、ハッシュの違う重みを同一扱いしません。

公開前に、作成時の学習コード・設定・データ生成元・素材条件を特定し、配布条件と公開可否を確認したうえで、上記ハッシュの重みを利用者が取得できる公開経路を整える必要があります。このコピーからは未確認のURLを作れないため、人物用の自動取得スクリプトは作成していません。

### 無人の4球デモにも必要な理由

- `tools/run_cases.py` は開始前に `models/model-manifest.json` の **両モデル** を検証します。
- `vision_config.json` は球用重みを指定します。
- `fixed_camera_config.json` は `person_safety_enabled: true` と別人物モデルを指定します。
- `fleet_mission.py` は `FixedBallDetector` を構築し、`fixed_detection.py` は人物用重みが無ければ停止します。

したがって無人配置でも人物モデルは現行v85の依存です。設定無効化、チェックの削除、標準YOLOの人物検出への置換は検証済み版の変更となるため行っていません。

## 3. 配置後の読取り専用ハッシュ確認

次のコマンドはネットワーク通信やモデルの読み込みをせず、ファイルのSHA-256だけを計算します。

```bash
python tools/check_model_files.py
```

両方の検証が成功し、README記載の環境が用意できた後に `run.ps1` / `run-wsl.sh` で実行します。単体テスト306件の成功は、GPUシミュレーションの再現成功を意味しません。
