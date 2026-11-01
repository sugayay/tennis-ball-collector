# モデルの配置と検証

匿名化済みの人物用重みを同梱しています。
球用の公式重みは取得スクリプトで別途配置します。
実行には、両モデルのハッシュ確認とREADME記載の環境が必要です。
人物停止機能の設定は変更していません。

| 必要ファイル | 由来・取得状況 | 現行実装での用途 |
| --- | --- | --- |
| `models/yolo26n.pt` | Ultralytics 公式 YOLO26n、v8.4.0 assets。公式URLとSHA固定の既存取得スクリプトあり | 固定・車載RGBから球を検出 |
| `models/yolo26n-person-synthetic-v1.pt` | 2026-10-24に公式YOLO26nから追加学習。匿名化済み重み、学習コード、535タイル画像と実ラベルを同梱。120原画像は目録のみ | 固定RGBから人物を検出する別モデル |

## 1. 球用モデル

利用条件を確認後、利用者が以下を明示的に実行します。今回の準備ではダウンロードしていません。

```bash
python fetch_yolo_model.py
```

取得先: https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt

公式リリース: https://github.com/ultralytics/assets/releases/tag/v8.4.0

SHA-256: `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`

既存スクリプトはハッシュ不一致なら一時ファイルを削除して停止します。正しい重みが存在すれば再取得しません。モデル形式はPyTorchチェックポイントなので、信頼できる配布元と固定ハッシュを確認し、任意のファイルを読み込まないでください。

## 2. 同梱した人物用モデル

提出用メタデータ除去版の必要SHA-256:
`2591906d7ca45eeb9cd199380cf42a73e42b82af0bce30f138e5ce8aa3abc1d5`

元の録画時モデルとの対応は `evidence/model-equivalence.json`、由来の記録は `models/person-model-provenance.json` にあります。元の録画時ハッシュと提出用ハッシュは異なります。モデル名が同じでも、ハッシュの違う重みを同一扱いしません。

学習コード、設定、合成データの生成元と素材条件は既存の実験記録から特定しました。
対応する資料を `person_training/` に収録しています。
人物用重みはリポジトリ内にあるため、追加の取得スクリプトは不要です。
利用条件は `LICENSE` と `THIRD_PARTY_NOTICES_PERSON.md` に記載しています。

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


## 2026-11-01の来歴確認と追加パッケージ

人物モデルは2026-10-24の追加学習出力 `best.pt` に由来します。リポジトリのルートで `python tools/check_person_model.py` と既存の `python tools/check_model_files.py` で確認してください。学習資料は `person_training/`、実行手順は `person_training/REPRODUCTION.md`、条件は `THIRD_PARTY_NOTICES_PERSON.md` です。同梱した535タイルPNGと実ラベルは当時の学習データと完全一致を確認済みです。本プロジェクトはAGPL-3.0-onlyで公開し、第三者素材の条件は出典の表示を維持しています。
