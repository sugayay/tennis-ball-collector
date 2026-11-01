# テニス球拾いロボ

固定RGBカメラから車載RGBカメラへ球の追跡を引き継ぎ、2台のショベル式ロボットで回収するGenesisの試作です。
検証対象は昇降かごを備えた v85 です。
コントローラー、機体、認識設定、しきい値は2026-10-29の検証版を維持しています。

## 確認できた範囲

- 2026-10-29の固定7配置で各4/4球、合計28/28球を回収しました。
- 球全体が実際のカゴ座標の内側に3秒間残り、画像による増加判定と一致しました。
- 2026-10-31に保存された120Hzの生計測を再判定し、7配置すべての結果とハッシュを照合しました。
- 7配置を1回ずつ実行した開発用の結果です。28球を同時に扱った結果や、ランダム配置の成功率ではありません。
- このソースの回帰テストは306件です。提出コピーでの結果は `evidence/submission-tests.txt` に記録します。

練習中にプレーヤーと共存する実用機の完成は主張しません。
固定7配置は無人の仮区域です。
自分の位置と姿勢はシミュレーターから取得します。
球の真値は初期配置と独立評価だけで使い、回収制御には渡しません。
人物は合成画像向けの既存の別モデルを使用し、実写性能は未確認です。

## 確認環境

保存記録の検証環境は Windows と WSL Ubuntu 24.04 です。実行用 Python は `PHYSICAL_AI_PYTHON` または PowerShell の `-PythonPath` で指定します。既定値は `python3` です。
Python 3.12、Genesis 1.4.4、Ultralytics 8.4.182、ROCm対応Torch 2.9.1、Radeon RX 7900 XTXで検証しています。
正確な実行時の版は `evidence/environment.json` に保存します。
既存Torchを一般的なCUDA版へ置き換えるインストーラーは含みません。

RGB描画はllvmpipe、認識は既存AMD GPUを使います。
既定の物理計算は元の7配置と同じAMD GPUです。
`-Backend cpu` も選べますが、物理バックエンドを変えた結果を同じ成功実績とは扱いません。
オフライン時計は最大0.5シミュレーション秒/実秒に制限します。
推論中も物理時計は進み、この上限は推論の完了や球の状態を見て変えません。
実時間性能を測ったものではありません。

公開前の整理版ではモデル重みを分離しており、`models/` に `.pt` ファイルは含めていません。両方のモデルを別途配置しない限り、シミュレーションは開始できません。
`models/model-manifest.json` のSHA-256で両方を検証します。
保存済みの人物モデルは、録画後に学習時の個人パス5文字列だけを除去した版です。
全712テンソル領域のバイト列、読み込み後の708テンソル、推論出力が元と一致することを確認しました。
元と提出版のチェックポイントのハッシュの対応は `evidence/model-equivalence.json` に保存します。
新しい学習は実行していません。
動画の日本語表示は、Windowsのシステム設定またはWSLのマウント先からMeiryoを探します。別の日本語フォントは環境変数 `PHYSICAL_AI_FONT` で指定できます。
別のOSではフォント経路とROCm環境の適合確認が必要です。

## 起動

このフォルダでPowerShellを開きます。
出力先を毎回新しく指定してください。
既存の出力がある場合は上書きせず停止します。

```powershell
.\run.ps1 -Case 02_diagonal -OutputDir .\runs\diagonal-01
.\run.ps1 -Case 01_straight -OutputDir .\runs\straight-01 -NoVideo
.\run.ps1 -AllCases -OutputDir .\runs\seven-layouts-01 -NoVideo
```

Pythonの場所を変える場合は `-PythonPath /absolute/path/to/python` を指定します。
WSL Ubuntu 24.04と必要ライブラリの準備は利用者の環境に合わせて行います。
このスクリプトは認証、OS設定、パッケージのインストールを変更しません。
YOLOの実行時設定はこのソース内の `.runtime/` に作ります。

WSLからは同じ内容を次のように実行できます。

```bash
bash run-wsl.sh --case 02_diagonal --output-dir runs/diagonal-01
PHYSICAL_AI_PYTHON=/absolute/path/to/python bash run-wsl.sh --all-cases --output-dir runs/all-01 --no-video
```

7配置は逐次実行します。
出力の各フォルダには `mission_check.json`、独立した `full-basket-independent-audit.json`、生計測、画像、任意の動画を保存します。
`suite-summary.json` で各実行の終了コード、回収数、成否を区別します。
回収失敗は終了コード1、実行できなかった場合も成功には数えません。
途中の失敗を隠すための再試行は行いません。

## テスト

NumPyを使えるPythonで実行します。
物理シミュレーターとGPUは、このテストだけなら不要です。

```powershell
python -B -m unittest discover -s tests -q
```

元のテストが参照していた別フォルダの旧機体URDFは、同じバイト列を `tests/fixtures/collector_rigid_v78.urdf` へ同梱しました。
判定や期待値は変えていません。
一時ファイルを作れない環境では、プロセス用のTEMP/TMPを書き込み可能な作業フォルダへ指定してください。

## 構成

| 場所 | 役割 |
| --- | --- |
| `fleet_mission.py` | 画像、制御、物理計測の接続 |
| `mission_control.py` / `handoff_control.py` | 2台の担当、接近、回収、停止 |
| `fixed_detection.py` / `onboard_vision.py` | RGB画像から球と人物を検出 |
| `basket_confirmation.py` / `basket_vision.py` | カゴ内の画像による増加判定 |
| `assets/collector_basket_lift.urdf` | 提出候補の機体 |
| `cases/` | 変更していない固定7配置 |
| `tools/offline_runtime.py` | 実行環境の記録とオフライン時計の上限 |
| `tools/audit_full_basket.py` | 保存した生計測から球全体の3秒保持を再計算 |
| `evidence/` | 実行環境、版の照合、試験の集計 |

過去の単体デモを構成するPythonも回帰テストのため残しています。
提出候補の入口はこのREADMEの `run.ps1` / `run-wsl.sh` です。
他の旧デモで提出実績を再現したと解釈しないでください。

## 残っている限界

回収中の最大傾きは固定7配置で約6.99度でした。
当時の8度未満の判定は通りましたが、追加目標の5度以下を全配置で満たしてはいません。
補充、手振り認識、受け渡し機構の個別実験を、この提出版で統合して完了した成果とは扱いません。
有人運用、実写、実機、未知配置、通信遅延に対する性能は未確認です。
v66で測った0.125秒の車載画像遅延による失敗は、v85の遅延耐性の測定値へ転用しません。
制動や人物認識の単体テストは、実際の人に対する安全保証ではありません。

## モデルと出典

球モデルはUltralytics配布のYOLO26nです。
人物モデルは既存の合成画像向け重みで、由来とハッシュを `models/person-model-provenance.json` に保存しています。
取得元のライブラリとモデルの利用条件は、それぞれの公式配布元を確認してください。

- Genesis: https://github.com/Genesis-Embodied-AI/Genesis
- Ultralytics: https://github.com/ultralytics/ultralytics
- 球モデルの配布: https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt



## モデルの配置と公開前の確認

- 球用重みは `python fetch_yolo_model.py` で公式配布元から取得し、記録済みハッシュで照合できます。この取得処理は今回の公開準備では実行していません。
- 人物用重みは `models/yolo26n-person-synthetic-v1.pt` に配置する必要があります。公開用の配布方法、学習用ソースと資産の権利条件は確認中です。未確認の重みを代用品へ差し替えて同じ結果を主張しないでください。
- Ultralytics のコード・モデルには利用条件があります。公式案内: https://www.ultralytics.com/license 。この整理版にプロジェクト全体のライセンスはまだ付与していません。公開前に条件を確認する必要があります。
- `assets/person-cesium/` は Cesium Man の第三者素材です。出典・CC BY 4.0・商標に関する元の表示を同フォルダに保持しています。作者情報の匿名化を理由に第三者の権利表示を削除していません。

## この整理版での確認

2026-11-01、ZIP展開時に、保存された v85 の中核50ファイルの SHA-256 が記録と一致することを確認しました。公開用コピーでは、描画に使うフォントの固定パスを環境から探す処理に置き換えました。7ファイルの変更前後のハッシュは `evidence/source-integrity.json` の `publication_font_path_cleanup` に記録しています。制御・認識のロジックと設定は変更していません。ランチャーの Python 既定パスを一般化し、README と公開除外設定を整理しました。

クラウド環境で `python -B -m unittest discover -s tests -q` を実行し、306件すべて通過しました。GPU・レンダラー・人物用重みを含むシミュレーションの再実行はしていません。保存済みの実験結果と今回の単体テストを区別してください。

実験証拠は `evidence/` の集計・ハッシュ記録です。120 Hz の生計測すべてはこの整理版に含めていません。PDF と動画は別の成果物として管理し、古い版はこのリポジトリに同梱していません。

詳細な取得手順・人物モデルが無人デモでも必須である根拠は [MODEL_SETUP.md](MODEL_SETUP.md)、第三者表示は [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) を参照してください。公開前の未完了項目は人物用重みの配布経路・対応する学習資料・ライセンス方針です。
