# 人物モデルの来歴

`yolo26n-person-synthetic-v1.pt` は本プロジェクトで2026-10-24に追加学習した人物検出用モデルです。
初期重みはUltralytics配布のYOLO26nで、球用には初期モデルをそのまま使用します。

- 初期URL: https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt
- 初期SHA-256: `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`
- 学習開始: 2026-10-24T06:47:45.534653+00:00
- 学習終了: 2026-10-24T07:01:43.954794+00:00（約838秒、37 epochs）
- 学習: `scripts/train_person_v1.py` の `model.train(**arguments)`。最大40 epochs、patience 12、imgsz 1280、batch 4、AdamW、seed 20261024、freeze 10、人物クラス0。元の80クラス出力を保持。
- データ・設定・ログ: `outputs/tennis-ball-collector-suite/person-training-v1-20261024/`。
- 合成生成: `scripts/render_person_training_v1.py`。切り出し: `scripts/build_person_training_tiles_v1.py`。
- 導入記録: `scripts/adopt_person_v1_into_v50.py`。今回の配布手順は `REPRODUCTION.md` を参照。
- 元学習成果物SHA-256: `df315a4e110a8f72e1fc0fe0ce6cef46ba24c99b40ae091ebd28192f7d4c5536`
- 今回同梱の匿名化版SHA-256: `2591906d7ca45eeb9cd199380cf42a73e42b82af0bce30f138e5ce8aa3abc1d5`
- 2026-10-31の変更: 学習時の個人パス5文字列を削除。再学習なし。今回も712個のテンソル保存領域と全非pickleメンバーが元モデルとバイト一致することを静的確認。
- 2026-11-01の資料変更: 絶対個人パスを相対化。生成アルゴリズム、学習パラメーター、画像・ラベルは変更していません。変更前後ハッシュは `provenance/file-transformations.json`。

元の学習コード、設定、データ、成果物のハッシュ対応を保存しています。
新しい環境での再学習が同じ重みのバイト列を必ず生成するとは主張しません。
追加学習元の単一静止姿勢、限られた身長・位置・向きの合成画像での結果です。
実写、実機、有人運用、安全距離、未知環境での性能・安全性は未検証です。
保存済みの独立評価はdevelopment 48枚、smoke 9枚、validation 24枚です。
smokeの無人5枚は学習にも含まれ、独立評価とは区別します。
development/validationの画像72枚はこの追加パッケージに含めず、目録と評価結果のみを収録しています。

元・匿名化版のモデル内には `AGPL-3.0 (https://ultralytics.com/license)` の表示があります。
第三者条件は `../LICENSES/AGPL-3.0.txt` と `../THIRD_PARTY_NOTICES_PERSON.md` を参照してください。
本プロジェクトの自作部分と人物モデルは `../LICENSE` のAGPL-3.0-onlyで公開します。
第三者素材の条件は `DATASET_ATTRIBUTION.md` を参照してください。
