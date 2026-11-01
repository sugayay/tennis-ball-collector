# 配置と再実行

この資料は既存提出コードに追加するものです。既存の制御・認識コード、フォント一般化、Python指定の変更を戻しません。
人物用重みと学習資料はリポジトリ内に配置済みです。
以下のコマンドはリポジトリのルートから実行します。

## 1. モデルを配置して確認

```bash
python tools/check_person_model.py
python fetch_yolo_model.py
python tools/check_model_files.py
```

1行目は今回同梱した人物重みの固定SHAだけを検査し、pickleを実行しません。
2行目は既存取得コードで公式の球モデルを取得します。今回の準備中に取得は実行していません。
3行目は既存公開版の両モデル確認です。人物モデルのハッシュは既存manifestから変更しません。
人物モデルは無人デモにも必要です。`fixed_camera_config.json` や人物停止設定を無効化しません。
既存のREADMEの起動方法を使用してください。全シミュレーションの再実行は今回行っていません。

## 2. 学習時の画像を確認

535枚の実際のタイルPNGを収録しています。検査に学習やGenesisは不要です。

```bash
python person_training/tools/verify_training_dataset.py
```

各PNGのSHA-256を元manifestと照合します。画像が不一致なら停止し、上書きしません。
同梱した学習ラベルと画像は、既知の人物/無人の包含関係を維持しています。

## 3. 保存データを用いた追加学習（任意）

記録した環境はPython 3.12.3、Genesis 1.4.4、Ultralytics 8.4.182、Torch 2.9.1+ROCm 7.2.1、Pillow 11.3.0、AMD Radeon RX 7900 XTXです。
既存の動作環境を利用し、TorchやGPUランタイムを自動で入れ替えないでください。
以下は利用者が必要な場合だけ実行する操作です。この準備では再学習していません。

```bash
python person_training/tools/prepare_training_workspace.py --output person-rerun-01
cd person-rerun-01
python scripts/train_person_v1.py
```

新しい作業フォルダへデータをコピーし、公式初期重みを配置し、dataset.yamlのパスだけをその環境に合わせます。
学習コードは `work/tennis-drive-v49/models/yolo26n.pt` を読み、`outputs/tennis-ball-collector-suite/person-training-v1-20261024/runs/n-person-v1/weights/best.pt` を生成します。
同梱した証跡を直接更新せず、新しい学習結果は新しい学習結果として区別します。
当時の起動シェルコマンド文字列は残っていません。上記は保存コードの配置に合わせた今回の実行手順です。
学習時の `training-plan.json` と `training-result.json` に保持したハッシュは歴史的記録であり、パスを相対化した設定の現在ハッシュとは区別してください。

## 4. 合成画像生成をやり直す場合（任意）

```bash
python person_training/tools/prepare_training_workspace.py --output person-render-01 --render-new
cd person-render-01
python scripts/render_person_training_v1.py
python scripts/build_person_training_tiles_v1.py
```

元の生成コード、乱数seed、60場面の宣言、固定カメラ設定、CesiumMan資産と学習済み無人画像5枚を利用します。
`work/tennis-drive-v49/` は保存されていたv49ヘルパーのコピーです。生成当時の全シーンソースとの完全同一性を示す独立のハッシュ一覧は残っていません。
したがって異なるレンダラーで生成した新画像が当時の原画像と同じバイト列になるとは主張しません。同梱した535タイルPNGとラベルは当時の学習データと完全一致を確認済みです。
当時の120枚のレンダリング原画像はこのパッケージには含めず、目録を収録しています。保存タイルを使う追加学習には不要です。
development/validationの72画像はこのパッケージには含めていないため、それらの評価スクリプトの再実行には元の凍結画像が別途必要です。
導入スクリプトは当時のv50向け証跡として保存し、今回の人物重みは既存公開版の `models/` へ配置します。
