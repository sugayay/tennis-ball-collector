# モデルと実行環境の区別

球検出はUltralytics公式の学習済みYOLO26nをそのまま使用し、この球モデルへの追加学習は行っていません。
人物検出は2026-10-24に同じ公式YOLO26nを初期重みとして、Genesisの合成画像で追加学習した別モデルを使用します。
「独自学習したものとして扱いません」という旧文書の記述は、公式の球用重みにだけ当てはまります。

人物モデルの由来、変更日、元/匿名化版ハッシュは `person_training/MODEL_CARD.md`、手順は `person_training/REPRODUCTION.md`、素材条件は `person_training/DATASET_ATTRIBUTION.md` に記録しています。
実行環境と起動方法は既存READMEを参照し、公開版のフォント探索とPython実行パス一般化は維持してください。
本追加パッケージはGPU環境のインストールや設定変更を行いません。
