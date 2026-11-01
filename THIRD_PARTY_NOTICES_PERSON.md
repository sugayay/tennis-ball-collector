# 人物モデル追加パッケージの第三者表示

Ultralytics由来のモデル内にはAGPL-3.0の表示があります。追加学習版の条件を取り除いたり、独自のモデルライセンスへ変更したりしていません。
第三者ライセンス原文は `LICENSES/AGPL-3.0.txt`（Ultralytics v8.4.182のLICENSE）です。
公式案内: https://www.ultralytics.com/license
原文出典: https://github.com/ultralytics/ultralytics/blob/v8.4.182/LICENSE
対応する学習コード・設定・データ・変更履歴は `person_training/` に収録しています。
本プロジェクトの自作部分に適用するAGPL-3.0-onlyの原文は `LICENSE` にあります。

Cesium Man: Copyright 2017 Cesium（CC BY 4.0）、Copyright 2015 Cesium（Cesium Trademark or Logo）。
出典とレンダリング加工の表示は `person_training/DATASET_ATTRIBUTION.md`、原文は `LICENSES/CesiumMan/` と学習用ソースの `assets/person-cesium/` に保持しています。
Genesis本体・Ultralytics本体・GPUランタイム・フォントのバイナリは同梱しません。

## 配布条件と再実行の範囲

1. 本プロジェクトの自作部分と人物モデルはAGPL-3.0-onlyで公開します。改変版を配布する場合なども、適用される対応ソースの提供条件を守ってください。
2. 人物重みと対応する学習ソース、535タイル画像と実ラベルを同じリポジトリで提供します。
3. AGPL原文、Cesiumの著作権・CC BY・加工表示・商標表示を外部配布でも保持すること。
4. 独立評価72画像の再実行が必要なら、今回含めなかった凍結画像も用意すること。
