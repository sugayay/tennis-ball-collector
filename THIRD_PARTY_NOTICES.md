# Third-party notices / 公開前確認

本プロジェクトの自作部分は `LICENSE` のAGPL-3.0-onlyで公開します。
この文書は第三者の出典と適用条件をまとめたものです。
確認日: 2026-11-01。

## Ultralytics

外部依存のUltralyticsとYOLOモデルの利用条件は公式 https://www.ultralytics.com/license を参照してください。公式は大学課題・個人プロジェクトについてもAGPL-3.0の条件と全体ソース公開を案内しています。モデル重みを除外すれば条件が消えるとは扱いません。人物モデルに対応する学習コード、設定、535枚のタイル画像と実ラベルは `person_training/` に収録しています。Ultralyticsライブラリ本体と公式の球用重みは同梱していません。
追加学習した人物用重みと対応する資料は同梱しています。

## Cesium Man

Copyright 2017 Cesium: CC BY 4.0。Copyright 2015 Cesium: Cesium Trademark or Logo notice。

出典: https://github.com/KhronosGroup/glTF-Sample-Assets/tree/4995a638c96dbc94376d3d6164e70fee3ac7f7a2/Models/CesiumMan

`assets/person-cesium/` の `README-source.md`、`CC-BY-4.0.txt`、`LicenseRef-LegalMark-Cesium.txt`、`provenance.json` をそのまま保持しています。第三者の必要な権利表示は匿名化対象から除外します。素材表示は支援・推奨・商標使用の許諾を意味しません。

## Genesis

外部依存であり本体は同梱していません。公式ライセンス: https://github.com/Genesis-Embodied-AI/Genesis/blob/main/LICENSE （Apache-2.0）。導入する版の条件も確認してください。

その他の外部依存は `requirements` 系ファイルと `evidence/environment.json` を参照してください。OSフォント、GPUランタイム、Python依存のバイナリは同梱しません。


## 人物モデル追加資料（2026-11-01）

匿名化済み人物モデル、対応する学習コード、設定、535枚のタイル画像と実ラベル、原画像の目録、第三者ライセンスの原文を同梱しています。AGPL原文は `LICENSES/AGPL-3.0.txt`、追加表示と残件は `THIRD_PARTY_NOTICES_PERSON.md`、Cesium由来データの加工表示は `person_training/DATASET_ATTRIBUTION.md` を参照してください。プロジェクトの自作部分に適用するAGPL-3.0-onlyの原文は `LICENSE` にあります。
