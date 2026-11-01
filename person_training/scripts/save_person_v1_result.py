from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/tennis-ball-collector-suite'
TRAIN=OUT/'person-training-v1-20261024';read=lambda p:json.loads(p.read_text())
reports={name:read(TRAIN/f'evaluation-{name}.json') for name in ('development','smoke','validation')}
trained=read(TRAIN/'training-result.json');audit=read(TRAIN/'dataset-audit.json')
assert audit['verified'] and all(r['all_supplied_images_pass'] and r['weights_sha256']==trained['best_sha256'] for r in reports.values())
summary=dict(status='offline_synthetic_gate_passed_integrated_validation_pending',
    person_weights_sha256=trained['best_sha256'],original_ball_weights_unchanged=trained['original_ball_weights_unchanged'],
    image_counts={k:r['images'] for k,r in reports.items()},person_detections={k:r['true_positives'] for k,r in reports.items()},
    false_positives={k:r['false_positives'] for k,r in reports.items()},false_negatives={k:r['false_negatives'] for k,r in reports.items()},
    reserved_validation_evaluated_once=True,confidence=.75,full_image_tile_size=1024,
    known_negative_training_overlap_images=5,real_people_or_operational_safety_verified=False,
    evaluated_at_utc=datetime.now(timezone.utc).isoformat())
(TRAIN/'acceptance.json').write_text(json.dumps(summary,indent=2)+'\n')
(TRAIN/'RESULTS.md').write_text('''# 人物用YOLOの追加学習結果

2026-10-24。合成画像でのオフライン評価を通過。実際の人・有人運転は未検証。

## 結果

| 評価 | 画像数 | 人物検出 | 見逃し | 誤検出 |
|---|---:|---:|---:|---:|
| 既存の開発用画像（学習未使用） | 48 | 16/16 | 0 | 0 |
| 既知の誤検出を含む再発確認 | 9 | 4/4 | 0 | 0 |
| 最後まで残した検証用画像（学習・モデル選び未使用） | 24 | 8/8 | 0 | 0 |

元のYOLO26nは開発用48画像の人物6/16件を検出。今回は同じ画像・信頼度0.75・1024pxの全画像タイル・1m以内の位置一致で16/16件を検出した。閾値や人物の検索範囲は変更していない。
再発確認の無人画像5枚は追加学習にも使用しており、未知画像の評価とは区別する。追加学習と、既存の開発用48枚・検証用24枚に同じ画像はない。

## 学習

- 60場面を2台の固定カメラで撮影し、120枚の合成画像を生成。
- 場面単位で分け、切り出し後は学習441枚・調整確認94枚。部分的に人物が切れた画像を無人として教えていない。
- 人物の正解枠はシミュレーターの人物メッシュから生成し、近・遠距離の画像と照合。ロボの制御にはこの正解データを渡さない。
- 既存YOLO26nの80クラスの出力層を維持し、人物ラベルで追加学習。最大40周・改善停止12周を設定し、37周で終了。
- 約838秒。調整確認用のスコアで最良モデルを選び、既存の開発用と再発確認を通過した後、検証用24枚を1回評価。
- 人物専用の別モデルとしてv50へ導入。球検出用の元モデルはハッシュ一致で未変更。

## 限界

人物はCesiumManの静止姿勢1種類。位置・向き・身長3段階・ロボの位置とショベル姿勢を変えているが、服装、歩行姿勢、照明、複数人、実写の多様性は検証していない。
固定画像の正答だけでは、走行中の停止・再開、画像更新の期限、積載球の保持を保証しない。v50で無人の回収制御との統合を検証中。

詳細はtraining-plan.json、dataset-audit.json、evaluation-*.json、acceptance.jsonを参照。
''',encoding='utf-8')
status_path=ROOT/'work/ACTIVE_PROGRESS.json';status=read(status_path)
status.update(status='verifying_v50_integrated_person_and_pickup',active_source='work/tennis-drive-v50',
    active_exec_session=89863,active_queue='outputs/tennis-ball-collector-suite/v50-acceptance-queue-20261024.json',
    person_model_status='Synthetic person-v1 passed development16/16, smoke4/4 and sealed validation8/8, zero FP/FN; v50 integration under test',
    current_step='Frozen v50: long twice, original7, full return, six deliberate empty attempts; stop on any failure',
    canonical_copy_proof='outputs/tennis-ball-collector-suite/v49-canonical-copy.json',
    updated_at_utc=datetime.now(timezone.utc).isoformat())
status_path.write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(summary,indent=2))
