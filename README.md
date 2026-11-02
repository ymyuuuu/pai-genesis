# 制約は知能を生むか — 有限エネルギーの下で学習した4脚歩行の汎化

Physical AI 応用編1 最終課題。Genesis 上で Unitree Go2 の歩行方策を強化学習（PPO）で学習し、
**エネルギーが有限であるという制約が、未知環境への汎化（＝知能）を高めるか**を検証する。

- 問い・仮説・設計の決定ログ: [docs/research_decisions.md](docs/research_decisions.md)
- 結果の要約: [results/summary.md](results/summary.md)

## 結果（要約）
- 本実験（6条件 × 3シード）: エネルギー制約は汎化を高めなかった。未知環境の平均維持率は C0 0.56 / C1 0.56 / C2 0.51 / C3 0.50 / C4 0.44（**H1〜H3 不支持**）
- **頑健に再現した結果**: 有限性だけから、発熱の安い膝へ負担を移す「配分の判断」と節約が生まれた（膝のトルク²分担 0.64 → 0.83、消費電力 最大−47%）。新シードでもほぼ同じ値
- **追加検証（事前登録）**: 3シードで見えた環境ごとの差から立てた H4（トルクの余力で汎化の向きを予測できる）を、新しい環境（重力増加・登り斜面）と新しいシード（4〜8）で検証したが**不支持**。環境ごとの差も新シードでは再現しなかった
- **探索的な発見（未確認）**: 登り斜面では、エネルギー制約で学習した方策だけが一様に弱く、非物理量の予算（CTRL）では弱くならなかった

詳細: [results/summary.md](results/summary.md)（本実験）、[results/summary_h4.md](results/summary_h4.md)（追加検証）、説明資料: [docs/report/report.pdf](docs/report/report.pdf)

## 仮説
- **H1**: 適度なエネルギー制約の下で学習した方策は、制約なしの方策より未知環境での汎化性能が高い
- **H2**: 制約の強さと汎化の関係は逆U字になる
- **H3**: H1の効果は、強さをそろえた非物理量の予算（行動の大きさの総和）では同程度に得られない

## エネルギーモデル
全関節で一つの予算 $E_0$ を共有し、0になったらエピソード終了（ハードな制約）。残りエネルギーの割合を観測に含める。

$$E_{t+1} = E_t - \Delta t \left[ \sum_j \left( |\tau_j \dot q_j| + \alpha\, w_{k(j)} \tau_j^2 \right) + P_\text{basal} \right]$$

- 機械的仕事 + ジュール熱（関節の種類ごとの重み）+ 基礎代謝
- 膝は実効減速比が股・腿の1.5倍（Go2 URDFのトルク・速度上限から推定）なので、同じトルクの発熱が安い: $w = (1, 1, 1/1.5^2)$
- α と $P_\text{basal}$ は制約なしの方策で較正（発熱 = 機械的仕事、基礎代謝 = 総消費の20%）

## 条件
| 条件 | 予算（ベースラインの1エピソード消費に対する割合） |
|---|---|
| C0 | なし |
| C1 / C2 / C3 / C4 | エネルギー 100% / 70% / 50% / 30% |
| CTRL | 行動の大きさの総和に 70% |

各条件3シード、4096並列環境 × 1000反復。

## 評価
未知環境（訓練では見ていない）を強度のグリッドで評価し、追従報酬の維持率（未知環境 / 訓練環境）を比べる。
- E1 低摩擦、E2 荷物、E3 トルク上限の低下、E4 外乱（押し）
- 主評価: 学習時と同じ予算ルール / 副評価: 予算なし（残りエネルギーの観測を1.0に固定）

## 実行方法
```bash
uv sync
./run_experiments.sh c0       # 制約なしの学習（3シード）
./run_experiments.sh calib    # エネルギーモデルの較正 -> results/calibration.json
./run_experiments.sh train    # 制約ありの学習（C1-C4, CTRL × 3シード）
./run_experiments.sh eval     # 評価 -> results/eval/
./run_experiments.sh analyze  # 図と要約 -> results/
./run_h4.sh                   # 追加検証（新シード4〜8、E5 重力増加・E6 登り斜面）-> results/summary_h4.md
./run_h5_train.sh             # 学習のみ（確認用の新シード9〜14）。確認は行わなかったため未評価
```
動画: `record_video.py`（1クリップ）と `compose_video.py`（構成）。

## ファイル
| ファイル | 内容 |
|---|---|
| `energy_env.py` | エネルギー予算つき Go2 環境（Genesis の locomotion サンプルを改変） |
| `train.py` | 条件ごとの学習 |
| `calibrate.py` | α・基礎代謝・基準消費量の較正 |
| `evaluate.py`, `rollout.py` | 未知環境での評価 |
| `analyze.py` | 図と要約表 |
| `record_video.py`, `compose_video.py` | 動画 |

## 環境
genesis-world 1.4.5 / torch 2.15 (CUDA) / rsl-rl-lib 5.5.2 / Python 3.13 / RTX 4080 SUPER

## ライセンス表記
`energy_env.py` と `train.py` は [Genesis](https://github.com/Genesis-Embodied-AI/Genesis) の
`examples/locomotion`（Apache-2.0）を元に改変している。
