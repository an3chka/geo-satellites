# Geo-satellite snapshots: classifying objects in 20 classes

Solution for a Kaggle competition from the *Introduction to Machine Learning* course
(3rd year, October 2026), team phase. Each image is a 256×256 satellite snapshot of one of
20 classes (bridge, church, lake, railway station, runway, …). The training set has 4,999 clean
images and the validation set 699; the 8,299 test images also contain distorted snapshots (blur,
low resolution, grayscale, darkening, noise). Metric: Accuracy.

**Result:** held-out validation accuracy **0.9742**, public leaderboard **0.9673**.

Course limits: only ImageNet-pretrained AlexNet / VGG / ResNet / Inception families, no external
data, training within 45 minutes and test prediction within 2 minutes on the free Kaggle GPU
(T4 ×2). The final notebook trains in 38.0 min and predicts the test set in 127 s (the
instructors announced a 30 s tolerance for the prediction step).

*Русская версия — [ниже](#русская-версия).*

## Approach

1. **Baselines.** An MLP on raw pixels reaches 0.36 on validation, a small CNN 0.55, a CNN with
   BatchNorm and Dropout 0.71. A pretrained ResNet-18 with the last blocks fine-tuned gives 0.91:
   with 5,000 training images, pretrained features matter far more than a custom architecture.
2. **Augmentations that imitate test defects.** The training images are clean, the test images
   are not, so every distortion is imitated at training time: Gaussian blur, down- and
   up-scaling, grayscale, darkening, pixel noise, plus flips and 90° rotations (a satellite
   image has no "up"). Without them a ResNet-18 scores 0.917 on validation, with them 0.953.
3. **Deeper networks at native resolution.** Shrinking the images costs accuracy (ResNet-18:
   0.930 at 128 px, 0.953 at 160–224 px on validation), and deeper networks score higher on the
   leaderboard. A single ResNet-152 at the native 256 px reaches 0.9606.
4. **Ensemble on two GPUs.** Six networks are split between the two T4 cards and trained at the
   same time, so the ensemble fits the 45-minute limit. At test time each network sees every
   image once, but under its own view (a rotation or a flip) — this gives test-time augmentation
   for free. Six ResNets: 0.9659.
5. **Networks of a different design.** Replacing two of the four ResNet-152 with ResNeXt-101 and
   Wide ResNet-101 (same `torchvision.models.resnet` module, confirmed as allowed by the
   instructor) adds diversity: 0.9670.
6. **Two input scales.** Feeding the two remaining ResNet-152 with images upscaled to 288 px
   makes the ensemble multi-scale: **0.9673** (final).

All networks use the same recipe: blocks `layer2–layer4` are fine-tuned, the head is replaced
with `Dropout(0.3) + Linear`, AdamW with OneCycleLR (max lr 5e-4), 10 epochs, label smoothing
0.1, mixed precision. The final models are trained on train + validation.

### Step by step

| Ensemble | Validation | Public LB | Training, min | Test prediction, s |
|---|---|---|---|---|
| 2× ResNet-18 + 2× ResNet-50, 224 px | 0.9628 | 0.9554 | 15.6 | — |
| 4× ResNet-152 + 2× ResNet-101, 256 px | 0.9757 | 0.9659 | 27.6 | 90 |
| two ResNet-152 at 288 px | 0.9728 | 0.9664 | 30.1 | 101 |
| ResNeXt-101 + Wide ResNet-101 instead of two ResNet-152 | 0.9771 | 0.9670 | 32.9 | 107 |
| **both changes together (final)** | 0.9742 | **0.9673** | 38.0 | 127 |
| final with ResNet-152 instead of ResNet-101 | — | 0.9671 | 39.6 | 132 |
| ResNeXt / Wide ResNet at 288 px, ResNet-50 instead of ResNet-101 | 0.9714 | 0.9639 | 39.3 | 134 |

Validation accuracy is measured on the supplied validation set with models trained on train
only. The ensembles differ by 1–3 validation images out of 699, which is below the noise level,
so the final model was chosen by the public leaderboard (70% of the test set, ≈ 5,800 images).

### Final ensemble

| Model | Input | Seed | Test view | Validation |
|---|---|---|---|---|
| ResNeXt-101 64x4d | 256 px | 42 | original | 0.9499 |
| Wide ResNet-101-2 | 256 px | 42 | flipped | 0.9642 |
| ResNet-152 | 288 px | 42 | rotated 90° | 0.9614 |
| ResNet-152 | 288 px | 1 | rotated 90° + flipped | 0.9614 |
| ResNet-101 | 256 px | 42 | rotated 180° | 0.9671 |
| ResNet-101 | 256 px | 1 | rotated 180° + flipped | 0.9599 |
| **Ensemble** (averaged probabilities) | | | | **0.9742** |

### What did not help

A smaller learning rate (0.9573 vs 0.9606 on the public leaderboard for a single ResNet-152),
Inception v3 in the ensemble (0.9645 vs 0.9659), ResNet-152 instead of ResNet-101 in the final
ensemble (0.9671 vs 0.9673), and moving ResNeXt / Wide ResNet to 288 px (0.9639).

## Repository structure

| File | Contents |
|---|---|
| `geo_satellites_clean.ipynb` | Final pipeline, executed on Kaggle (GPU T4 ×2): data, augmentations, training on two GPUs, validation, test prediction |
| `results/experiments.csv` | Experiment table: models, image size, seeds, validation accuracy, runtime, public score |
| `results/validation_runs.csv` | Validation accuracy and training time of every ensemble member |
| `results/single_model_search.csv` | Single-model search: architecture, image size, augmentations |
| `requirements.txt` | Library versions of the Kaggle run |

## How to run

The competition data are private and, under the course rules, may not be published, so they are
not included in this repository (the Kaggle run downloaded them from a course link, which is
removed from the notebook). To reproduce the results:

1. Put the competition files (`train.csv`, `val.csv`, `test.csv`, `classes.csv` and the image
   folders `train/`, `val/`, `test/`) into a `week-5-competition-geo-satellites-classification/`
   folder next to the notebook.
2. Install dependencies: `pip install -r requirements.txt` (Python 3.13). Internet access is
   needed once to download the ImageNet weights from torchvision.
3. Restart the kernel and run all cells. With two T4 GPUs training takes about 38 minutes;
   with one GPU the models are trained one after another.

`TRAIN_ON_VAL = True` trains on train + validation and writes the test predictions
(`week5_solution_v6c.csv` and `final_submission.csv`); `TRAIN_ON_VAL = False` trains on train
only and reports validation accuracy. The test set is used only for prediction. Seeds are set
for Python, NumPy and PyTorch; in our checks, separate Kaggle runs of the same models gave
identical test predictions.

---

## Русская версия

Решение учебного Kaggle-соревнования курса «Введение в машинное обучение»
(3 курс, октябрь 2026), командный этап. По спутниковому снимку 256×256 нужно определить один
из 20 классов (мост, церковь, озеро, вокзал, взлётная полоса и другие). В обучающей выборке
4999 чистых снимков, в валидации 699, а в тесте из 8299 снимков есть искажённые: размытые,
с низким разрешением, чёрно-белые, затемнённые, с шумом. Метрика — Accuracy.

**Результат:** точность на отложенной валидации **0.9742**, public leaderboard **0.9673**.

Ограничения курса: только предобученные на ImageNet семейства AlexNet / VGG / ResNet /
Inception, без внешних данных, обучение не дольше 45 минут и предсказание теста не дольше
2 минут на бесплатном GPU Kaggle (T4 ×2). Итоговый ноутбук учится 38.0 мин и предсказывает
тест за 127 с (для предсказания преподаватели объявили допуск 30 с).

**Ход решения:**
- свои сети дают мало: MLP 0.36, простая CNN 0.55, CNN с BatchNorm и Dropout 0.71;
  предобученная ResNet-18 с дообучением последних блоков — уже 0.91;
- аугментации, имитирующие дефекты теста (размытие, низкое разрешение, оттенки серого,
  затемнение, шум), поднимают ResNet-18 с 0.917 до 0.953 на валидации;
- более глубокие сети и исходный размер 256 px лучше уменьшенных картинок;
- шесть сетей делятся между двумя видеокартами и учатся одновременно, поэтому ансамбль
  укладывается в 45 минут; на тесте каждая сеть смотрит на снимок один раз под своим углом
  (поворот или отражение), вероятности усредняются;
- сети другого устройства из семейства ResNet (ResNeXt-101, Wide ResNet-101) добавляют
  разнообразия ансамблю, а две ResNet-152 на увеличенных до 288 px картинках — второй масштаб;
- не помогли: меньший шаг обучения, Inception v3 в ансамбле, ResNet-152 вместо ResNet-101
  в итоговом ансамбле, перевод ResNeXt и Wide ResNet на 288 px.

Все сравнения с цифрами — в таблицах выше и в `results/experiments.csv`.

**Данные** соревнования закрытые и по правилам курса не публикуются, поэтому их нет
в репозитории, а ссылка на архив из ноутбука убрана. Для запуска положите файлы соревнования
в папку `week-5-competition-geo-satellites-classification/` рядом с ноутбуком, установите
зависимости (`pip install -r requirements.txt`) и выполните Restart → Run All.
