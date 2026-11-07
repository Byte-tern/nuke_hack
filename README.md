# ДЕМО - Цифровые знахари: мультикамерный трекинг

Детектор YOLO11-pose + локальный трекер BoT-SORT на каждой камере + собственная склейка камер по плану офиса и времени (без Re-ID).
Видео из `camera_new_1.mkv`: 25 fps, 30001 кадр (~20 мин), 2592x1944 (в ТЗ - 1600x1200; код сам приводит кадры к 1600x1200,
координаты разметки нормированы). Подробности про библиотеки, модели и аномалии - в `libraries_and_models.txt`.

## Установка (Windows / Linux)
```
py -3.12 -m venv venv && venv\Scripts\activate          # Linux: python3.12 -m venv venv && source venv/bin/activate
pip install torch torchvision                           # CPU; для NVIDIA GPU - команда с https://pytorch.org (index-url .../whl/cu124 и т.п.)
pip install -r requirements.txt
```
Веса YOLO скачиваются автоматически при первом запуске (нужен интернет).

## Порядок запуска
```
# 0. Разметка (один раз; результат - calibration.json, он хранится в репозитории; пустой шаблон - calibration.template.json)
python calibrate.py k        --image koef.png            # k1 по прямым линиям; [c] подбор, [s] сохранить, [q]/Esc выйти (фокус - на окне OpenCV)
python calibrate.py walkable --plan officePlan.jpg       # полигон проходимой области (m - режим "препятствие")
python calibrate.py points   --camera 1 --root .         # для камер 1..5: >=6-8 пар "точка пола в кадре -> точка плана"
python calibrate.py mask     --camera 1 --root .         # ТВ на камерах 1 и 3 (и, при желании, зеркало на камере 2)
python calibrate.py zoom     --camera 3 --root .         # дальняя часть коридора: зона детального поиска далёких людей
python calibrate.py check    --root .                    # проверка: calib_check_cam*.jpg

# 1. Детекция + локальный трекинг (самое долгое). Для пробы: --max-frames 300
python stage1_detect_track.py --root . --out local_tracks.pkl
#    быстрый старт без разметки зоны:  --zoom 3:0.18,0.0,0.50,0.45   (x1,y1,x2,y2 - доли кадра; оценка по скриншоту, проверьте через calibrate.py zoom)

# 2. Склейка камер (секунды-минуты; перезапускается без этапа 1)
python fusion.py --inp local_tracks.pkl --out fusion_result.pkl

# 3. Интерфейс
python app.py --root . --result fusion_result.pkl --plan officePlan.jpg
```
Всё сразу: `python run_pipeline.py --root .` (после разметки; `--skip-stage1` пропускает этап 1).

## Интерфейс
Зоны 1-3 - видео (по умолчанию камеры 2, 4, 5), 4-6 - выпадающие списки (камера, уже выбранная в другой зоне, недоступна),
7 - название, число треков / активных / выбранный трек (+ пуск/пауза и ползунок), 8 - таблица
(ID, Started, Finished/ongoing, On camera 1-5: пусто - на камере не появлялся, `V` - сейчас на камере, `X` - появлялся, но сейчас не в её поле зрения),
9 - план: маршрут выбранного трека либо позиции всех активных ID. Клик по выбранной строке или Esc снимают выбор.
Выбранный трек на видео не подсвечивается. Треки остаются в таблице после finished. В конце видео время останавливается, активные остаются `ongoing`.
Время - от начала записи, `чч:мм:сс:мс`. Треки короче 1 с не учитываются.

## Настройка качества
- Далёкие люди не детектируются: `calibrate.py zoom` (зона обрабатывается повторно по вырезке полного разрешения), `--imgsz 1600`, `--conf 0.1`.
- Один человек -> два ID: `fusion.py --k-gate 3.5 --k-merge 3` (больше допуск на ошибку проекции вдали от камеры). Точность гомографии
  решает больше всего: добавьте пар точек в далёкой части кадра (`calibrate.py points`, ошибка в заголовке окна).
- Два разных человека склеились: уменьшите `--k-gate`, `--k-merge`, `--r0`.
- Остальные параметры - класс `P` в `fusion.py`.

## Воспроизведение на другом устройстве (GitHub)
В репозиторий входят: код, `calibration.json` (разметка - это часть решения), `officePlan.jpg`, `koef.png`, `office_camera.png`, `interface.png`.
НЕ входят (см. `.gitignore`): видео (большие, плюс на них люди - персональные данные), `*.pkl`, веса `*.pt`, `venv/`.
```
cd путь\к\проекту
git init
git add .
git status                      # убедитесь, что в списке нет *.mkv и *.pkl
git commit -m "Мультикамерный трекинг: ДЕМО - Цифровые знахари"
git branch -M main
git remote add origin https://github.com/<логин>/<репозиторий>.git     # репозиторий создайте на github.com (New repository, пустой, лучше Private)
git push -u origin main
```
Запуск на другом устройстве: `git clone <url>`, создать venv, установить torch и `pip install -r requirements.txt`, положить рядом
`camera_new_[1..5].mkv` (передаются отдельно: облако/диск) и выполнить `python run_pipeline.py --root <папка_с_видео>`.
Если нужно хранить видео в Git: Git LFS (`git lfs track "*.mkv"`) - у GitHub есть лимиты на размер.
Чтобы результаты были видны сразу, можно закоммитить `fusion_result.pkl` (небольшой): уберите `*.pkl` из `.gitignore`
и запустите только `python app.py ...` (видео всё равно нужны для картинки).
