1. Подготовка - работа через Терминал в папке проекта

py -3.12 -m venv venv && venv\Scripts\activate          # Linux: python3.12 -m venv venv && source venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu132    # для NVIDIA GPU, pip install torch torchvision для CPU
pip install -r requirements.txt
Также в папку проекта надо скачать видео

2. Калибровка данных

python calibrate.py k        --image koef.png            # k1 по прямым линиям; [c] подбор, [s] сохранить, [q]/Esc выйти, рядом окно с работой коэффициента
python calibrate.py walkable --plan officePlan.jpg       # полигон проходимой области (m - режим "препятствие")
python calibrate.py points   --camera 1 --root .         # для камер 1..5: от 8 пар "точка пола в кадре -> точка плана"
python calibrate.py mask     --camera 1 --root .         # маска на телевизор и зеркала
python calibrate.py zoom     --camera 3 --root .         # зона детального поиска далёких людей на камере 3 (проблемы с детекцией без него)
python calibrate.py check    --root .                    # проверка полигонов: calib_check_cam*.jpg

3. Детекция + локальный трекинг. Выполняется однократно. Для пробы: --max-frames 300
python stage1_detect_track.py --root . --out local_tracks.pkl 

4. Склейка камер
python fusion.py --inp local_tracks.pkl --out fusion_result.pkl

5. Запуск интерфейса
python app.py --root . --result fusion_result.pkl --plan officePlan.jpg

6. Настройка качества
- Далёкие люди не детектируются: `calibrate.py zoom --imgsz 1600 --conf 0.1`.
- Один человек -> два ID: `fusion.py --k-gate 3.5 --k-merge 3`. Стоит перекалибровать соответствующую камеру, добавив больше точек
- Два разных человека склеились: уменьшить `--k-gate`, `--k-merge`, `--r0`.
- Остальные параметры - редактировать класс `P` в `fusion.py`.

7. GitHub

Developer device:

cd <path>
git init                                                      # first time
git add .
git status
git commit -m "Version x.x"
git branch -M main
git remote add origin https://github.com/Byte-tern/nuke_hack.git   # first time
git push -u origin main

User device:

git clone https://github.com/Byte-tern/nuke_hack.git