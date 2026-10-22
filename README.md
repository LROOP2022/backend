# Backend: прогноз вероятности банкротства

FastAPI-обёртка над финальным ML-модулем `bankruptcy-final-one-match-v1` (public ROC-AUC 0.70861). Модели и справочники загружаются один раз при старте процесса. PostgreSQL больше не нужен.

## Запуск

Python 3.12, CPU. На Linux нужен `libgomp1` (в Docker-образе уже есть).

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
export ML_ARTIFACTS_DIR=/models/artifacts   # папка artifacts из ML-архива целиком (~3 ГБ)
.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8080 --workers 1
```

Папка `artifacts` (~3 ГБ) не хранится в Git: при сборке образа `Dockerfile` скачивает её из S3 (см. ниже). Не отдавайте её клиенту: `profiles.sqlite` содержит числовые профили TRAIN. Один worker, `MODEL_THREADS=2` по умолчанию; каждый дополнительный worker грузит модели заново.

Перед подключением, в итоговом окружении, из папки ML-архива:

```bash
python verify_package.py   # должен вывести PASS
```

## API

- `POST /api/v1/predict/by-id` `{"index":"194357"}` → `{"index":"194357","probability":0.0088…}`. Возвращает сохранённый финальный прогноз (`ScoreLookup`) только для ID из TEST. Неизвестный ID — `404`. Новых людей по одному ID рассчитать нельзя.
- `POST /api/v1/predict/csv` (multipart, поле `file`) → `{"predictions":[{"index":…,"probability":…}]}`. Нужны все `feature_0…feature_1366` (числа, без пустых и NaN), опционально `index` и `word_count` (`"слово 3 слово 2"`). Файл читается потоком и обрабатывается порциями по 1000 строк, общего лимита строк нет; лимит размера `MAX_CSV_BYTES` (по умолчанию 256 МиБ). Ошибки: `422` (с номером строки), `413`.
- `POST /api/v1/predict/records` `{"records":[…]}` — от 1 до 1000 объектов с теми же полями.
- `POST /api/v1/predict/explain` `{"record":{…}}` — для одного полного объекта возвращает 16 наиболее сильных изменений оценки при замене признака на медиану обучения. Это анализ чувствительности, не SHAP: числа не суммируются в итоговую оценку. Сначала отбираются 48 признаков по gain двух деревьев, затем пересчитывается весь ансамбль без правила точного совпадения.
- `POST /api/v1/predict/by-id/details` `{"index":"194357"}` — сохранённый балл, исходные 1367 признаков из TEST, `word_count` и анализ чувствительности. Требует дополнительный `test_features.sqlite`; пока файл отсутствует, отвечает `503`. Базовый `/by-id` остаётся доступным.
- `GET /health`, `/actuator/health/{liveness,readiness}`, `/actuator/prometheus`.

`probability` — оценка модели от 0 до 1 без калибровки; пороги риска не подбирались. Значения признаков не округлять: правило точного совпадения с TRAIN использует float64.

## Настройки

- `ML_ARTIFACTS_DIR` — путь к `artifacts` (по умолчанию `ml_backend/artifacts`).
- `MODEL_THREADS` — потоки модели, по умолчанию `2`.
- `MAX_CSV_BYTES` — лимит размера CSV.
- `VERIFY_ARTIFACTS_ON_START=0` — отключить проверку SHA-256 небольших файлов при старте.
- `CORS_ORIGINS` — разрешённые origin через запятую.

## Тесты

```bash
.venv/bin/python -m pip install -r requirements-dev.txt
ML_ARTIFACTS_DIR=/path/to/artifacts .venv/bin/python -m unittest discover -s tests -v
```

## Артефакты модели в S3 (один раз)

Образ при сборке скачивает tar по `MODEL_BUNDLE_URL` (по умолчанию `https://s3.hack.week-infra.ru/backend/ml_artifacts_v1.tar`), распаковывает в `/models/artifacts` и сверяет SHA-256 каждого файла с `ml_backend/manifest.json`. Несовпадение или отсутствие файла — сборка падает.

Создать архив из папки ML-пакета (команда одинакова в Linux/macOS и в PowerShell на Windows 10+):

```bash
tar -cf ml_artifacts_v1.tar -C ML_FINAL_20261021_233800/artifacts features.json raw_lightgbm.txt joint_lightgbm.txt text_C1.joblib text_C4.joblib text_svd.joblib linear.joblib linear_preprocessing.joblib scores.sqlite profiles.sqlite
```

Файлы лежат в корне tar, без папок. Загрузите `ml_artifacts_v1.tar` в бакет `backend` без переименования и проверьте, что URL открывается без авторизации. Новая модель: новый файл (`…_v2.tar`), новый `manifest.json` из ML-архива и смена URL в `Dockerfile`. Локально скачать и проверить без Docker: `python -m scripts.fetch_artifacts <URL> ./artifacts`.

## Признаки TEST по ID

Дополнительную базу строят один раз из исходных `TEST.csv` и `text_data.tsv`. Она содержит только TEST; исходные прогнозы в `scores.sqlite` не меняются. При наличии зашифрованного ZIP нужен `pyzipper` только для сборки базы:

```bash
python -m pip install pyzipper
python -m scripts.build_test_features --zip Hackathon-task.zip --password 'пароль-архива' test_features.sqlite
shasum -a 256 test_features.sqlite
```

Положите SHA-256 в `ml_backend/test_features.sha256`, загрузите `test_features.sqlite` в публичный бакет `backend` по адресу `https://s3.hack.week-infra.ru/backend/test_features.sqlite`. При сборке Docker скачивает базу по `TEST_FEATURES_URL` и проверяет контрольную сумму. База содержит данные клиентов: не размещайте ссылку на неё в интерфейсе, а публичный доступ к артефактам лучше закрыть после настройки приватной доставки в кластер.
