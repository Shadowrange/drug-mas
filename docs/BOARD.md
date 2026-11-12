# 📌 BOARD — первый комментарий

Скопировать в issue №1, озаглавить `📌 BOARD`, закрепить.

---

## Команда и GitHub-юзернеймы
| Роль | Человек | GitHub | Зона ответственности |
|---|---|---|---|
| **Р-A** — химия и инструменты | Кирилл Андреев | `@andkirill68` | `src/tools/chem.py`, `admet.py`, `qsar.py`, `docking.py`, `tests/`, эталонные прогоны `examples/scenario{1,2,3}.json` + `.png`, фактические результаты в `docs/scenarios.md`, `README.md` |
| **Р-B** — агенты и контракты | Ваня Андреев | `@Shadowrange` | `src/schemas.py`🔒, `src/cache.py`, `src/verifier.py`, `src/tools/databases.py`, `src/tools/generate.py`, `src/agents/*`, `src/graph.py`, `src/graph_state.py`, кэш и прогон с LLM |
| **Р-C** — фронт и демо | Рома | `@roma` *(уточнить при копировании в issue #1)* | `app.py`, дизайн карточек и бейджей происхождения, экран отказа Сценария 3, `.streamlit/config.toml`, 6 слайдов защиты |
| **Р-D** — защита и документация | Распределена (Р-A + Р-C по §8 плана) | `@andkirill68` / `@roma` | `README.md`, `docs/architecture.md`, `docs/scenarios.md`, `docs/BOARD.md` (Кирилл Р-A); презентация и тайминг демо (Рома Р-C) |
| **Таймкипер** | Ваня Андреев | `@Shadowrange` | Контроль дедлайнов и заморозки кода в 11:30 |

## Владение файлами
| Владелец | Файлы | Статус |
|---|---|---|
| **Р-A (Кирилл)** | `src/tools/chem.py`, `src/tools/admet.py`, `src/tools/qsar.py`, `src/tools/docking.py`, `tests/test_tools.py` | ✅ Готово (22 теста инструментов зелёные) |
| **Р-B (Ваня)** | `src/schemas.py`🔒, `src/cache.py`, `src/verifier.py`, `src/tools/databases.py`, `src/tools/generate.py`, `src/agents/*`, `src/graph.py`, `src/graph_state.py`, `tests/test_graph.py` | ✅ Готово (граф + 19 тестов графа зелёные, всего `41 passed`) |
| **Р-C (Рома)** | `app.py`, `.streamlit/config.toml`, слайды презентации | 🔄 В работе (подключён к реальному графу; финальная полировка UI и слайды) |
| **Р-A / Р-D** | `README.md`, `docs/scenarios.md`, `docs/architecture.md`, `docs/BOARD.md`, `examples/scenario{1,2,3}.{json,png}` | ✅ Готово |

🔒 `src/schemas.py` заморожен. Правка — только после поста `[decision]`.

## Формат постов
`[claim]` взял в работу · `[done]` готово + хеш коммита · `[blocked]` застрял · `[decision]` решение

Подробности — `docs/TEAM-AGENT-RULES.md` §3.

## Журнал выполнения (`[claim]` / `[done]` / `[decision]`)

- `[done]` **Р-B (`@Shadowrange`)** — Каркас репозитория, замороженный контракт `src/schemas.py`, `src/verifier.py`, `src/cache.py`, `src/tools/chem.py` (12 базовых тестов) · коммит `8e563f1`
- `[decision]` **Р-A (`@andkirill68`) + Р-B (`@Shadowrange`)** — По рискам R1 и R3:
  - **R1 (докинг):** при отсутствии в среде бинарника `AutoDock Vina` и `Meeko` не делать физически некорректный жёсткий докинг (`TORSDOF 0`) для гибких лигандов; сохранять 3D-конформер (`ETKDGv3` + `MMFF94`), проверять PDB `1IEP` и возвращать честный статус `Origin.HYPOTHESIS` (`value=None`, `не посчитано (Vina недоступна)`).
  - **R3 (ADMET):** растворимость считать по `ESOL (Delaney 2004)` с `Origin.PREDICTED` и метриками `AAE=0.83, n_train=2874` (согласно `docs/inf.pdf`); дескрипторы `TPSA` и `RotBonds` — `Origin.COMPUTED`; бинарные классификаторы без обученных весов (`BBB`, `hERG`, `Ames`, `CYP`) маркировать как `not_computed` (`Origin.HYPOTHESIS`, `value=None`).
- `[done]` **Р-A (`@andkirill68`)** — Реализованы `src/tools/qsar.py` (RandomForest + Morgan ECFP4 2048 бит, `R2=0.704`, `n_train=1063` на `CHEMBL203`), `src/tools/admet.py`, `src/tools/docking.py`, расширен `tests/test_tools.py` до 22 тестов, оформлен `docs/architecture.md` · коммит `689a3ee`
- `[done]` **Р-B (`@Shadowrange`)** — Реализованы `src/tools/databases.py`, `src/tools/generate.py` (STONED-SELFIES + фильтр `chemically_sane` с `Tanimoto >= 0.40`), `src/agents/*`, `src/graph_state.py`, `src/graph.py` и `tests/test_graph.py` (41 тест зелёный); `app.py` переведён с фикстуры на реальный `run_graph()` · коммиты `868621d`, `462b1aa`, `35b6471`, `375a5f2`, `d9304e2`
- `[done]` **Р-A (`@andkirill68`)** — Сгенерированы и проверены `examples/scenario1.json`, `examples/scenario2.json`, `examples/scenario3.json` через `ScenarioResult.model_dump_json(indent=2)` и скриншоты `examples/scenario{1,2,3}.png`; заполнена колонка и разделы «Фактический результат» в `docs/scenarios.md`; актуализированы `README.md` и `docs/BOARD.md`.

## Контрольные точки
- ✅ **чт 01:00** — статус докинга и ADMET зафиксирован (риски R1, R3 закрыты через честный План Б)
- ✅ **чт 20:00** — три сценария проходят end-to-end в офлайне, `41 passed`, фронт подключён к реальному прогону
- ✅ **чт вечер (Р-A)** — `examples/scenario{1,2,3}.{json,png}`, фактические результаты в `docs/scenarios.md`, `README.md` и `docs/BOARD.md` готовы
- ⏳ **пт 11:30** — 🔒 заморозка кода, две репетиции с секундомером и запись страховочного видео прогона
