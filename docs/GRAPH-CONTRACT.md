# Контракт графа и агентов. ВЛАДЕЛЕЦ: Р-B. Блок A1

Документ — ТЗ для реализации `src/graph.py` и `src/agents/*.py`.
`schemas.py` и `verifier.py` заморожены: ни одно поле ниже их не меняет.
Всё, что здесь описано, укладывается в 4–6 вызовов LLM на сценарий.

Новые типы заводить в `src/graph_state.py`, НЕ в `schemas.py`.

---

## 1. Объект состояния графа

`GraphState` — `TypedDict, total=False` в `src/graph_state.py`.
Редьюсер только у `tool_calls` (`Annotated[list[ToolCall], operator.add]`),
у остальных полей — перезапись последним писателем.

| Поле | Тип | Пишет | Читает | Обязательно к моменту |
|---|---|---|---|---|
| `query` | `str` | вход графа | все | START |
| `scenario` | `str` (`lead_optimization` \| `target_to_hit` \| `insufficient_data_demo`) | supervisor | reporter, `build_result` | после supervisor |
| `task` | `dict` (см. ниже) | supervisor | data_agent, generator | после supervisor |
| `parent` | `Molecule \| None` | data_agent | generator, evaluator | до generator (для `lead_optimization` — не None) |
| `activities` | `list[dict]` | data_agent | data_agent (QSAR) | до generator |
| `n_activities` | `int` | data_agent | `check_data_sufficiency` | до роутера достаточности |
| `has_structure` | `bool` | data_agent | `check_data_sufficiency`, evaluator | до роутера достаточности |
| `receptor` | `dict \| None` `{"pdbqt_path": str, "center": tuple[float,float,float], "box": tuple}` | data_agent | evaluator (докинг) | до evaluator, если `has_structure` |
| `qsar` | `dict \| None` `{"model": object, "metrics": ModelMetrics}` | data_agent | evaluator | до evaluator (может быть None) |
| `missing` | `list[MissingData]` | data_agent (только из `check_data_sufficiency`) | роутер, reporter, `build_result` | после data_agent |
| `generated` | `list[Molecule]` | generator | evaluator | до evaluator, непустой |
| `candidates` | `list[Candidate]` | evaluator | verifier_node | до verifier, непустой |
| `verification` | `VerificationReport \| None` | verifier_node | роутер, reporter | после verifier |
| `rejected_all` | `list[Candidate]` | verifier_node | reporter | накапливается между кругами |
| `feedback` | `list[str]` | verifier_node | generator | перед вторым кругом |
| `loop` | `int` (старт 0) | только generator (`+= 1`) | роутер после verifier | всегда |
| `tool_calls` | `list[ToolCall]` (add) | **каждый** узел | verifier_node, reporter, фронт | до verifier — обязательно |
| `summary` | `str` | reporter | `build_result` | после reporter |
| `result` | `ScenarioResult \| None` | reporter (через `build_result`) | вызывающий код / `app.py` | END |

`task` (пишет только supervisor, дальше read-only):
```python
{"parent_name": str | None, "parent_smiles": str | None,
 "target_name": str | None, "target_uniprot": str | None,
 "target_chembl_id": str | None, "pdb_id": str | None,
 "objectives": list[str], "constraints": list[str], "plan": list[str]}
```

### Почему `tool_calls` критичен

`verify(candidates, tool_calls)` сверяет `Value.source` с журналом: если в source
есть подстрока из `SOURCE_REQUIRES_TOOL`, а успешного вызова соответствующего
инструмента в прогоне не было — кандидат ДИСКВАЛИФИЦИРУЕТСЯ целиком, не
«с примечанием». Отсюда три жёстких требования к реализации:

1. **`ToolCall.tool` — ровно одна из этих строк**, без префиксов и суффиксов:
   `pubchem_smiles`, `chembl_activities`, `uniprot_target`, `pdb_structure`,
   `qsar.predict`, `admet.profile`, `dock`, `interactions`.
   Любые другие вызовы (`chem.describe`, `generate.mutate`) логировать можно и нужно —
   верификатор их просто не сверяет.
2. **Лог пишется до возврата из узла**, в том же `return {"tool_calls": [...]}`.
   Узел, который сходил в ChEMBL и не залогировал вызов, обнуляет свои же данные.
3. **Неудачный вызов логируется с `ok=False` и текстом ошибки.** Это не косметика:
   `ok=False` означает, что значения с этим источником не пройдут, и значит
   их нельзя записывать вообще — вместо них `schemas.not_computed(...)`.

Три мины, на которые наступит реализация, если не предупредить:

- `parent` с `source="PubChem CID 5291"` требует успешного `pubchem_smiles`
  в ЭТОМ прогоне. Хардкод SMILES иматиниба в обход инструмента → весь прогон
  уходит в `unverifiable`. Либо реально звать (кэш греет), либо ставить
  `source="входные данные"`.
- `REQUIRED_FOR_RANKING = ("LogP", "QED", "SA_Score")`. Если `sascorer` из
  RDKit Contrib не поднялся, `chem.describe` вернёт `not_computed("SA_Score")`,
  и **все** кандидаты будут отсеяны по «Полнота данных». Проверить импорт
  до прогона — это блокер уровня докинга.
- `verify()` МУТИРУЕТ `c.filters` (`c.filters = c.filters + checks`). Один и тот
  же `Candidate` нельзя прогонять через `verify()` дважды — на втором круге
  фильтры удвоятся. Кандидаты второго круга должны быть новыми объектами,
  а выжившие с первого круга через verify повторно не прогоняются.

---

## 2. Узлы и рёбра

```
START -> supervisor -> data_agent -> route_sufficiency
                                       |-- missing     --> reporter -> END
                                       |-- ok          --> generator
generator -> evaluator -> verifier_node -> route_verify
                                       |-- passed      --> reporter -> END
                                       |-- retry       --> generator
                                       |-- exhausted   --> reporter -> END
```

Все узлы синхронные, сигнатура `def node(state: GraphState) -> dict`.
Возвращают ТОЛЬКО изменённые ключи. Исключений не поднимают: ошибка инструмента —
это `ToolCall(ok=False)` + `not_computed`, а не падение графа.

### supervisor
`(state) -> {"scenario", "task", "tool_calls"}`
Предусловие: `query`. 1 вызов LLM, JSON-режим. Инструментов не вызывает.
Если LLM вернула невалидный JSON — один retry, затем дефолт
`scenario="target_to_hit"`, `task` из regex-экстракции, и строка в `limitations`.

### data_agent
`(state) -> {"parent", "activities", "n_activities", "has_structure", "receptor", "qsar", "missing", "tool_calls"}`
Предусловие: `task`. 1 вызов LLM (план вызовов), затем детерминированное исполнение.
Порядок: `pubchem_smiles` / `uniprot_target` → `chembl_activities` → `pdb_structure`
→ `chem.describe(parent)` → `qsar.train` **только если** `n_activities >= Thresholds.min_qsar_train`.
Последним действием узла — `missing = check_data_sufficiency(n_activities, has_structure)`.
Узел НЕ решает, отказываться ли, он только считает.

### route_sufficiency (функция, не LLM)
`state["missing"] != [] -> "reporter"`, иначе `"generator"`.
Ветка отказа гарантирует `candidates=[]`: `build_result` при `missing` обнуляет
кандидатов сам, но generator на этой ветке и не запускается.

### generator
`(state) -> {"generated", "loop", "tool_calls"}`
Предусловие: `parent is not None`, `missing == []`, `loop < MAX_LOOPS`.
1 вызов LLM на круг — выдаёт ТОЛЬКО параметры (`backend`, `n`, подсказки), не SMILES.
Дальше `tools.generate.mutate(...)`. Фильтрует: валидность по `chem.parse`,
уникальность по каноническому SMILES, исключение SMILES, уже отсеянных в
`rejected_all`. `loop += 1` — **единственное место инкремента в графе**.
Если на выходе 0 новых молекул — возвращает `generated=[]`.

### evaluator
`(state) -> {"candidates", "tool_calls"}`
Предусловие: `generated` непустой. 1 вызов LLM, и только на первом круге
(`loop == 1`): выбирает шортлист на докинг и помечает риски текстом.
На втором круге шортлист берётся кодом (top-k по `score`) — экономия 1 вызова.
Порядок: `chem.run_all_filters` → `admet.profile` (если доступен) →
`qsar.predict` (если `qsar`) → `docking.dock` только по шортлисту (k ≤ 5) и только
если `receptor` есть. `score` считает КОД, детерминированной формулой по
посчитанным величинам; отсутствующая величина в `score` не входит и её вес
перераспределяется. `Candidate.rank` не ставит — ранг проставляет `build_result`.

### verifier_node (не LLM, обёртка над `verifier.verify`)
`(state) -> {"verification", "feedback", "rejected_all", "tool_calls"}`
Предусловие: `candidates` непустой, `tool_calls` заполнен.
```python
rep = verify(state["candidates"], state["tool_calls"])
rejected_all = dedupe_by_smiles(state.get("rejected_all", []) + rep.rejected)
```
`feedback = rep.feedback` (перезапись, не накопление: подсказки второго круга
актуальнее). `rep.unverifiable` не трогать — `build_result` сам положит их
в `limitations`.
