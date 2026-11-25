"""Формирование отчёта (Markdown) и машиночитаемой копии (JSON)."""
from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from .criteria import ALL_CRITERIA, REQ_CRITERIA, SET_CRITERIA, SEVERITY_ORDER, SEVERITY_RULES
from .models import Finding, Location
from .pipeline import RunResult, _overlap, is_critical_type
from .version import agent_version

SEV_ICON = {"критичный": "🔴", "значительный": "🟠", "незначительный": "🟡"}


def _cell(s: str, limit: int | None = None) -> str:
    s = " ".join(str(s).split())
    if limit and len(s) > limit:
        s = s[:limit].rstrip() + "…"
    return s.replace("|", "\\|")


def _crit_name(code: str) -> str:
    c = ALL_CRITERIA.get(code)
    return f"{code} {c[0]}" if c else code


def _ev_lines(f: Finding) -> list[str]:
    return [f"  - «{_cell(q, 300)}» — {loc.human()}" for q, loc in f.evidence]


def _finding_block(f: Finding, related: list[str] | None = None) -> list[str]:
    if f.dup_of is not None:
        q, loc = f.evidence[0]
        return [f"**{_crit_name(f.criterion)}** · {SEV_ICON.get(f.severity, '')} {f.severity} — **{_cell(f.title)}** "
                f"↪ тот же дефект описан в проверке набора: **{f.dup_of.fid}** (в сводке учтён один раз)", "",
                f"- Фрагмент: «{_cell(q, 300)}» — {loc.human()}", f"- Суть: {f.explanation}", ""]
    out = [f"**{f.fid}** · {SEV_ICON.get(f.severity, '')} {f.severity} · {_crit_name(f.criterion)}"
           f"{' · источник: ' + f.source if f.source != 'LLM' else ''} — **{_cell(f.title)}**", ""]
    if len(f.evidence) == 1:
        q, loc = f.evidence[0]
        out.append(f"- Фрагмент: «{_cell(q, 300)}» — {loc.human()}")
    else:
        out.append("- Фрагменты:")
        out += _ev_lines(f)
    if f.req_ids and f.scope != "requirement":
        out.append(f"- Требования: {', '.join(f.req_ids)}")
    out.append(f"- Суть: {f.explanation}")
    out.append(f"- Рекомендация: {f.recommendation}")
    if related:
        out.append(f"- См. также: {', '.join(related)}")
    out.append("")
    return out


def build_markdown(res: RunResult) -> str:
    doc, reqs, fs, v = res.doc, res.requirements, res.findings, res.verdict
    p = res.provider
    req_f = {r.rid: [f for f in fs if f.scope == "requirement" and r.rid in f.req_ids] for r in reqs}
    set_f = [f for f in fs if f.scope in ("set", "rule")]
    dups = [f for f in fs if f.dup_of is not None]
    fs = res.unique_findings          # всё, что считается в итогах, — без повторов
    sev = Counter(f.severity for f in fs)
    dur = (res.finished - res.started).total_seconds()
    L: list[str] = []

    L += [f"# Отчёт о проверке требований: «{doc.name}»", ""]
    L += [f"> Итог: **{v.status}** · требований: **{len(reqs)}** · дефектов: **{len(fs)}** "
          f"(🔴 {sev['критичный']} · 🟠 {sev['значительный']} · 🟡 {sev['незначительный']})", ""]
    if not res.complete:
        gaps = res.failed_steps + ([f"нет оценки модели для {', '.join(res.unchecked)}"] if res.unchecked else [])
        L += [f"> ⚠ **Проверка выполнена не полностью:** {'; '.join(gaps)}. Подробности — в приложении.", ""]

    # 1. Паспорт ---------------------------------------------------------------
    L += ["## 1. Паспорт прогона", "", "| Параметр | Значение |", "| :--- | :--- |"]
    rows = [
        ("Документ", doc.name),
        ("Файл", f"{Path(doc.path).name} ({doc.fmt.upper()}, {len(doc.lines)} строк)"),
        ("SHA-256 файла", f"`{doc.sha256}`"),
        ("Дата и время прогона", f"{res.started:%d.%m.%Y %H:%M:%S %Z} (длительность {dur:.0f} с)"),
        ("Режим", res.mode + ("" if res.extraction in ("LLM", "правила") else f"; нарезка требований: {res.extraction}")),
        ("Модель", f"{p.model} ({p.name})" if p else "— (LLM не использовалась)"),
        ("Параметры модели", f"temperature={p.temperature}, seed=42, structured output (JSON-схема)" if p else "—"),
        ("Версия агента", agent_version()),
        ("Обращений к модели", (f"{p.usage.calls} (повторов: {p.usage.retries}); токенов: вход {p.usage.input_tokens:,}, "
                                f"выход {p.usage.output_tokens:,}".replace(",", " ")
                                + (f"; стоимость ${p.usage.cost_usd:.4f}" if p.usage.cost_usd else "")) if p else "0"),
        ("Стандарты", "ISO/IEC/IEEE 29148:2018 п. 5.2.5, 5.2.6, 5.2.7; IEEE Std 830-1998 п. 4.3"),
    ]
    L += [f"| {k} | {_cell(val)} |" for k, val in rows] + [""]

    # 2. Методика ------------------------------------------------------------
    if p:
        rej = len(res.rejected)
        method = ("Агент выполняет статическое ревью: (1) извлекает текст с номерами строк; (2) выделяет требования "
                  f"({'моделью' if res.extraction == 'LLM' else 'формальными правилами — ' + res.extraction}), "
                  "каждая цитата сверяется с исходным текстом; (3) формальные правила ищут заглушки, пустые поля, "
                  "размытые термины, требования без идентификаторов; (4) модель проверяет каждое требование по "
                  "критериям C1–C9 и (5) набор целиком по S1–S8; (6) цитата каждого дефекта ищется в документе — "
                  "дефект без подтверждённой цитаты в отчёт не попадает"
                  + (f" (отброшено замечаний: {rej}, перечислены в приложении)" if rej else "")
                  + "; (7) итог: есть критичные → «НЕ ГОТОВ»; часть проверки не выполнена → «ПРОВЕРКА НЕ ЗАВЕРШЕНА»; "
                  "есть значительные → «ГОТОВ С ОГОВОРКАМИ»; иначе «ГОТОВ».")
    else:
        method = ("Режим offline: модель не использовалась. Выполнены только формальные правила (заглушки, пустые поля, "
                  "незаполненный чек-лист, размытые термины, временные характеристики без условий измерения, "
                  "требования без идентификаторов). Проверка по смыслу критериев C1–C9 и S1–S8 не проводилась: "
                  "✔ в матрице означает только отсутствие формальных находок.")
    L += ["## 2. Методика", "", method, "",
          "| Код | Свойство | Источник |", "| :--- | :--- | :--- |"]
    L += [f"| {c} | {n} | {s} |" for c, (n, s, _) in {**REQ_CRITERIA, **SET_CRITERIA}.items()]
    L += ["", "Шкала критичности:", SEVERITY_RULES, "",
          "Место в документе указывается как «раздел, строка N» (для PDF — также страница). "
          "Строки нумеруются по извлечённому тексту файла, указанному в паспорте.", ""]

    # 3. Нарезка ---------------------------------------------------------------
    L += ["## 3. Как агент нарезал требования", "",
          f"Выделено требований: **{len(reqs)}**, из них с идентификатором в документе: "
          f"**{sum(1 for r in reqs if r.doc_id not in ('—', '-', ''))}**. Требованиям без номера агент присвоил собственные ID и сохранил цитату.", "",
          "| ID в отчёте | ID в документе | Тип | Краткая формулировка | Цитата | Место |",
          "| :--- | :--- | :--- | :--- | :--- | :--- |"]
    for r in reqs:
        L.append(f"| {r.rid} | {_cell(r.doc_id)} | {_cell(r.kind)} | {_cell(r.summary)} | «{_cell(r.quote, 260)}» | {_cell(r.location.human())} |")
    L.append("")

    # 4. Проверка требований -----------------------------------------------------
    L += ["## 4. Проверка каждого требования", "", "### 4.1. Матрица соответствия критериям", "",
          "✔ — нарушений не выявлено, ✖ — есть нарушение (число замечаний), — — требование не оценено моделью.", "",
          "| Требование | " + " | ".join(REQ_CRITERIA) + " | Дефекты |",
          "| :--- | " + " | ".join(":-:" for _ in REQ_CRITERIA) + " | :-: |"]
    for r in reqs:
        cnt = Counter(f.criterion for f in req_f[r.rid])
        ok = "—" if r.rid in res.unchecked else "✔"   # не оценено моделью → «нарушений нет» утверждать нельзя
        cells = [("✖" + (str(cnt[c]) if cnt[c] > 1 else "")) if cnt[c] else ok for c in REQ_CRITERIA]
        mark = "" if r.rid not in res.unchecked else " ⚠ не проверено"
        L.append(f"| {r.rid} ({_cell(r.doc_id)}){mark} | " + " | ".join(cells) + f" | {len(req_f[r.rid])} |")
    L += ["", "### 4.2. Замечания по требованиям", ""]
    for r in reqs:
        L += [f"#### {r.rid} · {_cell(r.doc_id)} — {_cell(r.summary)}", "",
              f"> {_cell(r.quote, 1200)}", "", f"Место: {r.location.human()}", ""]
        items = sorted(req_f[r.rid], key=lambda f: SEVERITY_ORDER.get(f.severity, 3))
        if r.rid in res.unchecked:
            L += ["⚠ Модель не вернула оценку этого требования — требуется ручная проверка"
                  + (". Ниже — только формальные находки." if items else "."), ""]
            for f in items:
                L += _finding_block(f)
            continue
        if not items:
            L += ["✔ Нарушений критериев C1–C9 не выявлено.", ""]
            continue
        for f in items:
            related = [s.fid for s in set_f if f.dup_of is None and r.rid in s.req_ids
                       and any(_overlap(loc, l2) for _, loc in s.evidence for _, l2 in f.evidence)]
            L += _finding_block(f, related)

    # 5. Набор ---------------------------------------------------------------
    L += ["## 5. Проверка всего набора требований", "", "| Свойство набора | Результат |", "| :--- | :--- |"]
    for c, (n, s, _) in SET_CRITERIA.items():
        lst = [f for f in set_f if f.criterion == c]
        res_txt = "✔ нарушений не выявлено" if not lst else "✖ " + ", ".join(f"{f.fid} ({f.severity})" for f in lst)
        L.append(f"| {c} {n} | {res_txt} |")
    L.append("")
    other = [f for f in set_f if f.criterion not in SET_CRITERIA]
    for c, (n, _, _) in SET_CRITERIA.items():
        lst = sorted([f for f in set_f if f.criterion == c], key=lambda f: SEVERITY_ORDER.get(f.severity, 3))
        if not lst:
            continue
        L += [f"### {c}. {n}", ""]
        for f in lst:
            L += _finding_block(f)
    if other:
        L += ["### Прочие замечания к набору", ""]
        for f in other:
            L += _finding_block(f)

    # 6. Сводка ---------------------------------------------------------------
    with_def = sum(1 for r in reqs if req_f[r.rid])
    L += ["## 6. Сводка", "", "| Показатель | Значение |", "| :--- | :--- |",
          f"| Требований выделено | {len(reqs)} |",
          f"| Требований с нарушениями | {with_def} из {len(reqs)} |",
          f"| Требований без нарушений | {sum(1 for r in reqs if not req_f[r.rid] and r.rid not in res.unchecked)} |",
          f"| Требований не оценено моделью | {len(res.unchecked)} |",
          f"| Всего дефектов | {len(fs)} |",
          f"| — по отдельным требованиям | {sum(1 for f in fs if f.scope == 'requirement')} |",
          f"| Повторов дефектов набора в замечаниях к требованиям (учтены один раз) | {len(dups)} |",
          f"| — по набору и документу | {len(set_f)} |",
          f"| 🔴 критичных | {sev['критичный']} |", f"| 🟠 значительных | {sev['значительный']} |",
          f"| 🟡 незначительных | {sev['незначительный']} |", ""]
    by_c = Counter(f.criterion for f in fs)
    L += ["Распределение по критериям: " + "; ".join(f"{_crit_name(c)} — {n}" for c, n in sorted(by_c.items())), ""]
    crit = [f for f in fs if f.severity == "критичный"]
    L += ["### Критичные дефекты", ""]
    if crit:
        L += ["| Дефект | Тип | Требования | Суть | Место |", "| :--- | :--- | :--- | :--- | :--- |"]
        for f in crit:
            L.append(f"| {f.fid} | {is_critical_type(f)} | {', '.join(f.req_ids) or 'документ'} | {_cell(f.title)} | "
                     f"{_cell('; '.join(loc.human() for _, loc in f.evidence), 200)} |")
    else:
        L.append("Критичных дефектов не выявлено.")
    L.append("")

    # 7. Итог ---------------------------------------------------------------
    L += ["## 7. Итог", "", f"**Решение: {v.status}**", "",
          f"- **Разработка:** {v.development}", f"- **Тестирование:** {v.testing}", "",
          "### Что исправить в первую очередь", ""]
    for i, (ids, action) in enumerate(v.priorities, 1):
        L.append(f"{i}. {action}" + (f" ({', '.join(ids)})" if ids else ""))
    L.append("")

    # Приложение ----------------------------------------------------------
    if res.notes or res.unchecked or res.rejected or res.failed_steps:
        L += ["## Приложение. Журнал прогона", ""]
        if res.failed_steps:
            L.append(f"- Шаги, на которых модель не ответила: {', '.join(res.failed_steps)}")
        if res.unchecked:
            L.append(f"- Не получена оценка модели: {', '.join(res.unchecked)}")
        L += [f"- {n}" for n in res.notes]
        L.append("")
        if res.rejected:
            L += ["### Отброшенные замечания модели", "",
                  "Цитата, на которую ссылалась модель, не найдена в документе (ни дословно, ни нечётко), поэтому "
                  "замечание не включено в отчёт: это защита от выдуманных дефектов. Список приведён для прозрачности.", "",
                  "| Область | Требование | Критерий | Замечание | Цитата модели (не найдена) |",
                  "| :--- | :--- | :--- | :--- | :--- |"]
            for j in res.rejected:
                L.append(f"| {'требование' if j.scope == 'requirement' else 'набор'} | {_cell(j.req_id)} | "
                         f"{_crit_name(j.criterion)} | {_cell(j.title)} | «{_cell(j.quote, 200)}» |")
            L.append("")
    L += ["---", f"*Отчёт сформирован автоматически агентом reqval {agent_version()}.*", ""]
    return "\n".join(L)


def build_json(res: RunResult) -> dict:
    def loc(l: Location) -> dict:
        return asdict(l)

    return {
        "document": {"name": res.doc.name, "file": os.path.basename(res.doc.path), "format": res.doc.fmt, "sha256": res.doc.sha256},
        "run": {"started": res.started.isoformat(), "finished": res.finished.isoformat(), "mode": res.mode,
                "model": res.provider.model if res.provider else None, "agent_version": agent_version(),
                "usage": asdict(res.provider.usage) if res.provider else None},
        "requirements": [{"id": r.rid, "doc_id": r.doc_id, "kind": r.kind, "summary": r.summary, "quote": r.quote,
                          "location": loc(r.location), "anchored": r.anchored} for r in res.requirements],
        "findings": [{"id": f.fid, "duplicate_of": f.dup_of.fid if f.dup_of else None, "scope": f.scope, "criterion": f.criterion, "severity": f.severity, "title": f.title,
                      "explanation": f.explanation, "recommendation": f.recommendation, "requirements": f.req_ids,
                      "evidence": [{"quote": q, "location": loc(l)} for q, l in f.evidence], "source": f.source}
                     for f in res.findings],
        "verdict": {"status": res.verdict.status, "development": res.verdict.development, "testing": res.verdict.testing,
                    "priorities": [{"findings": ids, "action": a} for ids, a in res.verdict.priorities]},
        "notes": res.notes, "unchecked": res.unchecked, "complete": res.complete, "failed_steps": res.failed_steps,
        "extraction": res.extraction, "rejected": [asdict(j) for j in res.rejected],
    }


def write_report(res: RunResult, out_path: str, with_json: bool = True) -> list[str]:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    md = build_markdown(res)
    written = []
    if out.suffix.lower() in (".html", ".htm"):
        import markdown
        html = markdown.markdown(md, extensions=["tables"])
        out.write_text(f"<!doctype html><html lang='ru'><head><meta charset='utf-8'><title>{res.doc.name}</title>"
                       "<style>body{font-family:system-ui,sans-serif;max-width:1100px;margin:2em auto;padding:0 1em;line-height:1.5}"
                       "table{border-collapse:collapse;font-size:14px}td,th{border:1px solid #ccc;padding:4px 8px;vertical-align:top}"
                       "blockquote{border-left:4px solid #ccc;margin:0;padding:0 1em;color:#444}</style></head><body>"
                       f"{html}</body></html>", encoding="utf-8")
    else:
        out.write_text(md, encoding="utf-8")
    written.append(str(out))
    if with_json:
        jp = out.with_suffix(".json")
        jp.write_text(json.dumps(build_json(res), ensure_ascii=False, indent=2), encoding="utf-8")
        written.append(str(jp))
    return written
