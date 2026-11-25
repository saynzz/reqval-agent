"""Конвейер проверки: документ → требования → проверки → дефекты → итог."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from . import prompts, rules
from .anchor import Anchorer
from .criteria import CRITICAL_CRITERIA, SEVERITY_ORDER
from .llm import BaseProvider, LLMError
from .models import (Document, Finding, LLMExtraction, LLMReqBatch, LLMSetResult, LLMVerdict,
                     Location, Requirement)

log = logging.getLogger("reqval")


@dataclass
class Verdict:
    status: str                    # НЕ ГОТОВ / ГОТОВ С ОГОВОРКАМИ / ГОТОВ
    development: str
    testing: str
    priorities: list[tuple[list[str], str]]


@dataclass
class RunResult:
    doc: Document
    requirements: list[Requirement]
    findings: list[Finding]
    verdict: Verdict
    started: datetime
    finished: datetime
    provider: Optional[BaseProvider]
    mode: str
    notes: list[str] = field(default_factory=list)     # заметки о прогоне (отброшенные цитаты и т.п.)
    unchecked: list[str] = field(default_factory=list)  # требования, которые LLM не оценила


def _fmt_req(r: Requirement, maxlen: int = 400) -> str:
    q = r.quote if len(r.quote) <= maxlen else r.quote[:maxlen] + "…"
    return f"{r.rid} [{r.doc_id}] (L{r.location.line_from:03d}–L{r.location.line_to:03d}, {r.kind}): «{q}»"


def _overlap(a: Location, b: Location) -> bool:
    return not (a.line_to < b.line_from or b.line_to < a.line_from)


class Pipeline:
    def __init__(self, doc: Document, provider: Optional[BaseProvider], batch_size: int = 6):
        self.doc = doc
        self.llm = provider
        self.batch_size = batch_size
        self.anchor = Anchorer(doc)
        self.notes: list[str] = []

    # ------------------------------------------------------------------ extraction
    def extract(self) -> list[Requirement]:
        if self.llm is None:
            return self._extract_offline()
        text = self.doc.numbered_text()
        res = self.llm.generate(prompts.EXTRACT_SYSTEM,
                                prompts.EXTRACT_USER.format(name=self.doc.name, text=text),
                                LLMExtraction, label="нарезка")
        reqs: list[Requirement] = []
        seen: set[tuple[int, int, str]] = set()
        for item in res.requirements:
            loc = self.anchor.locate(item.quote, hint_line=item.line_from)
            quote = item.quote.strip()
            anchored = loc is not None
            if loc is None:
                # цитата не нашлась — берём текст указанных строк, если они существуют
                lf, lt = item.line_from, max(item.line_from, item.line_to)
                cand = None
                if self.doc.line(lf) and self.doc.line(lt) and lt - lf < 15:
                    first = self.doc.line(lf)
                    cand = Location(lf, lt, first.page, first.section)
                    # строки принимаются, только если их текст похож на цитату модели
                    from rapidfuzz import fuzz
                    from .anchor import normalize
                    if fuzz.token_set_ratio(normalize(item.quote), normalize(self.anchor.text_of(cand))) < 70:
                        cand = None
                if cand is not None:
                    loc = cand
                    quote = self.anchor.text_of(loc)
                    self.notes.append(f"Нарезка: цитата для «{item.summary}» не совпала дословно, взят текст строк {lf}–{lt}.")
                else:
                    self.notes.append(f"Нарезка: отброшено требование «{item.summary}» — цитата не найдена в документе.")
                    continue
            key = (loc.line_from, loc.line_to, re.sub(r"\W+", "", quote.lower())[:80])
            if key in seen:
                continue
            seen.add(key)
            reqs.append(Requirement(rid="", doc_id=item.doc_id.strip() or "—", summary=item.summary.strip(),
                                    quote=quote, kind=item.kind.strip(), location=loc, anchored=anchored))
        reqs.sort(key=lambda r: (r.location.line_from, r.location.line_to))
        for i, r in enumerate(reqs, 1):
            r.rid = f"R-{i:02d}"
        return reqs

    def _extract_offline(self) -> list[Requirement]:
        """Нарезка правилами (без LLM): пункты списков, абзацы, строки таблиц требований."""
        skip_sections = re.compile(r"(общие сведения|общие параметры|ревью|согласующ|содержание)", re.I)
        reqs: list[Requirement] = []
        header_seen: set[str] = set()
        for ln in self.doc.lines:
            t = ln.text.strip()
            if not t or ln.is_heading or skip_sections.search(ln.section):
                continue
            if "|" in t and ln.section not in header_seen:   # первая строка таблицы — заголовок
                header_seen.add(ln.section)
                continue
            if len(t) < 12 and "|" not in t:
                continue
            m = re.match(r"^[*•\-]?\s*(Ф-\d+|[A-ZА-Я]{1,4}-\d+(?:\.\d+)*)\.?\s+", t)
            m2 = re.match(r"^(\d+)\.\s+", t)
            doc_id = m.group(1) if m else (f"{ln.section.split(' › ')[-1]}, п. {m2.group(1)}" if m2 else "—")
            loc = Location(ln.no, ln.no, ln.page, ln.section)
            kind = ln.section.split(" › ")[-1] if ln.section else "требование"
            reqs.append(Requirement("", doc_id, t[:120], t.lstrip("*• "), kind, loc))
        for i, r in enumerate(reqs, 1):
            r.rid = f"R-{i:02d}"
        return reqs

    # ------------------------------------------------------------------ helpers
    def _req_at(self, loc: Location, reqs: list[Requirement]) -> Optional[Requirement]:
        for r in reqs:
            if _overlap(r.location, loc):
                return r
        return None

    def _anchor_issue_quote(self, quote: str, req: Optional[Requirement]) -> tuple[str, Optional[Location]]:
        hint = req.location.line_from if req else None
        loc = self.anchor.locate(quote, hint_line=hint)
        if loc is not None:
            return quote, loc
        return quote, None

    # ------------------------------------------------------------------ per-requirement
    def check_requirements(self, reqs: list[Requirement], hints: dict[str, list[rules.RuleHit]]) -> tuple[list[Finding], list[str]]:
        findings: list[Finding] = []
        unchecked: list[str] = []
        all_reqs = "\n".join(_fmt_req(r, 250) for r in reqs)
        by_id = {r.rid: r for r in reqs}
        text = self.doc.numbered_text()
        batches = [reqs[i:i + self.batch_size] for i in range(0, len(reqs), self.batch_size)]
        for bi, batch in enumerate(batches, 1):
            pending = list(batch)
            for attempt in range(2):
                if not pending:
                    break
                hint_txt = "\n".join(f"- {h.req_id}: {h.title} — {h.explanation}"
                                     for r in pending for h in hints.get(r.rid, [])) or "нет"
                label = f"проверка требований {bi}/{len(batches)}" + (" (повтор)" if attempt else "")
                log.info("%s: %s", label, ", ".join(r.rid for r in pending))
                try:
                    res = self.llm.generate(prompts.CHECK_SYSTEM, prompts.CHECK_USER.format(
                        name=self.doc.name, text=text, all_reqs=all_reqs, n=len(pending),
                        batch="\n".join(_fmt_req(r, 2000) for r in pending), hints=hint_txt),
                        LLMReqBatch, label=label)
                except LLMError as e:
                    self.notes.append(f"{label}: ошибка LLM — {e}")
                    break
                got = set()
                for a in res.assessments:
                    req = by_id.get(a.req_id.strip())
                    if req is None or req not in pending:
                        continue
                    got.add(req.rid)
                    for iss in a.issues:
                        quote, loc = self._anchor_issue_quote(iss.quote, req)
                        if loc is None:
                            quote, loc = req.quote, req.location
                            self.notes.append(f"{req.rid}: цитата модели «{iss.quote[:60]}» не найдена дословно — дефект привязан к месту требования.")
                        findings.append(Finding("", "requirement", iss.criterion.strip().upper()[:3], iss.severity,
                                                iss.title.strip(), iss.explanation.strip(), iss.recommendation.strip(),
                                                [req.rid], [(quote, loc)]))
                pending = [r for r in pending if r.rid not in got]
            unchecked += [r.rid for r in pending]
        return findings, unchecked

    # ------------------------------------------------------------------ set
    def check_set(self, reqs: list[Requirement], rule_hits: list[rules.RuleHit]) -> list[Finding]:
        findings: list[Finding] = []
        all_reqs = "\n".join(_fmt_req(r, 600) for r in reqs)
        rh = "\n".join(f"- {h.title}: «{h.quote[:80]}» ({h.location.human()})" for h in rule_hits) or "нет"
        text = self.doc.numbered_text()
        by_id = {r.rid: r for r in reqs}
        for label, system in (("проверка набора: согласованность", prompts.SET_SYSTEM_CONSISTENCY),
                              ("проверка набора: полнота", prompts.SET_SYSTEM_COMPLETENESS)):
            log.info(label)
            try:
                res = self.llm.generate(system, prompts.SET_USER.format(name=self.doc.name, text=text,
                                        all_reqs=all_reqs, rule_hits=rh), LLMSetResult, label=label)
            except LLMError as e:
                self.notes.append(f"{label}: ошибка LLM — {e}")
                continue
            for iss in res.issues:
                ev: list[tuple[str, Location]] = []
                rids: list[str] = []
                for e in iss.evidence:
                    req = by_id.get(e.req_id.strip())
                    quote, loc = self._anchor_issue_quote(e.quote, req)
                    if loc is None:
                        self.notes.append(f"Набор: цитата «{e.quote[:60]}» к дефекту «{iss.title}» не найдена — исключена.")
                        continue
                    ev.append((quote, loc))
                    owner = self._req_at(loc, reqs)
                    if owner and owner.rid not in rids:
                        rids.append(owner.rid)
                if not ev:
                    self.notes.append(f"Набор: дефект «{iss.title}» отброшен — ни одна цитата не подтверждена текстом.")
                    continue
                findings.append(Finding("", "set", iss.criterion.strip().upper()[:3], iss.severity, iss.title.strip(),
                                        iss.explanation.strip(), iss.recommendation.strip(), rids, ev))
        return findings

    # ------------------------------------------------------------------ verdict
    @staticmethod
    def auto_status(findings: list[Finding]) -> str:
        if any(f.severity == "критичный" for f in findings):
            return "НЕ ГОТОВ"
        if any(f.severity == "значительный" for f in findings):
            return "ГОТОВ С ОГОВОРКАМИ"
        return "ГОТОВ"

    def verdict(self, reqs: list[Requirement], findings: list[Finding]) -> Verdict:
        status = self.auto_status(findings)
        crit = [f for f in findings if f.severity == "критичный"]
        cnt = {s: sum(1 for f in findings if f.severity == s) for s in SEVERITY_ORDER}
        fallback_dev = {
            "НЕ ГОТОВ": f"Начинать разработку нельзя: {len(crit)} критичных дефектов ({', '.join(f.fid for f in crit[:10])}) допускают разное толкование поведения системы.",
            "ГОТОВ С ОГОВОРКАМИ": "Разработку можно начинать после согласования значительных дефектов с аналитиком.",
            "ГОТОВ": "Критичных и значительных дефектов не найдено — разработку можно начинать.",
        }[status]
        fallback_test = ("Проектирование тест-кейсов заблокировано для требований с критичными дефектами."
                         if crit else "Тестирование можно планировать по текущей версии.")
        fallback_prio = [([f.fid], f.title) for f in crit[:8]]
        if self.llm is None:
            return Verdict(status, fallback_dev, fallback_test, fallback_prio)
        ids = {f.fid for f in findings}
        defects = "\n".join(f"{f.fid} [{f.severity}; {f.criterion}; {', '.join(f.req_ids) or 'документ'}] {f.title}: {f.explanation[:300]}"
                            for f in findings)
        try:
            v = self.llm.generate(prompts.VERDICT_SYSTEM, prompts.VERDICT_USER.format(
                name=self.doc.name, n_req=len(reqs), n_def=len(findings), n_crit=cnt["критичный"],
                n_major=cnt["значительный"], n_minor=cnt["незначительный"], auto_verdict=status, defects=defects),
                LLMVerdict, label="итог")
        except LLMError as e:
            self.notes.append(f"Итог: ошибка LLM — {e}; использован итог по правилам.")
            return Verdict(status, fallback_dev, fallback_test, fallback_prio)
        prio = [([i for i in p.finding_ids if i in ids], p.action.strip()) for p in v.priorities]
        prio = [p for p in prio if p[1]]
        return Verdict(status, v.development_comment.strip(), v.testing_comment.strip(), prio or fallback_prio)

    # ------------------------------------------------------------------ run
    def run(self) -> RunResult:
        started = datetime.now().astimezone()
        log.info("Документ: %s (%s, %d строк)", self.doc.name, self.doc.fmt, len(self.doc.lines))
        reqs = self.extract()
        log.info("Выделено требований: %d", len(reqs))

        doc_hits = rules.check_document(self.doc)
        id_hit = rules.check_ids(reqs)
        if id_hit:
            doc_hits.append(id_hit)
        req_hits: dict[str, list[rules.RuleHit]] = {}
        for r in reqs:
            hs = rules.check_requirement(r)
            if hs:
                req_hits[r.rid] = hs

        findings: list[Finding] = []
        unchecked: list[str] = []
        if self.llm is not None:
            fr, unchecked = self.check_requirements(reqs, req_hits)
            fs = self.check_set(reqs, doc_hits)
            findings += fr + fs
        else:
            for rid, hs in req_hits.items():
                for h in hs:
                    findings.append(Finding("", "requirement", h.criterion, h.severity, h.title, h.explanation,
                                            h.recommendation, [rid], [(h.quote, h.location)], source="правило"))

        # формальные находки уровня документа: не дублируем то, что уже описала модель
        llm_set = [f for f in findings if f.scope == "set"]
        for h in doc_hits:
            dup = any(_overlap(h.location, loc) and (f.criterion == h.criterion or h.rule in ("R1", "R4", "R2"))
                      for f in llm_set for _, loc in f.evidence)
            if dup:
                continue
            owner = self._req_at(h.location, reqs)
            findings.append(Finding("", "rule", h.criterion, h.severity, h.title, h.explanation, h.recommendation,
                                    [owner.rid] if owner and h.rule != "R8" else [], [(h.quote, h.location)], source="правило"))

        # нумерация дефектов: по требованиям (в порядке документа), затем набор, затем формальные
        order = {r.rid: i for i, r in enumerate(reqs)}
        scope_rank = {"requirement": 0, "set": 1, "rule": 2}
        findings.sort(key=lambda f: (scope_rank[f.scope],
                                     order.get(f.req_ids[0], 10**6) if f.scope == "requirement" else 0,
                                     SEVERITY_ORDER.get(f.severity, 3),
                                     min(loc.line_from for _, loc in f.evidence)))
        for i, f in enumerate(findings, 1):
            f.fid = f"D-{i:03d}"

        verdict = self.verdict(reqs, findings)
        finished = datetime.now().astimezone()
        return RunResult(self.doc, reqs, findings, verdict, started, finished, self.llm,
                         "LLM" if self.llm else "offline (только правила)", self.notes, unchecked)


def is_critical_type(f: Finding) -> Optional[str]:
    """Тип критичного дефекта для сводки (непроверяемое / противоречивое / неполное ...)."""
    if f.severity != "критичный":
        return None
    return CRITICAL_CRITERIA.get(f.criterion, "критичный")
