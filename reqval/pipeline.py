"""Конвейер проверки: документ → требования → проверки → дефекты → итог."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from . import prompts, rules
from .anchor import MIN_EVIDENCE_CHARS, Anchorer
from .dedup import dedupe
from .criteria import CRITICAL_CRITERIA, SEVERITY_ORDER, normalize_rid
from .llm import BaseProvider, LLMError
from .models import (Document, Finding, LLMExtraction, LLMReqBatch, LLMSetResult, LLMVerdict,
                     Location, Requirement)

log = logging.getLogger("reqval")


NOT_READY = "НЕ ГОТОВ"
READY_WITH_REMARKS = "ГОТОВ С ОГОВОРКАМИ"
READY = "ГОТОВ"
INCOMPLETE = "ПРОВЕРКА НЕ ЗАВЕРШЕНА"


@dataclass
class Rejected:
    """Замечание модели, отброшенное потому, что его цитаты нет в документе (возможная галлюцинация)."""
    scope: str          # requirement / set
    req_id: str         # R-xx или «—»
    criterion: str
    title: str
    quote: str          # цитата модели, которую не нашли в тексте


@dataclass
class Verdict:
    status: str                    # НЕ ГОТОВ / ГОТОВ С ОГОВОРКАМИ / ГОТОВ / ПРОВЕРКА НЕ ЗАВЕРШЕНА
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
    rejected: list[Rejected] = field(default_factory=list)  # отброшенные замечания модели
    failed_steps: list[str] = field(default_factory=list)   # шаги, на которых LLM не ответила
    extraction: str = "LLM"                                  # LLM / правила / правила (ошибка LLM)

    @property
    def unique_findings(self) -> list[Finding]:
        """Дефекты без повторов: замечание по требованию, повторяющее дефект набора, считается один раз."""
        return [f for f in self.findings if f.dup_of is None]

    @property
    def complete(self) -> bool:
        """Все шаги проверки выполнены. Если нет — отчёт честно помечается как неполный."""
        return not self.unchecked and not self.failed_steps


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
        self.rejected: list[Rejected] = []
        self.failed_steps: list[str] = []
        self.extraction = "правила" if provider is None else "LLM"

    # ------------------------------------------------------------------ extraction
    def extract(self) -> list[Requirement]:
        if self.llm is None:
            return self._extract_offline()
        try:
            reqs = self._extract_llm()
        except LLMError as e:
            self.failed_steps.append("нарезка")
            self.notes.append(f"Нарезка: ошибка LLM — {e}. Требования выделены формальными правилами.")
            log.warning("Нарезка моделью не удалась (%s) — нарезаю правилами", str(e)[:160])
            self.extraction = "правила (ошибка LLM)"
            return self._extract_offline()
        if not reqs:
            self.failed_steps.append("нарезка")
            self.notes.append("Нарезка: модель не вернула ни одного требования с подтверждённой цитатой. "
                              "Требования выделены формальными правилами.")
            log.warning("Модель не выделила требований — нарезаю правилами")
            self.extraction = "правила (модель не выделила требований)"
            return self._extract_offline()
        return reqs

    def _extract_llm(self) -> list[Requirement]:
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
        has_sections = any(l.is_heading and l.section for l in self.doc.lines)
        for ln in self.doc.lines:
            t = ln.text.strip()
            if not t or ln.is_heading or skip_sections.search(ln.section):
                continue
            if has_sections and not ln.section:   # преамбула до первого раздела — служебный текст
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

    @staticmethod
    def _resolve_req(raw: str, by_id: dict[str, Requirement], candidates: list[Requirement]) -> Optional[Requirement]:
        """ID требования из ответа модели → требование. Модель пишет «R-1», «R01», «Р-01» (кириллица)
        или вместо нашего ID даёт номер из документа («Ф-2») — всё это сводится к одному требованию."""
        rid = normalize_rid(raw)
        req = by_id.get(rid) if rid else None
        if req is None:
            key = re.sub(r"\W+", "", str(raw)).lower()
            same = [r for r in candidates if key and re.sub(r"\W+", "", r.doc_id).lower() == key]
            req = same[0] if len(same) == 1 else None
        return req

    def _anchor_issue_quote(self, quote: str, req: Optional[Requirement]) -> tuple[str, Optional[Location]]:
        """Цитата-доказательство: сначала в строках своего требования; короткая — только там."""
        hint = req.location.line_from if req else None
        loc = self.anchor.locate(quote, hint_line=hint, within=req.location if req else None,
                                 min_global=MIN_EVIDENCE_CHARS)
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
                    req = self._resolve_req(a.req_id, by_id, pending)
                    if req is None:
                        self.notes.append(f"{label}: модель вернула оценку для неизвестного требования «{a.req_id}» — пропущена.")
                        continue
                    if req not in pending:      # уже оценено или не из этого пакета
                        continue
                    got.add(req.rid)
                    for iss in a.issues:
                        quote, loc = self._anchor_issue_quote(iss.quote, req)
                        if loc is None:
                            self.rejected.append(Rejected("requirement", req.rid, iss.criterion, iss.title.strip(),
                                                          iss.quote.strip()))
                            continue
                        findings.append(Finding("", "requirement", iss.criterion, iss.severity,
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
        # Один проход по всем S1–S8: два отдельных прохода давали дубли и лишний запрос к модели.
        for label, system in (("проверка набора", prompts.SET_SYSTEM),):
            log.info(label)
            try:
                res = self.llm.generate(system, prompts.SET_USER.format(name=self.doc.name, text=text,
                                        all_reqs=all_reqs, rule_hits=rh), LLMSetResult, label=label)
            except LLMError as e:
                self.notes.append(f"{label}: ошибка LLM — {e}")
                self.failed_steps.append(label)
                continue
            for iss in res.issues:
                ev: list[tuple[str, Location]] = []
                rids: list[str] = []
                for e in iss.evidence:
                    req = self._resolve_req(e.req_id, by_id, reqs)
                    quote, loc = self._anchor_issue_quote(e.quote, req)
                    if loc is None:
                        self.notes.append(f"Набор: цитата «{e.quote[:60]}» к дефекту «{iss.title}» не найдена — исключена.")
                        continue
                    ev.append((quote, loc))
                    owner = self._req_at(loc, reqs)
                    if owner and owner.rid not in rids:
                        rids.append(owner.rid)
                if not ev:
                    self.rejected.append(Rejected("set", ", ".join(e.req_id for e in iss.evidence) or "—", iss.criterion,
                                                  iss.title.strip(), " … ".join(e.quote.strip() for e in iss.evidence)))
                    continue
                findings.append(Finding("", "set", iss.criterion, iss.severity, iss.title.strip(),
                                        iss.explanation.strip(), iss.recommendation.strip(), rids, ev))
        return findings

    # ------------------------------------------------------------------ verdict
    @staticmethod
    def auto_status(findings: list[Finding], complete: bool = True) -> str:
        """Критичные дефекты → НЕ ГОТОВ в любом случае. Если часть проверки не выполнена (модель не ответила),
        «ГОТОВ» выдавать нельзя: отсутствие найденных дефектов ничего не доказывает."""
        if any(f.severity == "критичный" for f in findings):
            return NOT_READY
        if not complete:
            return INCOMPLETE
        if any(f.severity == "значительный" for f in findings):
            return READY_WITH_REMARKS
        return READY

    @staticmethod
    def fallback_priorities(findings: list[Finding], limit: int = 8) -> list[tuple[list[str], str]]:
        """«Что исправить в первую очередь» без модели: критичные, затем значительные; внутри уровня —
        противоречия, непроверяемые и неполные формулировки (группы ТЗ) раньше остальных, затем по номеру."""
        from .criteria import tz_group
        group_rank = {"contradictory": 0, "unverifiable": 1, "incomplete": 2}
        top = sorted((f for f in findings if f.severity in ("критичный", "значительный")),
                     key=lambda f: (SEVERITY_ORDER.get(f.severity, 3), group_rank.get(tz_group(f.criterion), 3), f.fid))
        return [([f.fid], f"{f.title}. {f.recommendation}".strip() if f.recommendation else f.title)
                for f in top[:limit]]

    def verdict(self, reqs: list[Requirement], findings: list[Finding], unchecked: list[str] | None = None) -> Verdict:
        complete = not unchecked and not self.failed_steps
        status = self.auto_status(findings, complete)
        crit = [f for f in findings if f.severity == "критичный"]
        cnt = {s: sum(1 for f in findings if f.severity == s) for s in SEVERITY_ORDER}
        gaps = ", ".join(self.failed_steps + ([f"оценка {len(unchecked)} требований"] if unchecked else []))
        fallback_dev = {
            NOT_READY: f"Начинать разработку нельзя: {len(crit)} критичных дефектов ({', '.join(f.fid for f in crit[:10])}) допускают разное толкование поведения системы.",
            READY_WITH_REMARKS: "Разработку можно начинать после согласования значительных дефектов с аналитиком.",
            READY: "Критичных и значительных дефектов не найдено — разработку можно начинать.",
            INCOMPLETE: f"Вывод о готовности сделать нельзя: часть проверки не выполнена ({gaps}). Повторите прогон.",
        }[status]
        fallback_test = ("Проектирование тест-кейсов заблокировано для требований с критичными дефектами."
                         if crit else "Тестирование можно планировать по текущей версии.")
        fallback_prio = self.fallback_priorities(findings)
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
        rule_only = set(req_hits) if self.llm is None else set(unchecked) & set(req_hits)
        if rule_only:   # модель эти требования не оценила — оставляем хотя бы формальные находки
            for rid in [r.rid for r in reqs if r.rid in rule_only]:
                for h in req_hits[rid]:
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

        findings, n_merged, n_linked = dedupe(findings)
        if n_merged or n_linked:
            self.notes.append(f"Дедупликация: склеено одинаковых замечаний — {n_merged}; замечаний по требованиям, "
                              f"повторяющих дефект набора, — {n_linked} (в сводке учтены один раз).")

        # нумерация дефектов: по требованиям (в порядке документа), затем набор, затем формальные.
        # Повторы дефекта набора номера не получают — ссылаются на номер дефекта набора.
        order = {r.rid: i for i, r in enumerate(reqs)}
        scope_rank = {"requirement": 0, "set": 1, "rule": 2}
        findings.sort(key=lambda f: (scope_rank[f.scope],
                                     order.get(f.req_ids[0], 10**6) if f.scope == "requirement" else 0,
                                     SEVERITY_ORDER.get(f.severity, 3),
                                     min(loc.line_from for _, loc in f.evidence)))
        unique = [f for f in findings if f.dup_of is None]
        for i, f in enumerate(unique, 1):
            f.fid = f"D-{i:03d}"
        for f in findings:
            if f.dup_of is not None:
                f.fid = f"↪{f.dup_of.fid}"

        verdict = self.verdict(reqs, unique, unchecked)
        if self.rejected:
            log.warning("Отброшено замечаний модели без подтверждённой цитаты: %d (см. приложение отчёта)",
                        len(self.rejected))
        finished = datetime.now().astimezone()
        return RunResult(self.doc, reqs, findings, verdict, started, finished, self.llm,
                         "LLM" if self.llm else "offline (только правила)", self.notes, unchecked,
                         self.rejected, self.failed_steps, self.extraction)


def is_critical_type(f: Finding) -> Optional[str]:
    """Тип критичного дефекта для сводки (непроверяемое / противоречивое / неполное ...)."""
    if f.severity != "критичный":
        return None
    return CRITICAL_CRITERIA.get(f.criterion, "критичный")
