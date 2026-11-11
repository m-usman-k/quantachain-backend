"""Static analysis of smart contracts (FE-1, "QuantaAI Guard" scanner).

Two engines share the ``Finding`` / ``ScanResult`` output shape:

* ``SolidityScanner`` - a light, dependency-free Solidity front-end (comments and
  string literals are blanked while preserving offsets, then contracts,
  functions, modifiers and state variables are recovered with brace matching)
  feeding rule-based detectors for SWC-registry weaknesses and the
  centralisation / honeypot patterns typical of rug pulls.
* ``BytecodeScanner`` - for unverified contracts: a minimal EVM disassembler
  (skips PUSH immediates and the trailing CBOR metadata) that reports dangerous
  opcodes and known 4-byte selectors. Bytecode is lossy, so its findings are
  capped at ``medium`` severity.

Risk score = ``min(100, sum(SEVERITY_WEIGHTS[severity]))`` over deduplicated
findings; labels follow ``RISK_LABEL_THRESHOLDS``. Both scanners never raise on
arbitrary input: the worst case is a single ``info`` finding ``unparseable``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import structlog

from app.integrations.evm import keccak256
from app.modules.fraud.models import Finding, FindingSeverity, RiskLabel

logger = structlog.get_logger(__name__)

MAX_SOURCE_CHARS = 1_500_000
SNIPPET_CHARS = 160

SEVERITY_WEIGHTS: dict[FindingSeverity, int] = {
    FindingSeverity.CRITICAL: 30,
    FindingSeverity.HIGH: 15,
    FindingSeverity.MEDIUM: 7,
    FindingSeverity.LOW: 3,
    FindingSeverity.INFO: 0,
}
# (minimum score, label) - evaluated top-down.
RISK_LABEL_THRESHOLDS: tuple[tuple[int, RiskLabel], ...] = (
    (75, RiskLabel.CRITICAL),
    (50, RiskLabel.HIGH),
    (25, RiskLabel.MEDIUM),
    (10, RiskLabel.LOW),
    (0, RiskLabel.SAFE),
)
# Distinct honeypot indicators needed before the composite critical finding fires.
HONEYPOT_COMPOSITE_MIN = 3
HONEYPOT_INDICATOR_IDS = frozenset(
    {
        "blacklist",
        "trading-toggle",
        "tx-limits",
        "fee-setter-unbounded",
        "fee-high",
        "pair-restriction",
        "hidden-owner",
        "approve-all",
    }
)
FEE_HIGH_PCT = 10.0
FEE_CRITICAL_PCT = 50.0
FEE_CAP_WARN_PCT = 25.0

IDENT = r"[A-Za-z_$][A-Za-z0-9_$]*"
_CLEAN_RE = re.compile(r"//[^\n]*|/\*.*?\*/|\"(?:\\.|[^\"\\\n])*\"|'(?:\\.|[^'\\\n])*'", re.DOTALL)
_PRAGMA_RE = re.compile(r"\bpragma\s+solidity\s+([^;]+);")
_CONTRACT_RE = re.compile(
    rf"\b(abstract\s+contract|contract|library|interface)\s+({IDENT})\s*(?:is\s+([^{{]*?))?\s*\{{"
)
_FUNC_RE = re.compile(rf"\b(?:function\s+({IDENT})|(constructor)|(fallback)|(receive)|modifier\s+({IDENT}))\s*\(")
_BRACE_RE = re.compile(r"[{}]")
_PAREN_RE = re.compile(r"[()]")
_HEADER_TOKEN_RE = re.compile(rf"({IDENT})(\s*\((?:[^()]|\([^()]*\))*\))?")
_STATE_VAR_RE = re.compile(rf"^(?P<type>.+?)\s+(?P<name>{IDENT})\s*(?:=(?![=>])(?P<value>.*))?$", re.DOTALL)
_STATEMENT_RE = re.compile(r"[^;{}]+;")
_BLOCK_DECL_RE = re.compile(rf"\b(struct|enum)\s+{IDENT}\s*\{{")
_VERSION_RE = re.compile(r"(\d+)\.(\d+)(?:\.(\d+))?")
_WRITE_RE = re.compile(
    rf"(?<![\w.$])({IDENT})\s*((?:\[[^\]]*\]\s*)*)(?:\.{IDENT}\s*)*"
    r"(?:(?<![=!<>])=(?![=>])|\+=|-=|\*=|/=|%=|\|=|&=|\^=|<<=|>>=|\+\+|--)"
)
_PREFIX_WRITE_RE = re.compile(rf"(?:\+\+|--|\bdelete)\s*({IDENT})")
_EXT_CALL_RE = re.compile(
    r"\.(?:(call)\s*\{[^}]*\}\s*\(|(call)\s*\.\s*value\s*\([^)]*\)\s*\(|(call)\s*\(|(send)\s*\(|(transfer)\s*\()"
)
_LOW_LEVEL_RE = re.compile(r"\.(call|delegatecall|staticcall|send)\b\s*(?:\{[^}]*\}\s*)?\(")
_TX_ORIGIN_RE = re.compile(r"\btx\s*\.\s*origin\b")
_SELFDESTRUCT_RE = re.compile(r"\b(selfdestruct|suicide)\s*\(")
_DELEGATECALL_RE = re.compile(rf"(?:({IDENT})\s*\.\s*)?\b(delegatecall|callcode)\s*\(")
_TIMESTAMP_RE = re.compile(r"\bblock\s*\.\s*timestamp\b|\bnow\b")
_CONDITION_RE = re.compile(r"\b(require|if|while|assert)\s*\(")
_RANDOM_RE = re.compile(r"keccak256|random|seed|lottery|winner", re.IGNORECASE)
_KEYWORDS = frozenset(
    {
        "public",
        "private",
        "internal",
        "external",
        "view",
        "pure",
        "payable",
        "constant",
        "virtual",
        "override",
        "returns",
        "memory",
        "storage",
        "calldata",
    }
)
_GUARD_MODIFIER_RE = re.compile(r"reentran|^lock|^mutex|^nonReentrant|^noReentrancy", re.IGNORECASE)
_ACCESS_MODIFIER_RE = re.compile(r"^only|^auth|^restricted|^admin|^owner|^governance|^whenOwner|Role$", re.IGNORECASE)
_ACCESS_CHECK_RE = re.compile(r"require\s*\(\s*(?:msg\.sender|_msgSender\(\))\s*==|_checkOwner\(|onlyOwner|hasRole\(")
_BLACKLIST_RE = re.compile(
    r"\b_?(?:is)?black_?list\w*|\b_?is_?bot\b|\badd_?bots?\b|\bdel_?bots?\b|\bset_?bots?\b|\b_?bots\b"
    r"|\b_?is_?blacklisted\w*|\b_?banned\w*|\b_?is_?sniper\w*|\b_?snipers\b|\bblock_?list\w*|\bisBlocked\b",
    re.IGNORECASE,
)
_TRADING_TOGGLE_RE = re.compile(
    r"\b_?trading_?(?:enabled|open|active|started|allowed|live)\b|\b(?:enable|open|start|set|toggle)_?trading\w*\b"
    r"|\b_?can_?trade\b|\bisTradingEnabled\b|\blaunch\s*\(|\b_?launched\b|\bswapEnabled\b",
    re.IGNORECASE,
)
_TX_LIMIT_RE = re.compile(
    r"\bset_?max_?tx\w*|\b_?max_?tx_?amount\b|\b_?max_?tx\b|\bmaxTransactionAmount\b|\b_?max_?wallet\w*"
    r"|\bcool_?down\w*|\bremoveLimits\b|\blimitsInEffect\b|\btransferDelayEnabled\b|\b_?max_?sell\w*|\b_?max_?buy\w*",
    re.IGNORECASE,
)
_FEE_SETTER_NAME_RE = re.compile(r"^(?:set|update|change|adjust|modify|_set)\w*(?:fee|tax|taxes)\w*$", re.IGNORECASE)
_FEE_VAR_NAME_RE = re.compile(r"fee|tax", re.IGNORECASE)
_FEE_EXCLUDE_NAME_RE = re.compile(
    r"denominator|divisor|wallet|receiver|address|recipient|enabled|exempt|excluded|max|cap", re.IGNORECASE
)
_BOUND_CHECK_RE = re.compile(
    r"(?:require|assert)\s*\([^;]*?(<=|<)\s*(\d+)|if\s*\([^;]*?(>=|>)\s*(\d+)[^;]*?\)\s*revert"
)
_PAIR_IDENT_RE = re.compile(r"\b\w*pair\w*\b", re.IGNORECASE)
_COMPARE_RE = re.compile(r"==|!=")
_TRANSFER_FUNCS = frozenset(
    {"_transfer", "transfer", "transferFrom", "_beforeTokenTransfer", "_update", "_tokenTransfer", "_transferStandard"}
)
_ALLOWANCE_FUNCS = frozenset(
    {"approve", "_approve", "increaseAllowance", "decreaseAllowance", "_spendAllowance", "permit", "transferFrom"}
)
_APPROVE_CALL_RE = re.compile(r"\b_approve\s*\(\s*([^,()]+?)\s*,\s*([^,()]+?)\s*,")
_ALLOWANCE_WRITE_RE = re.compile(r"\b_?allowances?\s*\[\s*([^\]]+?)\s*\]\s*\[\s*([^\]]+?)\s*\]\s*=(?!=)")
_OWNER_LIKE_RE = re.compile(
    r"owner\s*\(\s*\)|\b_owner\b|\bmsg\.sender\b|_msgSender\s*\(\s*\)|tx\.origin|[Ww]allet\b|\b_?dev\w*|\b_?marketing\w*|\b_?team\w*"
)
_SELF_LIKE_RE = re.compile(r"^(?:msg\.sender|_msgSender\(\)|owner|sender|account|holder)$")
_PAUSE_RE = re.compile(
    r"\bwhenNotPaused\b|\b_requireNotPaused\s*\(|\bpaused\s*\(\s*\)|!\s*paused\b|\brequire\s*\(\s*!\s*_?paused\b"
)
_PAUSE_FUNC_RE = re.compile(r"\bfunction\s+pause\s*\(|\b_pause\s*\(\s*\)")
_PROXY_RE = re.compile(
    r"\bupgradeTo(?:AndCall)?\b|\b_upgradeTo\w*\b|\b_implementation\b|\b_setImplementation\b|\bIMPLEMENTATION_SLOT\b"
    r"|\bERC1967\w*|\bUUPSUpgradeable\b|\bTransparentUpgradeableProxy\b|\b_getImplementation\b"
)
_RENOUNCE_REAL_RE = re.compile(
    r"_owner\s*=\s*address\s*\(\s*0\s*\)|\bowner\s*=\s*address\s*\(\s*0\s*\)|_transferOwnership\s*\(\s*address\s*\(\s*0\s*\)\s*\)"
    r"|_setOwner\s*\(\s*address\s*\(\s*0\s*\)\s*\)|transferOwnership\s*\(\s*address\s*\(\s*0\s*\)\s*\)"
)
_PREVIOUS_OWNER_RE = re.compile(r"\b_previousOwner\b|\bpreviousOwner\b")
_UNLOCK_RE = re.compile(r"\bfunction\s+unlock\s*\(|\bfunction\s+lock\s*\(")
_MINT_RE = re.compile(r"\b_mint\s*\(|\b_?totalSupply\s*(?:\+=|=\s*_?totalSupply\s*(?:\+|\.add))")
_SUPPLY_CAP_RE = re.compile(
    r"\bcap\s*\(\s*\)|\b_cap\b|MAX_SUPPLY|\bmax_?Supply\b|\b_maxSupply\b|ERC20Capped|totalSupply\s*\(\s*\)\s*\+\s*\w+\s*<="
)
_SAFEMATH_RE = re.compile(r"\busing\s+SafeMath\b|\bSafeMath\s*\.|\.add\s*\(|\.sub\s*\(|\.mul\s*\(")
_ARITH_RE = re.compile(r"[^+]\+[^+=]|[^-]-[^-=>]|[^*/]\*[^*=]")
_MINIMAL_PROXY_RE = re.compile(r"^363d3d373d3d3d363d73([0-9a-f]{40})5af43d82803e903d91602b57fd5bf3", re.IGNORECASE)


# ----------------------------------------------------------------------- data
@dataclass(slots=True)
class StateVar:
    name: str
    type: str
    line: int
    value: str | None
    contract: str
    is_mapping: bool = False


@dataclass(slots=True)
class FunctionInfo:
    name: str
    kind: str  # function | constructor | fallback | receive | modifier
    contract: str
    visibility: str
    modifiers: list[str]
    params: str
    mutability: str | None
    body: str | None
    body_offset: int  # absolute offset of the body text (after the opening brace)
    start_line: int
    end_line: int
    start_offset: int
    end_offset: int

    @property
    def is_view(self) -> bool:
        return self.mutability in ("view", "pure")

    @property
    def param_names(self) -> set[str]:
        names: set[str] = set()
        for part in self.params.split(","):
            tokens = re.findall(IDENT, part)
            if tokens:
                names.add(tokens[-1])
        return names


@dataclass(slots=True)
class ContractInfo:
    name: str
    kind: str  # contract | abstract contract | library | interface
    bases: list[str]
    start_offset: int
    end_offset: int
    body_offset: int
    body: str
    start_line: int
    end_line: int
    functions: list[FunctionInfo] = field(default_factory=list)
    state_vars: list[StateVar] = field(default_factory=list)
    using_for: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ParsedSource:
    original: str
    text: str  # cleaned (comments/strings blanked, same length as original)
    pragma: str | None
    contracts: list[ContractInfo]
    lines: int

    @property
    def functions(self) -> list[FunctionInfo]:
        return [fn for c in self.contracts for fn in c.functions]

    @property
    def state_var_names(self) -> set[str]:
        return {v.name for c in self.contracts for v in c.state_vars}

    def line_of(self, offset: int) -> int:
        return self.text.count("\n", 0, max(0, min(offset, len(self.text)))) + 1

    def snippet_at(self, offset: int) -> str:
        line = self.line_of(offset)
        rows = self.original.splitlines()
        if 1 <= line <= len(rows):
            return rows[line - 1].strip()[:SNIPPET_CHARS]
        return ""

    def contract_at(self, offset: int) -> ContractInfo | None:
        for contract in self.contracts:
            if contract.start_offset <= offset < contract.end_offset:
                return contract
        return None

    def function_at(self, offset: int) -> FunctionInfo | None:
        contract = self.contract_at(offset)
        if contract is None:
            return None
        for fn in contract.functions:
            if fn.start_offset <= offset < fn.end_offset:
                return fn
        return None


@dataclass
class ScanResult:
    findings: list[Finding]
    risk_score: int
    risk_label: RiskLabel
    stats: dict[str, Any]
    kind: str = "source"  # source | bytecode
    contract_name: str | None = None
    pragma: str | None = None
    contracts: list[str] = field(default_factory=list)
    is_proxy: bool = False
    implementation: str | None = None

    @property
    def counts(self) -> dict[str, int]:
        out = {s.value: 0 for s in FindingSeverity}
        for finding in self.findings:
            out[finding.severity.value] += 1
        return out


# -------------------------------------------------------------------- scoring
def score_findings(findings: list[Finding]) -> int:
    return min(100, sum(SEVERITY_WEIGHTS[f.severity] for f in findings))


def risk_label(score: int) -> RiskLabel:
    for minimum, label in RISK_LABEL_THRESHOLDS:
        if score >= minimum:
            return label
    return RiskLabel.SAFE


def dedupe_findings(findings: list[Finding]) -> list[Finding]:
    """Merge findings sharing (id, contract, function); union their lines, keep the first text."""
    merged: dict[tuple[str, str | None, str | None], Finding] = {}
    for finding in findings:
        key = (finding.id, finding.contract, finding.function)
        existing = merged.get(key)
        if existing is None:
            merged[key] = finding.model_copy(update={"lines": sorted(set(finding.lines))})
            continue
        existing.lines = sorted(set(existing.lines) | set(finding.lines))
        if _severity_rank(finding.severity) > _severity_rank(existing.severity):
            existing.severity = finding.severity
    ordered = sorted(merged.values(), key=lambda f: (-_severity_rank(f.severity), f.lines[:1] or [0], f.id))
    return ordered


def _severity_rank(severity: FindingSeverity) -> int:
    return {
        FindingSeverity.INFO: 0,
        FindingSeverity.LOW: 1,
        FindingSeverity.MEDIUM: 2,
        FindingSeverity.HIGH: 3,
        FindingSeverity.CRITICAL: 4,
    }[severity]


# -------------------------------------------------------------------- parsing
def clean_source(text: str) -> str:
    """Blank comments and string-literal contents while preserving every offset and newline."""

    def blank(match: re.Match[str]) -> str:
        token = match.group(0)
        if token[0] in "\"'" and len(token) >= 2:
            return token[0] + re.sub(r"[^\n]", " ", token[1:-1]) + token[-1]
        return re.sub(r"[^\n]", " ", token)

    return _CLEAN_RE.sub(blank, text)


def _match_bracket(text: str, open_pos: int, pattern: re.Pattern[str], open_ch: str, end: int | None = None) -> int:
    depth = 0
    for match in pattern.finditer(text, open_pos, end if end is not None else len(text)):
        if match.group(0) == open_ch:
            depth += 1
        else:
            depth -= 1
            if depth == 0:
                return match.start()
    return -1


def parse_pragma_version(pragma: str | None) -> tuple[int, int, int] | None:
    """Lowest version mentioned in a pragma, e.g. ``>=0.6.0 <0.8.0`` -> (0, 6, 0)."""
    if not pragma:
        return None
    versions = [(int(a), int(b), int(c or 0)) for a, b, c in _VERSION_RE.findall(pragma)]
    return min(versions) if versions else None


def is_floating_pragma(pragma: str | None) -> bool:
    return bool(pragma) and any(ch in pragma for ch in "^~<>") if pragma else False


def _parse_header(header: str) -> tuple[str, list[str], str | None]:
    visibility = "public"
    mutability: str | None = None
    modifiers: list[str] = []
    for match in _HEADER_TOKEN_RE.finditer(header):
        name = match.group(1)
        if name in ("public", "private", "internal", "external"):
            visibility = name
        elif name in ("view", "pure", "payable", "constant"):
            mutability = name
        elif name in _KEYWORDS:
            continue
        else:
            modifiers.append(name)
    return visibility, modifiers, mutability


def _parse_functions(parsed_text: str, contract: ContractInfo, line_of: Any) -> list[FunctionInfo]:
    functions: list[FunctionInfo] = []
    pos = contract.body_offset
    end = contract.end_offset
    while pos < end:
        match = _FUNC_RE.search(parsed_text, pos, end)
        if match is None:
            break
        name = match.group(1) or match.group(5) or match.group(2) or match.group(3) or match.group(4) or "function"
        if match.group(5):
            kind = "modifier"
        elif match.group(1):
            kind = "function"
        else:
            kind = name
        paren_open = match.end() - 1
        paren_close = _match_bracket(parsed_text, paren_open, _PAREN_RE, "(", end)
        if paren_close == -1:
            break
        params = parsed_text[paren_open + 1 : paren_close]
        terminator = re.compile(r"[{;]").search(parsed_text, paren_close + 1, end)
        if terminator is None:
            break
        header = parsed_text[paren_close + 1 : terminator.start()]
        visibility, modifiers, mutability = _parse_header(header)
        if terminator.group(0) == "{":
            body_close = _match_bracket(parsed_text, terminator.start(), _BRACE_RE, "{", end)
            if body_close == -1:
                body_close = end - 1
            body = parsed_text[terminator.start() + 1 : body_close]
            body_offset = terminator.start() + 1
            end_offset = body_close + 1
        else:
            body = None
            body_offset = terminator.start()
            end_offset = terminator.end()
        functions.append(
            FunctionInfo(
                name=name,
                kind=kind,
                contract=contract.name,
                visibility=visibility,
                modifiers=modifiers,
                params=params,
                mutability=mutability,
                body=body,
                body_offset=body_offset,
                start_line=line_of(match.start()),
                end_line=line_of(end_offset),
                start_offset=match.start(),
                end_offset=end_offset,
            )
        )
        pos = max(end_offset, match.end())
    return functions


def _parse_state_vars(parsed_text: str, contract: ContractInfo, line_of: Any) -> tuple[list[StateVar], list[str]]:
    body = list(contract.body)
    base = contract.body_offset
    # Blank function / modifier bodies and struct / enum blocks so only top-level declarations remain.
    for fn in contract.functions:
        for i in range(max(fn.start_offset - base, 0), min(fn.end_offset - base, len(body))):
            if body[i] != "\n":
                body[i] = " "
    top = "".join(body)
    for match in _BLOCK_DECL_RE.finditer(top):
        close = _match_bracket(top, match.end() - 1, _BRACE_RE, "{")
        close = len(top) - 1 if close == -1 else close
        for i in range(match.start(), close + 1):
            if top[i] != "\n":
                body[i] = " "
    top = "".join(body)
    state_vars: list[StateVar] = []
    using_for: list[str] = []
    for match in _STATEMENT_RE.finditer(top):
        stmt = match.group(0)[:-1].strip()
        if not stmt:
            continue
        first = re.match(IDENT, stmt)
        head = first.group(0) if first else ""
        if head == "using":
            using_for.append(stmt)
            continue
        if head in (
            "event",
            "error",
            "function",
            "modifier",
            "constructor",
            "pragma",
            "import",
            "emit",
            "return",
            "mapping_",
        ):
            continue
        decl = _STATE_VAR_RE.match(stmt)
        if decl is None:
            continue
        type_text = re.sub(
            r"\b(public|private|internal|constant|immutable|override|payable)\b", " ", decl.group("type")
        )
        type_text = re.sub(r"\s+", " ", type_text).strip()
        if not type_text or decl.group("name") in _KEYWORDS:
            continue
        value = decl.group("value")
        offset = base + match.start() + (len(match.group(0)) - len(match.group(0).lstrip()))
        state_vars.append(
            StateVar(
                name=decl.group("name"),
                type=type_text,
                line=line_of(offset),
                value=value.strip() if value else None,
                contract=contract.name,
                is_mapping=type_text.startswith("mapping"),
            )
        )
    return state_vars, using_for


def parse_solidity(source: str) -> ParsedSource:
    text = clean_source(source)
    pragma_match = _PRAGMA_RE.search(text)
    pragma = re.sub(r"\s+", " ", pragma_match.group(1)).strip() if pragma_match else None
    parsed = ParsedSource(original=source, text=text, pragma=pragma, contracts=[], lines=source.count("\n") + 1)
    pos = 0
    while pos < len(text):
        match = _CONTRACT_RE.search(text, pos)
        if match is None:
            break
        brace_open = match.end() - 1
        brace_close = _match_bracket(text, brace_open, _BRACE_RE, "{")
        if brace_close == -1:
            brace_close = len(text) - 1
        kind = re.sub(r"\s+", " ", match.group(1))
        bases = [b.strip().split("(")[0].strip() for b in (match.group(3) or "").split(",") if b.strip()]
        contract = ContractInfo(
            name=match.group(2),
            kind=kind,
            bases=bases,
            start_offset=match.start(),
            end_offset=brace_close + 1,
            body_offset=brace_open + 1,
            body=text[brace_open + 1 : brace_close],
            start_line=parsed.line_of(match.start()),
            end_line=parsed.line_of(brace_close),
        )
        contract.functions = _parse_functions(text, contract, parsed.line_of)
        contract.state_vars, contract.using_for = _parse_state_vars(text, contract, parsed.line_of)
        parsed.contracts.append(contract)
        pos = brace_close + 1
    return parsed


# ------------------------------------------------------------------ detectors
class SolidityScanner:
    """Rule-based Solidity analyser. ``scan`` never raises."""

    def scan(self, source: str) -> ScanResult:
        if not isinstance(source, str) or not source.strip():
            return self._unparseable("Empty source", stats={"contracts": 0, "functions": 0, "lines": 0, "pragma": None})
        truncated = False
        if len(source) > MAX_SOURCE_CHARS:
            source = source[:MAX_SOURCE_CHARS]
            truncated = True
        try:
            parsed = parse_solidity(source)
            findings: list[Finding] = []
            for detector in (
                self._reentrancy,
                self._tx_origin,
                self._selfdestruct,
                self._delegatecall,
                self._unchecked_calls,
                self._integer_overflow,
                self._floating_pragma,
                self._timestamp,
                self._mint,
                self._blacklist,
                self._trading_toggle,
                self._tx_limits,
                self._fees,
                self._pair_restriction,
                self._pausable,
                self._proxy,
                self._hidden_owner,
                self._approve_all,
            ):
                try:
                    findings.extend(detector(parsed))
                except Exception as exc:  # pragma: no cover - a single rule must never sink the scan
                    logger.warning("scanner_rule_failed", rule=detector.__name__, error=str(exc))
            findings = dedupe_findings(findings)
            findings.extend(self._composite(findings))
            stats = {
                "contracts": len(parsed.contracts),
                "functions": sum(1 for fn in parsed.functions if fn.kind != "modifier"),
                "state_variables": sum(len(c.state_vars) for c in parsed.contracts),
                "lines": parsed.lines,
                "pragma": parsed.pragma,
                "bytes": len(source),
                "truncated": truncated,
            }
            if not parsed.contracts and not parsed.functions:
                findings.append(
                    self._finding(
                        "unparseable",
                        None,
                        "No Solidity contracts recognised",
                        FindingSeverity.INFO,
                        "The input does not contain a contract, library or interface definition the parser could "
                        "recognise; only text-level heuristics were applied.",
                        "Submit the flattened, verified Solidity source of the contract.",
                        category="info",
                    )
                )
            score = score_findings(findings)
            main = next((c for c in reversed(parsed.contracts) if c.kind in ("contract", "abstract contract")), None)
            main = main or (parsed.contracts[0] if parsed.contracts else None)
            return ScanResult(
                findings=findings,
                risk_score=score,
                risk_label=risk_label(score),
                stats=stats,
                kind="source",
                contract_name=main.name if main else None,
                pragma=parsed.pragma,
                contracts=[c.name for c in parsed.contracts],
                is_proxy=any(f.id == "upgradeable" for f in findings),
            )
        except Exception as exc:  # pragma: no cover - defensive; parser is exercised with garbage in tests
            logger.warning("scanner_failed", error=f"{type(exc).__name__}: {exc}")
            return self._unparseable(
                f"{type(exc).__name__}: {exc}"[:200],
                stats={"contracts": 0, "functions": 0, "lines": source.count("\n") + 1, "pragma": None},
            )

    # ------------------------------------------------------------ helpers
    @staticmethod
    def _finding(
        fid: str,
        swc: str | None,
        title: str,
        severity: FindingSeverity,
        description: str,
        recommendation: str,
        *,
        lines: list[int] | None = None,
        snippet: str | None = None,
        function: str | None = None,
        contract: str | None = None,
        category: str = "vulnerability",
    ) -> Finding:
        return Finding(
            id=fid,
            swc=swc,
            title=title,
            severity=severity,
            category=category,
            description=description,
            recommendation=recommendation,
            lines=sorted(set(lines or [])),
            snippet=snippet or None,
            function=function,
            contract=contract,
        )

    def _unparseable(self, detail: str, *, stats: dict[str, Any]) -> ScanResult:
        finding = self._finding(
            "unparseable",
            None,
            "Source could not be analysed",
            FindingSeverity.INFO,
            f"The scanner could not parse the submitted input ({detail}).",
            "Submit valid, flattened Solidity source code.",
            category="info",
        )
        return ScanResult(findings=[finding], risk_score=0, risk_label=RiskLabel.SAFE, stats=stats, kind="source")

    @staticmethod
    def _statement_bounds(body: str, pos: int) -> tuple[int, int]:
        start = max(body.rfind(";", 0, pos), body.rfind("{", 0, pos), body.rfind("}", 0, pos)) + 1
        end = body.find(";", pos)
        return start, len(body) if end == -1 else end

    @staticmethod
    def _enclosing_condition(body: str, pos: int) -> str | None:
        """Return the condition text of the innermost ``require(`` / ``if (`` wrapping ``pos``, if any."""
        for match in _CONDITION_RE.finditer(body, 0, pos + 1):
            close = _match_bracket(body, match.end() - 1, _PAREN_RE, "(")
            if close != -1 and match.end() <= pos < close:
                return body[match.end() : close]
        return None

    def _function_findings(self, parsed: ParsedSource) -> list[tuple[FunctionInfo, str]]:
        return [(fn, fn.body) for fn in parsed.functions if fn.body is not None]

    # ------------------------------------------------------- SWC detectors
    def _reentrancy(self, parsed: ParsedSource) -> list[Finding]:
        out: list[Finding] = []
        state_names = parsed.state_var_names
        for fn, body in self._function_findings(parsed):
            if fn.kind in ("constructor", "modifier") or fn.is_view:
                continue
            if any(_GUARD_MODIFIER_RE.search(m) for m in fn.modifiers):
                continue
            for call in _EXT_CALL_RE.finditer(body):
                kind = next(g for g in call.groups() if g)
                after = body[call.end() :]
                write = self._state_write_after(after, state_names)
                if write is None:
                    continue
                offset = fn.body_offset + call.start()
                write_offset = fn.body_offset + call.end() + write
                severity = FindingSeverity.CRITICAL if kind == "call" else FindingSeverity.MEDIUM
                out.append(
                    self._finding(
                        "reentrancy",
                        "SWC-107",
                        "Reentrancy: state updated after external call",
                        severity,
                        f"`{fn.name}` makes an external `{kind}` on line {parsed.line_of(offset)} and only updates "
                        f"contract state afterwards (line {parsed.line_of(write_offset)}). A malicious callee can "
                        "re-enter before the balance/flag is updated.",
                        "Apply checks-effects-interactions (update state before the call) and/or add a "
                        "`nonReentrant` guard.",
                        lines=[parsed.line_of(offset), parsed.line_of(write_offset)],
                        snippet=parsed.snippet_at(offset),
                        function=fn.name,
                        contract=fn.contract,
                    )
                )
                break
        return out

    @staticmethod
    def _state_write_after(after: str, state_names: set[str]) -> int | None:
        for match in _WRITE_RE.finditer(after):
            name = match.group(1)
            if name in state_names or (not state_names and match.group(2).strip()):
                return match.start()
        for match in _PREFIX_WRITE_RE.finditer(after):
            if match.group(1) in state_names:
                return match.start()
        return None

    def _tx_origin(self, parsed: ParsedSource) -> list[Finding]:
        out: list[Finding] = []
        for fn, body in self._function_findings(parsed):
            for match in _TX_ORIGIN_RE.finditer(body):
                start, end = self._statement_bounds(body, match.start())
                stmt = body[start:end]
                auth = bool(_CONDITION_RE.search(stmt) or _COMPARE_RE.search(stmt))
                offset = fn.body_offset + match.start()
                out.append(
                    self._finding(
                        "tx-origin-auth" if auth else "tx-origin",
                        "SWC-115",
                        "Authorisation through tx.origin" if auth else "tx.origin used",
                        FindingSeverity.HIGH if auth else FindingSeverity.MEDIUM,
                        "`tx.origin` identifies the transaction sender, not the caller; a phishing contract can "
                        "relay a victim's transaction and pass this check."
                        if auth
                        else "`tx.origin` is referenced; it is unreliable with smart-contract wallets and relayers.",
                        "Use `msg.sender` for authorisation checks.",
                        lines=[parsed.line_of(offset)],
                        snippet=parsed.snippet_at(offset),
                        function=fn.name,
                        contract=fn.contract,
                    )
                )
        return out

    def _selfdestruct(self, parsed: ParsedSource) -> list[Finding]:
        out: list[Finding] = []
        for fn, body in self._function_findings(parsed):
            for match in _SELFDESTRUCT_RE.finditer(body):
                protected = any(_ACCESS_MODIFIER_RE.search(m) for m in fn.modifiers) or bool(
                    _ACCESS_CHECK_RE.search(body)
                )
                exposed = fn.visibility in ("public", "external") and fn.kind == "function"
                offset = fn.body_offset + match.start()
                severity = FindingSeverity.CRITICAL if exposed and not protected else FindingSeverity.HIGH
                out.append(
                    self._finding(
                        "selfdestruct",
                        "SWC-106",
                        "Unprotected SELFDESTRUCT"
                        if severity is FindingSeverity.CRITICAL
                        else "Contract can self-destruct",
                        severity,
                        f"`{match.group(1)}` in `{fn.name}` destroys the contract and sends its balance away"
                        + (
                            " - and the function has no access control."
                            if severity is FindingSeverity.CRITICAL
                            else "."
                        ),
                        "Remove `selfdestruct` or restrict it behind multi-sig / timelock governance.",
                        lines=[parsed.line_of(offset)],
                        snippet=parsed.snippet_at(offset),
                        function=fn.name,
                        contract=fn.contract,
                    )
                )
        return out

    def _delegatecall(self, parsed: ParsedSource) -> list[Finding]:
        out: list[Finding] = []
        for fn, body in self._function_findings(parsed):
            contract = parsed.contract_at(fn.start_offset)
            if contract is not None and contract.kind == "library":
                continue
            for match in _DELEGATECALL_RE.finditer(body):
                target = match.group(1)
                user_controlled = target is not None and target in fn.param_names
                offset = fn.body_offset + match.start()
                out.append(
                    self._finding(
                        "delegatecall",
                        "SWC-112",
                        "Delegatecall to user-supplied address" if user_controlled else "Delegatecall used",
                        FindingSeverity.HIGH if user_controlled else FindingSeverity.MEDIUM,
                        f"`{match.group(2)}` executes foreign code in this contract's storage context"
                        + (f"; the target `{target}` comes from a function parameter." if user_controlled else "."),
                        "Restrict delegatecall targets to trusted, immutable addresses.",
                        lines=[parsed.line_of(offset)],
                        snippet=parsed.snippet_at(offset),
                        function=fn.name,
                        contract=fn.contract,
                    )
                )
        return out

    def _unchecked_calls(self, parsed: ParsedSource) -> list[Finding]:
        out: list[Finding] = []
        for fn, body in self._function_findings(parsed):
            for match in _LOW_LEVEL_RE.finditer(body):
                start, end = self._statement_bounds(body, match.start())
                stmt = body[start:end]
                before = body[start : match.start()]
                checked = (
                    bool(re.search(r"\(\s*bool\b", stmt))
                    or bool(re.match(r"\s*(require|assert|if|return|bool|while)\b", stmt))
                    or "=" in before
                    or self._enclosing_condition(body, match.start()) is not None
                )
                if checked:
                    continue
                offset = fn.body_offset + match.start()
                out.append(
                    self._finding(
                        "unchecked-call",
                        "SWC-104",
                        f"Unchecked return value of `{match.group(1)}`",
                        FindingSeverity.MEDIUM,
                        f"The result of `.{match.group(1)}(...)` in `{fn.name}` is ignored; a failed call "
                        "will not revert and execution continues as if it succeeded.",
                        "Capture the boolean result and `require(success)`.",
                        lines=[parsed.line_of(offset)],
                        snippet=parsed.snippet_at(offset),
                        function=fn.name,
                        contract=fn.contract,
                    )
                )
        return out

    def _integer_overflow(self, parsed: ParsedSource) -> list[Finding]:
        version = parse_pragma_version(parsed.pragma)
        if version is None or version >= (0, 8, 0):
            return []
        if _SAFEMATH_RE.search(parsed.text):
            return []
        arithmetic = any(_ARITH_RE.search(fn.body) for fn in parsed.functions if fn.body)
        if not arithmetic:
            return []
        pragma_match = _PRAGMA_RE.search(parsed.text)
        offset = pragma_match.start() if pragma_match else 0
        return [
            self._finding(
                "integer-overflow",
                "SWC-101",
                "Unchecked arithmetic (pre-0.8 compiler without SafeMath)",
                FindingSeverity.MEDIUM,
                f"Pragma `{parsed.pragma}` predates built-in overflow checks and no SafeMath usage was found; "
                "additions and subtractions can silently wrap around.",
                "Compile with Solidity >= 0.8 or use SafeMath for all arithmetic.",
                lines=[parsed.line_of(offset)],
                snippet=parsed.snippet_at(offset),
            )
        ]

    def _floating_pragma(self, parsed: ParsedSource) -> list[Finding]:
        if not parsed.pragma or not is_floating_pragma(parsed.pragma):
            return []
        pragma_match = _PRAGMA_RE.search(parsed.text)
        offset = pragma_match.start() if pragma_match else 0
        return [
            self._finding(
                "floating-pragma",
                "SWC-103",
                "Floating compiler pragma",
                FindingSeverity.LOW,
                f"`pragma solidity {parsed.pragma}` allows compilation with a range of compiler versions, so the "
                "deployed bytecode may differ from what was audited.",
                "Lock the pragma to the exact compiler version used for deployment.",
                lines=[parsed.line_of(offset)],
                snippet=parsed.snippet_at(offset),
            )
        ]

    def _timestamp(self, parsed: ParsedSource) -> list[Finding]:
        out: list[Finding] = []
        for fn, body in self._function_findings(parsed):
            for match in _TIMESTAMP_RE.finditer(body):
                start, end = self._statement_bounds(body, match.start())
                stmt = body[start:end]
                in_condition = self._enclosing_condition(body, match.start()) is not None or bool(
                    re.match(r"\s*(if|while|require|assert)\b", stmt)
                )
                randomness = bool(_RANDOM_RE.search(stmt))
                if not in_condition and not randomness:
                    continue
                offset = fn.body_offset + match.start()
                if randomness:
                    out.append(
                        self._finding(
                            "weak-randomness",
                            "SWC-120",
                            "Block timestamp used as a source of randomness",
                            FindingSeverity.MEDIUM,
                            "Miners/validators can influence `block.timestamp`; it must not seed randomness.",
                            "Use a verifiable randomness source (e.g. Chainlink VRF).",
                            lines=[parsed.line_of(offset)],
                            snippet=parsed.snippet_at(offset),
                            function=fn.name,
                            contract=fn.contract,
                        )
                    )
                else:
                    out.append(
                        self._finding(
                            "timestamp-dependence",
                            "SWC-116",
                            "Control flow depends on block timestamp",
                            FindingSeverity.LOW,
                            f"`{fn.name}` branches on `{match.group(0)}`; validators can skew it by several seconds.",
                            "Avoid tight timestamp comparisons or tolerate a drift window.",
                            lines=[parsed.line_of(offset)],
                            snippet=parsed.snippet_at(offset),
                            function=fn.name,
                            contract=fn.contract,
                        )
                    )
        return out

    # ------------------------------------------- centralisation / honeypot
    def _mint(self, parsed: ParsedSource) -> list[Finding]:
        out: list[Finding] = []
        for fn, body in self._function_findings(parsed):
            if fn.kind != "function" or fn.visibility in ("internal", "private"):
                continue
            match = _MINT_RE.search(body)
            if match is None:
                continue
            contract = parsed.contract_at(fn.start_offset)
            scope = (contract.body if contract else "") + " ".join(
                c.body for c in parsed.contracts if contract and c.name in contract.bases
            )
            capped = bool(_SUPPLY_CAP_RE.search(scope)) or bool(_SUPPLY_CAP_RE.search(body))
            protected = any(_ACCESS_MODIFIER_RE.search(m) for m in fn.modifiers) or bool(_ACCESS_CHECK_RE.search(body))
            offset = fn.body_offset + match.start()
            if capped:
                severity, title = FindingSeverity.MEDIUM, "Privileged minting (capped supply)"
            elif protected:
                severity, title = FindingSeverity.HIGH, "Owner can mint unlimited tokens"
            else:
                severity, title = FindingSeverity.CRITICAL, "Anyone can mint tokens"
            out.append(
                self._finding(
                    "owner-mint",
                    None,
                    title,
                    severity,
                    f"`{fn.name}` increases the token supply"
                    + (" without any supply cap" if not capped else "")
                    + (" and without access control." if not protected else "."),
                    "Remove minting, enforce a hard cap, or move mint authority behind a timelock/multi-sig.",
                    lines=[parsed.line_of(offset)],
                    snippet=parsed.snippet_at(offset),
                    function=fn.name,
                    contract=fn.contract,
                    category="centralisation",
                )
            )
        return out

    def _pattern_finding(
        self,
        parsed: ParsedSource,
        pattern: re.Pattern[str],
        *,
        fid: str,
        title: str,
        severity: FindingSeverity,
        description: str,
        recommendation: str,
        category: str,
        max_lines: int = 12,
    ) -> list[Finding]:
        lines: list[int] = []
        snippet: str | None = None
        contract: str | None = None
        for match in pattern.finditer(parsed.text):
            if snippet is None:
                snippet = parsed.snippet_at(match.start())
                found = parsed.contract_at(match.start())
                contract = found.name if found else None
            lines.append(parsed.line_of(match.start()))
        if not lines:
            return []
        unique = sorted(set(lines))
        return [
            self._finding(
                fid,
                None,
                title,
                severity,
                description + f" ({len(unique)} occurrence{'s' if len(unique) != 1 else ''}).",
                recommendation,
                lines=unique[:max_lines],
                snippet=snippet,
                contract=contract,
                category=category,
            )
        ]

    def _blacklist(self, parsed: ParsedSource) -> list[Finding]:
        return self._pattern_finding(
            parsed,
            _BLACKLIST_RE,
            fid="blacklist",
            title="Blacklist / anti-bot controls",
            severity=FindingSeverity.HIGH,
            description="Addresses can be blacklisted or flagged as bots, which lets a privileged account block "
            "specific holders from selling",
            recommendation="Treat as a honeypot indicator unless the blacklist is governed transparently.",
            category="honeypot",
        )

    def _trading_toggle(self, parsed: ParsedSource) -> list[Finding]:
        return self._pattern_finding(
            parsed,
            _TRADING_TOGGLE_RE,
            fid="trading-toggle",
            title="Trading can be switched on/off",
            severity=FindingSeverity.HIGH,
            description="A trading toggle lets the deployer disable transfers for everyone except themselves",
            recommendation="Verify trading is enabled and the toggle cannot be flipped back (or ownership is renounced).",
            category="honeypot",
        )

    def _tx_limits(self, parsed: ParsedSource) -> list[Finding]:
        return self._pattern_finding(
            parsed,
            _TX_LIMIT_RE,
            fid="tx-limits",
            title="Adjustable transaction / wallet limits",
            severity=FindingSeverity.MEDIUM,
            description="Max-transaction, max-wallet or cooldown limits can be tightened by the owner to block sells",
            recommendation="Check the current limits and whether they can be changed after launch.",
            category="honeypot",
        )

    def _fees(self, parsed: ParsedSource) -> list[Finding]:
        out: list[Finding] = []
        for fn, body in self._function_findings(parsed):
            if fn.kind != "function" or not _FEE_SETTER_NAME_RE.search(fn.name):
                continue
            if fn.visibility in ("internal", "private"):
                continue
            bound = _BOUND_CHECK_RE.search(body)
            offset = fn.start_offset
            if bound is None:
                out.append(
                    self._finding(
                        "fee-setter-unbounded",
                        None,
                        "Fee setter without an upper bound",
                        FindingSeverity.HIGH,
                        f"`{fn.name}` changes buy/sell fees without a `require(fee <= max)` check, so fees can be "
                        "raised to 100% at any time (classic honeypot exit).",
                        "Enforce a hard maximum fee in the setter or renounce ownership.",
                        lines=[parsed.line_of(offset)],
                        snippet=parsed.snippet_at(offset),
                        function=fn.name,
                        contract=fn.contract,
                        category="honeypot",
                    )
                )
            else:
                cap = int(bound.group(2) or bound.group(4) or 0)
                if FEE_CAP_WARN_PCT <= cap <= 100:
                    out.append(
                        self._finding(
                            "fee-setter-high-cap",
                            None,
                            f"Fee setter allows up to {cap}%",
                            FindingSeverity.MEDIUM,
                            f"`{fn.name}` bounds the fee at {cap}, which still permits punitive taxation.",
                            "Lower the fee ceiling to a reasonable level (<= 10%).",
                            lines=[parsed.line_of(offset)],
                            snippet=parsed.snippet_at(offset),
                            function=fn.name,
                            contract=fn.contract,
                            category="honeypot",
                        )
                    )
        denominator = self._fee_denominator(parsed.text)
        for contract in parsed.contracts:
            for var in contract.state_vars:
                if not _FEE_VAR_NAME_RE.search(var.name) or _FEE_EXCLUDE_NAME_RE.search(var.name):
                    continue
                if var.value is None or not re.fullmatch(r"\d+", var.value.strip()):
                    continue
                pct = int(var.value.strip()) * 100 / denominator
                if pct <= FEE_HIGH_PCT:
                    continue
                severity = FindingSeverity.CRITICAL if pct >= FEE_CRITICAL_PCT else FindingSeverity.HIGH
                out.append(
                    self._finding(
                        "fee-high",
                        None,
                        f"Fee of {pct:g}% configured",
                        severity,
                        f"`{var.name}` is initialised to {var.value.strip()} (~{pct:g}% with denominator "
                        f"{denominator}). Fees above {FEE_HIGH_PCT:g}% make selling uneconomic.",
                        "Avoid tokens with double-digit buy/sell taxes.",
                        lines=[var.line],
                        snippet=parsed.snippet_at(self._offset_of_line(parsed, var.line)),
                        contract=contract.name,
                        category="honeypot",
                    )
                )
        return out

    @staticmethod
    def _fee_denominator(text: str) -> int:
        explicit = re.search(r"\b\w*(?:denominator|divisor)\w*\s*=\s*(\d+)", text, re.IGNORECASE)
        if explicit:
            try:
                value = int(explicit.group(1))
                if value in (100, 1000, 10000, 100000):
                    return value
            except ValueError:  # pragma: no cover
                pass
        if re.search(r"/\s*10000\b|\b10000\b", text):
            return 10000
        if re.search(r"/\s*1000\b|\b1000\b", text):
            return 1000
        return 100

    @staticmethod
    def _offset_of_line(parsed: ParsedSource, line: int) -> int:
        offset = 0
        for _ in range(line - 1):
            nxt = parsed.text.find("\n", offset)
            if nxt == -1:
                break
            offset = nxt + 1
        return offset

    def _pair_restriction(self, parsed: ParsedSource) -> list[Finding]:
        out: list[Finding] = []
        for fn, body in self._function_findings(parsed):
            in_transfer = fn.name in _TRANSFER_FUNCS
            for match in _CONDITION_RE.finditer(body):
                close = _match_bracket(body, match.end() - 1, _PAREN_RE, "(")
                if close == -1:
                    continue
                condition = body[match.end() : close]
                if not _PAIR_IDENT_RE.search(condition) or not _COMPARE_RE.search(condition):
                    continue
                offset = fn.body_offset + match.start()
                out.append(
                    self._finding(
                        "pair-restriction",
                        None,
                        "Transfer rules target the DEX pair",
                        FindingSeverity.HIGH if in_transfer else FindingSeverity.MEDIUM,
                        f"`{fn.name}` applies conditions when the counterparty is the liquidity pair, i.e. buys "
                        "and sells are treated differently from plain transfers - the shape of sell-blocking honeypots.",
                        "Inspect the condition: legitimate uses are fee routing; blocking or taxing sells only is not.",
                        lines=[parsed.line_of(offset)],
                        snippet=parsed.snippet_at(offset),
                        function=fn.name,
                        contract=fn.contract,
                        category="honeypot",
                    )
                )
        return out

    def _pausable(self, parsed: ParsedSource) -> list[Finding]:
        lines: list[int] = []
        snippet = None
        contract_name = None
        for fn, body in self._function_findings(parsed):
            if fn.name not in _TRANSFER_FUNCS:
                continue
            if any(m == "whenNotPaused" for m in fn.modifiers) or _PAUSE_RE.search(body):
                lines.append(fn.start_line)
                snippet = snippet or parsed.snippet_at(fn.start_offset)
                contract_name = contract_name or fn.contract
        if not lines and _PAUSE_FUNC_RE.search(parsed.text) and _PAUSE_RE.search(parsed.text):
            match = _PAUSE_FUNC_RE.search(parsed.text)
            if match:
                lines.append(parsed.line_of(match.start()))
                snippet = parsed.snippet_at(match.start())
                found = parsed.contract_at(match.start())
                contract_name = found.name if found else None
        if not lines:
            return []
        return [
            self._finding(
                "pausable-transfers",
                None,
                "Transfers can be paused",
                FindingSeverity.MEDIUM,
                "A privileged account can pause token transfers, freezing every holder.",
                "Confirm who holds the pauser role and whether it is time-locked.",
                lines=lines,
                snippet=snippet,
                contract=contract_name,
                category="centralisation",
            )
        ]

    def _proxy(self, parsed: ParsedSource) -> list[Finding]:
        return self._pattern_finding(
            parsed,
            _PROXY_RE,
            fid="upgradeable",
            title="Upgradeable / proxy contract",
            severity=FindingSeverity.MEDIUM,
            description="The contract logic can be replaced after deployment, so this scan only reflects the current "
            "implementation",
            recommendation="Check the proxy admin and whether upgrades are time-locked or governed by a multi-sig.",
            category="centralisation",
        )

    def _hidden_owner(self, parsed: ParsedSource) -> list[Finding]:
        out: list[Finding] = []
        for fn, body in self._function_findings(parsed):
            if fn.name != "renounceOwnership" or fn.kind != "function":
                continue
            if _RENOUNCE_REAL_RE.search(body):
                continue
            out.append(
                self._finding(
                    "hidden-owner",
                    None,
                    "renounceOwnership does not renounce ownership",
                    FindingSeverity.HIGH,
                    "`renounceOwnership` is overridden without clearing the owner, so the project can claim ownership "
                    "was renounced while keeping full control.",
                    "Verify on-chain that `owner()` returns the zero address.",
                    lines=[fn.start_line],
                    snippet=parsed.snippet_at(fn.start_offset),
                    function=fn.name,
                    contract=fn.contract,
                    category="honeypot",
                )
            )
        if _PREVIOUS_OWNER_RE.search(parsed.text) and _UNLOCK_RE.search(parsed.text):
            match = _UNLOCK_RE.search(parsed.text)
            assert match is not None
            found = parsed.contract_at(match.start())
            out.append(
                self._finding(
                    "hidden-owner",
                    None,
                    "Ownership can be reclaimed after renouncing",
                    FindingSeverity.HIGH,
                    "The `_previousOwner` + `lock()/unlock()` pattern stores the owner before 'renouncing' so it can "
                    "be restored later.",
                    "Treat ownership as not renounced.",
                    lines=[parsed.line_of(match.start())],
                    snippet=parsed.snippet_at(match.start()),
                    contract=found.name if found else None,
                    category="honeypot",
                )
            )
        return out

    def _approve_all(self, parsed: ParsedSource) -> list[Finding]:
        out: list[Finding] = []
        for fn, body in self._function_findings(parsed):
            if fn.name in _ALLOWANCE_FUNCS or fn.kind == "modifier":
                continue
            hits: list[int] = []
            for match in _APPROVE_CALL_RE.finditer(body):
                holder, spender = match.group(1).strip(), match.group(2).strip()
                if _SELF_LIKE_RE.match(holder):
                    continue
                if _OWNER_LIKE_RE.search(spender):
                    hits.append(match.start())
            for match in _ALLOWANCE_WRITE_RE.finditer(body):
                holder, spender = match.group(1).strip(), match.group(2).strip()
                if not _SELF_LIKE_RE.match(holder) and _OWNER_LIKE_RE.search(spender):
                    hits.append(match.start())
            if not hits:
                continue
            offset = fn.body_offset + hits[0]
            out.append(
                self._finding(
                    "approve-all",
                    None,
                    "Holders' tokens approved to a privileged account",
                    FindingSeverity.CRITICAL,
                    f"`{fn.name}` grants a privileged address an allowance over other holders' balances, allowing "
                    "their tokens to be pulled without consent.",
                    "Never interact with the token; this is a theft primitive.",
                    lines=[parsed.line_of(fn.body_offset + h) for h in hits],
                    snippet=parsed.snippet_at(offset),
                    function=fn.name,
                    contract=fn.contract,
                    category="honeypot",
                )
            )
        return out

    def _composite(self, findings: list[Finding]) -> list[Finding]:
        present = {f.id for f in findings if f.id in HONEYPOT_INDICATOR_IDS}
        if len(present) < HONEYPOT_COMPOSITE_MIN:
            return []
        lines = sorted({line for f in findings if f.id in present for line in f.lines})[:12]
        return [
            self._finding(
                "honeypot-composite",
                None,
                f"Multiple honeypot indicators ({len(present)})",
                FindingSeverity.CRITICAL,
                "The contract combines several deployer-controlled sell restrictions: "
                + ", ".join(sorted(present))
                + ".",
                "Do not buy: tokens with this combination are overwhelmingly honeypots or planned rug pulls.",
                lines=lines,
                category="honeypot",
            )
        ]


# ------------------------------------------------------------------- bytecode
def selector(signature: str) -> str:
    return keccak256(signature.encode()).hex()[:8]


SELECTOR_SIGNATURES: dict[str, tuple[str, FindingSeverity, str]] = {
    "blacklist(address)": ("blacklist", FindingSeverity.MEDIUM, "Blacklist function exposed"),
    "blacklist(address,bool)": ("blacklist", FindingSeverity.MEDIUM, "Blacklist function exposed"),
    "setBlacklist(address,bool)": ("blacklist", FindingSeverity.MEDIUM, "Blacklist function exposed"),
    "addBot(address)": ("blacklist", FindingSeverity.MEDIUM, "Anti-bot blacklist exposed"),
    "isBot(address)": ("blacklist", FindingSeverity.LOW, "Bot flagging exposed"),
    "setMaxTxAmount(uint256)": ("tx-limits", FindingSeverity.LOW, "Adjustable max transaction"),
    "setMaxWallet(uint256)": ("tx-limits", FindingSeverity.LOW, "Adjustable max wallet"),
    "removeLimits()": ("tx-limits", FindingSeverity.LOW, "Adjustable limits"),
    "setFee(uint256)": ("fee-setter", FindingSeverity.MEDIUM, "Fee setter exposed"),
    "setFees(uint256,uint256)": ("fee-setter", FindingSeverity.MEDIUM, "Fee setter exposed"),
    "setTaxes(uint256,uint256)": ("fee-setter", FindingSeverity.MEDIUM, "Fee setter exposed"),
    "setSellFee(uint256)": ("fee-setter", FindingSeverity.MEDIUM, "Sell fee setter exposed"),
    "setBuyFee(uint256)": ("fee-setter", FindingSeverity.MEDIUM, "Buy fee setter exposed"),
    "pause()": ("pausable", FindingSeverity.LOW, "Pause function exposed"),
    "unpause()": ("pausable", FindingSeverity.LOW, "Unpause function exposed"),
    "mint(address,uint256)": ("mint", FindingSeverity.MEDIUM, "Mint function exposed"),
    "enableTrading()": ("trading-toggle", FindingSeverity.MEDIUM, "Trading toggle exposed"),
    "openTrading()": ("trading-toggle", FindingSeverity.MEDIUM, "Trading toggle exposed"),
    "setTradingEnabled(bool)": ("trading-toggle", FindingSeverity.MEDIUM, "Trading toggle exposed"),
    "upgradeTo(address)": ("upgradeable", FindingSeverity.MEDIUM, "Upgradeable (UUPS) contract"),
    "upgradeToAndCall(address,bytes)": ("upgradeable", FindingSeverity.MEDIUM, "Upgradeable (UUPS) contract"),
    "renounceOwnership()": ("ownable", FindingSeverity.INFO, "Ownable contract"),
    "transferOwnership(address)": ("ownable", FindingSeverity.INFO, "Ownable contract"),
}
SELECTORS: dict[str, str] = {sig: selector(sig) for sig in SELECTOR_SIGNATURES}

OPCODE_FINDINGS: dict[int, tuple[str, FindingSeverity, str, str, str]] = {
    0xFF: (
        "op-selfdestruct",
        FindingSeverity.MEDIUM,
        "SWC-106",
        "SELFDESTRUCT opcode present",
        "The contract contains SELFDESTRUCT; it can be destroyed and its balance moved.",
    ),
    0xF4: (
        "op-delegatecall",
        FindingSeverity.MEDIUM,
        "SWC-112",
        "DELEGATECALL opcode present",
        "The contract delegates execution to other code (proxy pattern or library call).",
    ),
    0xF2: (
        "op-callcode",
        FindingSeverity.LOW,
        "SWC-112",
        "CALLCODE opcode present",
        "Deprecated CALLCODE is used; behaviour resembles delegatecall with msg.sender semantics.",
    ),
}


def disassemble(code: bytes) -> list[tuple[int, int, bytes]]:
    """Return ``(offset, opcode, push_data)`` triples; PUSH immediates are skipped as data."""
    ops: list[tuple[int, int, bytes]] = []
    i = 0
    n = len(code)
    while i < n:
        op = code[i]
        if 0x60 <= op <= 0x7F:
            size = op - 0x5F
            ops.append((i, op, code[i + 1 : i + 1 + size]))
            i += 1 + size
        else:
            ops.append((i, op, b""))
            i += 1
    return ops


def strip_metadata(code: bytes) -> bytes:
    """Drop the trailing CBOR metadata (``a264697066735822...0033``) appended by solc."""
    if len(code) < 4:
        return code
    length = int.from_bytes(code[-2:], "big")
    if 0 < length + 2 <= len(code):
        blob = code[-(length + 2) : -2]
        if blob[:1] in (b"\xa1", b"\xa2", b"\xa3") and (b"ipfs" in blob or b"bzzr" in blob or b"solc" in blob):
            return code[: -(length + 2)]
    return code


class BytecodeScanner:
    """Opcode and selector heuristics for unverified contracts. ``scan`` never raises."""

    def scan(self, code_hex: str) -> ScanResult:
        try:
            raw = (code_hex or "").strip()
            raw = raw[2:] if raw.lower().startswith("0x") else raw
            if not raw:
                return ScanResult(
                    findings=[
                        Finding(
                            id="empty-bytecode",
                            title="No bytecode",
                            severity=FindingSeverity.INFO,
                            category="info",
                            description="The address holds no code (externally owned account or destroyed contract).",
                            recommendation="Verify the address; it is not a contract.",
                        )
                    ],
                    risk_score=0,
                    risk_label=RiskLabel.SAFE,
                    stats={"bytes": 0, "opcodes": 0, "selectors": []},
                    kind="bytecode",
                )
            if len(raw) % 2:
                raw = raw[:-1]
            code = bytes.fromhex(raw)
        except ValueError:
            return ScanResult(
                findings=[
                    Finding(
                        id="unparseable",
                        title="Bytecode is not valid hex",
                        severity=FindingSeverity.INFO,
                        category="info",
                        description="The input could not be decoded as EVM bytecode.",
                        recommendation="Provide the `eth_getCode` result of the contract.",
                    )
                ],
                risk_score=0,
                risk_label=RiskLabel.SAFE,
                stats={"bytes": 0, "opcodes": 0, "selectors": []},
                kind="bytecode",
            )
        findings: list[Finding] = []
        implementation: str | None = None
        proxy = _MINIMAL_PROXY_RE.match(raw)
        if proxy:
            implementation = "0x" + proxy.group(1).lower()
            findings.append(
                Finding(
                    id="minimal-proxy",
                    title="EIP-1167 minimal proxy",
                    severity=FindingSeverity.INFO,
                    category="info",
                    description=f"This is a clone forwarding every call to {implementation}; scan that address instead.",
                    recommendation="Analyse the implementation contract.",
                )
            )
        body = strip_metadata(code)
        ops = disassemble(body)
        present = {op for _, op, _ in ops}
        for opcode, (fid, severity, swc, title, description) in OPCODE_FINDINGS.items():
            if opcode in present and not (proxy and opcode == 0xF4):
                offsets = [offset for offset, op, _ in ops if op == opcode][:5]
                findings.append(
                    Finding(
                        id=fid,
                        swc=swc,
                        title=title,
                        severity=severity,
                        description=description + f" (offset{'s' if len(offsets) > 1 else ''} {offsets}).",
                        recommendation="Obtain and review the verified source; bytecode analysis cannot show the guard conditions.",
                    )
                )
        pushed = {data.hex() for _, op, data in ops if op == 0x63 and len(data) == 4}
        matched: list[str] = []
        for signature, sel in SELECTORS.items():
            if sel not in pushed:
                continue
            matched.append(signature)
            fid, severity, title = SELECTOR_SIGNATURES[signature]
            findings.append(
                Finding(
                    id=f"selector-{fid}",
                    title=title,
                    severity=severity,
                    category="honeypot"
                    if fid in ("blacklist", "trading-toggle", "fee-setter", "tx-limits")
                    else "centralisation",
                    description=f"Function selector `0x{sel}` matches `{signature}`.",
                    recommendation="Selector matches are indicative only (bytecode is lossy); confirm with verified source.",
                )
            )
        findings = dedupe_findings(findings)
        score = score_findings(findings)
        return ScanResult(
            findings=findings,
            risk_score=score,
            risk_label=risk_label(score),
            stats={
                "bytes": len(code),
                "opcodes": len(ops),
                "selectors": sorted(matched),
                "metadata_stripped": len(body) != len(code),
            },
            kind="bytecode",
            is_proxy=bool(proxy) or 0xF4 in present,
            implementation=implementation,
        )


__all__ = [
    "HONEYPOT_INDICATOR_IDS",
    "MAX_SOURCE_CHARS",
    "OPCODE_FINDINGS",
    "RISK_LABEL_THRESHOLDS",
    "SELECTORS",
    "SELECTOR_SIGNATURES",
    "SEVERITY_WEIGHTS",
    "BytecodeScanner",
    "ContractInfo",
    "FunctionInfo",
    "ParsedSource",
    "ScanResult",
    "SolidityScanner",
    "StateVar",
    "clean_source",
    "dedupe_findings",
    "disassemble",
    "is_floating_pragma",
    "parse_pragma_version",
    "parse_solidity",
    "risk_label",
    "score_findings",
    "selector",
    "strip_metadata",
]
