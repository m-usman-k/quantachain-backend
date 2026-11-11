"""Module 6 - Risk Assessment & Fraud Detection endpoints."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query, status

from app.api.deps import DB, AdminUser, CurrentUser, OptionalUser, SettingsDep
from app.core.pagination import Page, Pagination
from app.modules.fraud.models import AlertSeverity, AlertType, BlacklistStatus
from app.modules.fraud.schemas import (
    BlacklistIn,
    BlacklistOut,
    FraudAlertAck,
    FraudAlertOut,
    FraudStatsOut,
    ProjectScoreOut,
    ScanIn,
    ScanOut,
    WatchedPairIn,
    WatchedPairOut,
)
from app.modules.fraud.service import FraudService

router = APIRouter(prefix="/fraud", tags=["fraud"])


def get_fraud_service(db: DB, settings: SettingsDep) -> FraudService:
    return FraudService(db, settings)


Service = Annotated[FraudService, Depends(get_fraud_service)]


@router.get("/stats", response_model=FraudStatsOut, summary="Fraud & risk summary statistics")
async def stats(service: Service) -> FraudStatsOut:
    data = await service.stats()
    return FraudStatsOut(**data)


# ----------------------------------------------------------- smart contract scan
@router.post(
    "/scan",
    response_model=ScanOut,
    status_code=status.HTTP_201_CREATED,
    summary="Scan smart contract for security vulnerabilities",
)
async def scan_contract(payload: ScanIn, user: OptionalUser, service: Service) -> ScanOut:
    user_id = user.id if user and user.id else "anonymous"
    scan = await service.scan_contract(user_id, payload)
    return ScanOut(**scan.model_dump())


@router.get("/scans/{scan_id}", response_model=ScanOut, summary="Retrieve scan results by scan ID")
async def get_scan(scan_id: str, service: Service) -> ScanOut:
    scan = await service.get_scan(scan_id)
    return ScanOut(**scan.model_dump())


@router.get(
    "/scans/address/{chain}/{address}", response_model=ScanOut | None, summary="Get latest scan for a contract address"
)
async def get_scan_by_address(chain: str, address: str, service: Service) -> ScanOut | None:
    scan = await service.latest_scan_for_address(chain, address)
    if not scan:
        return None
    return ScanOut(**scan.model_dump())


# ----------------------------------------------------------- project scores
@router.get(
    "/project-scores/{chain}/{address}",
    response_model=ProjectScoreOut,
    summary="Get or calculate Transparency & Liquidity Score",
)
async def get_project_score(chain: str, address: str, service: Service) -> ProjectScoreOut:
    score = await service.get_project_score(chain, address)
    return ProjectScoreOut(**score.model_dump())


# ----------------------------------------------------------- alerts
@router.get("/alerts", response_model=Page[FraudAlertOut], summary="List fraud & threat alerts")
async def list_alerts(
    service: Service,
    pagination: Pagination,
    type: Annotated[AlertType | None, Query()] = None,
    severity: Annotated[AlertSeverity | None, Query()] = None,
    acknowledged: Annotated[bool | None, Query()] = None,
) -> Page[FraudAlertOut]:
    items, total = await service.list_alerts(
        type=type,
        severity=severity,
        acknowledged=acknowledged,
        limit=pagination.limit,
        offset=pagination.offset,
    )
    return Page.build([FraudAlertOut(**item.model_dump()) for item in items], total, pagination)


@router.post("/alerts/{alert_id}/acknowledge", response_model=FraudAlertOut, summary="Acknowledge a fraud alert")
async def acknowledge_alert(
    alert_id: str,
    user: CurrentUser,
    service: Service,
    payload: FraudAlertAck | None = None,
) -> FraudAlertOut:
    note = payload.note if payload else None
    updated = await service.acknowledge_alert(alert_id, user.email, note=note)
    return FraudAlertOut(**updated.model_dump())


# ----------------------------------------------------------- watched pairs
@router.get("/watched-pairs", response_model=list[WatchedPairOut], summary="List monitored DEX pairs")
async def list_watched_pairs(service: Service) -> list[WatchedPairOut]:
    pairs = await service.list_watched_pairs(active_only=False)
    return [WatchedPairOut(**p.model_dump()) for p in pairs]


@router.post(
    "/watched-pairs",
    response_model=WatchedPairOut,
    status_code=status.HTTP_201_CREATED,
    summary="Add DEX pair to fraud monitoring",
)
async def add_watched_pair(payload: WatchedPairIn, user: CurrentUser, service: Service) -> WatchedPairOut:
    pair = await service.add_watched_pair(payload, added_by=user.email)
    return WatchedPairOut(**pair.model_dump())


# ----------------------------------------------------------- AML blacklist
@router.get("/blacklist", response_model=list[BlacklistOut], summary="List AML blacklisted addresses")
async def list_blacklist(
    service: Service,
    status: Annotated[BlacklistStatus | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
) -> list[BlacklistOut]:
    entries = await service.list_blacklist(status=status, limit=limit)
    return [BlacklistOut(**e.model_dump()) for e in entries]


@router.post(
    "/blacklist",
    response_model=BlacklistOut,
    status_code=status.HTTP_201_CREATED,
    summary="Add address to AML blacklist (admin)",
)
async def add_blacklist(payload: BlacklistIn, admin: AdminUser, service: Service) -> BlacklistOut:
    entry = await service.add_to_blacklist(payload, actor=admin.email)
    return BlacklistOut(**entry.model_dump())


@router.get(
    "/blacklist/check/{address}", response_model=BlacklistOut | None, summary="Check if an address is blacklisted"
)
async def check_blacklist(address: str, service: Service) -> BlacklistOut | None:
    entry = await service.check_blacklist(address)
    if not entry:
        return None
    return BlacklistOut(**entry.model_dump())
