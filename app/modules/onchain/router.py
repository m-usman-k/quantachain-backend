"""Module 3 - On-Chain Intelligence Engine endpoints."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Header, Query, Request

from app.api.deps import DB, SettingsDep
from app.core.pagination import Page, Pagination
from app.modules.onchain.models import TransferFlow, TransferRisk, Wallet
from app.modules.onchain.schemas import (
    ClusterGraphOut,
    InvestigationIn,
    InvestigationOut,
    NetflowOut,
    WalletProfileOut,
    WhaleTransferOut,
)
from app.modules.onchain.service import OnchainService

router = APIRouter(prefix="/onchain", tags=["onchain"])


def get_onchain_service(db: DB, settings: SettingsDep) -> OnchainService:
    return OnchainService(db, settings)


Service = Annotated[OnchainService, Depends(get_onchain_service)]


# ----------------------------------------------------------- whale transfers
@router.get("/transfers", response_model=Page[WhaleTransferOut], summary="List on-chain whale transfers")
async def list_transfers(
    service: Service,
    pagination: Pagination,
    symbol: Annotated[str | None, Query()] = None,
    min_amount_usd: Annotated[float | None, Query()] = None,
    flow: Annotated[TransferFlow | None, Query()] = None,
    risk_level: Annotated[TransferRisk | None, Query()] = None,
    address: Annotated[str | None, Query()] = None,
) -> Page[WhaleTransferOut]:
    items, total = await service.list_transfers(
        symbol=symbol,
        min_amount_usd=min_amount_usd,
        flow=flow,
        risk_level=risk_level,
        address=address,
        limit=pagination.limit,
        offset=pagination.offset,
    )
    return Page.build([WhaleTransferOut(**t.model_dump()) for t in items], total, pagination)


@router.get("/transfers/{tx_hash}", response_model=list[WhaleTransferOut], summary="Get transfers by transaction hash")
async def get_transfer_by_tx(tx_hash: str, service: Service) -> list[WhaleTransferOut]:
    transfers = await service.get_transfers_by_tx(tx_hash)
    return [WhaleTransferOut(**t.model_dump()) for t in transfers]


# ----------------------------------------------------------- wallets & smart money
@router.get("/wallets/{address}", response_model=WalletProfileOut, summary="Get wallet profile and smart money score")
async def get_wallet(address: str, service: Service) -> WalletProfileOut:
    return await service.get_wallet_profile(address)


@router.get(
    "/wallets/{address}/graph",
    response_model=ClusterGraphOut,
    summary="Generate D3 visual cluster network graph for a wallet",
)
async def get_wallet_graph(address: str, service: Service) -> ClusterGraphOut:
    return await service.build_cluster_graph(address)


@router.get("/clusters", response_model=list[Wallet], summary="List top smart-money wallets and whale clusters")
async def top_smart_money(
    service: Service,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
) -> list[Wallet]:
    return await service.top_smart_money(limit=limit)


# ----------------------------------------------------------- exchange netflows
@router.get(
    "/netflow/{symbol}", response_model=NetflowOut, summary="Exchange netflow summary and trend (inflow vs outflow)"
)
async def get_netflow(
    symbol: str,
    service: Service,
    window_hours: Annotated[int, Query(ge=1, le=168)] = 24,
) -> NetflowOut:
    return await service.get_netflow(symbol, window_hours=window_hours)


# ----------------------------------------------------------- investigation
@router.post(
    "/investigate", response_model=InvestigationOut, summary="Investigate transaction or wallet address relationships"
)
async def investigate(payload: InvestigationIn, service: Service) -> InvestigationOut:
    return await service.investigate(payload)


# ----------------------------------------------------------- Moralis Webhooks
@router.post("/webhooks/moralis", summary="Moralis Streams webhook for live on-chain transfers")
async def moralis_webhook(
    request: Request,
    service: Service,
    x_signature: Annotated[str | None, Header()] = None,
) -> dict:
    body = await request.body()
    return await service.handle_moralis_webhook(body, x_signature)
