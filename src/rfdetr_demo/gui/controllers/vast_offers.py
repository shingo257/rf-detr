# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Vast.ai GPU offer search helpers for the GUI controller."""

from __future__ import annotations

from dataclasses import dataclass

from rfdetr_demo.vast.offers import search_gpu_offers
from rfdetr_demo.vast.types import VastGpuOffer


@dataclass(frozen=True)
class OfferSearchUiOutcome:
    """GPU offer search result formatted for GUI application."""

    labels: list[str]
    default_label: str
    log_lines: list[tuple[str, str]]
    show_empty_info_dialog: bool


def find_offer(offers: list[VastGpuOffer], label: str) -> VastGpuOffer | None:
    """Return the offer whose label matches ``label``, if any."""
    stripped = label.strip()
    if not stripped:
        return None
    for offer in offers:
        if offer.label == stripped:
            return offer
    return None


def search_offers(
    *,
    api_key: str | None,
    max_dph: float,
    gpu_name: str,
) -> list[VastGpuOffer]:
    """Search Vast.ai GPU offers for the GUI."""
    return search_gpu_offers(
        api_key=api_key,
        max_dph=max_dph,
        gpu_name=gpu_name,
    )


def build_offer_search_ui(offers: list[VastGpuOffer]) -> OfferSearchUiOutcome:
    """Format offer search results for Tk combo / log application."""
    labels = [offer.label for offer in offers]
    if labels:
        return OfferSearchUiOutcome(
            labels=labels,
            default_label=labels[0],
            log_lines=[("info", f"Vast.ai: {len(labels)} 件の GPU オファーを取得")],
            show_empty_info_dialog=False,
        )
    return OfferSearchUiOutcome(
        labels=[],
        default_label="",
        log_lines=[("info", "Vast.ai: 条件に合う GPU が見つかりませんでした")],
        show_empty_info_dialog=True,
    )


__all__ = [
    "OfferSearchUiOutcome",
    "build_offer_search_ui",
    "find_offer",
    "search_offers",
]
