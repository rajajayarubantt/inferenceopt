# SPDX-License-Identifier: Apache-2.0
"""Versioned API routers, all mounted under ``/api/v1``."""

from fastapi import APIRouter

from inferenceopt.server.routes import auth, inventory, jobs, system

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(system.router)
api_router.include_router(auth.router)
api_router.include_router(jobs.router)
api_router.include_router(inventory.router)

__all__ = ["api_router"]
