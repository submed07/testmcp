"""Azure Data Factory MCP Server.

Exposes ADF pipeline analysis as MCP tools via StreamableHTTP transport.

Run standalone:
    uv run uvicorn adf_tools:app --reload --port 8001

Or integrate into api.py by importing `mcp` and mounting its sub-app.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI
from mcp.server.fastmcp import FastMCP
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

MAX_WORKERS = 10

# Azure SDKs — optional; server starts without them (tools return an error message)
try:
    from azure.identity import DefaultAzureCredential
    from azure.mgmt.datafactory import DataFactoryManagementClient
    from azure.mgmt.datafactory.models import RunFilterParameters
except ImportError:
    DefaultAzureCredential = None  # type: ignore
    DataFactoryManagementClient = None  # type: ignore
    RunFilterParameters = None  # type: ignore


# =============================================================================
# Settings
# =============================================================================


class ADFSettings(BaseSettings):
    """Azure Data Factory configuration settings."""

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False, extra="ignore")

    subscription_id: Optional[str] = Field(default=None, validation_alias="AZURE_SUBSCRIPTION_ID")
    resource_group: Optional[str] = Field(default=None, validation_alias="AZURE_RESOURCE_GROUP")
    factory_name: Optional[str] = Field(default=None, validation_alias="AZURE_DATA_FACTORY_NAME")


# =============================================================================
# Sync ADF Client (used internally by the async client for SDK compatibility)
# =============================================================================


class ADFClient:
    """Azure Data Factory client wrapper."""

    def __init__(self, settings: Optional[ADFSettings] = None) -> None:
        self.settings = settings or ADFSettings()
        self._client = None

        if not all(
            [
                self.settings.subscription_id,
                self.settings.resource_group,
                self.settings.factory_name,
                DefaultAzureCredential,
                DataFactoryManagementClient,
            ]
        ):
            logger.warning("[ADF] Client not initialized (missing config or SDK)")
            return

        try:
            credential = DefaultAzureCredential()
            self._client = DataFactoryManagementClient(credential, self.settings.subscription_id)
            logger.info("[ADF] Client initialized successfully")
        except Exception as e:
            logger.error(f"[ADF] Failed to initialize client: {e}")
            self._client = None

    @property
    def is_available(self) -> bool:
        return self._client is not None

    def get_pipeline_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        if not self.is_available:
            return None
        try:
            run = self._client.pipeline_runs.get(
                resource_group_name=self.settings.resource_group,
                factory_name=self.settings.factory_name,
                run_id=run_id,
            )
            return self._serialize_pipeline_run(run)
        except Exception as e:
            logger.error(f"[ADF] Failed to get pipeline run {run_id}: {e}")
            return None

    def get_activity_runs(self, run_id: str) -> List[Dict[str, Any]]:
        if not self.is_available or not RunFilterParameters:
            return []
        try:
            filter_params = RunFilterParameters(
                last_updated_after=datetime.now(timezone.utc) - timedelta(days=30),
                last_updated_before=datetime.now(timezone.utc) + timedelta(days=1),
            )
            result = self._client.activity_runs.query_by_pipeline_run(
                resource_group_name=self.settings.resource_group,
                factory_name=self.settings.factory_name,
                run_id=run_id,
                filter_parameters=filter_params,
            )
            return [self._serialize_activity_run(ar) for ar in result.value]
        except Exception as e:
            logger.error(f"[ADF] Failed to get activity runs for {run_id}: {e}")
            return []

    def get_pipeline_hierarchy(self, run_id: str) -> Dict[str, Any]:
        total_start = time.time()
        logger.info(f"[ADF] Starting hierarchy retrieval for {run_id}")

        if not self.is_available:
            return {"error": "ADF client not available", "run_id": run_id}

        pipeline_run = self.get_pipeline_run(run_id)
        if not pipeline_run:
            return {"error": f"Pipeline run not found: {run_id}", "run_id": run_id}

        root_run = pipeline_run
        visited = {run_id}
        parent_depth = 0

        while root_run.get("invokedBy", {}).get("invokedByType") == "PipelineActivity":
            parent_run_id = root_run.get("invokedBy", {}).get("pipelineRunId")
            if not parent_run_id or parent_run_id in visited:
                break
            visited.add(parent_run_id)
            parent_run = self.get_pipeline_run(parent_run_id)
            if parent_run:
                root_run = parent_run
                parent_depth += 1
            else:
                break

        logger.debug(f"[ADF] Found root after traversing {parent_depth} parents")

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            hierarchy = self._build_hierarchy(root_run, executor)

        summary = self._calculate_hierarchy_summary(hierarchy)
        hierarchy["_metadata"] = {
            "requested_run_id": run_id,
            "root_run_id": root_run.get("runId"),
            "is_root": run_id == root_run.get("runId"),
        }
        hierarchy["_summary"] = summary

        logger.info(
            f"[ADF] Hierarchy retrieved in {time.time() - total_start:.2f}s "
            f"({summary['total_pipelines']} pipelines)"
        )
        return hierarchy

    def _calculate_hierarchy_summary(self, hierarchy: Dict[str, Any]) -> Dict[str, Any]:
        total = succeeded = failed = in_progress = cancelled = 0
        pipelines_list: List[Dict[str, Any]] = []

        def traverse(node: Dict[str, Any], depth: int = 0) -> None:
            nonlocal total, succeeded, failed, in_progress, cancelled
            total += 1
            status = node.get("status", "").lower()
            pipelines_list.append({
                "pipelineName": node.get("pipelineName"),
                "runId": node.get("runId"),
                "status": node.get("status"),
                "depth": depth,
            })
            if status == "succeeded":
                succeeded += 1
            elif status == "failed":
                failed += 1
            elif status in ("inprogress", "in progress", "running", "queued"):
                in_progress += 1
            elif status == "cancelled":
                cancelled += 1
            for child in node.get("childPipelines", []):
                traverse(child, depth + 1)

        traverse(hierarchy)
        return {
            "total_pipelines": total,
            "succeeded": succeeded,
            "failed": failed,
            "in_progress": in_progress,
            "cancelled": cancelled,
            "pipelines": pipelines_list,
        }

    def _build_hierarchy(
        self, pipeline_run: Dict[str, Any], executor: Optional[ThreadPoolExecutor] = None
    ) -> Dict[str, Any]:
        run_id = pipeline_run.get("runId")
        pipeline_name = pipeline_run.get("pipelineName")

        hierarchy = {
            "runId": run_id,
            "pipelineName": pipeline_name,
            "status": pipeline_run.get("status"),
            "runStart": pipeline_run.get("runStart"),
            "runEnd": pipeline_run.get("runEnd"),
            "durationInMs": pipeline_run.get("durationInMs"),
            "message": pipeline_run.get("message"),
            "invokedBy": pipeline_run.get("invokedBy"),
            "parameters": pipeline_run.get("parameters"),
            "activities": [],
            "childPipelines": [],
        }

        activity_runs = self.get_activity_runs(run_id)
        child_run_ids = []

        for activity in activity_runs:
            child_run_id = None
            if activity.get("activityType") == "ExecutePipeline":
                output = activity.get("output", {})
                child_run_id = output.get("pipelineRunId") if isinstance(output, dict) else None
                if child_run_id:
                    child_run_ids.append(child_run_id)
            hierarchy["activities"].append({
                "activityName": activity.get("activityName"),
                "activityType": activity.get("activityType"),
                "status": activity.get("status"),
                "activityRunStart": activity.get("activityRunStart"),
                "activityRunEnd": activity.get("activityRunEnd"),
                "durationInMs": activity.get("durationInMs"),
                "error": activity.get("error"),
                "childPipelineRunId": child_run_id,
            })

        if child_run_ids:
            owns_executor = executor is None
            if owns_executor:
                executor = ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(child_run_ids)))
            try:
                future_to_id = {
                    executor.submit(self.get_pipeline_run, cid): cid for cid in child_run_ids
                }
                child_runs = {}
                for future in as_completed(future_to_id):
                    cid = future_to_id[future]
                    try:
                        child_runs[cid] = future.result()
                    except Exception as e:
                        logger.error(f"[ADF] Failed to fetch child {cid}: {e}")
                        child_runs[cid] = None

                hierarchy_futures = {
                    executor.submit(self._build_hierarchy, cr, executor): cid
                    for cid, cr in child_runs.items() if cr
                }
                for future in as_completed(hierarchy_futures):
                    try:
                        hierarchy["childPipelines"].append(future.result())
                    except Exception as e:
                        logger.error(f"[ADF] Failed to build child hierarchy: {e}")
            finally:
                if owns_executor:
                    executor.shutdown(wait=False)

        return hierarchy

    @staticmethod
    def _serialize_pipeline_run(run) -> Dict[str, Any]:
        return {
            "runId": run.run_id,
            "pipelineName": run.pipeline_name,
            "status": run.status,
            "runStart": run.run_start.isoformat() if run.run_start else None,
            "runEnd": run.run_end.isoformat() if run.run_end else None,
            "durationInMs": run.duration_in_ms,
            "message": run.message,
            "invokedBy": (
                {
                    "name": run.invoked_by.name if run.invoked_by else None,
                    "invokedByType": run.invoked_by.invoked_by_type if run.invoked_by else None,
                    "pipelineRunId": run.invoked_by.pipeline_run_id if run.invoked_by else None,
                }
                if run.invoked_by
                else {}
            ),
            "parameters": run.parameters or {},
        }

    @staticmethod
    def _serialize_activity_run(activity) -> Dict[str, Any]:
        return {
            "activityName": activity.activity_name,
            "activityType": activity.activity_type,
            "status": activity.status,
            "activityRunStart": (
                activity.activity_run_start.isoformat() if activity.activity_run_start else None
            ),
            "activityRunEnd": (
                activity.activity_run_end.isoformat() if activity.activity_run_end else None
            ),
            "durationInMs": activity.duration_in_ms,
            "error": activity.error,
            "output": activity.output,
        }


# =============================================================================
# Async ADF Client
# =============================================================================


class AsyncADFClient:
    """Async Azure Data Factory client — wraps the sync SDK with asyncio."""

    def __init__(self, settings: Optional[ADFSettings] = None) -> None:
        self.settings = settings or ADFSettings()
        self._client = None

        if not all(
            [
                self.settings.subscription_id,
                self.settings.resource_group,
                self.settings.factory_name,
                DefaultAzureCredential,
                DataFactoryManagementClient,
            ]
        ):
            logger.warning("[ADF Async] Client not initialized (missing config or SDK)")
            return

        try:
            self._client = DataFactoryManagementClient(
                DefaultAzureCredential(), self.settings.subscription_id
            )
            logger.info("[ADF Async] Client initialized successfully")
        except Exception as e:
            logger.error(f"[ADF Async] Failed to initialize client: {e}")

    @property
    def is_available(self) -> bool:
        return self._client is not None

    async def get_pipeline_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        if not self.is_available:
            return None
        try:
            loop = asyncio.get_event_loop()
            run = await loop.run_in_executor(
                None,
                lambda: self._client.pipeline_runs.get(
                    resource_group_name=self.settings.resource_group,
                    factory_name=self.settings.factory_name,
                    run_id=run_id,
                ),
            )
            return ADFClient._serialize_pipeline_run(run)
        except Exception as e:
            logger.error(f"[ADF Async] Failed to get pipeline run {run_id}: {e}")
            return None

    async def get_activity_runs(self, run_id: str) -> List[Dict[str, Any]]:
        if not self.is_available or not RunFilterParameters:
            return []
        try:
            filter_params = RunFilterParameters(
                last_updated_after=datetime.now(timezone.utc) - timedelta(days=30),
                last_updated_before=datetime.now(timezone.utc) + timedelta(days=1),
            )
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self._client.activity_runs.query_by_pipeline_run(
                    resource_group_name=self.settings.resource_group,
                    factory_name=self.settings.factory_name,
                    run_id=run_id,
                    filter_parameters=filter_params,
                ),
            )
            return [ADFClient._serialize_activity_run(ar) for ar in result.value]
        except Exception as e:
            logger.error(f"[ADF Async] Failed to get activity runs for {run_id}: {e}")
            return []

    async def get_pipeline_hierarchy(self, run_id: str) -> Dict[str, Any]:
        total_start = time.time()
        if not self.is_available:
            return {"error": "ADF client not available", "run_id": run_id}

        pipeline_run = await self.get_pipeline_run(run_id)
        if not pipeline_run:
            return {"error": f"Pipeline run not found: {run_id}", "run_id": run_id}

        root_run = pipeline_run
        visited = {run_id}
        parent_depth = 0

        while root_run.get("invokedBy", {}).get("invokedByType") == "PipelineActivity":
            parent_run_id = root_run.get("invokedBy", {}).get("pipelineRunId")
            if not parent_run_id or parent_run_id in visited:
                break
            visited.add(parent_run_id)
            parent_run = await self.get_pipeline_run(parent_run_id)
            if parent_run:
                root_run = parent_run
                parent_depth += 1
            else:
                break

        hierarchy = await self._build_hierarchy_async(root_run)
        summary = self._calculate_hierarchy_summary(hierarchy)
        hierarchy["_metadata"] = {
            "requested_run_id": run_id,
            "root_run_id": root_run.get("runId"),
            "is_root": run_id == root_run.get("runId"),
        }
        hierarchy["_summary"] = summary

        logger.info(
            f"[ADF Async] Hierarchy retrieved in {time.time() - total_start:.2f}s "
            f"({summary['total_pipelines']} pipelines)"
        )
        return hierarchy

    async def _build_hierarchy_async(self, pipeline_run: Dict[str, Any]) -> Dict[str, Any]:
        run_id = pipeline_run.get("runId")

        hierarchy = {
            "runId": run_id,
            "pipelineName": pipeline_run.get("pipelineName"),
            "status": pipeline_run.get("status"),
            "runStart": pipeline_run.get("runStart"),
            "runEnd": pipeline_run.get("runEnd"),
            "durationInMs": pipeline_run.get("durationInMs"),
            "message": pipeline_run.get("message"),
            "invokedBy": pipeline_run.get("invokedBy"),
            "parameters": pipeline_run.get("parameters"),
            "activities": [],
            "childPipelines": [],
        }

        activity_runs = await self.get_activity_runs(run_id)
        child_run_ids = []

        for activity in activity_runs:
            child_run_id = None
            if activity.get("activityType") == "ExecutePipeline":
                output = activity.get("output", {})
                child_run_id = output.get("pipelineRunId") if isinstance(output, dict) else None
                if child_run_id:
                    child_run_ids.append(child_run_id)
            hierarchy["activities"].append({
                "activityName": activity.get("activityName"),
                "activityType": activity.get("activityType"),
                "status": activity.get("status"),
                "activityRunStart": activity.get("activityRunStart"),
                "activityRunEnd": activity.get("activityRunEnd"),
                "durationInMs": activity.get("durationInMs"),
                "error": activity.get("error"),
                "childPipelineRunId": child_run_id,
            })

        if child_run_ids:
            child_runs = await asyncio.gather(
                *[self.get_pipeline_run(cid) for cid in child_run_ids],
                return_exceptions=True,
            )
            valid_child_runs = [
                cr for cr in child_runs if cr is not None and not isinstance(cr, Exception)
            ]
            if valid_child_runs:
                child_hierarchies = await asyncio.gather(
                    *[self._build_hierarchy_async(cr) for cr in valid_child_runs],
                    return_exceptions=True,
                )
                hierarchy["childPipelines"] = [
                    ch for ch in child_hierarchies
                    if ch is not None and not isinstance(ch, Exception)
                ]

        return hierarchy

    def _calculate_hierarchy_summary(self, hierarchy: Dict[str, Any]) -> Dict[str, Any]:
        total = succeeded = failed = in_progress = cancelled = 0
        pipelines_list: List[Dict[str, Any]] = []

        def traverse(node: Dict[str, Any], depth: int = 0) -> None:
            nonlocal total, succeeded, failed, in_progress, cancelled
            total += 1
            status = node.get("status", "").lower()
            pipelines_list.append({
                "pipelineName": node.get("pipelineName"),
                "runId": node.get("runId"),
                "status": node.get("status"),
                "depth": depth,
            })
            if status == "succeeded":
                succeeded += 1
            elif status == "failed":
                failed += 1
            elif status in ("inprogress", "in progress", "running", "queued"):
                in_progress += 1
            elif status == "cancelled":
                cancelled += 1
            for child in node.get("childPipelines", []):
                traverse(child, depth + 1)

        traverse(hierarchy)
        return {
            "total_pipelines": total,
            "succeeded": succeeded,
            "failed": failed,
            "in_progress": in_progress,
            "cancelled": cancelled,
            "pipelines": pipelines_list,
        }


# =============================================================================
# Singleton instances
# =============================================================================

_async_adf_client: Optional[AsyncADFClient] = None


def get_async_adf_client() -> AsyncADFClient:
    global _async_adf_client
    if _async_adf_client is None:
        _async_adf_client = AsyncADFClient()
    return _async_adf_client


# =============================================================================
# MCP Server
# =============================================================================

mcp = FastMCP("adf-mcp-server")


# CHANGE 1: @tool  →  @mcp.tool()
# CHANGE 2: def    →  async def  (MCP supports native async; no loop management needed)
# CHANGE 3: removed `use_async` param — the MCP tool is always async
# CHANGE 4: return type Dict[str, Any]  →  str  (MCP transports text; we JSON-serialize)
@mcp.tool()
async def get_pipeline_hierarchy(pipeline_run_id: str) -> str:
    """Get the complete hierarchy of an Azure Data Factory pipeline run.

    Retrieves the full pipeline execution tree including:
    - Parent pipeline details (auto-discovered if a child run ID is provided)
    - All activity runs within each pipeline
    - Child pipeline executions spawned by ExecutePipeline activities
    - Status, timing, and error information at every level

    Args:
        pipeline_run_id: ADF pipeline run ID — parent or child, the tool finds the root.

    Returns:
        JSON string containing the complete hierarchy with a _summary and _metadata block.
    """
    logger.info(f"[ADF MCP] get_pipeline_hierarchy called for {pipeline_run_id}")
    client = get_async_adf_client()
    result = await client.get_pipeline_hierarchy(pipeline_run_id)
    # json.dumps with default=str handles datetime/UUID objects that may appear in the payload
    return json.dumps(result, default=str)


# =============================================================================
# FastAPI app — needed to serve the MCP StreamableHTTP transport
# =============================================================================

_mcp_asgi_app = mcp.streamable_http_app()


@asynccontextmanager
async def lifespan(_: FastAPI):
    async with mcp.session_manager.run():
        yield


app = FastAPI(
    title="ADF MCP Server",
    description="Azure Data Factory pipeline analysis exposed as MCP tools",
    version="1.0.0",
    lifespan=lifespan,
)

app.mount("/", _mcp_asgi_app)


@app.get("/health")
def health():
    return {"status": "ok", "adf_available": get_async_adf_client().is_available}


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    load_dotenv()

    port = int(os.getenv("PORT", 8001))
    print(f"ADF MCP Server → http://localhost:{port}/mcp")
    uvicorn.run(app, host="0.0.0.0", port=port)
