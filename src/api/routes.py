"""Versioned HTTP routes for checkpointed agent sessions."""

from typing import Annotated, Any
from urllib.parse import quote
from uuid import UUID

from fastapi import APIRouter, Depends, Header, Response, status
from fastapi.responses import FileResponse

from src.api.dependencies import (
    get_session_service,
    provide_request_id,
    require_project_id,
    require_user_id,
)
from src.api.schemas import CancelSessionResponse, MessageRequest, SessionResponse


router = APIRouter(prefix="/api/v1/projects/{project_id}", tags=["sessions"])

ProjectId = Annotated[str, Depends(require_project_id)]
UserId = Annotated[str, Depends(require_user_id)]
RequestId = Annotated[str, Depends(provide_request_id)]
Service = Annotated[Any, Depends(get_session_service)]


@router.post(
    "/sessions",
    response_model=SessionResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_session(
    project_id: ProjectId,
    message: MessageRequest,
    user_id: UserId,
    request_id: RequestId,
    service: Service,
    x_thread_id: Annotated[
        UUID | None,
        Header(alias="X-Thread-ID"),
    ] = None,
) -> SessionResponse:
    return await service.create_session(
        user_id=user_id,
        project_id=project_id.strip(),
        request_id=request_id,
        message=message,
        thread_id=str(x_thread_id) if x_thread_id is not None else None,
    )


@router.post("/sessions/{thread_id}/messages", response_model=SessionResponse)
async def send_message(
    project_id: ProjectId,
    thread_id: UUID,
    message: MessageRequest,
    user_id: UserId,
    request_id: RequestId,
    service: Service,
) -> SessionResponse:
    return await service.send_message(
        user_id=user_id,
        project_id=project_id.strip(),
        thread_id=str(thread_id),
        request_id=request_id,
        message=message,
    )


@router.get("/sessions/{thread_id}", response_model=SessionResponse)
async def get_session(
    project_id: ProjectId,
    thread_id: UUID,
    user_id: UserId,
    service: Service,
) -> SessionResponse:
    return await service.get_session(
        user_id=user_id,
        project_id=project_id.strip(),
        thread_id=str(thread_id),
    )


@router.get("/sessions/{thread_id}/report", response_class=FileResponse)
async def download_report(
    project_id: ProjectId,
    thread_id: UUID,
    user_id: UserId,
    service: Service,
) -> FileResponse:
    report = await service.get_current_report_download(
        user_id=user_id,
        project_id=project_id.strip(),
        thread_id=str(thread_id),
    )
    return FileResponse(
        path=report.path,
        filename=report.filename,
        media_type=(
            "application/vnd.openxmlformats-officedocument."
            "wordprocessingml.document"
        ),
    )


@router.post("/sessions/{thread_id}/cancel", response_model=CancelSessionResponse)
async def cancel_session(
    project_id: ProjectId,
    thread_id: UUID,
    user_id: UserId,
    service: Service,
) -> CancelSessionResponse:
    return await service.cancel_session(
        user_id=user_id,
        project_id=project_id.strip(),
        thread_id=str(thread_id),
    )


@router.get("/sessions/{thread_id}/reports/{report_id}")
async def download_historical_report(
    project_id: ProjectId,
    thread_id: UUID,
    report_id: str,
    user_id: UserId,
    service: Service,
) -> Response:
    report = await service.get_report_download(
        user_id=user_id,
        project_id=project_id.strip(),
        thread_id=str(thread_id),
        report_id=report_id,
    )
    if report.path is not None:
        return FileResponse(
            path=report.path,
            filename=report.filename,
            media_type=(
                "application/vnd.openxmlformats-officedocument."
                "wordprocessingml.document"
            ),
        )
    return Response(
        content=report.content or "",
        media_type="text/markdown; charset=utf-8",
        headers={
            "Content-Disposition": (
                "attachment; filename*=UTF-8''" + quote(report.filename)
            )
        },
    )


@router.delete(
    "/sessions/{thread_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
)
async def close_session(
    project_id: ProjectId,
    thread_id: UUID,
    user_id: UserId,
    service: Service,
) -> Response:
    await service.close_session(
        user_id=user_id,
        project_id=project_id.strip(),
        thread_id=str(thread_id),
    )
    return Response(status_code=status.HTTP_204_NO_CONTENT)
