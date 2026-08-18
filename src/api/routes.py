"""Versioned HTTP routes for checkpointed agent sessions."""

from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Depends, Response, status
from fastapi.responses import FileResponse

from src.api.dependencies import (
    get_session_service,
    provide_request_id,
    require_project_id,
    require_user_id,
)
from src.api.schemas import MessageRequest, SessionResponse


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
def create_session(
    project_id: ProjectId,
    message: MessageRequest,
    user_id: UserId,
    request_id: RequestId,
    service: Service,
) -> SessionResponse:
    return service.create_session(
        user_id=user_id,
        project_id=project_id.strip(),
        request_id=request_id,
        message=message,
    )


@router.post("/sessions/{thread_id}/messages", response_model=SessionResponse)
def send_message(
    project_id: ProjectId,
    thread_id: UUID,
    message: MessageRequest,
    user_id: UserId,
    request_id: RequestId,
    service: Service,
) -> SessionResponse:
    return service.send_message(
        user_id=user_id,
        project_id=project_id.strip(),
        thread_id=str(thread_id),
        request_id=request_id,
        message=message,
    )


@router.get("/sessions/{thread_id}", response_model=SessionResponse)
def get_session(
    project_id: ProjectId,
    thread_id: UUID,
    user_id: UserId,
    service: Service,
) -> SessionResponse:
    return service.get_session(
        user_id=user_id,
        project_id=project_id.strip(),
        thread_id=str(thread_id),
    )


@router.get("/sessions/{thread_id}/report", response_class=FileResponse)
def download_report(
    project_id: ProjectId,
    thread_id: UUID,
    user_id: UserId,
    service: Service,
) -> FileResponse:
    report_path = service.get_report_path(
        user_id=user_id,
        project_id=project_id.strip(),
        thread_id=str(thread_id),
    )
    return FileResponse(
        path=report_path,
        filename=report_path.name,
        media_type=(
            "application/vnd.openxmlformats-officedocument."
            "wordprocessingml.document"
        ),
    )


@router.delete(
    "/sessions/{thread_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
)
def close_session(
    project_id: ProjectId,
    thread_id: UUID,
    user_id: UserId,
    service: Service,
) -> Response:
    service.close_session(
        user_id=user_id,
        project_id=project_id.strip(),
        thread_id=str(thread_id),
    )
    return Response(status_code=status.HTTP_204_NO_CONTENT)
