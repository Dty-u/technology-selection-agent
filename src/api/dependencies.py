"""FastAPI dependencies shared by versioned routes."""

from typing import Annotated, Any
from uuid import UUID, uuid4

from fastapi import Header, HTTPException, Path, Request


async def get_session_service(request: Request) -> Any:
    return request.app.state.session_service


async def require_project_id(
    project_id: Annotated[str, Path(min_length=1, max_length=128)],
) -> str:
    normalized = project_id.strip()
    if not normalized:
        raise HTTPException(status_code=422, detail="project_id must not be blank")
    return normalized


async def require_user_id(
    x_user_id: Annotated[
        str | None,
        Header(alias="X-User-ID", min_length=1, max_length=128),
    ] = None,
) -> str:
    normalized = str(x_user_id or "").strip()
    if not normalized:
        raise HTTPException(status_code=422, detail="X-User-ID is required")
    return normalized


async def provide_request_id(
    x_request_id: Annotated[
        str | None,
        Header(alias="X-Request-ID"),
    ] = None,
) -> str:
    if x_request_id is None:
        return str(uuid4())
    try:
        return str(UUID(x_request_id))
    except (TypeError, ValueError, AttributeError) as exc:
        raise HTTPException(
            status_code=422,
            detail="X-Request-ID must be a UUID",
        ) from exc
