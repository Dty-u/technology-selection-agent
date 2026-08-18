export type ApiErrorCode =
  | 'session_not_found'
  | 'session_busy'
  | 'service_unavailable'
  | 'validation_error'
  | 'invalid_response'
  | 'network_error'
  | 'request_aborted'
  | 'unsafe_report_url'
  | 'internal_error'

export class ApiError extends Error {
  readonly status: number
  readonly code: ApiErrorCode | string

  constructor(status: number, code: ApiErrorCode | string, message: string) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.code = code
  }
}

const messagesByStatus: Record<number, string> = {
  404: '服务端找不到这个会话，它可能已关闭或不属于当前项目。',
  409: '该会话仍在处理上一条消息，请稍后同步最新状态。',
  422: '提交内容不符合接口要求，请检查输入。',
  500: '服务暂时无法完成请求，内部细节已隐藏。',
  503: 'Agent 依赖的服务暂时不可用，请稍后再试。',
}

export async function apiErrorFromResponse(response: Response): Promise<ApiError> {
  let code: string | undefined
  try {
    const body = (await response.json()) as { detail?: { code?: unknown } }
    if (typeof body.detail?.code === 'string') code = body.detail.code
  } catch {
    // Error bodies are untrusted and optional; never expose the raw text.
  }
  return new ApiError(
    response.status,
    code ?? (response.status === 422 ? 'validation_error' : 'internal_error'),
    messagesByStatus[response.status] ?? '请求失败，请稍后重试。',
  )
}

export function apiErrorFromUnknown(error: unknown): ApiError {
  if (error instanceof ApiError) return error
  if (error instanceof DOMException && error.name === 'AbortError') {
    return new ApiError(0, 'request_aborted', '已停止在当前页面等待，服务端任务可能仍在继续。')
  }
  return new ApiError(0, 'network_error', '无法连接 Agent API，请检查地址和服务状态。')
}
