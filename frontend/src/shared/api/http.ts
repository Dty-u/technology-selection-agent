import {
  ApiError,
  apiErrorFromResponse,
  apiErrorFromStreamEvent,
  apiErrorFromUnknown,
} from './errors'

export async function requestJson<T>(
  fetcher: typeof fetch,
  url: string,
  init: RequestInit,
  parse: (value: unknown) => T,
): Promise<T> {
  try {
    const response = await fetcher(url, init)
    if (!response.ok) throw await apiErrorFromResponse(response)
    const body: unknown = await response.json()
    try {
      return parse(body)
    } catch {
      throw new ApiError(502, 'invalid_response', '后端返回了无法识别的数据。')
    }
  } catch (error) {
    throw apiErrorFromUnknown(error)
  }
}

export async function requestVoid(
  fetcher: typeof fetch,
  url: string,
  init: RequestInit,
): Promise<void> {
  try {
    const response = await fetcher(url, init)
    if (!response.ok) throw await apiErrorFromResponse(response)
  } catch (error) {
    throw apiErrorFromUnknown(error)
  }
}

export async function requestBlob(
  fetcher: typeof fetch,
  url: string,
  init: RequestInit,
): Promise<{ blob: Blob; filename: string }> {
  try {
    const response = await fetcher(url, init)
    if (!response.ok) throw await apiErrorFromResponse(response)
    const disposition = response.headers.get('Content-Disposition') ?? ''
    const encodedName = disposition.match(/filename\*=UTF-8''([^;]+)/i)?.[1]
    const basicName = disposition.match(/filename="?([^";]+)"?/i)?.[1]
    let filename = 'technology-selection-report.docx'
    if (encodedName) {
      try {
        filename = decodeURIComponent(encodedName)
      } catch {
        filename = 'technology-selection-report.docx'
      }
    } else if (basicName) {
      filename = basicName
    }
    return { blob: await response.blob(), filename }
  } catch (error) {
    throw apiErrorFromUnknown(error)
  }
}

export type ServerSentEvent = {
  event: string
  data: string
}

function parseEventFrame(frame: string): ServerSentEvent | null {
  let event = 'message'
  const data: string[] = []
  for (const line of frame.split('\n')) {
    if (!line || line.startsWith(':')) continue
    const separator = line.indexOf(':')
    const field = separator === -1 ? line : line.slice(0, separator)
    let value = separator === -1 ? '' : line.slice(separator + 1)
    if (value.startsWith(' ')) value = value.slice(1)
    if (field === 'event') event = value
    else if (field === 'data') data.push(value)
  }
  return data.length ? { event, data: data.join('\n') } : null
}

export async function* readEventStream(
  body: ReadableStream<Uint8Array>,
): AsyncGenerator<ServerSentEvent> {
  const reader = body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  try {
    while (true) {
      const { done, value } = await reader.read()
      buffer += decoder.decode(value, { stream: !done }).replace(/\r\n?/g, '\n')
      let boundary = buffer.indexOf('\n\n')
      while (boundary !== -1) {
        const parsed = parseEventFrame(buffer.slice(0, boundary))
        buffer = buffer.slice(boundary + 2)
        if (parsed) yield parsed
        boundary = buffer.indexOf('\n\n')
      }
      if (done) break
    }
    const trailing = parseEventFrame(buffer)
    if (trailing) yield trailing
  } finally {
    reader.releaseLock()
  }
}

/**
 * POST to an SSE endpoint and resolve with its `final` event.
 *
 * Pre-stream failures (busy, not found) still arrive as JSON error statuses;
 * a plain JSON success body is accepted so older backends keep working.
 */
export async function requestEventStream<T>(
  fetcher: typeof fetch,
  url: string,
  init: RequestInit,
  parse: (value: unknown) => T,
  onEvent: (event: ServerSentEvent) => void,
): Promise<T> {
  const parseFinal = (value: unknown) => {
    try {
      return parse(value)
    } catch {
      throw new ApiError(502, 'invalid_response', '后端返回了无法识别的数据。')
    }
  }
  try {
    const response = await fetcher(url, init)
    if (!response.ok) throw await apiErrorFromResponse(response)
    const contentType = response.headers.get('Content-Type') ?? ''
    if (!contentType.includes('text/event-stream') || !response.body) {
      return parseFinal(await response.json())
    }
    for await (const event of readEventStream(response.body)) {
      if (event.event === 'final') return parseFinal(JSON.parse(event.data))
      if (event.event === 'error') {
        let body: unknown = null
        try {
          body = JSON.parse(event.data)
        } catch {
          // Fall through to a generic internal error.
        }
        throw apiErrorFromStreamEvent(body)
      }
      onEvent(event)
    }
    throw new ApiError(502, 'invalid_response', '回答流在完成前中断。')
  } catch (error) {
    throw apiErrorFromUnknown(error)
  }
}
