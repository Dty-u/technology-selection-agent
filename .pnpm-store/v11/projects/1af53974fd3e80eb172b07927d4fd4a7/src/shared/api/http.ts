import { ApiError, apiErrorFromResponse, apiErrorFromUnknown } from './errors'

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
