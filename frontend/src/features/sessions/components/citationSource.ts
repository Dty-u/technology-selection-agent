import type { SessionSource } from '../api/sessionSchemas'

function isLocalPath(value: string): boolean {
  return /^(?:[a-z]:[\\/]|\\\\|\/|file:)/i.test(value)
}

export function sourceLocation(source: SessionSource): string {
  return source.source || source.url || ''
}

export function isLocalSource(source: SessionSource): boolean {
  return isLocalPath(sourceLocation(source))
}

export function sourceTitle(source: SessionSource): string {
  if (source.title) return source.title
  const location = sourceLocation(source)
  if (!location) return '未命名来源'
  return isLocalPath(location) ? (location.split(/[\\/]/).pop() ?? location) : location
}

// Citation numbers are not positions in `sources`: several [Sn] can share one document,
// so a citation only resolves through its own line in the answer's reference list.
export function resolveCitationSource(
  content: string,
  citation: number,
  sources: readonly SessionSource[],
): number | null {
  const reference = new RegExp(`^\\s*(?:[-*+]|\\d+\\.)?\\s*\\[S${citation}\\](.*)$`, 'm').exec(
    content,
  )?.[1]
  if (!reference) return null
  const index = sources.findIndex((source) => {
    const location = sourceLocation(source)
    return (
      (location !== '' && reference.includes(location)) ||
      (!!source.title && reference.includes(source.title))
    )
  })
  return index === -1 ? null : index
}
