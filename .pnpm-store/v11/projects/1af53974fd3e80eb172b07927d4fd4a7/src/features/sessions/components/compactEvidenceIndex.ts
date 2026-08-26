const EVIDENCE_INDEX_HEADING = /^##\s+(?:8[.、]\s*)?证据(?:索引|来源概览)\s*$/
const SECTION_HEADING = /^##\s+/
const EVIDENCE_TITLE = /^###\s+/
const SOURCE_LINE = /^\s*-\s*来源[：:]\s*(.+?)\s*$/
const MARKDOWN_LINK = /\[[^\]]*\]\(([^)]+)\)/
const SOURCE_PREVIEW_LIMIT = 3

type ExternalSource = {
  hostname: string
  url: string
}

function externalSource(value: string): ExternalSource | null {
  const rawUrl = MARKDOWN_LINK.exec(value)?.[1]
  if (!rawUrl) return null

  try {
    const url = new URL(rawUrl)
    if (!['http:', 'https:'].includes(url.protocol)) return null
    return {
      hostname: url.hostname.toLowerCase().replace(/^www\./, ''),
      url: url.toString(),
    }
  } catch {
    return null
  }
}

export function compactEvidenceIndex(content: string): string {
  const lines = content.split(/\r?\n/)
  const sectionStart = lines.findIndex((line) => EVIDENCE_INDEX_HEADING.test(line.trim()))
  if (sectionStart < 0) return content

  const nextSectionOffset = lines
    .slice(sectionStart + 1)
    .findIndex((line) => SECTION_HEADING.test(line.trim()))
  const sectionEnd = nextSectionOffset < 0 ? lines.length : sectionStart + 1 + nextSectionOffset
  let evidenceCount = 0
  const sources = new Map<string, ExternalSource>()

  for (const line of lines.slice(sectionStart + 1, sectionEnd)) {
    if (EVIDENCE_TITLE.test(line.trim())) {
      evidenceCount += 1
      continue
    }

    const sourceValue = SOURCE_LINE.exec(line)?.[1]
    if (!sourceValue) continue
    const source = externalSource(sourceValue)
    if (source && !sources.has(source.hostname)) sources.set(source.hostname, source)
  }

  const summary = (() => {
    if (!evidenceCount) return '证据来源暂无法概括，请在右侧证据面板查看。'
    if (!sources.size) {
      return `本报告共参考 ${evidenceCount} 条证据，未发现可展示的外部来源。`
    }

    const preview = [...sources.values()]
      .slice(0, SOURCE_PREVIEW_LIMIT)
      .map((source) => `[${source.hostname}](${source.url})`)
      .join('、')
    const suffix = sources.size > SOURCE_PREVIEW_LIMIT ? ' 等。' : '。'
    return `本报告共参考 ${evidenceCount} 条证据，来自 ${sources.size} 个来源：${preview}${suffix}`
  })()

  return [
    ...lines.slice(0, sectionStart),
    '## 8. 证据来源概览',
    '',
    summary,
    '',
    ...lines.slice(sectionEnd),
  ].join('\n')
}
