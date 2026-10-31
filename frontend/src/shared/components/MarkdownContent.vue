<script setup lang="ts">
import { computed } from 'vue'
import MarkdownIt from 'markdown-it'

const props = defineProps<{ content: string; citable?: (citation: number) => boolean }>()
const emit = defineEmits<{ citation: [citation: number] }>()

const markdown = new MarkdownIt({
  html: false,
  linkify: true,
  breaks: true,
})

const defaultLinkOpen =
  markdown.renderer.rules.link_open ??
  ((tokens, index, options, _environment, renderer) => renderer.renderToken(tokens, index, options))

markdown.renderer.rules.link_open = (tokens, index, options, environment, renderer) => {
  const href = String(tokens[index]?.attrGet('href') ?? '')
  if (/^https?:\/\//i.test(href)) {
    tokens[index]?.attrSet('target', '_blank')
    tokens[index]?.attrSet('rel', 'noopener noreferrer')
  } else {
    tokens[index]?.attrSet('href', '')
  }
  return defaultLinkOpen(tokens, index, options, environment, renderer)
}

// Only plain text outside links is scanned, so code spans and link labels keep their text.
markdown.core.ruler.push('citation_links', (state) => {
  const citable = state.env?.citable as ((citation: number) => boolean) | undefined
  if (!citable) return
  for (const block of state.tokens) {
    if (block.type !== 'inline' || !block.children) continue
    const children: typeof block.children = []
    let linkDepth = 0
    for (const child of block.children) {
      if (child.type === 'link_open') linkDepth += 1
      if (child.type === 'link_close') linkDepth -= 1
      if (child.type !== 'text' || linkDepth > 0) {
        children.push(child)
        continue
      }
      let cursor = 0
      for (const match of child.content.matchAll(/\[S(\d+)\]/g)) {
        const citation = Number(match[1])
        if (!citable(citation)) continue
        const before = new state.Token('text', '', 0)
        before.content = child.content.slice(cursor, match.index)
        const link = new state.Token('html_inline', '', 0)
        link.content = `<button type="button" class="citation-link" data-citation="${citation}">[S${citation}]</button>`
        children.push(before, link)
        cursor = match.index + match[0].length
      }
      const rest = new state.Token('text', '', 0)
      rest.content = child.content.slice(cursor)
      children.push(rest)
    }
    block.children = children
  }
})

const rendered = computed(() => {
  const protocolSanitized = props.content.replace(/\b(?:javascript|vbscript|data)\s*:/gi, '')
  return markdown.render(protocolSanitized, { citable: props.citable })
})

function onClick(event: MouseEvent) {
  const trigger = (event.target as HTMLElement).closest<HTMLElement>('[data-citation]')
  if (trigger) emit('citation', Number(trigger.dataset.citation))
}
</script>

<template>
  <!-- Raw HTML is escaped and unsafe URL schemes are removed before this render step;
       the only generated markup is the fixed citation button built from a parsed number. -->
  <div class="markdown-content" @click="onClick" v-html="rendered" />
</template>
