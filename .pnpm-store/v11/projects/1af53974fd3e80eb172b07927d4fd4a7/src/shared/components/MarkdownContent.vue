<script setup lang="ts">
import { computed } from 'vue'
import MarkdownIt from 'markdown-it'

const props = defineProps<{ content: string }>()

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

const rendered = computed(() => {
  const protocolSanitized = props.content.replace(/\b(?:javascript|vbscript|data)\s*:/gi, '')
  return markdown.render(protocolSanitized)
})
</script>

<template>
  <!-- Raw HTML is escaped and unsafe URL schemes are removed before this render step. -->
  <div class="markdown-content" v-html="rendered" />
</template>
