import { computed, onBeforeUnmount, onMounted, ref, type CSSProperties } from 'vue'

import { readJson, writeJson } from '@/shared/storage/localStorage'

type ResizablePanel = 'sidebar' | 'inspector'

type PanelLayout = {
  sidebar: number
  inspector: number
}

const STORAGE_KEY = 'rag-agent.workspace-panels.v1'
const SIDEBAR_MIN = 200
const SIDEBAR_MAX = 420
const INSPECTOR_MIN = 300
const INSPECTOR_MAX = 760
const CENTER_MIN = 410
const KEYBOARD_STEP = 16

function defaultLayout(viewportWidth: number): PanelLayout {
  return viewportWidth <= 1180 ? { sidebar: 230, inspector: 360 } : { sidebar: 258, inspector: 430 }
}

function parseLayout(value: unknown): PanelLayout {
  if (!value || typeof value !== 'object') throw new Error('invalid workspace panel layout')
  const candidate = value as Partial<PanelLayout>
  if (!Number.isFinite(candidate.sidebar) || !Number.isFinite(candidate.inspector)) {
    throw new Error('invalid workspace panel widths')
  }
  return { sidebar: Number(candidate.sidebar), inspector: Number(candidate.inspector) }
}

function parseInspectorCollapsed(value: unknown): boolean {
  return (value as { inspectorCollapsed?: unknown } | null)?.inspectorCollapsed === true
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.min(Math.max(value, minimum), maximum)
}

export function useWorkspacePanelResize() {
  const initial = defaultLayout(globalThis.innerWidth || 1280)
  const sidebarWidth = ref(initial.sidebar)
  const inspectorWidth = ref(initial.inspector)
  const inspectorCollapsed = ref(false)
  const resizingPanel = ref<ResizablePanel | null>(null)
  let pointerStartX = 0
  let widthAtPointerStart = 0

  const sidebarMaximum = computed(() =>
    Math.max(
      SIDEBAR_MIN,
      Math.min(
        SIDEBAR_MAX,
        window.innerWidth - (inspectorCollapsed.value ? 0 : inspectorWidth.value) - CENTER_MIN,
      ),
    ),
  )
  const inspectorMaximum = computed(() =>
    Math.max(
      INSPECTOR_MIN,
      Math.min(INSPECTOR_MAX, window.innerWidth - sidebarWidth.value - CENTER_MIN),
    ),
  )
  const layoutStyle = computed(
    () =>
      ({
        '--workspace-sidebar-width': `${sidebarWidth.value}px`,
        '--workspace-inspector-width': `${inspectorWidth.value}px`,
      }) as CSSProperties,
  )

  function constrainToViewport() {
    sidebarWidth.value = clamp(sidebarWidth.value, SIDEBAR_MIN, sidebarMaximum.value)
    inspectorWidth.value = clamp(inspectorWidth.value, INSPECTOR_MIN, inspectorMaximum.value)
  }

  function persist() {
    writeJson(localStorage, STORAGE_KEY, {
      sidebar: Math.round(sidebarWidth.value),
      inspector: Math.round(inspectorWidth.value),
      inspectorCollapsed: inspectorCollapsed.value,
    })
  }

  function setInspectorCollapsed(collapsed: boolean) {
    if (inspectorCollapsed.value === collapsed) return
    inspectorCollapsed.value = collapsed
    constrainToViewport()
    persist()
  }

  function setWidth(panel: ResizablePanel, requestedWidth: number) {
    if (panel === 'sidebar') {
      sidebarWidth.value = clamp(requestedWidth, SIDEBAR_MIN, sidebarMaximum.value)
      return
    }
    inspectorWidth.value = clamp(requestedWidth, INSPECTOR_MIN, inspectorMaximum.value)
  }

  function startResize(panel: ResizablePanel, event: PointerEvent) {
    if (event.button !== 0) return
    event.preventDefault()
    resizingPanel.value = panel
    pointerStartX = event.clientX
    widthAtPointerStart = panel === 'sidebar' ? sidebarWidth.value : inspectorWidth.value
    ;(event.currentTarget as HTMLElement | null)?.setPointerCapture?.(event.pointerId)
    document.body.classList.add('workspace-is-resizing')
    window.addEventListener('pointermove', continueResize)
    window.addEventListener('pointerup', stopResize)
    window.addEventListener('pointercancel', stopResize)
  }

  function continueResize(event: PointerEvent) {
    if (!resizingPanel.value) return
    const delta = event.clientX - pointerStartX
    setWidth(
      resizingPanel.value,
      widthAtPointerStart + (resizingPanel.value === 'sidebar' ? delta : -delta),
    )
  }

  function stopResize() {
    if (!resizingPanel.value) return
    resizingPanel.value = null
    document.body.classList.remove('workspace-is-resizing')
    window.removeEventListener('pointermove', continueResize)
    window.removeEventListener('pointerup', stopResize)
    window.removeEventListener('pointercancel', stopResize)
    persist()
  }

  function resizeWithKeyboard(panel: ResizablePanel, event: KeyboardEvent) {
    if (!['ArrowLeft', 'ArrowRight', 'Home'].includes(event.key)) return
    event.preventDefault()
    if (event.key === 'Home') {
      setWidth(panel, defaultLayout(window.innerWidth)[panel])
    } else {
      const direction = event.key === 'ArrowRight' ? 1 : -1
      const visualDirection = panel === 'sidebar' ? direction : -direction
      const currentWidth = panel === 'sidebar' ? sidebarWidth.value : inspectorWidth.value
      setWidth(panel, currentWidth + visualDirection * KEYBOARD_STEP)
    }
    persist()
  }

  function resetWidth(panel: ResizablePanel) {
    setWidth(panel, defaultLayout(window.innerWidth)[panel])
    persist()
  }

  function restore() {
    const saved = readJson(localStorage, STORAGE_KEY, parseLayout).value
    if (saved) {
      sidebarWidth.value = saved.sidebar
      inspectorWidth.value = saved.inspector
    }
    inspectorCollapsed.value = readJson(localStorage, STORAGE_KEY, parseInspectorCollapsed).value === true
    constrainToViewport()
  }

  onMounted(() => {
    restore()
    window.addEventListener('resize', constrainToViewport)
  })

  onBeforeUnmount(() => {
    window.removeEventListener('resize', constrainToViewport)
    window.removeEventListener('pointermove', continueResize)
    window.removeEventListener('pointerup', stopResize)
    window.removeEventListener('pointercancel', stopResize)
    document.body.classList.remove('workspace-is-resizing')
  })

  return {
    inspectorCollapsed,
    setInspectorCollapsed,
    inspectorMaximum,
    inspectorMinimum: INSPECTOR_MIN,
    inspectorWidth,
    layoutStyle,
    sidebarMaximum,
    sidebarMinimum: SIDEBAR_MIN,
    sidebarWidth,
    resetWidth,
    resizeWithKeyboard,
    startResize,
  }
}
