import { defineStore } from 'pinia'
import { ref } from 'vue'

import { useSettingsStore } from '@/features/settings/settingsStore'
import { ApiError, apiErrorFromUnknown } from '@/shared/api/errors'
import { createSessionApi } from '../api/sessionApi'
import type { ReportContent, ReportLibraryItem } from '../api/sessionSchemas'

function reportFilename(report: ReportLibraryItem): string {
  return `${report.title}.${report.format === 'docx' ? 'docx' : 'md'}`
}

export const useReportLibraryStore = defineStore('reportLibrary', () => {
  const settingsStore = useSettingsStore()
  const reports = ref<ReportLibraryItem[]>([])
  const loading = ref(false)
  const loaded = ref(false)
  const error = ref<ApiError | null>(null)
  let latestLoad = 0

  function api() {
    if (!settingsStore.settings) {
      throw new ApiError(0, 'validation_error', '请先完成工作区设置。')
    }
    return createSessionApi(settingsStore.settings)
  }

  async function load() {
    const token = ++latestLoad
    loading.value = true
    error.value = null
    try {
      const list = await api().listReports()
      if (token !== latestLoad) return
      reports.value = list
      loaded.value = true
    } catch (caught) {
      if (token === latestLoad) error.value = apiErrorFromUnknown(caught)
    } finally {
      if (token === latestLoad) loading.value = false
    }
  }

  async function readContent(report: ReportLibraryItem): Promise<ReportContent | null> {
    error.value = null
    try {
      return await api().getReportContent(report.project_id, report.thread_id, report.report_id)
    } catch (caught) {
      error.value = apiErrorFromUnknown(caught)
      return null
    }
  }

  async function download(report: ReportLibraryItem) {
    error.value = null
    try {
      return await api().downloadReport(report.download_url, reportFilename(report))
    } catch (caught) {
      error.value = apiErrorFromUnknown(caught)
      return null
    }
  }

  async function remove(report: ReportLibraryItem): Promise<boolean> {
    error.value = null
    try {
      await api().deleteReport(report.project_id, report.thread_id, report.report_id)
      reports.value = reports.value.filter(
        (item) => item.thread_id !== report.thread_id || item.report_id !== report.report_id,
      )
      return true
    } catch (caught) {
      const failure = apiErrorFromUnknown(caught)
      error.value =
        failure.code === 'session_busy'
          ? new ApiError(
              failure.status,
              failure.code,
              '这个会话正在处理或等待补充信息，完成后才能删除这份报告。',
            )
          : failure
      return false
    }
  }

  return { reports, loading, loaded, error, load, readContent, download, remove }
})
