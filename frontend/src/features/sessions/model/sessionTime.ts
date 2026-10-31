export type SessionGroupLabel = '今天' | '昨天' | '更早'

const DAY_MS = 86_400_000

function calendarDaysAgo(date: Date, now: Date): number {
  const startOf = (value: Date) =>
    new Date(value.getFullYear(), value.getMonth(), value.getDate()).getTime()
  return Math.round((startOf(now) - startOf(date)) / DAY_MS)
}

function clockTime(date: Date): string {
  return date.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' })
}

function monthDay(date: Date): string {
  return date.toLocaleDateString('zh-CN', { month: 'short', day: 'numeric' })
}

export function sessionGroupLabel(iso: string, now = new Date()): SessionGroupLabel {
  const days = calendarDaysAgo(new Date(iso), now)
  if (days <= 0) return '今天'
  return days === 1 ? '昨天' : '更早'
}

export function formatSessionTime(iso: string, now = new Date()): string {
  const date = new Date(iso)
  const days = calendarDaysAgo(date, now)
  if (days <= 0) return clockTime(date)
  return days === 1 ? `昨天 ${clockTime(date)}` : monthDay(date)
}

export function formatMessageTime(iso: string, now = new Date()): string {
  const date = new Date(iso)
  return calendarDaysAgo(date, now) <= 0 ? clockTime(date) : `${monthDay(date)} ${clockTime(date)}`
}
