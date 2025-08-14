import { create } from 'zustand'

export type Depth = 'cleaned' | 'full'

type TaskState = 'PENDING' | 'STARTED' | 'RETRY' | 'PROGRESS' | 'SUCCESS' | 'FAILURE'

export type PollPayload = {
  task_id: string
  state: TaskState
  meta?: { stage?: string; progress?: number }
  error?: string
  result?: {
    task_id: string
    stage_completed: string[]
    width: number
    height: number
    num_bubbles: number
    json_url: string
    overlay_url: string
    cleaned_url?: string
    final_url?: string
    typeset_debug_url?: string
    editor_payload_url?: string
  }
}

const API_BASE = import.meta.env.VITE_API_BASE || 'http://127.0.0.1:8000'

async function uploadAndStart(file: File, depth: Depth, debug = false, force = false): Promise<string> {
  const form = new FormData()
  form.append('file', file)
  form.append('depth', depth)
  form.append('debug', String(debug))
  form.append('force', String(force))
  const resp = await fetch(`${API_BASE}/api/v1/process`, { method: 'POST', body: form })
  if (!resp.ok) {
    const text = await resp.text()
    throw new Error(`Upload failed (${resp.status}): ${text}`)
  }
  const data = (await resp.json()) as { task_id: string }
  return data.task_id
}

async function pollOnce(taskId: string): Promise<PollPayload> {
  const resp = await fetch(`${API_BASE}/api/v1/process/${taskId}`)
  if (!resp.ok) {
    const text = await resp.text()
    throw new Error(`Poll failed (${resp.status}): ${text}`)
  }
  return (await resp.json()) as PollPayload
}

export type EditorBubble = {
  id: number
  polygon: [number, number][]
  ja_text?: string
  en_text?: string
  font_size?: number
  rect?: { x: number; y: number; w: number; h: number }
  crop_url?: string
}

export type EditorPayload = {
  image_url: string
  width: number
  height: number
  bubbles: EditorBubble[]
}

type EditsMap = Record<number, { en_text?: string; font_size?: number }>

export type ExportPayload = {
  final_url?: string
  text_layer_url?: string
}

type StoreState = {
  depth: Depth
  setDepth: (d: Depth) => void

  taskId: string
  state: TaskState
  meta?: { stage?: string; progress?: number }
  result?: PollPayload['result']
  error?: string

  start: (file: File) => Promise<void>
  reset: () => void

  // Editor state
  editor?: EditorPayload
  selectedBubbleId?: number
  setSelectedBubbleId: (id?: number) => void
  edits: EditsMap
  updateEdit: (id: number, patch: { en_text?: string; font_size?: number }) => void
  loadEditor: (taskId: string, result: PollPayload['result']) => Promise<void>

  // Apply edits & exports
  applyingEdits: boolean
  exports?: ExportPayload
  applyEdits: () => Promise<void>
  downloadUrl: () => string | undefined
}

export const useAppStore = create<StoreState>((set, get) => ({
  depth: 'cleaned',
  setDepth: (d) => set({ depth: d }),

  taskId: '',
  state: 'PENDING',
  meta: undefined,
  result: undefined,
  error: undefined,

  editor: undefined,
  selectedBubbleId: undefined,
  setSelectedBubbleId: (id) => set({ selectedBubbleId: id }),
  edits: {},
  updateEdit: (id, patch) => {
    const next = { ...get().edits, [id]: { ...get().edits[id], ...patch } }
    set({ edits: next })
    const taskId = get().taskId
    if (taskId) {
      try {
        localStorage.setItem(`mf_edits_${taskId}`, JSON.stringify(next))
      } catch {
        // ignore storage failures in MVP
      }
    }
  },
  async loadEditor(taskId, result) {
    if (!result) return
    const base = result.editor_payload_url || `/artifacts/jobs/${taskId}/editor_payload.json`
    // Cache bust to avoid stale payload after edits
    const url = `${base}?t=${Date.now()}`
    try {
      const resp = await fetch(`${API_BASE}${url}`)
      if (!resp.ok) return
      const payload = (await resp.json()) as EditorPayload
      // restore edits
      let saved: EditsMap = {}
      try {
        const raw = localStorage.getItem(`mf_edits_${taskId}`)
        if (raw) saved = JSON.parse(raw)
      } catch {
        // ignore parse/storage failures
      }
      // apply edits onto payload bubbles
      const bubbles = payload.bubbles.map((b) => {
        const e = saved[b.id]
        return e ? { ...b, en_text: e.en_text ?? b.en_text, font_size: e.font_size ?? b.font_size } : b
      })
      set({ editor: { ...payload, bubbles }, edits: saved })
    } catch {
      // ignore for MVP if missing
    }
  },

  applyingEdits: false,
  exports: undefined,
  async applyEdits() {
    const taskId = get().taskId
    const currentResult = get().result
    if (!taskId || !currentResult) return
    // Build edits array from local edits map; include only changed fields
    const editsMap = get().edits
    const editor = get().editor
    const editsArr: { id: number; en_text?: string; font_size?: number }[] = []
    const bubbles = editor?.bubbles || []
    for (const b of bubbles) {
      const e = editsMap[b.id]
      if (!e) continue
      const payload: { id: number; en_text?: string; font_size?: number } = { id: b.id }
      if (typeof e.en_text === 'string') payload.en_text = e.en_text
      if (typeof e.font_size === 'number') payload.font_size = e.font_size
      editsArr.push(payload)
    }
    // Always proceed to trigger server-side typesetting even if no local diffs
    set({ applyingEdits: true })
    try {
      // Enqueue apply-edits task
      const resp = await fetch(`${API_BASE}/api/v1/jobs/${taskId}/edits`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ edits: editsArr }),
      })
      if (!resp.ok) throw new Error(`Apply edits failed (${resp.status})`)
      const data = (await resp.json()) as { task_id: string }
      const editsTaskId = data.task_id
      // Poll until the edits task is done
      while (true) {
        const poll = await pollOnce(editsTaskId)
        if (poll.state === 'SUCCESS' || poll.state === 'FAILURE') break
        await new Promise((r) => setTimeout(r, 1000))
      }
      // Fetch latest exports
      const expResp = await fetch(`${API_BASE}/api/v1/jobs/${taskId}/exports`)
      if (expResp.ok) {
        const expData = (await expResp.json()) as { final_url?: string; text_layer_url?: string }
        set({ exports: { final_url: expData.final_url, text_layer_url: expData.text_layer_url } })
      }
      // Refresh editor payload to reflect normalized font sizes/text
      await get().loadEditor(taskId, currentResult)
    } catch (err) {
      // surface error in store.error without breaking app
      const message = err instanceof Error ? err.message : String(err)
      set({ error: message })
    } finally {
      set({ applyingEdits: false })
    }
  },
  downloadUrl() {
    const id = get().taskId
    if (!id) return undefined
    return `${API_BASE}/api/v1/jobs/${id}/download`
  },

  async start(file: File) {
    set({ error: undefined, result: undefined, state: 'PENDING' })
    const depth = get().depth
    const id = await uploadAndStart(file, depth)
    set({ taskId: id, state: 'STARTED' })
    const timer = window.setInterval(async () => {
      try {
        const data = await pollOnce(id)
        set({ state: data.state, meta: data.meta })
        if (data.state === 'SUCCESS') {
          set({ result: data.result })
          // attempt to load editor payload for After state
          if (data.result) {
            await get().loadEditor(id, data.result)
          }
          window.clearInterval(timer)
        } else if (data.state === 'FAILURE') {
          set({ error: data.error || 'Task failed' })
          window.clearInterval(timer)
        }
      } catch (err: unknown) {
        const message = err instanceof Error ? err.message : String(err)
        set({ error: message })
        window.clearInterval(timer)
      }
    }, 1000)
  },

  reset() {
    set({ taskId: '', state: 'PENDING', meta: undefined, result: undefined, error: undefined, editor: undefined, selectedBubbleId: undefined, edits: {} })
  },
}))


