import { create } from 'zustand'

export type Depth = 'cleaned' | 'full'

type ProjectStatus = 'PENDING' | 'PROCESSING' | 'COMPLETED' | 'FAILED'

export type PollPayload = {
  project_id: string
  status: ProjectStatus
  task_state?: string
  meta?: { stage?: string; progress?: number }
  error?: string
  artifacts?: {
    [key: string]: string
  }
  editor_data?: EditorPayload
}

const API_BASE = import.meta.env.VITE_API_BASE || 'http://127.0.0.1:8000'

async function uploadAndStart(file: File, getToken: () => Promise<string | null>): Promise<{ projectId: string; taskId: string }> {
  const token = await getToken()
  if (!token) throw new Error('Not authenticated')
  const headers = { Authorization: `Bearer ${token}` }

  // 1. Get upload URL
  const uploadUrlResp = await fetch(`${API_BASE}/api/v1/projects/upload-url?filename=${encodeURIComponent(file.name)}`, {
    method: 'POST',
    headers,
  })
  if (!uploadUrlResp.ok) throw new Error('Failed to get upload URL')
  const { project_id: projectId, url: uploadUrl } = await uploadUrlResp.json()

  // 2. Upload file (multipart/form-data with field name 'file')
  const form = new FormData()
  form.append('file', file)
  const uploadResp = await fetch(`${API_BASE}${uploadUrl}`, { method: 'POST', body: form, headers })
  if (!uploadResp.ok) throw new Error('Upload failed')
  const { storage_key: storageKey } = await uploadResp.json()

  // 3. Create project and start processing
  const createProjectResp = await fetch(`${API_BASE}/api/v1/projects?project_id=${projectId}&filename=${encodeURIComponent(file.name)}&storage_key=${storageKey}`, {
    method: 'POST',
    headers,
  })
  if (!createProjectResp.ok) throw new Error('Failed to create project')
  const { task_id: taskId } = await createProjectResp.json()

  return { projectId, taskId }
}

async function pollOnce(projectId: string, getToken: () => Promise<string | null>): Promise<PollPayload> {
  const token = await getToken()
  if (!token) throw new Error('Not authenticated')
  const headers = { Authorization: `Bearer ${token}` }
  const resp = await fetch(`${API_BASE}/api/v1/projects/${projectId}`, { headers })
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

  projectId: string
  taskId: string
  state: ProjectStatus
  meta?: { stage?: string; progress?: number }
  result?: PollPayload
  error?: string

  start: (file: File, getToken: () => Promise<string | null>) => Promise<void>
  reset: () => void

  // Editor state
  editor?: EditorPayload
  selectedBubbleId?: number
  setSelectedBubbleId: (id?: number) => void
  edits: EditsMap
  updateEdit: (id: number, patch: { en_text?: string; font_size?: number }) => void
  loadEditor: (projectId: string, result: PollPayload) => Promise<void>

  // Apply edits & exports
  applyingEdits: boolean
  exports?: ExportPayload
  applyEdits: (getToken: () => Promise<string | null>) => Promise<void>
  downloadUrl: () => string | undefined
  downloadFile: (getToken: () => Promise<string | null>) => Promise<void>
}

export const useAppStore = create<StoreState>((set, get) => ({
  depth: 'cleaned',
  setDepth: (d) => set({ depth: d }),

  projectId: '',
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
    const projectId = get().projectId
    if (projectId) {
      try {
        localStorage.setItem(`mf_edits_${projectId}`, JSON.stringify(next))
      } catch {
        // ignore storage failures in MVP
      }
    }
  },
  async loadEditor(projectId, result) {
    // Prefer payload embedded in the API response
    if (result.editor_data) {
      let saved: EditsMap = {}
      try {
        const raw = localStorage.getItem(`mf_edits_${projectId}`)
        if (raw) saved = JSON.parse(raw)
      } catch {}
      const bubbles = result.editor_data.bubbles.map((b) => {
        const e = saved[b.id]
        return e ? { ...b, en_text: e.en_text ?? b.en_text, font_size: e.font_size ?? b.font_size } : b
      })
      set({ editor: { ...result.editor_data, bubbles }, edits: saved })
      return
    }
    // Fallback: load from a public URL when provided (not typical in Phase 2)
    const editorPayloadUrl = result.artifacts?.["EDITOR_PAYLOAD"]
    if (editorPayloadUrl) {
      const url = `${API_BASE}${editorPayloadUrl}?t=${Date.now()}`
      try {
        const resp = await fetch(url)
        if (!resp.ok) return
        const payload = (await resp.json()) as EditorPayload
        let saved: EditsMap = {}
        try {
          const raw = localStorage.getItem(`mf_edits_${projectId}`)
          if (raw) saved = JSON.parse(raw)
        } catch {}
        const bubbles = payload.bubbles.map((b) => {
          const e = saved[b.id]
          return e ? { ...b, en_text: e.en_text ?? b.en_text, font_size: e.font_size ?? b.font_size } : b
        })
        set({ editor: { ...payload, bubbles }, edits: saved })
      } catch {}
    }
  },

  applyingEdits: false,
  exports: undefined,
  async applyEdits(getToken) {
    const projectId = get().projectId
    if (!projectId) return

    const editsMap = get().edits
    const editsArr = Object.entries(editsMap).map(([id, patch]) => ({ id: Number(id), ...patch }))

    set({ applyingEdits: true })
    try {
      const token = await getToken()
      if (!token) throw new Error('Not authenticated')
      const headers = { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` }

      const resp = await fetch(`${API_BASE}/api/v1/projects/${projectId}`, {
        method: 'PUT',
        headers,
        body: JSON.stringify({ edits: editsArr }),
      })
      if (!resp.ok) throw new Error('Apply edits failed')
      
      // Poll for updated project status
      let pollCount = 0
      while (pollCount < 30) { // 30 second timeout
        const data = await pollOnce(projectId, getToken)
        if (data.status === 'COMPLETED' || data.status === 'FAILED') {
          set({ result: data, state: data.status })
          if (data.status === 'COMPLETED') {
            await get().loadEditor(projectId, data)
          }
          break
        }
        await new Promise((r) => setTimeout(r, 1000))
        pollCount++
      }
    } catch (err) {
      // surface error in store.error without breaking app
      const message = err instanceof Error ? err.message : String(err)
      set({ error: message })
    } finally {
      set({ applyingEdits: false })
    }
  },
  downloadUrl() {
    const id = get().projectId
    if (!id) return undefined
    return `${API_BASE}/api/v1/projects/${id}/download`
  },

  async downloadFile(getToken) {
    const projectId = get().projectId
    if (!projectId) return
    const token = await getToken()
    if (!token) throw new Error('Not authenticated')
    const resp = await fetch(`${API_BASE}/api/v1/projects/${projectId}/download`, {
      headers: { Authorization: `Bearer ${token}` },
    })
    if (!resp.ok) throw new Error(`Download failed (${resp.status})`)
    const blob = await resp.blob()
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `mangafuse_${projectId}.zip`
    document.body.appendChild(a)
    a.click()
    a.remove()
    URL.revokeObjectURL(url)
  },

  async start(file, getToken) {
    set({ error: undefined, result: undefined, state: 'PENDING' })
    try {
      const { projectId, taskId } = await uploadAndStart(file, getToken)
      set({ projectId, taskId, state: 'PROCESSING' })
      const timer = window.setInterval(async () => {
        try {
          const data = await pollOnce(projectId, getToken)
          set({ state: data.status, meta: data.meta })
          if (data.status === 'COMPLETED') {
            set({ result: data })
            await get().loadEditor(projectId, data)
            window.clearInterval(timer)
          } else if (data.status === 'FAILED') {
            set({ error: data.error || 'Task failed' })
            window.clearInterval(timer)
          }
        } catch (err: unknown) {
          const message = err instanceof Error ? err.message : String(err)
          set({ error: message })
          window.clearInterval(timer)
        }
      }, 1000)
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : String(err)
      set({ error: message })
    }
  },

  reset() {
    set({
      projectId: '',
      taskId: '',
      state: 'PENDING',
      meta: undefined,
      result: undefined,
      error: undefined,
      editor: undefined,
      selectedBubbleId: undefined,
      edits: {},
    })
  },
}))
