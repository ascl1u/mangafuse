import { create } from 'zustand'
import { getUploader } from './uploader'
import {
  POLL_DECORRELATED_JITTER,
  POLL_FAST_PATH_ATTEMPTS,
  POLL_INITIAL_DELAY_MS,
  POLL_MAX_DELAY_MS,
  POLL_MAX_TIME_MS,
} from './constants'

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
  const { project_id: projectId, url: uploadUrl, storage_key: storageKey } = await uploadUrlResp.json()

  // 2. Upload the file using the appropriate strategy for the environment.
  const uploader = getUploader()
  await uploader.upload(file, uploadUrl, headers)

  // 3. Create project and start processing
  const idempotencyKey = projectId // stable per project creation
  const createProjectResp = await fetch(
    `${API_BASE}/api/v1/projects?project_id=${projectId}&filename=${encodeURIComponent(file.name)}&storage_key=${storageKey}`,
    {
      method: 'POST',
      headers: { ...headers, 'X-Idempotency-Key': idempotencyKey },
    },
  )
  if (!createProjectResp.ok) throw new Error('Failed to create project')
  const { task_id: taskId } = await createProjectResp.json()

  return { projectId, taskId }
}

// legacy helper kept for reference; not used after backoff integration

// Poll variant that returns headers/status for backoff and does not throw on 429/503
async function pollOnceWithHeaders(
  projectId: string,
  getToken: () => Promise<string | null>,
  signal?: AbortSignal,
): Promise<{ data?: PollPayload; status: number; retryAfterMs?: number }> {
  const token = await getToken()
  if (!token) throw new Error('Not authenticated')
  const headers = { Authorization: `Bearer ${token}` }
  const resp = await fetch(`${API_BASE}/api/v1/projects/${projectId}`, { headers, signal })
  const status = resp.status
  let retryAfterMs: number | undefined
  const retryAfter = resp.headers.get('Retry-After')
  if (retryAfter) {
    const seconds = Number(retryAfter)
    if (!Number.isNaN(seconds)) retryAfterMs = seconds * 1000
  }
  if (!resp.ok) {
    try {
      // consume body to free connection
      await resp.text()
    } catch {
      // ignore
    }
    return { status, retryAfterMs }
  }
  const data = (await resp.json()) as PollPayload
  return { data, status, retryAfterMs }
}

function nextDelayDecorrelated(prevMs: number): number {
  const low = POLL_INITIAL_DELAY_MS
  const high = Math.max(low, prevMs * 3)
  const jitter = low + Math.random() * (high - low)
  return Math.min(POLL_MAX_DELAY_MS, Math.max(low, Math.floor(jitter)))
}

async function sleep(ms: number, signal?: AbortSignal) {
  return new Promise<void>((resolve, reject) => {
    const id = window.setTimeout(() => resolve(), ms)
    if (signal) {
      const onAbort = () => {
        window.clearTimeout(id)
        signal.removeEventListener('abort', onAbort)
        reject(new DOMException('Aborted', 'AbortError'))
      }
      signal.addEventListener('abort', onAbort)
    }
  })
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
  text_layer_url?: string
}

type EditsMap = Record<number, { en_text?: string }>

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
  updateEdit: (id: number, patch: { en_text?: string }) => void
  loadEditor: (projectId: string, result: PollPayload) => Promise<void>

  // Apply edits & exports
  applyingEdits: boolean
  pendingEditIds: number[]
  exports?: ExportPayload
  applyEdits: (getToken: () => Promise<string | null>) => Promise<void>
  downloadUrl: () => string | undefined
  downloadFile: (getToken: () => Promise<string | null>) => Promise<void>
  // internal: track active polling for cancellation
  pollController?: AbortController
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
    function resolveAssetPath(path: string | undefined): string | undefined {
      if (!path) return undefined
      if (/^https?:\/\//i.test(path)) return path
      if (path.startsWith('/')) return path
      // treat as storage_key
      return `/artifacts/${path}`
    }

    function extractTextLayerUrlFromResult(r: PollPayload): string | undefined {
      // 1) embedded in editor_data
      const ed = r.editor_data
      if (ed && typeof ed.text_layer_url === 'string' && ed.text_layer_url.length > 0) {
        return resolveAssetPath(ed.text_layer_url)
      }
      // 2) artifacts: direct common keys
      const artifacts = r.artifacts || {}
      const direct = artifacts['TEXT_LAYER'] || artifacts['text_layer_url'] || artifacts['TEXT_LAYER_URL']
      if (typeof direct === 'string' && direct.length > 0) return resolveAssetPath(direct)
      // 3) any key that contains "text_layer" (case-insensitive)
      for (const [k, v] of Object.entries(artifacts)) {
        if (k.toLowerCase().includes('text_layer') && typeof v === 'string' && v.length > 0) {
          return resolveAssetPath(v)
        }
      }
      // 4) any value that ends with text_layer.png
      for (const v of Object.values(artifacts)) {
        if (typeof v === 'string' && v.toLowerCase().endsWith('text_layer.png')) {
          return resolveAssetPath(v)
        }
      }
      return undefined
    }

    // Prefer payload embedded in the API response
    if (result.editor_data) {
      let saved: EditsMap = {}
      try {
        const raw = localStorage.getItem(`mf_edits_${projectId}`)
        if (raw) saved = JSON.parse(raw)
      } catch {
        // ignore storage failures in MVP
      }
      const bubbles = result.editor_data.bubbles.map((b) => {
        const e = saved[b.id]
        return e ? { ...b, en_text: e.en_text ?? b.en_text } : b
      })
      // discover server text layer URL from payload/artifacts
      const tl = extractTextLayerUrlFromResult(result)
      set({ editor: { ...result.editor_data, bubbles, text_layer_url: tl }, edits: saved })
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
        } catch {
          // ignore storage failures in MVP
        }
        const bubbles = payload.bubbles.map((b) => {
          const e = saved[b.id]
          return e ? { ...b, en_text: e.en_text ?? b.en_text } : b
        })
        set({ editor: { ...payload, bubbles }, edits: saved })
      } catch {
        // ignore fetch/parse failures
      }
    }
  },

  applyingEdits: false,
  pendingEditIds: [],
  exports: undefined,
  async applyEdits(getToken) {
    const projectId = get().projectId
    if (!projectId) return

    const editsMap = get().edits
    const editor = get().editor
    // Filter to only bubbles with actual text changes
    const editsArr = Object.entries(editsMap)
      .map(([id, patch]) => ({ id: Number(id), en_text: patch.en_text }))
      .filter(({ id, en_text }) => {
        if (!en_text || !editor) return false
        const bubble = editor.bubbles.find((b) => b.id === id)
        const current = (bubble?.en_text ?? '').trim()
        return en_text.trim() !== current
      })
    if (editsArr.length === 0) return

    set({ applyingEdits: true })
    try {
      const token = await getToken()
      if (!token) throw new Error('Not authenticated')
      const headers = { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` }

      const idempotencyKey = (crypto && 'randomUUID' in crypto) ? crypto.randomUUID() : `${projectId}-${Date.now()}`
      const resp = await fetch(`${API_BASE}/api/v1/projects/${projectId}`, {
        method: 'PUT',
        headers: { ...headers, 'X-Idempotency-Key': idempotencyKey },
        body: JSON.stringify({ edits: editsArr }),
      })
      if (!resp.ok) throw new Error('Apply edits failed')

      // mark all edited bubble ids as pending for spinner overlays
      set({ pendingEditIds: editsArr.map((e) => e.id) })

      // Cancel any existing polling
      const prev = get().pollController
      if (prev) prev.abort()
      const controller = new AbortController()
      set({ pollController: controller })

      const t0 = Date.now()
      let prevDelay = POLL_INITIAL_DELAY_MS
      let attempts = 0
      for (;;) {
        if (Date.now() - t0 > POLL_MAX_TIME_MS) break
        try {
          const { data, retryAfterMs } = await pollOnceWithHeaders(projectId, getToken, controller.signal)
          if (data) {
            set({ state: data.status, meta: data.meta })
            if (data.status === 'COMPLETED' || data.status === 'FAILED') {
              set({ result: data, state: data.status, pendingEditIds: [] })
              if (data.status === 'COMPLETED') {
                await get().loadEditor(projectId, data)
              }
              break
            }
          }
          // compute next delay
          const nextBase = POLL_INITIAL_DELAY_MS
          let next = nextBase
          if (attempts < POLL_FAST_PATH_ATTEMPTS) {
            next = nextBase
          } else {
            next = POLL_DECORRELATED_JITTER ? nextDelayDecorrelated(prevDelay) : Math.min(POLL_MAX_DELAY_MS, prevDelay * 2)
          }
          if (retryAfterMs !== undefined) next = Math.min(next, retryAfterMs)
          prevDelay = next
          await sleep(next, controller.signal)
          attempts++
        } catch (err) {
          // Abort stops loop
          if (err instanceof DOMException && err.name === 'AbortError') break
          // On transient errors, back off and continue
          const nextDelay = POLL_DECORRELATED_JITTER ? nextDelayDecorrelated(prevDelay) : Math.min(POLL_MAX_DELAY_MS, prevDelay * 2)
          prevDelay = nextDelay
          try { await sleep(nextDelay, controller.signal) } catch { break }
          attempts++
        }
      }
    } catch (err) {
      // surface error in store.error without breaking app
      const message = err instanceof Error ? err.message : String(err)
      set({ error: message, pendingEditIds: [] })
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

      // Cancel any existing polling
      const prev = get().pollController
      if (prev) prev.abort()
      const controller = new AbortController()
      set({ pollController: controller })

      const t0 = Date.now()
      let prevDelay = POLL_INITIAL_DELAY_MS
      let attempts = 0
      for (;;) {
        if (Date.now() - t0 > POLL_MAX_TIME_MS) break
        try {
          const { data, retryAfterMs } = await pollOnceWithHeaders(projectId, getToken, controller.signal)
          if (data) {
            set({ state: data.status, meta: data.meta })
            if (data.status === 'COMPLETED') {
              set({ result: data })
              await get().loadEditor(projectId, data)
              break
            } else if (data.status === 'FAILED') {
              set({ error: data.error || 'Task failed' })
              break
            }
          }
          const nextBase = POLL_INITIAL_DELAY_MS
          let next = nextBase
          if (attempts < POLL_FAST_PATH_ATTEMPTS) {
            next = nextBase
          } else {
            next = POLL_DECORRELATED_JITTER ? nextDelayDecorrelated(prevDelay) : Math.min(POLL_MAX_DELAY_MS, prevDelay * 2)
          }
          if (retryAfterMs !== undefined) next = Math.min(next, retryAfterMs)
          prevDelay = next
          await sleep(next, controller.signal)
          attempts++
        } catch (err) {
          if (err instanceof DOMException && err.name === 'AbortError') break
          const nextDelay = POLL_DECORRELATED_JITTER ? nextDelayDecorrelated(prevDelay) : Math.min(POLL_MAX_DELAY_MS, prevDelay * 2)
          prevDelay = nextDelay
          try { await sleep(nextDelay, controller.signal) } catch { break }
          attempts++
        }
      }
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
      pendingEditIds: [],
      pollController: undefined,
    })
  },
}))
