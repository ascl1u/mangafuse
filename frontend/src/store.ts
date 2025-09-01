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

type ProjectStatus = 'PENDING' | 'PROCESSING' | 'TRANSLATING' | 'UPDATING' | 'COMPLETED' | 'FAILED'
type EditorStatus = 'IDLE' | 'UPDATING' | 'EDIT_FAILED'

export type PollPayload = {
  project_id: string
  status: ProjectStatus
  task_state?: string
  meta?: { stage?: string; progress?: number }
  error?: string
  editor_data_rev?: number
  artifacts?: {
    [key: string]: string
  }
  editor_data?: EditorPayload
  editor_payload_url?: string
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
  error?: string
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
  submitting: boolean

  start: (file: File, getToken: () => Promise<string | null>) => Promise<void>
  reset: () => void

  // Editor state
  editor?: EditorPayload
  editorStatus: EditorStatus
  selectedBubbleId?: number
  setSelectedBubbleId: (id?: number) => void
  edits: EditsMap
  updateEdit: (id: number, patch: { en_text?: string }) => void
  loadEditor: (projectId: string, result: PollPayload, updatedBubbleIds?: number[]) => Promise<void>

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
  submitting: false,

  editor: undefined,
  editorStatus: 'IDLE',
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
  async loadEditor(projectId, result, updatedBubbleIds: number[] = []) {
    function resolveAssetPath(path: string | undefined): string | undefined {
      if (!path) return undefined
      if (/^https?:\/\//i.test(path)) return path
      if (path.startsWith('/')) return path
      // treat as storage_key
      return `/artifacts/${path}`
    }

    function extractTextLayerUrlFromResult(r: PollPayload): string | undefined {
      const artifacts = r.artifacts || {}
      const direct = artifacts['TEXT_LAYER_PNG'] || artifacts['text_layer_url']
      if (typeof direct === 'string' && direct.length > 0) return resolveAssetPath(direct)
      return undefined
    }

    const dataToLoad = result.editor_data
    if (dataToLoad) {
      let saved: EditsMap = {}
      try {
        const raw = localStorage.getItem(`mf_edits_${projectId}`)
        if (raw) saved = JSON.parse(raw)
      } catch {
        // ignore
      }
      const oldBubblesById = new Map(get().editor?.bubbles.map((b) => [b.id, b]));
      const bubbles = dataToLoad.bubbles.map((newBubble) => {
        const oldBubble = oldBubblesById.get(newBubble.id);
        const userEdit = saved[newBubble.id];
        let finalError = newBubble.error; // Trust the error from the API first.
        // If the API reports no error for this bubble, check if we should persist an old one.
        if (!finalError) {
            // If the bubble had an error before and it was NOT part of this specific update, keep the old error.
            if (oldBubble?.error && !updatedBubbleIds.includes(newBubble.id)) {
                finalError = oldBubble.error;
            }
        }

        return { 
          ...newBubble, 
          en_text: userEdit?.en_text ?? newBubble.en_text, 
          error: finalError,
        };
      });
      
      const tl = extractTextLayerUrlFromResult(result)
      set({ editor: { ...dataToLoad, bubbles, text_layer_url: tl }, edits: saved })
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
    const editsArr = Object.entries(editsMap)
      .map(([id, patch]) => ({ id: Number(id), en_text: patch.en_text }))
      .filter(({ id, en_text }) => {
        if (en_text === undefined || !editor) return false
        const bubble = editor.bubbles.find((b) => b.id === id)
        const current = (bubble?.en_text ?? '').trim()
        return en_text.trim() !== current
      })
    if (editsArr.length === 0) return

    set({ applyingEdits: true, editorStatus: 'UPDATING', error: undefined })
    // mark all edited bubble ids as pending immediately for spinner overlays
    set({ pendingEditIds: editsArr.map((e) => e.id) })
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
      // pendingEditIds already set; keep them until the new editor_data arrives
      // Cancel any existing polling to prevent duplicate attempts
      const prev = get().pollController
      if (prev) prev.abort()
      const controller = new AbortController()
      set({ pollController: controller })

      const t0 = Date.now()
      let prevDelay = POLL_INITIAL_DELAY_MS
      let attempts = 0
      // Wait until we see a COMPLETED with a higher editor_data_rev than before
      const startRev = get().result?.editor_data_rev ?? 0
      for (;;) {
        if (Date.now() - t0 > POLL_MAX_TIME_MS) break
        try {
          const { data, retryAfterMs } = await pollOnceWithHeaders(projectId, getToken, controller.signal)
          if (data) {
            set({ state: data.status, meta: data.meta });
            if (data.status === 'COMPLETED' || data.status === 'FAILED') {
              const idsInFlight = get().pendingEditIds;
              set({ result: data, state: data.status });
              
              // Only load the editor once the revision increases to avoid stale payloads
              const rev = data.editor_data_rev ?? 0
              if (data.status === 'COMPLETED' && rev <= startRev) {
                // keep polling until we see the new revision
              } else {
                await get().loadEditor(projectId, data, idsInFlight);
                set({ pendingEditIds: [] });
                if (data.status === 'FAILED') {
                  const apiError = data.error || 'Edit failed';
                  let userFriendlyError = apiError;
                  if (apiError.includes('typeset_failed')) {
                    userFriendlyError = 'Text is too long to fit in the errored bubbles.';
                  }
                  set({ error: userFriendlyError, editorStatus: 'EDIT_FAILED' });
                } else {
                  set({ editorStatus: 'IDLE' });
                }
                break;
              }
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
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err)
      set({ error: message, pendingEditIds: [], editorStatus: 'EDIT_FAILED' })
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
    set({ error: undefined, result: undefined, state: 'PENDING', submitting: true })
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
    } finally {
      set({ submitting: false })
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
      submitting: false,
      editor: undefined,
      selectedBubbleId: undefined,
      edits: {},
      pendingEditIds: [],
      pollController: undefined,
    })
  },
}))