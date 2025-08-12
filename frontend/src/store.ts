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
}

export const useAppStore = create<StoreState>((set, get) => ({
  depth: 'cleaned',
  setDepth: (d) => set({ depth: d }),

  taskId: '',
  state: 'PENDING',
  meta: undefined,
  result: undefined,
  error: undefined,

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
    set({ taskId: '', state: 'PENDING', meta: undefined, result: undefined, error: undefined })
  },
}))


