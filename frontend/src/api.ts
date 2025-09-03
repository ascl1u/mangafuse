import type { PollPayload } from './store'

const API_BASE = import.meta.env.VITE_API_BASE || 'http://127.0.0.1:8000'

export async function fetchProjectById(projectId: string, token?: string): Promise<PollPayload> {
  if (!projectId) throw new Error('projectId is required')

  const headers: HeadersInit = {}
  if (token) headers['Authorization'] = `Bearer ${token}`

  try {
    const resp = await fetch(`${API_BASE}/api/v1/projects/${encodeURIComponent(projectId)}`, {
      headers,
    })
    if (!resp.ok) {
      // Consume body to free the connection before throwing
      try { await resp.text() } catch {}
      throw new Error(`Failed to fetch project (${resp.status})`)
    }
    return resp.json()
  } catch (err) {
    // Network errors (e.g., ECONNREFUSED) surface as TypeError in fetch
    const message = err instanceof Error ? err.message : String(err)
    if (message.includes('Failed to fetch') || err instanceof TypeError) {
      throw new Error('Service unavailable. Please try again shortly.')
    }
    throw err
  }
}


