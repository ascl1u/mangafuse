import type { PollPayload, ProjectListResponse, BillingStatus } from './types'

const API_BASE = import.meta.env.VITE_API_BASE || 'http://127.0.0.1:8000'

export async function uploadAndStart(
	file: File,
	getToken: () => Promise<string | null>,
	depth: 'cleaned' | 'full' = 'full'
): Promise<{ projectId: string; taskId: string }> {
	const token = await getToken()
	if (!token) throw new Error('Not authenticated')
	const headers = { Authorization: `Bearer ${token}` }

	const uploadUrlResp = await fetch(`${API_BASE}/api/v1/projects/upload-url?filename=${encodeURIComponent(file.name)}`, {
		method: 'POST',
		headers,
	})
	if (!uploadUrlResp.ok) throw new Error('Failed to get upload URL')
	const { project_id: projectId, url: uploadUrl, storage_key: storageKey } = await uploadUrlResp.json()

	// For now keep the uploader contract in the store; this function only owns network API contracts
	// The caller will perform the actual upload where needed, but to keep behavior parity with the existing app,
	// we also support direct upload here when uploadUrl is an absolute URL.
	const isAbsolute = /^https?:\/\//i.test(uploadUrl)
	if (isAbsolute) {
		const resp = await fetch(uploadUrl, {
			method: 'PUT',
			body: file,
			headers: { 'Content-Type': file.type },
		})
		if (!resp.ok) throw new Error(`Upload failed (${resp.status})`)
	} else {
		// Local dev path expected; forward to backend with auth header
		const form = new FormData()
		form.append('file', file)
		const resp = await fetch(`${API_BASE}${uploadUrl}`, { method: 'POST', body: form, headers })
		if (!resp.ok) throw new Error('Upload failed')
	}

	const idempotencyKey = projectId
	const createProjectResp = await fetch(
		`${API_BASE}/api/v1/projects?project_id=${projectId}&filename=${encodeURIComponent(file.name)}&storage_key=${storageKey}&depth=${depth}`,
		{ method: 'POST', headers: { ...headers, 'X-Idempotency-Key': idempotencyKey } },
	)

	if (createProjectResp.status === 429 || createProjectResp.status === 403) {
		const errorData = await createProjectResp.json();
		throw new Error(errorData.detail || 'You have reached your project limit.');
	}

	if (!createProjectResp.ok) throw new Error('Failed to create project')
	const { task_id: taskId } = await createProjectResp.json()

	return { projectId, taskId }
}

export async function fetchProjectById(projectId: string, token?: string): Promise<PollPayload> {
  if (!projectId) throw new Error('projectId is required')

  const headers: HeadersInit = {}
  if (token) headers['Authorization'] = `Bearer ${token}`

  try {
    const resp = await fetch(`${API_BASE}/api/v1/projects/${encodeURIComponent(projectId)}`, {
      headers,
    })
    if (!resp.ok) {
      // Consume body to free the connection before throwing (ignore errors)
      await resp.text().catch(() => undefined)
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
export async function applyEditsAndWaitForRev(
	projectId: string,
	edits: Array<{ id: number; en_text?: string }>,
	startRev: number,
	getToken: () => Promise<string | null>,
): Promise<PollPayload> {
	if (!projectId) throw new Error('projectId is required')
	if (!Array.isArray(edits) || edits.length === 0) throw new Error('No edits to apply')
	const token = await getToken()
	if (!token) throw new Error('Not authenticated')
	const headers = { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` }

	const idempotencyKey = (crypto && 'randomUUID' in crypto) ? crypto.randomUUID() : `${projectId}-${Date.now()}`
	const resp = await fetch(`${API_BASE}/api/v1/projects/${projectId}`, {
		method: 'PUT',
		headers: { ...headers, 'X-Idempotency-Key': idempotencyKey },
		body: JSON.stringify({ edits }),
	})
	if (!resp.ok) throw new Error('Apply edits failed')

	// Short, bounded polling loop until revision increases or terminal state
	const t0 = Date.now()
	const MAX_MS = 15 * 1000
	let prevDelay = 1000
	async function pollOnce(): Promise<PollPayload | undefined> {
		const r = await fetch(`${API_BASE}/api/v1/projects/${projectId}`, { headers: { Authorization: `Bearer ${token}` } })
		if (!r.ok) return undefined
		return r.json()
	}
	for (;;) {
		if (Date.now() - t0 > MAX_MS) break
		const data = await pollOnce()
		if (data) {
			const rev = data.editor_data_rev ?? 0
			if (data.status === 'FAILED') return data
			if (data.status === 'COMPLETED' && rev > startRev) return data
		}
		await new Promise((res) => setTimeout(res, prevDelay))
		prevDelay = Math.min(5000, Math.max(1000, Math.floor(prevDelay * 1.5)))
	}
	// If we time out, surface the latest (may be stale); caller can decide UX
	return fetchProjectById(projectId, token)
}

export async function downloadZip(projectId: string, getToken: () => Promise<string | null>): Promise<void> {
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
}

export async function listProjects(
	getToken: () => Promise<string | null>,
	page = 1,
	limit = 20,
): Promise<ProjectListResponse> {
	const token = await getToken()
	if (!token) throw new Error('Not authenticated')
	const params = new URLSearchParams({ page: String(Math.max(1, page)), limit: String(Math.max(1, Math.min(50, limit))) })
	const resp = await fetch(`${API_BASE}/api/v1/projects?${params.toString()}`, {
		headers: { Authorization: `Bearer ${token}` },
	})
	if (!resp.ok) {
		await resp.text().catch(() => undefined)
		throw new Error(`Failed to list projects (${resp.status})`)
	}
	return resp.json()
}

export async function createCheckoutSession(getToken: () => Promise<string | null>): Promise<{ url: string }> {
	const token = await getToken()
	if (!token) throw new Error('Not authenticated')
	const resp = await fetch(`${API_BASE}/api/v1/billing/create-checkout-session`, {
		method: 'POST',
		headers: { Authorization: `Bearer ${token}` },
	})
	if (!resp.ok) throw new Error('Failed to create checkout session')
	return resp.json()
}

export async function createPortalSession(getToken: () => Promise<string | null>): Promise<{ url: string }> {
	const token = await getToken()
	if (!token) throw new Error('Not authenticated')
	const resp = await fetch(`${API_BASE}/api/v1/billing/create-portal-session`, {
		method: 'POST',
		headers: { Authorization: `Bearer ${token}` },
	})
	if (!resp.ok) throw new Error('Failed to create portal session')
	return resp.json()
}

export async function syncSubscription(getToken: () => Promise<string | null>): Promise<void> {
	const token = await getToken()
	if (!token) throw new Error('Not authenticated')
	const resp = await fetch(`${API_BASE}/api/v1/billing/sync-subscription`, {
		method: 'POST',
		headers: { Authorization: `Bearer ${token}` },
	})
	if (!resp.ok) {
		const errorData = await resp.json().catch(() => ({ detail: 'Failed to sync subscription' }))
		throw new Error(errorData.detail)
	}
}


export async function fetchBillingStatus(getToken: () => Promise<string | null>): Promise<BillingStatus> {
	const token = await getToken()
	if (!token) throw new Error('Not authenticated')
	const headers = { Authorization: `Bearer ${token}` }

	const resp = await fetch(`${API_BASE}/api/v1/billing/status`, { headers })
	if (!resp.ok) {
		throw new Error(`Failed to fetch billing status (${resp.status})`)
	}
	return resp.json()
}