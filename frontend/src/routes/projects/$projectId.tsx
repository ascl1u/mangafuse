import { createFileRoute } from '@tanstack/react-router'
import { useEffect, useMemo, useState } from 'react'
import { useAuth } from '@clerk/clerk-react'
import { applyEditsAndWaitForRev, downloadZip, fetchProjectById } from '../../api'
import type { EditorPayload, PollPayload } from '../../types'
import { EditorCanvas } from '../../editor/EditorCanvas'
import { SelectedBubblePanel } from '../../editor/SelectedBubblePanel'
import { useAppStore } from '../../store'

type RouterContext = { getToken: () => Promise<string | null> }

function resolveAssetPath(path: string | undefined): string | undefined {
  if (!path) return undefined
  if (/^https?:\/\//i.test(path)) return path
  if (path.startsWith('/')) return path
  return `/artifacts/${path}`
}

function extractTextLayerUrlFromResult(r: PollPayload): string | undefined {
  const artifacts = r.artifacts || {}
  const direct = (artifacts['TEXT_LAYER_PNG'] as unknown as string) || (artifacts['text_layer_url'] as unknown as string)
  if (typeof direct === 'string' && direct.length > 0) return resolveAssetPath(direct)
  return undefined
}

function seedEditorFromData(data: PollPayload | undefined): EditorPayload | undefined {
  if (!data?.editor_data) return undefined
  const tl = extractTextLayerUrlFromResult(data)
  return { ...data.editor_data, text_layer_url: tl }
}

function ProjectPage() {
  const { projectId } = Route.useParams()
  const data = Route.useLoaderData() as PollPayload
  const { getToken } = useAuth()

  // UI store only
  const selectedBubbleId = useAppStore((s) => s.selectedBubbleId)
  const setSelectedBubbleId = useAppStore((s) => s.setSelectedBubbleId)
  const edits = useAppStore((s) => s.edits)
  const updateEdit = useAppStore((s) => s.updateEdit)
  const resetProjectState = useAppStore((s) => s.resetProjectState)

  const [snapshot, setSnapshot] = useState<PollPayload>(data)
  const [editor, setEditor] = useState<EditorPayload | undefined>(() => seedEditorFromData(data))
  const [pendingEditIds, setPendingEditIds] = useState<number[]>([])
  const [updating, setUpdating] = useState(false)
  const [projectError, setProjectError] = useState<string | undefined>(undefined)

  // If loader changes (hard refresh or revalidation), update editor seed
  const seededFromLoader = useMemo(() => seedEditorFromData(data), [data])
  useEffect(() => {
    setEditor(seededFromLoader)
  }, [seededFromLoader])

  // Auto-poll while non-terminal: refresh snapshot until COMPLETED or FAILED
  useEffect(() => {
    const terminal = snapshot?.status === 'COMPLETED' || snapshot?.status === 'FAILED'
    if (terminal) return
    let alive = true
    const interval = window.setInterval(async () => {
      try {
        const token = await getToken()
        const next = await fetchProjectById(projectId, token || undefined)
        if (!alive) return
        setSnapshot(next)
      } catch {
        // ignore transient errors; try again on next tick
      }
    }, 5000)
    return () => {
      alive = false
      window.clearInterval(interval)
    }
  }, [projectId, getToken, snapshot?.status])

  // When snapshot reaches COMPLETED with editor_data, hydrate editor state
  useEffect(() => {
    if (snapshot?.status === 'COMPLETED' && snapshot.editor_data) {
      const seeded = seedEditorFromData(snapshot)
      if (seeded) setEditor(seeded)
    }
    // Check for errors and warnings regardless of project status
    if (snapshot?.error) {
      setProjectError(snapshot.error)
    } else if (snapshot?.completion_warnings) {
      setProjectError(snapshot.completion_warnings)
    }
  }, [snapshot])

  // Load saved edits for this project into the UI store
  useEffect(() => {
    try {
      const raw = localStorage.getItem(`mf_edits_${projectId}`)
      if (!raw) return
      const saved = JSON.parse(raw) as Record<number, { en_text?: string }>
      for (const [idStr, patch] of Object.entries(saved)) {
        const id = Number(idStr)
        if (!Number.isNaN(id)) updateEdit(id, patch)
      }
    } catch {
      // ignore corrupt storage
    }
  }, [projectId, updateEdit])

  // Persist edits per projectId whenever they change
  useEffect(() => {
    try {
      localStorage.setItem(`mf_edits_${projectId}`, JSON.stringify(edits))
    } catch {
      // ignore quota/storage errors in MVP
    }
  }, [projectId, edits])

  // Reset project-specific state when navigating to a different project
  useEffect(() => {
    resetProjectState()
  }, [projectId, resetProjectState])

  const status = snapshot?.status
  const stage = snapshot?.meta?.stage

  // Compute the text field values shown in the editor: user edit overrides payload
  const effectiveEditor = useMemo<EditorPayload | undefined>(() => {
    if (!editor) return undefined
    const bubbles = editor.bubbles.map((b) => ({
      ...b,
      en_text: (edits[b.id]?.en_text ?? b.en_text),
    }))
    return { ...editor, bubbles }
  }, [editor, edits])

  async function onApplyEdits() {
    if (!editor) return
    const diffs = Object.entries(edits)
      .map(([idStr, patch]) => ({ id: Number(idStr), en_text: patch.en_text }))
      .filter(({ id, en_text }) => {
        const bubble = editor.bubbles.find((b) => b.id === id)
        if (!bubble) return false
        if (en_text === undefined) return false
        return (en_text?.trim() ?? '') !== (bubble.en_text?.trim() ?? '')
      })
    if (diffs.length === 0) return
    setPendingEditIds(diffs.map((d) => d.id))
    setUpdating(true)
    try {
      const next = await applyEditsAndWaitForRev(projectId, diffs, data?.editor_data_rev ?? 0, getToken)
      // Update error state based on API response, prioritizing errors over warnings
      if (next.status === 'FAILED') {
        const apiError = next.error || 'Edit failed'
        setProjectError(apiError)
      } else if (next.completion_warnings) {
        // If the operation succeeded but has warnings, display them
        setProjectError(next.completion_warnings)
      } else {
        // If the operation succeeded and there are no warnings, clear any previous error
        setProjectError(undefined)
      }
      const seeded = seedEditorFromData(next)
      if (seeded) setEditor(seeded)
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err)
      setProjectError(message)
    } finally {
      setPendingEditIds([])
      setUpdating(false)
    }
  }

  async function onDownload() {
    try { await downloadZip(projectId, getToken) } catch (err) {
      const message = err instanceof Error ? err.message : String(err)
      alert(message)
    }
  }

  if (status !== 'COMPLETED' || !effectiveEditor) {
    const progress = typeof snapshot?.meta?.progress === 'number' ? Math.max(0, Math.min(1, snapshot.meta!.progress!)) : undefined
    return (
      <div className="max-w-2xl">
        <h1 className="text-xl font-semibold mb-3">Project {projectId}</h1>
        <div className="text-sm">Status: {stage || status}</div>
        {typeof progress === 'number' && (
          <div className="mt-3">
            <div className="w-full h-2 bg-gray-200 rounded" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(progress * 100)} role="progressbar">
              <div className="h-2 bg-blue-600 rounded" style={{ width: `${Math.round(progress * 100)}%` }} />
            </div>
          </div>
        )}
        {status === 'FAILED' && snapshot?.error && (
          <div className="mt-3 p-3 rounded border border-red-200 bg-red-50 text-sm text-red-700">
            <strong>Error:</strong> {snapshot.error}
          </div>
        )}
      </div>
    )
  }

  return (
    <div className="grid" style={{ gridTemplateColumns: '320px 1fr', columnGap: 24 }}>
      <div className="space-y-4">
        <div className="bg-white p-4 rounded border">
          <div className="font-medium mb-2">Download</div>
          <div className="flex items-center gap-2">
            <button
              className="px-3 py-2 rounded bg-gray-800 text-white disabled:opacity-50 disabled:cursor-not-allowed"
              onClick={onDownload}
              disabled={updating}
            >
              {updating ? 'Updating…' : 'Download'}
            </button>
          </div>
        </div>
        <div className="bg-white p-4 rounded border">
          <div className="font-medium mb-2">Actions</div>
          <button
            className="px-3 py-2 rounded bg-blue-600 text-white disabled:opacity-50"
            onClick={onApplyEdits}
            disabled={updating}
          >
            {updating ? 'Applying…' : 'Apply edits'}
          </button>
          {projectError && (
            <div className="mt-2 p-2 text-sm bg-red-50 text-red-700 rounded border border-red-200">
              <strong>Error:</strong> {projectError}
            </div>
          )}
        </div>
        <div className="bg-white p-4 rounded border">
          <div className="font-medium mb-2">Selected bubble</div>
          <SelectedBubblePanel
            editor={effectiveEditor}
            selectedId={selectedBubbleId}
            edits={edits}
            onChangeText={(id, value) => updateEdit(id, { en_text: value })}
          />
        </div>
      </div>
      <div className="bg-white p-2 rounded border min-w-0">
        <EditorCanvas
          editor={effectiveEditor}
          selectedId={selectedBubbleId}
          onSelect={setSelectedBubbleId}
          pendingEditIds={pendingEditIds}
          disabled={updating}
          hasEditFailed={!!projectError}
        />
      </div>
    </div>
  )
}

function ProjectError({ error }: { error: unknown }) {
  const { projectId } = Route.useParams()
  const message = error instanceof Error ? error.message : 'Unknown error'
  return (
    <div className="max-w-2xl">
      <h1 className="text-xl font-semibold mb-3">Project {projectId}</h1>
      <div className="p-3 rounded border border-red-200 bg-red-50 text-sm text-red-700">{message}</div>
    </div>
  )
}

export const Route = createFileRoute('/projects/$projectId')({
  loader: async ({ params, context }) => {
    console.log('[loader] fetching project', params.projectId)
    const token = await (context as RouterContext).getToken()
    const data = await fetchProjectById(params.projectId, token || undefined)
    return data
  },
  errorComponent: ProjectError,
  component: ProjectPage,
})


