import { useState } from 'react'
import { useAppStore } from './store'
import { EditorCanvas } from './editor/EditorCanvas'
import { SelectedBubblePanel } from './editor/SelectedBubblePanel'
import { SignedIn, SignedOut, SignInButton, SignUpButton, UserButton, useAuth } from '@clerk/clerk-react'

function ProgressBar({ progress }: { progress: number }) {
  const pct = Math.max(0, Math.min(100, Math.round(progress * 100)))
  return (
    <div className="w-full h-2 bg-gray-200 rounded">
      <div className="h-2 bg-blue-600 rounded" style={{ width: `${pct}%` }} />
    </div>
  )
}

export default function App() {
  const { isSignedIn, getToken } = useAuth()
  const [file, setFile] = useState<File | null>(null)
  const depth = useAppStore((s) => s.depth)
  const setDepth = useAppStore((s) => s.setDepth)
  const projectId = useAppStore((s) => s.projectId)
  const state = useAppStore((s) => s.state)
  const meta = useAppStore((s) => s.meta)
  const error = useAppStore((s) => s.error)
  const start = useAppStore((s) => s.start)
  const submitting = useAppStore((s) => s.submitting)

  const onSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!file) {
      alert('Please choose a file')
      return
    }
    await start(file, getToken)
  }

  const editor = useAppStore((s) => s.editor)
  const edits = useAppStore((s) => s.edits)
  const selectedBubbleId = useAppStore((s) => s.selectedBubbleId)
  const setSelectedBubbleId = useAppStore((s) => s.setSelectedBubbleId)
  const updateEdit = useAppStore((s) => s.updateEdit)
  const applyEdits = useAppStore((s) => s.applyEdits)
  const downloadFile = useAppStore((s) => s.downloadFile)
  const pendingEditIds = useAppStore((s) => s.pendingEditIds)
  const editorStatus = useAppStore((s) => s.editorStatus)
  const applyingEdits = useAppStore((s) => s.applyingEdits)

  const showEditor = !!editor

  return (
    <div className="min-h-screen w-full bg-gray-50">
      <header className="w-full bg-white border-b">
        <div className="mx-auto max-w-[1280px] px-6 h-16 flex items-center justify-between">
          <div className="font-semibold">MangaFuse</div>
          <div className="flex items-center gap-3">
            <a href="#" aria-label="X" className="text-gray-600">X</a>
            <a href="#" className="text-gray-600">Credits</a>
            <SignedOut>
              <SignInButton />
              <SignUpButton />
            </SignedOut>
            <SignedIn>
              <UserButton />
            </SignedIn>
          </div>
        </div>
      </header>
      <div className="w-full bg-white border-b">
        <div className="mx-auto max-w-[1280px] px-6 h-10 flex items-center text-gray-500 gap-3">
          <div className="px-3 py-1 rounded border bg-gray-100 opacity-60 cursor-not-allowed">Current job</div>
          <button onClick={() => window.location.reload()} className="px-3 py-1.5 rounded bg-blue-600 text-white">New job</button>
        </div>
      </div>

      <main className="mx-auto max-w-[1280px] px-6 py-6">
        {!showEditor && (
          <form onSubmit={onSubmit} className="grid" style={{ gridTemplateColumns: '320px 1fr', columnGap: 24 }}>
            <div className="space-y-4">
              <div className="bg-white p-4 rounded border">
                <div className="font-medium mb-2">Upload</div>
                <SignedIn>
                  <input
                    type="file"
                    accept="image/*"
                    onChange={(e) => setFile(e.target.files?.[0] || null)}
                    className="block w-full text-sm text-gray-900 disabled:opacity-50 disabled:cursor-not-allowed file:mr-4 file:py-2 file:px-4 file:rounded file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
                    disabled={!isSignedIn || submitting}
                    required
                  />
                </SignedIn>
                <SignedOut>
                  <div className="text-sm text-gray-600">Please sign in to upload and process pages.</div>
                </SignedOut>
              </div>
              <div className="bg-white p-4 rounded border">
                <div className="font-medium mb-2">Mode</div>
                <label className="inline-flex items-center gap-3 text-sm">
                  <input
                    type="checkbox"
                    checked={depth === 'cleaned'}
                    onChange={(e) => setDepth(e.target.checked ? 'cleaned' : 'full')}
                  />
                  <span>Clean text only</span>
                </label>
              </div>
              <div className="bg-white p-4 rounded border">
                <button type="submit" className="px-4 py-2 rounded bg-blue-600 text-white hover:bg-blue-700 disabled:opacity-50 w-full" disabled={!isSignedIn || submitting}>
                  {submitting ? 'Submitting…' : 'Submit'}
                </button>
                {projectId && (
                  <div className="mt-3 space-y-2">
                    <div className="text-sm text-gray-600">Project: {projectId}</div>
                    {meta?.progress !== undefined && <ProgressBar progress={meta.progress} />}
                    <div className="text-sm">State: {state === 'TRANSLATING' ? 'Translating…' : state}{meta?.stage ? ` — ${meta.stage}` : ''}</div>
                  </div>
                )}
                {error && (
                  <div className="text-sm text-red-600 mt-2">{error}</div>
                )}
              </div>
            </div>
            <div className="bg-white p-4 rounded border min-h-[300px]">
              <div className="font-medium mb-2">Preview</div>
              {file ? (
                <img src={URL.createObjectURL(file)} alt="Preview" className="w-full object-contain" style={{ maxHeight: '70vh' }} />
              ) : (
                <div className="text-sm text-gray-500">No image uploaded</div>
              )}
            </div>
          </form>
        )}

        {showEditor && editor && (
          <div className="grid" style={{ gridTemplateColumns: '320px 1fr', columnGap: 24 }}>
            <div className="space-y-4">
              <div className="bg-white p-4 rounded border">
                <div className="font-medium mb-2">Download</div>
                <div className="flex items-center gap-2">
                  <button
                    className="px-3 py-2 rounded bg-gray-800 text-white disabled:opacity-50 disabled:cursor-not-allowed"
                    onClick={() => downloadFile(getToken)}
                    disabled={editorStatus === 'UPDATING' || applyingEdits}
                  >
                    {editorStatus === 'UPDATING' ? 'Updating…' : 'Download'}
                  </button>
                </div>
              </div>
              <div className="bg-white p-4 rounded border">
                <div className="font-medium mb-2">Actions</div>
                <button
                  className="px-3 py-2 rounded bg-blue-600 text-white disabled:opacity-50"
                  onClick={() => applyEdits(getToken)}
                  disabled={editorStatus === 'UPDATING'}
                >
                  {editorStatus === 'UPDATING' ? 'Applying…' : 'Apply edits'}
                </button>
                {editorStatus === 'EDIT_FAILED' && (
                    <div className="mt-2 p-2 text-sm bg-red-50 text-red-700 rounded border border-red-200">
                        <strong>Error:</strong> {error}
                    </div>
                )}
              </div>
              <div className="bg-white p-4 rounded border">
                <div className="font-medium mb-2">Selected bubble</div>
                <SelectedBubblePanel
                  editor={editor}
                  selectedId={selectedBubbleId}
                  edits={edits}
                  onChangeText={(id, value) => updateEdit(id, { en_text: value })}
                />
              </div>
            </div>
            <div className="bg-white p-2 rounded border">
              <EditorCanvas
                editor={editor}
                selectedId={selectedBubbleId}
                onSelect={setSelectedBubbleId}
                pendingEditIds={pendingEditIds}
                disabled={editorStatus === 'UPDATING'}
                hasEditFailed={editorStatus === 'EDIT_FAILED'}
              />
            </div>
          </div>
        )}
      </main>
    </div>
  )
}