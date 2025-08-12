import { useState } from 'react'
import { useAppStore } from './store'

function ProgressBar({ progress }: { progress: number }) {
  const pct = Math.max(0, Math.min(100, Math.round(progress * 100)))
  return (
    <div className="w-full h-2 bg-gray-200 rounded">
      <div className="h-2 bg-blue-600 rounded" style={{ width: `${pct}%` }} />
    </div>
  )
}

export default function App() {
  const [file, setFile] = useState<File | null>(null)
  const depth = useAppStore((s) => s.depth)
  const setDepth = useAppStore((s) => s.setDepth)
  const taskId = useAppStore((s) => s.taskId)
  const state = useAppStore((s) => s.state)
  const meta = useAppStore((s) => s.meta)
  const result = useAppStore((s) => s.result)
  const error = useAppStore((s) => s.error)
  const start = useAppStore((s) => s.start)

  const onSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!file) {
      alert('Please choose a file')
      return
    }
    await start(file)
  }

  const mainImageUrl = result?.final_url && depth === 'full' ? result.final_url : result?.cleaned_url
  const API_BASE = import.meta.env.VITE_API_BASE || 'http://127.0.0.1:8000'
  const toAbsolute = (url?: string) => (url ? `${API_BASE}${url}` : undefined)

  return (
    <div className="min-h-screen w-full">
      <div className="mx-auto max-w-5xl p-6">
        <h1 className="text-2xl font-semibold mb-4">MangaFuse — Upload</h1>

        <form onSubmit={onSubmit} className="space-y-4 bg-white p-4 rounded border">
          <div className="flex flex-col md:flex-row md:items-center gap-4">
            <input
              type="file"
              accept="image/*"
              onChange={(e) => setFile(e.target.files?.[0] || null)}
              className="block w-full text-sm text-gray-900 file:mr-4 file:py-2 file:px-4 file:rounded file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
              required
            />
            <div className="flex items-center gap-4">
              <label className="inline-flex items-center gap-2">
                <input
                  type="radio"
                  name="depth"
                  value="cleaned"
                  checked={depth === 'cleaned'}
                  onChange={() => setDepth('cleaned')}
                />
                <span>Cleaned only</span>
              </label>
              <label className="inline-flex items-center gap-2">
                <input
                  type="radio"
                  name="depth"
                  value="full"
                  checked={depth === 'full'}
                  onChange={() => setDepth('full')}
                />
                <span>Full</span>
              </label>
            </div>
            <button type="submit" className="px-4 py-2 rounded bg-blue-600 text-white hover:bg-blue-700 disabled:opacity-50">
              Start
            </button>
          </div>

          {taskId && (
            <div className="space-y-2">
              <div className="text-sm text-gray-600">Task: {taskId}</div>
              {meta?.progress !== undefined && <ProgressBar progress={meta.progress} />}
              <div className="text-sm">State: {state}{meta?.stage ? ` — ${meta.stage}` : ''}</div>
            </div>
          )}

          {error && (
            <div className="text-sm text-red-600">{error}</div>
          )}
        </form>

        {result && (
          <div className="mt-6 grid gap-6 md:grid-cols-[1fr,320px]">
            <div className="bg-white p-3 rounded border">
              {mainImageUrl ? (
                <img src={toAbsolute(mainImageUrl)} alt="Result" className="w-full h-auto" />
              ) : (
                <div className="text-gray-500">No image available</div>
              )}
            </div>
            <div className="bg-white p-3 rounded border space-y-2">
              <div className="font-medium">Artifacts</div>
              <ul className="text-sm list-disc pl-4 space-y-1">
                <li>
                  <a className="text-blue-700 hover:underline" href={toAbsolute(result.overlay_url)} target="_blank" rel="noreferrer">Segmentation overlay</a>
                </li>
                {result.cleaned_url && (
                  <li>
                    <a className="text-blue-700 hover:underline" href={toAbsolute(result.cleaned_url)} target="_blank" rel="noreferrer">Cleaned</a>
                  </li>
                )}
                {result.final_url && (
                  <li>
                    <a className="text-blue-700 hover:underline" href={toAbsolute(result.final_url)} target="_blank" rel="noreferrer">Final</a>
                  </li>
                )}
                <li>
                  <a className="text-blue-700 hover:underline" href={toAbsolute(result.json_url)} target="_blank" rel="noreferrer">text.json</a>
                </li>
                {result.typeset_debug_url && (
                  <li>
                    <a className="text-blue-700 hover:underline" href={toAbsolute(result.typeset_debug_url)} target="_blank" rel="noreferrer">Typeset debug</a>
                  </li>
                )}
              </ul>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

