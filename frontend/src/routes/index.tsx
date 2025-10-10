import { createFileRoute, useNavigate } from '@tanstack/react-router'
import { useState } from 'react'
import { useAuth, SignedIn, SignedOut } from '@clerk/clerk-react'
import { useAppStore } from '../store'
import { uploadAndStart } from '../api'
import { GuestDemo } from '../components/GuestDemo'

// Home page: upload + mode toggle. Project viewing/editing happens at /projects/:projectId

function IndexPage() {
  const { getToken } = useAuth()
  const [file, setFile] = useState<File | null>(null)
  const [submitting, setSubmitting] = useState(false)
  const depth = useAppStore((s) => s.depth)
  const setDepth = useAppStore((s) => s.setDepth)
  const navigate = useNavigate()

  const onSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!file) {
      alert('Please choose a file')
      return
    }
    setSubmitting(true)
    try {
      const { projectId } = await uploadAndStart(file, getToken, depth)
      navigate({ to: '/projects/$projectId', params: { projectId } })
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Upload failed'
      alert(message)
    } finally {
      setSubmitting(false)
    }
  }

  // Removed sample project shortcut for authenticated users to avoid changing their flow

  return (
    <>
      <SignedOut>
        <GuestDemo />
      </SignedOut>

      <SignedIn>
        <form onSubmit={onSubmit} className="grid" style={{ gridTemplateColumns: '320px 1fr', columnGap: 24 }}>
          <div className="space-y-4">
            <div className="bg-white p-4 rounded border">
              <div className="font-medium mb-2">Upload a Page</div>
              <input
                type="file"
                accept="image/*"
                onChange={(e) => setFile(e.target.files?.[0] || null)}
                className="block w-full text-sm text-gray-900 disabled:opacity-50 disabled:cursor-not-allowed file:mr-4 file:py-2 file:px-4 file:rounded file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
                disabled={submitting}
                required
              />
            </div>
            <div className="bg-white p-4 rounded border">
              <div className="font-medium mb-2">Mode</div>
              <label className="inline-flex items-center gap-3 text-sm">
                <input
                  type="checkbox"
                  checked={depth === 'cleaned'}
                  onChange={(e) => setDepth(e.target.checked ? 'cleaned' : 'full')}
                  disabled={submitting}
                />
                <span>Clean text only</span>
              </label>
            </div>
            <div className="bg-white p-4 rounded border">
              <button type="submit" className="px-4 py-2 rounded bg-blue-600 text-white hover:bg-blue-700 disabled:opacity-50 w-full" disabled={!file || submitting}>
                {submitting ? 'Processing…' : 'Start Project'}
              </button>
            </div>
          </div>
          <div className="bg-white p-4 rounded border min-h-[300px]">
            <div className="font-medium mb-2">Preview</div>
            {file ? (
              <img src={URL.createObjectURL(file)} alt="Preview" className="w-full object-contain" style={{ maxHeight: '70vh' }} />
            ) : (
              <div className="text-sm text-gray-500 flex items-center justify-center h-full">No image uploaded</div>
            )}
          </div>
        </form>
      </SignedIn>
    </>
  )
}

export const Route = createFileRoute('/')({
  component: IndexPage,
})