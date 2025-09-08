import { useMemo } from 'react'
import type { EditorPayload } from '../types'

// Helper function to determine error type from error message
function getErrorType(errorMessage: string): 'translation' | 'typeset' {
  return errorMessage === "Translation failed" ? 'translation' : 'typeset'
}

// Helper function to get error styling based on type
function getErrorStyle(errorType: 'translation' | 'typeset'): { bgColor: string; textColor: string; borderColor: string } {
  return {
    translation: {
      bgColor: "bg-yellow-50",
      textColor: "text-yellow-700",
      borderColor: "border-yellow-200"
    },
    typeset: {
      bgColor: "bg-red-50",
      textColor: "text-red-700",
      borderColor: "border-red-200"
    }
  }[errorType]
}


type Props = {
  editor?: EditorPayload
  selectedId?: number
  edits: Record<number, { en_text?: string }>
  onChangeText: (id: number, value: string) => void
}

export function SelectedBubblePanel({ editor, selectedId, edits, onChangeText }: Props) {
  const API_BASE = import.meta.env.VITE_API_BASE || 'http://127.0.0.1:8000'
  const selected = useMemo(() => editor?.bubbles.find((b) => b.id === selectedId), [editor, selectedId])
  if (!editor) return null
  if (!selected) {
    return <p className="text-sm text-gray-600">Click a bubble on the image to edit text.</p>
  }
  const cropUrl = selected?.crop_url ? `${API_BASE}${selected.crop_url}` : `${API_BASE}${editor.image_url}`
  const patch = edits[selected.id] || {}
  const text = patch.en_text ?? selected.en_text ?? ''
  return (
    <div className="space-y-3">
      <div className="w-full h-44 bg-gray-100 overflow-hidden flex items-center justify-center">
        <img src={cropUrl} alt="Bubble" className="object-contain w-full h-full opacity-75" />
      </div>
      <div className="space-y-1">
        <label className="text-sm font-medium">Text</label>
        {selected.error && (
          <div className={`p-2 text-sm rounded border ${getErrorStyle(getErrorType(selected.error)).bgColor} ${getErrorStyle(getErrorType(selected.error)).textColor} ${getErrorStyle(getErrorType(selected.error)).borderColor}`}>
            <strong>Error:</strong> {selected.error}
          </div>
        )}
        <textarea
          className="w-full border rounded p-2 text-sm"
          rows={4}
          value={text}
          onChange={(e) => onChangeText(selected.id, e.target.value)}
        />
      </div>
    </div>
  )
}