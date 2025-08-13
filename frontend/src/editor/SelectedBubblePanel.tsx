import { useMemo } from 'react'
import type { EditorPayload } from '../store'

type Props = {
  editor?: EditorPayload
  selectedId?: number
  edits: Record<number, { en_text?: string; font_size?: number }>
  onChangeText: (id: number, value: string) => void
  onChangeFont: (id: number, value: number) => void
}

export function SelectedBubblePanel({ editor, selectedId, edits, onChangeText, onChangeFont }: Props) {
  const API_BASE = import.meta.env.VITE_API_BASE || 'http://127.0.0.1:8000'
  const selected = useMemo(() => editor?.bubbles.find((b) => b.id === selectedId), [editor, selectedId])
  if (!editor) return null
  if (!selected) {
    return <p className="text-sm text-gray-600">Click a bubble on the image to edit text.</p>
  }
  const cropUrl = selected?.crop_url ? `${API_BASE}${selected.crop_url}` : `${API_BASE}${editor.image_url}`
  const patch = edits[selected.id] || {}
  const text = patch.en_text ?? selected.en_text ?? ''
  const fontSize = Math.max(6, Math.min(48, patch.font_size ?? selected.font_size ?? 18))
  return (
    <div className="space-y-3">
      <div className="w-full h-44 bg-gray-100 overflow-hidden flex items-center justify-center">
        <img src={cropUrl} alt="Bubble" className="object-contain w-full h-full opacity-75" />
      </div>
      <div className="space-y-1">
        <label className="text-sm font-medium">Text</label>
        <textarea
          className="w-full border rounded p-2 text-sm"
          rows={4}
          value={text}
          onChange={(e) => onChangeText(selected.id, e.target.value)}
        />
      </div>
      <div className="space-y-1">
        <label className="text-sm font-medium">Font size: {fontSize}</label>
        <input
          type="range"
          min={6}
          max={48}
          step={1}
          value={fontSize}
          onChange={(e) => onChangeFont(selected.id, Number(e.target.value))}
          className="w-full"
        />
      </div>
    </div>
  )
}


